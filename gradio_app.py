
import os
import json
import time
import secrets
import logging
import warnings
from pathlib import Path
from typing import Tuple, Any
from functools import partial

# see lines ~40-50 for running on lower VRAM GPUs

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import gradio as gr
import torch
import torchaudio

TEXT_PRESETS_PATH = Path("./text_presets.txt")
SAMPLER_PRESETS_PATH = Path("./sampler_presets.json")
DEVICE_CONFIG_PATH = Path("./device_config.json")
RUNTIME_CONFIG_PATH = Path("./runtime_config.json")

from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    ae_reconstruct,
    sample_pipeline,
    compile_model,
    compile_fish_ae,
    sample_euler_cfg_independent_guidances,
    sample_euler_cfg_joint_unconditional,
    sample_euler_cfg_apg_independent,
)

def load_runtime_config() -> dict:
    """Load runtime configuration options (like low VRAM mode).

    This is separate from device selection and can include flags that affect
    how models are loaded (e.g., dtypes) and generation defaults.
    """
    if RUNTIME_CONFIG_PATH.exists():
        try:
            with open(RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def save_runtime_config(config: dict) -> None:
    try:
        with open(RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f)
    except Exception:
        # Non-fatal
        pass


_runtime_cfg = load_runtime_config()
LOW_VRAM_DEFAULT = bool(_runtime_cfg.get("low_vram_mode", True))

# Configuration
if LOW_VRAM_DEFAULT:
    # 8GB-friendly defaults
    MODEL_DTYPE = torch.bfloat16
    FISH_AE_DTYPE = torch.bfloat16
    DEFAULT_SAMPLE_LATENT_LENGTH = 576  # ~27s (safer default on 8GB)
else:
    # Higher precision mode (more VRAM, slightly higher quality)
    MODEL_DTYPE = torch.float32
    FISH_AE_DTYPE = torch.float32

    # Prefer the full in-distribution length when not in Low VRAM mode.
    DEFAULT_SAMPLE_LATENT_LENGTH = 640

#EFAULT_SAMPLE_LATENT_LENGTH = 640 # decrease if OOM on 8GB vram GPU

# Safety limits
# 640 is the model's in-distribution max (~30s). Low VRAM mode sets a smaller
# *default* length, but we should not silently cap below 640 or long text will
# truncate unexpectedly.
MAX_SAFE_LATENT_LENGTH = 640
MAX_SAFE_STEPS = 40
MAX_SPEAKER_SECONDS = 120  # max duration of reference audio passed to the model

# NOTE peak S1-DAC decoding VRAM > peak latent sampling VRAM, so decoding in chunks (which is posisble as S1-DAC is causal) would allow for full 640-length generation on lower VRAM GPUs

# --------------------------------------------------------------------

# Audio Prompt Library for Custom Audio Panel (included in repo)
AUDIO_PROMPT_FOLDER = Path("./audio_prompts")

# --------------------------------------------------------------------

TEMP_AUDIO_DIR = Path("./temp_gradio_audio")
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Device selection & model loading

def get_available_devices() -> list[str]:
    """Return a list of device labels: CPU + available CUDA devices.

    Labels are of the form "CPU" and "GPU 0 (cuda:0)" etc. The first CUDA
    device (if any) is considered the default when no config is present.
    """
    devices: list[str] = ["CPU"]
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            devices.append(f"GPU {idx} (cuda:{idx}) - {name}")
    return devices


def _device_label_to_torch(device_label: str) -> str:
    """Map a human-readable device label from the dropdown to a torch device string."""
    if not device_label:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_label.upper() == "CPU":
        return "cpu"
    if "cuda:" in device_label:
        # Expect substring like "cuda:0" inside the label
        start = device_label.find("cuda:")
        if start != -1:
            # Grab up to the next closing parenthesis or end of string
            end = device_label.find(")", start)
            if end == -1:
                return device_label[start:]
            return device_label[start:end]
    # Fallbacks
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_device_config() -> str:
    """Load persisted device label from config file, defaulting to best available.

    Returns a label that should match one of the choices from get_available_devices().
    If the stored label is invalid, returns a safe default.
    """
    devices = get_available_devices()
    default_label = devices[1] if len(devices) > 1 else devices[0]

    if DEVICE_CONFIG_PATH.exists():
        try:
            with open(DEVICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            label = data.get("device_label")
            if isinstance(label, str) and label in devices:
                return label
        except Exception:
            pass

    return default_label


def save_device_config(device_label: str) -> None:
    """Persist the chosen device label so it is reused on next startup (UI & API)."""
    try:
        with open(DEVICE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"device_label": device_label}, f)
    except Exception:
        # Non-fatal – app should still run even if config can't be written
        pass


def _init_models_for_device(device_label: str):
    """(Re)load models for the specified device label.

    This function updates global model, fish_ae, and pca_state variables, and is
    used both at startup and when the device is changed via the UI. It ensures
    that API usage always reflects the current, persisted device.
    """
    global model, fish_ae, pca_state

    torch_device = _device_label_to_torch(device_label)

    # Free any existing CUDA allocations before reloading on a different GPU
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    model = load_model_from_hf(device=torch_device, dtype=MODEL_DTYPE, delete_blockwise_modules=True)
    fish_ae = load_fish_ae_from_hf(device=torch_device, dtype=FISH_AE_DTYPE)
    pca_state = load_pca_state_from_hf(device=torch_device)


def _init_models_for_torch_device(torch_device: str):
    """(Re)load models directly for a given torch device string.

    This is used by the OOM fallback path, which works with raw torch
    device strings (e.g., "cuda:1" or "cpu") instead of UI labels.
    """
    global model, fish_ae, pca_state

    # Best-effort cache clear before moving to a different CUDA device
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    model = load_model_from_hf(device=torch_device, dtype=MODEL_DTYPE, delete_blockwise_modules=True)
    fish_ae = load_fish_ae_from_hf(device=torch_device, dtype=FISH_AE_DTYPE)
    pca_state = load_pca_state_from_hf(device=torch_device)


# Initialize device & models at import time so API users get a ready model
INITIAL_DEVICE_LABEL = load_device_config()
_init_models_for_device(INITIAL_DEVICE_LABEL)

model_compiled = None
fish_ae_compiled = None


def _get_device_priority_list(primary_device: str | None) -> list[str]:
    """Return a prioritized list of devices to try for generation.

    Order: primary device, then other CUDA devices, then CPU.
    """
    devices: list[str] = []

    if primary_device:
        devices.append(primary_device)
    else:
        if torch.cuda.is_available():
            devices.append("cuda")
        else:
            devices.append("cpu")

    # Add other CUDA devices (if any) that aren't already listed
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            dev = f"cuda:{idx}"
            if dev not in devices:
                devices.append(dev)

    # Always allow CPU as a final fallback
    if "cpu" not in devices:
        devices.append("cpu")

    return devices

# --------------------------------------------------------------------
# Helper functions
def make_stem(prefix: str, user_id: str | None = None) -> str:
    """Create unique filename stem: prefix__user__timestamp_random or prefix__timestamp_random if no user_id."""
    ts = int(time.time() * 1000)
    rand = secrets.token_hex(4)
    if user_id:
        return f"{prefix}__{user_id}__{ts}_{rand}"
    return f"{prefix}__{ts}_{rand}"


def cleanup_temp_audio(dir_: Path, user_id: str | None, max_age_sec: int = 60 * 5):
    """Remove old files globally and all previous files for this user."""
    now = time.time()

    for p in dir_.glob("*"):
        try:
            if p.is_file() and (now - p.stat().st_mtime) > max_age_sec:
                p.unlink(missing_ok=True)
        except Exception:
            pass

    if user_id:
        for p in dir_.glob(f"*__{user_id}__*"):
            try:
                if p.is_file():
                    p.unlink(missing_ok=True)
            except Exception:
                pass


def save_audio_with_format(audio_tensor: torch.Tensor, base_path: Path, filename: str, sample_rate: int, audio_format: str) -> Path:
    """Save audio in specified format, fallback to WAV if MP3 encoding fails."""
    # Ensure tensor is on CPU and 2D (channels, time) for torchaudio
    audio_tensor = audio_tensor.detach().cpu()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    if audio_format == "mp3":
        try:
            output_path = base_path / f"{filename}.mp3"
            torchaudio.save(
                str(output_path),
                audio_tensor,
                sample_rate,
                format="mp3",
                encoding="mp3",
                bits_per_sample=None,
            )
            return output_path
        except Exception as e:
            print(f"MP3 encoding failed: {e}, falling back to WAV")
            output_path = base_path / f"{filename}.wav"
            # Try torchaudio first; if no backend, fall back to ffmpeg-python
            try:
                torchaudio.save(str(output_path), audio_tensor, sample_rate)
                return output_path
            except Exception as e2:
                print(f"torchaudio WAV save failed ({e2}), trying ffmpeg fallback")

                import ffmpeg

                tmp_raw = audio_tensor.squeeze(0).numpy().astype("float32").tobytes()
                ffmpeg.input(
                    "pipe:",
                    format="f32le",
                    ac=1,
                    ar=sample_rate,
                ).output(
                    str(output_path),
                    format="wav",
                    acodec="pcm_s16le",
                ).run(input=tmp_raw, capture_stdout=True, capture_stderr=True, quiet=True)
                return output_path

    # Default: save as WAV
    output_path = base_path / f"{filename}.wav"
    try:
        torchaudio.save(str(output_path), audio_tensor, sample_rate)
    except Exception as e:
        print(f"torchaudio WAV save failed ({e}), trying ffmpeg fallback")

        import ffmpeg

        tmp_raw = audio_tensor.squeeze(0).numpy().astype("float32").tobytes()
        ffmpeg.input(
            "pipe:",
            format="f32le",
            ac=1,
            ar=sample_rate,
        ).output(
            str(output_path),
            format="wav",
            acodec="pcm_s16le",
        ).run(input=tmp_raw, capture_stdout=True, capture_stderr=True, quiet=True)

    return output_path


def to_bool(val: Any) -> bool:
    """Parse truthy values from common string/bool inputs."""
    return str(val).strip().lower() not in {"", "0", "false", "off", "none", "no"}


def find_min_bucket_gte(values_str: str, actual_length: int) -> int | None:
    """Parse comma-separated values and find minimum value >= actual_length.
    
    If a single value is provided (no comma), returns that value directly.
    If comma-separated, finds the smallest bucket that can fit the content.
    Returns None if empty string.
    """
    if not values_str or not values_str.strip():
        return None
    
    values_str = values_str.strip()
    
    # Single value case - return as-is
    if "," not in values_str:
        return int(values_str)
    
    # Multiple values - find minimum >= actual_length
    values = [int(v.strip()) for v in values_str.split(",") if v.strip()]
    if not values:
        return None
    
    # Find minimum value >= actual_length
    candidates = [v for v in values if v >= actual_length]
    if candidates:
        return min(candidates)
    
    # If no value is >=, return the maximum (best effort)
    return max(values)


def synthesize_to_file(
    text_prompt: str,
    speaker_audio_path: str,
    num_steps: int,
    rng_seed: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float,
    rescale_k: float,
    rescale_sigma: float,
    force_speaker: bool,
    speaker_kv_scale: float,
    speaker_kv_min_t: float,
    speaker_kv_max_layers: int,
    reconstruct_first_30_seconds: bool,
    use_custom_shapes: bool,
    max_text_byte_length: str,
    max_speaker_latent_length: str,
    sample_latent_length: str,
    audio_format: str,
    use_compile: bool,
    show_original_audio: bool,
    session_id: str,
    fade_out_seconds: float = 0.0,
    cfg_mode: str = "independent",
):
    """Core generation logic assuming models are already on the correct device.

    This function performs TTS synthesis and writes the main generated audio to a
    file under TEMP_AUDIO_DIR, returning the output Path and related metadata.
    It is used by both the Gradio UI wrapper and the HTTP API server.
    """
    global model_compiled, fish_ae_compiled

    if use_compile:
        if model_compiled is None:
            try:
                model_compiled = compile_model(model)
                fish_ae_compiled = compile_fish_ae(fish_ae)
            except Exception as e:
                print(f"Compilation wrapping failed: {str(e)}")
                model_compiled = None
                fish_ae_compiled = None
                use_compile = False

    active_model = model_compiled if use_compile else model
    active_fish_ae = fish_ae_compiled if use_compile else fish_ae

    cleanup_temp_audio(TEMP_AUDIO_DIR, session_id)

    start_time = time.time()

    # Clamp steps to a range that works well on <=8GB GPUs
    num_steps_int = min(max(int(num_steps), 5), MAX_SAFE_STEPS)
    rng_seed_int = int(rng_seed) if rng_seed is not None else 0
    cfg_scale_text_val = float(cfg_scale_text)
    cfg_scale_speaker_val = float(cfg_scale_speaker) if cfg_scale_speaker is not None else None
    cfg_min_t_val = float(cfg_min_t)
    cfg_max_t_val = float(cfg_max_t)
    truncation_factor_val = float(truncation_factor)
    rescale_k_val = float(rescale_k) if rescale_k != 1.0 else None
    rescale_sigma_val = float(rescale_sigma)

    speaker_kv_enabled = bool(force_speaker)
    if speaker_kv_enabled:
        speaker_kv_scale_val = float(speaker_kv_scale) if speaker_kv_scale is not None else None
        speaker_kv_min_t_val = float(speaker_kv_min_t) if speaker_kv_min_t is not None else None
        speaker_kv_max_layers_val = int(speaker_kv_max_layers) if speaker_kv_max_layers is not None else None
    else:
        speaker_kv_scale_val = None
        speaker_kv_min_t_val = None
        speaker_kv_max_layers_val = None

    # Load speaker audio early so we can compute actual lengths for bucket selection
    use_zero_speaker = not speaker_audio_path or speaker_audio_path == ""
    speaker_audio = (
        load_audio(speaker_audio_path, max_duration=MAX_SPEAKER_SECONDS).cuda()
        if not use_zero_speaker
        else None
    )

    if use_custom_shapes:
        # Compute actual text byte length
        actual_text_byte_length = len(text_prompt.encode("utf-8")) + 1  # +1 for BOS token
        
        # Compute actual speaker latent length (audio_samples // 2048)
        AE_DOWNSAMPLE_FACTOR = 2048
        if speaker_audio is not None:
            actual_speaker_latent_length = (speaker_audio.shape[-1] // AE_DOWNSAMPLE_FACTOR) // 4 * 4
        else:
            actual_speaker_latent_length = 0
        
        # Find appropriate bucket sizes from comma-separated values
        pad_to_max_text_length = find_min_bucket_gte(max_text_byte_length, actual_text_byte_length)
        pad_to_max_speaker_latent_length = find_min_bucket_gte(max_speaker_latent_length, actual_speaker_latent_length)
        sample_latent_length_val = int(sample_latent_length) if sample_latent_length.strip() else (DEFAULT_SAMPLE_LATENT_LENGTH or 640)
    else:
        pad_to_max_text_length = None
        pad_to_max_speaker_latent_length = None
        sample_latent_length_val = DEFAULT_SAMPLE_LATENT_LENGTH or 640

    # Hard safety cap (in-distribution max length)
    sample_latent_length_val = min(sample_latent_length_val, MAX_SAFE_LATENT_LENGTH)

    cfg_mode_norm = (cfg_mode or "independent").strip().lower()
    if cfg_mode_norm == "joint-unconditional":
        sampler = sample_euler_cfg_joint_unconditional
    elif cfg_mode_norm == "apg-independent":
        sampler = sample_euler_cfg_apg_independent
    else:
        sampler = sample_euler_cfg_independent_guidances

    sample_fn = partial(
        sampler,
        num_steps=num_steps_int,
        cfg_scale_text=cfg_scale_text_val,
        cfg_scale_speaker=cfg_scale_speaker_val,
        cfg_min_t=cfg_min_t_val,
        cfg_max_t=cfg_max_t_val,
        truncation_factor=truncation_factor_val,
        rescale_k=rescale_k_val,
        rescale_sigma=rescale_sigma_val,
        speaker_kv_scale=speaker_kv_scale_val,
        speaker_kv_min_t=speaker_kv_min_t_val,
        speaker_kv_max_layers=speaker_kv_max_layers_val,
        sequence_length=sample_latent_length_val,
    )

    audio_out, normalized_text = sample_pipeline(
        model=active_model,
        fish_ae=active_fish_ae,
        pca_state=pca_state,
        sample_fn=sample_fn,
        text_prompt=text_prompt,
        speaker_audio=speaker_audio,
        rng_seed=rng_seed_int,
        pad_to_max_text_length=pad_to_max_text_length,
        pad_to_max_speaker_latent_length=pad_to_max_speaker_latent_length,
        normalize_text=True,
    )

    audio_to_save = audio_out[0].cpu()

    # Optional short silence before fade-out to give a natural pause at the end.
    tail_silence_seconds = 0.12  # small pause, tuned for Voxta utterances
    sr = 44100
    if tail_silence_seconds > 0.0:
        silence_samples = int(tail_silence_seconds * sr)
        if silence_samples > 0:
            import torch

            silence = torch.zeros_like(audio_to_save[..., :silence_samples])
            audio_to_save = torch.cat([audio_to_save, silence], dim=-1)

    # Optional fade-out at the end to suppress any residual tail noise.
    if fade_out_seconds and fade_out_seconds > 0.0:
        # Desired fade length based on requested fade duration
        desired_fade_len = max(1, int(fade_out_seconds * sr))
        num_samples = audio_to_save.shape[-1]

        # Never try to fade more samples than we actually have
        fade_len = min(desired_fade_len, num_samples)

        if fade_len > 1:
            import torch

            fade = torch.linspace(
                1.0,
                0.0,
                steps=fade_len,
                device=audio_to_save.device,
                dtype=audio_to_save.dtype,
            )

            # Apply fade along the last dimension only (time axis)
            audio_to_save[..., -fade_len:] *= fade

    stem = make_stem("generated", session_id)
    output_path = save_audio_with_format(audio_to_save, TEMP_AUDIO_DIR, stem, 44100, audio_format)

    generation_time = time.time() - start_time
    recon_output_path = None
    original_output_path = None

    if reconstruct_first_30_seconds and speaker_audio is not None:
        audio_recon = ae_reconstruct(
            fish_ae=fish_ae,
            pca_state=pca_state,
            audio=torch.nn.functional.pad(
                speaker_audio[..., :2048 * 640],
                (0, max(0, 2048 * 640 - speaker_audio.shape[-1])),
            )[None],
        )[..., : speaker_audio.shape[-1]]

        recon_stem = make_stem("speaker_recon", session_id)
        recon_output_path = save_audio_with_format(audio_recon.cpu()[0], TEMP_AUDIO_DIR, recon_stem, 44100, audio_format)

    if show_original_audio and speaker_audio is not None:
        original_stem = make_stem("original_audio", session_id)
        original_output_path = save_audio_with_format(speaker_audio.cpu(), TEMP_AUDIO_DIR, original_stem, 44100, audio_format)

    return output_path, normalized_text, generation_time, original_output_path, recon_output_path, speaker_audio


def _generate_audio_on_active_model(
    text_prompt: str,
    speaker_audio_path: str,
    num_steps: int,
    rng_seed: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float,
    rescale_k: float,
    rescale_sigma: float,
    force_speaker: bool,
    speaker_kv_scale: float,
    speaker_kv_min_t: float,
    speaker_kv_max_layers: int,
    reconstruct_first_30_seconds: bool,
    use_custom_shapes: bool,
    max_text_byte_length: str,
    max_speaker_latent_length: str,
    sample_latent_length: str,
    audio_format: str,
    use_compile: bool,
    show_original_audio: bool,
    session_id: str,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Wrapper around synthesize_to_file that formats Gradio UI outputs."""

    output_path, normalized_text, generation_time, original_output_path, recon_output_path, speaker_audio = synthesize_to_file(
        text_prompt=text_prompt,
        speaker_audio_path=speaker_audio_path,
        num_steps=num_steps,
        rng_seed=rng_seed,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        truncation_factor=truncation_factor,
        rescale_k=rescale_k,
        rescale_sigma=rescale_sigma,
        force_speaker=force_speaker,
        speaker_kv_scale=speaker_kv_scale,
        speaker_kv_min_t=speaker_kv_min_t,
        speaker_kv_max_layers=speaker_kv_max_layers,
        reconstruct_first_30_seconds=reconstruct_first_30_seconds,
        use_custom_shapes=use_custom_shapes,
        max_text_byte_length=max_text_byte_length,
        max_speaker_latent_length=max_speaker_latent_length,
        sample_latent_length=sample_latent_length,
        audio_format=audio_format,
        use_compile=use_compile,
        show_original_audio=show_original_audio,
        session_id=session_id,
    )

    time_str = f"⏱️ Total generation time: {generation_time:.2f}s"
    text_display = f"**Text Prompt (normalized):**\n\n{normalized_text}"
    show_reference_section = (show_original_audio or reconstruct_first_30_seconds) and speaker_audio is not None

    return (
        gr.update(),
        gr.update(value=str(output_path), visible=True),
        gr.update(value=text_display, visible=True),
        gr.update(value=str(original_output_path) if original_output_path else None, visible=True),
        gr.update(value=time_str, visible=True),
        gr.update(value=str(recon_output_path) if recon_output_path else None, visible=True),
        gr.update(visible=(show_original_audio and speaker_audio is not None)),
        gr.update(visible=(reconstruct_first_30_seconds and speaker_audio is not None)),
        gr.update(visible=show_reference_section),
    )


def generate_audio(
    text_prompt: str,
    speaker_audio_path: str,
    num_steps: int,
    rng_seed: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float,
    rescale_k: float,
    rescale_sigma: float,
    force_speaker: bool,
    speaker_kv_scale: float,
    speaker_kv_min_t: float,
    speaker_kv_max_layers: int,
    reconstruct_first_30_seconds: bool,
    use_custom_shapes: bool,
    max_text_byte_length: str,
    max_speaker_latent_length: str,
    sample_latent_length: str,
    audio_format: str,
    use_compile: bool,
    show_original_audio: bool,
    session_id: str,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Generate audio with OOM-aware device fallback.

    Tries the primary selected device first, then other GPUs, then CPU.
    Only CUDA out-of-memory errors trigger fallback; other errors are raised.
    """
    global model, fish_ae, pca_state, model_compiled, fish_ae_compiled

    # Determine the primary torch device from the initial UI label
    primary_torch_device = _device_label_to_torch(INITIAL_DEVICE_LABEL)
    device_candidates = _get_device_priority_list(primary_torch_device)

    last_oom: Exception | None = None

    for dev in device_candidates:
        try:
            # If we are moving to a different device, re-init models there
            # (skip if dev is the same as current primary device on first iteration).
            if dev != primary_torch_device:
                _init_models_for_torch_device(dev)
                model_compiled = None
                fish_ae_compiled = None

            return _generate_audio_on_active_model(
                text_prompt,
                speaker_audio_path,
                num_steps,
                rng_seed,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_min_t,
                cfg_max_t,
                truncation_factor,
                rescale_k,
                rescale_sigma,
                force_speaker,
                speaker_kv_scale,
                speaker_kv_min_t,
                speaker_kv_max_layers,
                reconstruct_first_30_seconds,
                use_custom_shapes,
                max_text_byte_length,
                max_speaker_latent_length,
                sample_latent_length,
                audio_format,
                use_compile,
                show_original_audio,
                session_id,
            )

        except RuntimeError as e:
            msg = str(e)
            is_cuda_oom = "CUDA out of memory" in msg or "CUDA error: out of memory" in msg
            is_cuda_device = dev.startswith("cuda")
            if is_cuda_oom and is_cuda_device:
                last_oom = e
                # Best-effort cache clear before trying next device
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            # For non-OOM or non-CUDA errors, raise immediately
            raise

    # If we exhausted all devices and only saw OOMs, re-raise the last one
    if last_oom is not None:
        raise last_oom

    raise RuntimeError("Failed to generate audio on any available device.")


# UI Helper Functions
def load_text_presets():
    """Load text presets from file with category and word count."""
    if TEXT_PRESETS_PATH.exists():
        with open(TEXT_PRESETS_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        result = []
        for line in lines:
            if " | " in line:
                parts = line.split(" | ", 1)
                category = parts[0]
                text = parts[1]
            else:
                category = "Uncategorized"
                text = line

            word_count = len(text.split())
            result.append([category, str(word_count), text])

        return result
    return []


def select_text_preset(evt: gr.SelectData):
    """Handle text preset selection - extract text from the row."""
    if evt.value:
        if isinstance(evt.index, (tuple, list)) and len(evt.index) >= 2:
            row_index = evt.index[0]
        else:
            row_index = evt.index

        presets_data = load_text_presets()
        if isinstance(row_index, int) and row_index < len(presets_data):
            text = presets_data[row_index][2]
            return gr.update(value=text)
    return gr.update()


def toggle_mode(mode):
    """Toggle advanced settings section visibility."""
    show_advanced = mode == "Advanced Mode"
    return gr.update(visible=show_advanced)


def update_force_row(force_speaker):
    """Show KV scaling controls when Force Speaker is enabled."""
    return gr.update(visible=bool(force_speaker))


def apply_cfg_preset(preset_name):
    """Apply CFG guidance preset."""
    presets = {
        "higher speaker": (3.0, 8.0, 0.5, 1.0),
        "large guidances": (8.0, 8.0, 0.5, 1.0),
    }

    if preset_name not in presets:
        return [gr.update()] * 5

    text_scale, speaker_scale, min_t, max_t = presets[preset_name]

    return [
        gr.update(value=text_scale),
        gr.update(value=speaker_scale),
        gr.update(value=min_t),
        gr.update(value=max_t),
        gr.update(value="Custom"),
    ]


def apply_speaker_kv_preset(preset_name):
    """Apply speaker KV attention control preset."""
    if preset_name == "enable":
        return [
            gr.update(value=True),
            gr.update(visible=True),
            gr.update(value="Custom"),
        ]
    if preset_name == "off":
        return [
            gr.update(value=False),
            gr.update(visible=False),
            gr.update(value="Custom"),
        ]
    return [gr.update()] * 3


def apply_truncation_preset(preset_name):
    """Apply truncation & temporal rescaling preset."""
    presets = {
        "flat": (0.8, 1.2, 3.0),
        "sharp": (0.9, 0.96, 3.0),
        "baseline(sharp)": (1.0, 1.0, 3.0),
    }

    if preset_name == "custom" or preset_name not in presets:
        return [gr.update()] * 4

    truncation, rescale_k, rescale_sigma = presets[preset_name]

    return [
        gr.update(value=truncation),
        gr.update(value=rescale_k),
        gr.update(value=rescale_sigma),
        gr.update(value="Custom"),
    ]


def load_sampler_presets():
    """Load sampler presets from JSON file."""
    if SAMPLER_PRESETS_PATH.exists():
        with open(SAMPLER_PRESETS_PATH, "r") as f:
            return json.load(f)

    default_presets = {
        "Independent-High-Speaker-CFG": {
            "num_steps": "40",
            "cfg_scale_text": "3.0",
            "cfg_scale_speaker": "8.0",
            "cfg_min_t": "0.5",
            "cfg_max_t": "1.0",
            "truncation_factor": "1.",
            "rescale_k": "1.",
            "rescale_sigma": "3.0"
        }
    }
    with open(SAMPLER_PRESETS_PATH, "w") as f:
        json.dump(default_presets, f, indent=2)
    return default_presets


def apply_sampler_preset(preset_name):
    """Apply a sampler preset to all fields."""
    presets = load_sampler_presets()
    if preset_name == "Custom" or preset_name not in presets:
        return [gr.update()] * 13

    preset = presets[preset_name]
    speaker_kv_enabled = to_bool(preset.get("speaker_kv_enable", False))

    def to_num(val, default):
        try:
            return float(val) if isinstance(val, str) else val
        except (ValueError, TypeError):
            return default

    return [
        gr.update(value=int(to_num(preset.get("num_steps", "40"), 40))),
        gr.update(value=to_num(preset.get("cfg_scale_text", "3.0"), 3.0)),
        gr.update(value=to_num(preset.get("cfg_scale_speaker", "5.0"), 5.0)),
        gr.update(value=to_num(preset.get("cfg_min_t", "0.5"), 0.5)),
        gr.update(value=to_num(preset.get("cfg_max_t", "1.0"), 1.0)),
        gr.update(value=to_num(preset.get("truncation_factor", "0.8"), 0.8)),
        gr.update(value=to_num(preset.get("rescale_k", "1.2"), 1.2)),
        gr.update(value=to_num(preset.get("rescale_sigma", "3.0"), 3.0)),
        gr.update(value=speaker_kv_enabled),
        gr.update(visible=speaker_kv_enabled),
        gr.update(value=to_num(preset.get("speaker_kv_scale", "1.5"), 1.5)),
        gr.update(value=to_num(preset.get("speaker_kv_min_t", "0.9"), 0.9)),
        gr.update(value=int(to_num(preset.get("speaker_kv_max_layers", "24"), 24))),
    ]


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}


def get_audio_prompt_files(search_query: str = ""):
    """Get list of audio files from the audio prompt folder, optionally filtered by search query."""
    if AUDIO_PROMPT_FOLDER is None or not AUDIO_PROMPT_FOLDER.exists():
        return []

    files = sorted([f.name for f in AUDIO_PROMPT_FOLDER.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS], key=str.lower)
    
    # Filter by search query if provided
    if search_query.strip():
        query_lower = search_query.lower()
        files = [f for f in files if query_lower in f.lower()]
    
    return [[file] for file in files]


def filter_audio_prompts(search_query: str):
    """Filter audio prompts based on search query."""
    return gr.update(value=get_audio_prompt_files(search_query))


def select_audio_prompt_file(evt: gr.SelectData):
    """Handle audio prompt file selection from table."""
    if evt.value and AUDIO_PROMPT_FOLDER is not None:
        file_path = AUDIO_PROMPT_FOLDER / evt.value
        if file_path.exists():
            path_str = str(file_path)
            # Return both the Audio component update and the raw path for State.
            return gr.update(value=path_str), path_str
    return gr.update(), None


def upload_audio_prompts(files):
    """Save uploaded audio files into the AUDIO_PROMPT_FOLDER and refresh the table.

    Gradio passes a list of file dicts for `gr.Files`, each with `name` and
    `path` keys. We copy them into the prompts folder and return the updated
    table plus a human-readable status message.
    """
    if AUDIO_PROMPT_FOLDER is None:
        return gr.update(), "Audio prompt folder is not configured."

    AUDIO_PROMPT_FOLDER.mkdir(parents=True, exist_ok=True)

    if not files:
        return gr.update(value=get_audio_prompt_files()), "No files provided to upload."

    import shutil
    from pathlib import Path

    saved = 0
    for f in files:
        # Gradio's Files can yield different types depending on version:
        #  - dict-like: {"name": ..., "path": ...}
        #  - NamedString / str: path to the temp file
        #  - object with .name / .path attributes
        name = None
        src = None

        # dict-like
        if isinstance(f, dict):
            name = f.get("name")
            src = f.get("path")
        else:
            # hasattr-style access first
            if hasattr(f, "name"):
                name = getattr(f, "name")
            if hasattr(f, "path"):
                src = getattr(f, "path")

            # Fallback: treat as a plain path string
            if src is None and isinstance(f, (str, bytes)):
                src = f

        if not src:
            continue

        if not name:
            name = Path(src).name or "uploaded_audio"

        name = Path(name).name
        dest = AUDIO_PROMPT_FOLDER / name
        try:
            shutil.copyfile(src, dest)
            saved += 1
        except Exception:
            # Best-effort; skip problematic files.
            continue

    status = "Uploaded 0 files." if saved == 0 else f"Uploaded {saved} file{'s' if saved != 1 else ''} into audio_prompts/."
    return gr.update(value=get_audio_prompt_files()), status


# UI styling and helpers
LINK_CSS = """
.preset-inline { display:flex; align-items:baseline; gap:6px; margin-top:-4px; margin-bottom:-12px; }
.preset-inline .title { font-weight:600; font-size:.95rem; }
.preset-inline .dim   { color:#666; margin:0 4px; }
a.preset-link { color: #0a5bd8; text-decoration: underline; cursor: pointer; font-weight: 400; }
a.preset-link:hover { text-decoration: none; opacity: 0.8; }
.dark a.preset-link,
[data-theme="dark"] a.preset-link { color: #60a5fa !important; }
.dark a.preset-link:hover,
[data-theme="dark"] a.preset-link:hover { color: #93c5fd !important; }
.dark .preset-inline .dim,
[data-theme="dark"] .preset-inline .dim { color: #9ca3af !important; }
.proxy-btn { position:absolute; width:0; height:0; overflow:hidden; padding:0 !important; margin:0 !important; border:0 !important; opacity:0; pointer-events:none; }
.gr-group { border: 1px solid #d1d5db !important; background: #f3f4f6 !important; }
.dark .gr-group,
[data-theme="dark"] .gr-group { border: 1px solid #4b5563 !important; background: #1f2937 !important; }
.generated-audio-player { border: 3px solid #667eea !important; border-radius: 12px !important; padding: 20px !important; background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.05) 100%) !important; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important; margin: 1rem 0 !important; }
.generated-audio-player > div { background: transparent !important; }
#component-mode-selector { text-align: center; padding: 1rem 0; }
#component-mode-selector label { font-size: 1.1rem !important; font-weight: 600 !important; margin-bottom: 0.5rem !important; }
#component-mode-selector .wrap { justify-content: center !important; }
#component-mode-selector fieldset { border: 2px solid #e5e7eb !important; border-radius: 8px !important; padding: 1rem !important; background: #f9fafb !important; }
.dark #component-mode-selector fieldset,
[data-theme="dark"] #component-mode-selector fieldset { border: 2px solid #4b5563 !important; background: #1f2937 !important; }
.section-separator { height: 3px !important; background: linear-gradient(90deg, transparent 0%, #667eea 20%, #764ba2 80%, transparent 100%) !important; border: none !important; margin: 2rem 0 !important; }
.dark .section-separator,
[data-theme="dark"] .section-separator { background: linear-gradient(90deg, transparent 0%, #667eea 20%, #764ba2 80%, transparent 100%) !important; }
.gradio-container h1,
.gradio-container h2 { font-weight: 700 !important; margin-top: 1.5rem !important; margin-bottom: 1rem !important; }
.tip-box { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important; border-left: 4px solid #f59e0b !important; border-radius: 8px !important; padding: 1rem 1.5rem !important; margin: 1rem 0 !important; box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1) !important; }
.tip-box strong { color: #92400e !important; }
.dark .tip-box,
[data-theme="dark"] .tip-box { background: linear-gradient(135deg, #451a03 0%, #78350f 100%) !important; border-left: 4px solid #f59e0b !important; }
.dark .tip-box strong,
[data-theme="dark"] .tip-box strong { color: #fbbf24 !important; }
"""

JS_CODE = r"""
function () {
  const appEl = document.querySelector("gradio-app");
  const root  = appEl && appEl.shadowRoot ? appEl.shadowRoot : document;
  function clickHiddenButtonById(id) {
    if (!id) return;
    const host = root.getElementById(id);
    if (!host) return;
    const realBtn = host.querySelector("button, [role='button']") || host;
    realBtn.click();
  }
  root.addEventListener("click", (ev) => {
    const a = ev.target.closest("a.preset-link");
    if (!a) return;
    ev.preventDefault();
    ev.stopPropagation();
    ev.stopImmediatePropagation();
    clickHiddenButtonById(a.getAttribute("data-fire"));
    return false;
  }, true);
}
"""


# -------------------------------------------------------------
# API server management helpers (for Voxta/HTTP TTS)

_api_process = None


def _normalize_host_for_server(host: str) -> str:
    """Normalize host value for uvicorn --host (no scheme, no slashes)."""
    host = host.strip()
    if host.startswith("http://"):
        host = host[len("http://") :]
    elif host.startswith("https://"):
        host = host[len("https://") :]
    return host.strip().strip("/") or "0.0.0.0"


def _normalize_host_for_url(host: str) -> str:
    """Normalize host value for use in HTTP URLs (ensure scheme, no trailing slash)."""
    host = host.strip()
    if not host.startswith("http://") and not host.startswith("https://"):
        host = "http://" + host
    return host.rstrip("/")


def start_api_server(host: str, port: int) -> str:
    """Start the FastAPI-based Echo-TTS API server in a background process.

    This uses `uvicorn` to run `api_server:app`. If the server is already
    running, this is a no-op.
    """
    global _api_process
    server_host = _normalize_host_for_server(host)
    if _api_process is not None and _api_process.poll() is None:
        return f"API server already running at {host}:{port}."

    import subprocess, sys

    cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "api_server:app",
    "--host",
    server_host,
        "--port",
        str(int(port)),
    ]

    try:
        _api_process = subprocess.Popen(cmd)
        return f"Started API server at {host}:{port}."
    except Exception as e:
        _api_process = None
        return f"Failed to start API server: {e}"


def stop_api_server() -> str:
    """Stop the background API server process if it's running."""
    global _api_process
    if _api_process is None or _api_process.poll() is not None:
        _api_process = None
        return "API server is not running."

    try:
        _api_process.terminate()
        _api_process.wait(timeout=5)
        msg = "Stopped API server."
    except Exception:
        msg = "Failed to stop API server cleanly; it may have already exited."
    finally:
        _api_process = None

    return msg


def generate_voxta_config(label: str, host: str, port: int, fmt: str) -> str:
    """Generate a Voxta provider JSON config string for Echo-TTS API."""
    url_host = _normalize_host_for_url(host)
    base_url = f"{url_host}:" + str(int(port))
    if fmt not in {"wav", "mp3"}:
        fmt = "wav"

    content_type = "audio/wav" if fmt == "wav" else "audio/mpeg"

    # RequestBody string patterned after working chatterbox_tts.json example
    # Fields are aligned with api_server.TTSRequest and Echo-TTS parameters.
    request_body = (
        '{\n'
        '"text": "{{ text }}",\n'
        '"voice_mode": "predefined",\n'
        '"predefined_voice_id": "{{ voice_id }}",\n'
        '"reference_audio_filename": "string",\n'
        f'"output_format": "{fmt}",\n'
        '"split_text": false,\n'
        '"chunk_size": 120,\n'
        '"temperature":0.8,\n'
        '"exaggeration": 0.8,\n'
        '"cfg_weight": 0.5,\n'
        '"seed": 0, "speed_factor": 1,\n'
        '"num_steps": 20,\n'
        '"culture": "{{ culture }}",\n'
        '"language": "{{ language }}"\n'
        "}"
    )

    voices_format = (
        '{\n'
        '  "label": "{{display_name}}",\n'
        '  "parameters": {\n'
        '    "voice_id": "{{filename}}"\n'
        '  }\n'
        "}"
    )

    cfg = {
        "label": label,
        "values": {
            "ContentType": content_type,
            "ForceConversion": "true",
            "UrlTemplate": f"{base_url}/tts",
            "Request ContentType": "application/json",
            "RequestBody": request_body,
            "VoicesFormat": voices_format,
            "ThinkingSpeech": "uh, ...\nmmh...\nhum...\nhuh...\noh...",
            "AudioGap": "0",
            "VoicesUrl": f"{base_url}/get_predefined_voices",
            "AuthorizationHeader": " ",
            "": "",
        },
    }

    import json

    return json.dumps(cfg, indent=2)


def init_session():
    """Initialize session ID for this browser tab/session."""
    return secrets.token_hex(8)


with gr.Blocks(title="Echo-TTS", css=LINK_CSS, js=JS_CODE) as demo:
    gr.Markdown("# Echo-TTS")
    gr.Markdown("*Jordan Darefsky, 2025. See technical details [here](https://jordandarefsky.com/blog/2025/echo/)*")

    gr.Markdown("**License Notice:** All audio outputs are subject to non-commercial use [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).")

    gr.Markdown("**Responsible Use:** Do not use this model to impersonate real people without their explicit consent or to generate deceptive audio.")

    with gr.Row():
        low_vram_checkbox = gr.Checkbox(
            label="Low VRAM Mode (<= 8GB)",
            value=LOW_VRAM_DEFAULT,
            info=(
                "Runs Echo-TTS in bfloat16 for the main model and autoencoder, "
                "with conservative sequence lengths and reference durations. "
                "Changing this requires restarting the app to fully reload models."
            ),
        )

    with gr.Accordion("📖 Quick Start Instructions", open=True):
        gr.Markdown(
            """
            1. Upload or record a short reference clip (or leave blank for no speaker reference).
            2. Pick a text preset or type your own prompt.
            3. Click **Generate Audio**.

            <div class="tip-box">
            💡 **Tip:** If the generated voice does not match the reference, enable "Force Speaker" and regenerate.
            </div>
            """
        )

    session_id_state = gr.State(None)
    # Track the current speaker reference filepath explicitly so generation
    # always uses the latest selected prompt (table selection or upload/mic).
    speaker_audio_path_state = gr.State("")

    # Device selection row (applies to UI and API usage; persisted between runs)
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("#### Compute Device")
            device_choices = get_available_devices()
            device_dropdown = gr.Dropdown(
                choices=device_choices,
                value=INITIAL_DEVICE_LABEL if INITIAL_DEVICE_LABEL in device_choices else device_choices[0],
                label="Select GPU / CPU",
                info="Applies to all generations and API calls; stored in `device_config.json`.",
            )
        with gr.Column(scale=2):
            gr.Markdown(
                """
                *Choose where to run Echo-TTS. GPU is recommended when available; CPU will be much slower.*
                """
            )

    gr.Markdown("# Speaker Reference")
    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            if AUDIO_PROMPT_FOLDER is not None:
                gr.Markdown("#### Audio Library (click to load)")
                audio_prompt_search = gr.Textbox(
                    label="",
                    placeholder="🔍 Search audio prompts...",
                    lines=1,
                    max_lines=1,
                )
                audio_prompt_table = gr.Dataframe(
                    value=get_audio_prompt_files(),
                    headers=["Filename"],
                    datatype=["str"],
                    row_count=(10, "dynamic"),
                    col_count=(1, "fixed"),
                    interactive=False,
                    label="",
                )

                gr.Markdown("#### Add Audio Prompts")
                audio_prompt_uploader = gr.Files(
                    label="Drag & drop or select audio files",
                    file_count="multiple",
                    file_types=["audio"],
                )
                upload_btn = gr.Button("Upload Voice Files", variant="secondary")
                upload_status = gr.Markdown("", visible=False)
        with gr.Column(scale=2):
            custom_audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Speaker Reference Audio (first five minutes used; blank for no speaker reference)",
                max_length=600,
            )

    gr.HTML('<hr class="section-separator">')
    gr.Markdown("# Text Prompt")
    with gr.Accordion("Text Presets", open=True):
        text_presets_table = gr.Dataframe(
            value=load_text_presets(),
            headers=["Category", "Words", "Preset Text"],
            datatype=["str", "str", "str"],
            row_count=(3, "dynamic"),
            col_count=(3, "fixed"),
            interactive=False,
            column_widths=["12%", "6%", "82%"],
        )
    text_prompt = gr.Textbox(label="Text Prompt", placeholder="[S1] Enter your text prompt here...", lines=4)

    gr.HTML('<hr class="section-separator">')
    gr.Markdown("# Generation")

    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            mode_selector = gr.Radio(
                choices=["Simple Mode", "Advanced Mode"],
                value="Simple Mode",
                label="",
                info=None,
                elem_id="component-mode-selector",
            )
        with gr.Column(scale=1):
            pass

    with gr.Accordion("⚙️ Generation Parameters", open=True):
        with gr.Row(equal_height=False):
            presets = load_sampler_presets()
            preset_keys = list(presets.keys())
            first_preset = preset_keys[0] if preset_keys else "Custom"

            with gr.Column(scale=2):
                preset_dropdown = gr.Dropdown(
                    choices=["Custom"] + preset_keys,
                    value=first_preset,
                    label="Sampler Preset",
                    info="Load preset configurations",
                )

            with gr.Column(scale=0.8, min_width=100):
                num_steps = gr.Number(
                    label="Steps",
                    value=40,
                    info="Sampling steps (Try 20-80)",
                    precision=0,
                    minimum=5,
                    step=5,
                    maximum=80,
                )

            with gr.Column(scale=0.8, min_width=100):
                rng_seed = gr.Number(label="RNG Seed", value=0, info="Seed for noise", precision=0)

            with gr.Column(scale=3):
                with gr.Group():
                    gr.HTML(
                        """
                    <div class="preset-inline">
                      <span class="title">Speaker KV Attention Scaling</span>
                    </div>
                    """
                    )
                    spk_kv_preset_enable = gr.Button("", elem_id="spk_kv_enable", elem_classes=["proxy-btn"])
                    spk_kv_preset_off = gr.Button("", elem_id="spk_kv_off", elem_classes=["proxy-btn"])
                    force_speaker = gr.Checkbox(
                        label='"Force Speaker" (KV scaling)',
                        value=False,
                        info="Enable to more strongly match the reference speaker (though higher values may degrade quality)",
                    )
                    with gr.Row(visible=False) as speaker_kv_row:
                        speaker_kv_scale = gr.Number(label="KV Scale", value=1.5, info="Scale factor (>1 -> larger effect; try 1.5, 1.2, ...)", minimum=0, step=0.1)
                        speaker_kv_min_t = gr.Number(
                            label="KV Min t",
                            value=0.9,
                            info="(0-1), scale applied from steps t=1. to val",
                            minimum=0,
                            maximum=1,
                            step=0.05,
                        )
                        speaker_kv_max_layers = gr.Number(
                            label="Max Layers",
                            value=24,
                            info="(0-24), scale applied in first N layers",
                            precision=0,
                            minimum=0,
                            maximum=24,
                        )

        with gr.Column(visible=False) as advanced_mode_column:
            compile_checkbox = gr.Checkbox(
                label="Compile Model",
                value=False,
                info="Compile for faster runs (~10-30% faster); forces Custom Shapes on to avoid excessive recompilation.",
            )
            use_custom_shapes_checkbox = gr.Checkbox(
                label="Use Custom Shapes (Advanced)",
                value=False,
                info="Override default generation length and/or force latent and text padding (if unchecked, no padding is used and latent generation length is 640≈30s.)",
            )

            with gr.Row(visible=False) as custom_shapes_row:
                max_text_byte_length = gr.Textbox(
                    label="Max Text Byte Length (padded)",
                    value="768",
                    info="Single value or comma-separated buckets (auto-selects min >= length); 768 = max; leave blank for no padding",
                    scale=1,
                )
                max_speaker_latent_length = gr.Textbox(
                    label="Max Speaker Latent Length (padded)",
                    value="640, 2816, 6400",
                    info="Single value or comma-separated buckets (auto-selects min >= length); 640≈30s, 2560≈2min, 6400≈5min (max); leave blank for no padding",
                    scale=1,
                )
                sample_latent_length = gr.Textbox(
                    label="Sample Latent Length",
                    value=str(DEFAULT_SAMPLE_LATENT_LENGTH),
                    info="Maximum sample latent length (640≈30s max seen during training; smaller works well for generating prefixes)",
                    scale=1,
                )

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML(
                            """
                        <div class="preset-inline">
                          <span class="title">Truncation &amp; Temporal Rescaling</span><span class="dim">(</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="trunc_flat">flat</a>
                          <span class="dim">,</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="trunc_sharp">sharp</a>
                          <span class="dim">,</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="trunc_baseline">baseline(sharp)</a>
                          <span class="dim">)</span>
                        </div>
                        """
                        )
                        trunc_preset_flat = gr.Button("", elem_id="trunc_flat", elem_classes=["proxy-btn"])
                        trunc_preset_sharp = gr.Button("", elem_id="trunc_sharp", elem_classes=["proxy-btn"])
                        trunc_preset_baseline = gr.Button("", elem_id="trunc_baseline", elem_classes=["proxy-btn"])
                        with gr.Row():
                            truncation_factor = gr.Number(
                                label="Truncation Factor",
                                value=0.8,
                                info="Multiply initial noise (<1 helps artifacts)",
                                minimum=0,
                                step=0.05,
                            )
                            rescale_k = gr.Number(
                                label="Rescale k", value=1.2, info="<1=sharpen, >1=flatten, 1=off", minimum=0, step=0.05
                            )
                            rescale_sigma = gr.Number(
                                label="Rescale σ", value=3.0, info="Sigma parameter", minimum=0, step=0.1
                            )

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.HTML(
                            """
                        <div class="preset-inline">
                          <span class="title">CFG Guidance</span><span class="dim">(</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="cfg_higher">higher speaker</a>
                          <span class="dim">,</span>
                          <a href="javascript:void(0)" class="preset-link" data-fire="cfg_large">large guidances</a>
                          <span class="dim">)</span>
                        </div>
                        """
                        )
                        cfg_preset_higher_speaker = gr.Button("", elem_id="cfg_higher", elem_classes=["proxy-btn"])
                        cfg_preset_large_guidances = gr.Button("", elem_id="cfg_large", elem_classes=["proxy-btn"])
                        with gr.Row():
                            cfg_scale_text = gr.Number(
                                label="Text CFG Scale", value=3.0, info="Guidance strength for text", minimum=0, step=0.5
                            )
                            cfg_scale_speaker = gr.Number(
                                label="Speaker CFG Scale",
                                value=5.0,
                                info="Guidance strength for speaker",
                                minimum=0,
                                step=0.5,
                            )

                        with gr.Row():
                            cfg_min_t = gr.Number(
                                label="CFG Min t", value=0.5, info="(0-1), CFG applied when t >= val", minimum=0, maximum=1, step=0.05
                            )
                            cfg_max_t = gr.Number(
                                label="CFG Max t", value=1.0, info="(0-1), CFG applied when t <= val", minimum=0, maximum=1, step=0.05
                            )

    with gr.Row(equal_height=True):
        audio_format = gr.Radio(choices=["wav", "mp3"], value="wav", label="Format", scale=1, min_width=90)
        generate_btn = gr.Button("Generate Audio", variant="primary", size="lg", scale=10)
        with gr.Column(scale=1):
            show_original_audio = gr.Checkbox(label="Re-display Original Audio (full 5-minute cropped mono)", value=False)
            reconstruct_first_30_seconds = gr.Checkbox(
                label="Show Autoencoder Reconstruction (only first 30s of reference)", value=False
            )

    gr.HTML('<hr class="section-separator">')
    with gr.Accordion("Generated Audio", open=True, visible=True) as generated_section:
        generation_time_display = gr.Markdown("", visible=False)
        with gr.Group(elem_classes=["generated-audio-player"]):
            generated_audio = gr.Audio(label="Generated Audio", visible=True)
        text_prompt_display = gr.Markdown("", visible=False)

        gr.Markdown("---")
        reference_audio_header = gr.Markdown("#### Reference Audio", visible=False)

        with gr.Accordion("Original Audio (5 min Cropped Mono)", open=False, visible=False) as original_accordion:
            original_audio = gr.Audio(label="Original Reference Audio (5 min)", visible=True)

        with gr.Accordion("Autoencoder Reconstruction of First 30s of Reference", open=False, visible=False) as reference_accordion:
            reference_audio = gr.Audio(label="Decoded Reference Audio (30s)", visible=True)

    # ---------------------------------------------------------
    # Voxta / HTTP API section
    gr.HTML('<hr class="section-separator">')
    gr.Markdown("# Voxta / HTTP API Integration")

    with gr.Accordion("🔌 Echo-TTS API Server & Voxta Config", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                api_host = gr.Textbox(label="API Host", value="http://localhost", lines=1)
                api_port = gr.Number(label="API Port", value=8004, precision=0)
                api_format = gr.Radio(choices=["wav", "mp3"], value="wav", label="Output Format")

                start_api_btn = gr.Button("Start API Server", variant="primary")
                stop_api_btn = gr.Button("Stop API Server")
                api_status = gr.Markdown("🔴 API server is stopped.")

                def _start_api(host, port):
                    msg = start_api_server(host, int(port))
                    # Heuristic based on the returned text to pick an indicator.
                    if "Started API server" in msg or "already running" in msg:
                        return f"🟢 {msg}"
                    return f"🔴 {msg}"

                def _stop_api():
                    msg = stop_api_server()
                    if "Stopped API server" in msg:
                        return f"🟡 {msg}"
                    if "not running" in msg:
                        return f"🔴 {msg}"
                    return f"⚠️ {msg}"

                start_api_btn.click(_start_api, inputs=[api_host, api_port], outputs=[api_status])
                stop_api_btn.click(_stop_api, inputs=None, outputs=[api_status])

            with gr.Column(scale=1):
                provider_label = gr.Textbox(label="Voxta Provider Label", value="Echo-TTS", lines=1)
                generate_cfg_btn = gr.Button("Generate Voxta Provider JSON")
                cfg_output = gr.Code(label="Voxta Provider JSON", language="json")

                def _gen_cfg(label, host, port, fmt):
                    return generate_voxta_config(label, host, int(port), fmt)

                generate_cfg_btn.click(
                    _gen_cfg,
                    inputs=[provider_label, api_host, api_port, api_format],
                    outputs=[cfg_output],
                )

    # Event handlers
    if AUDIO_PROMPT_FOLDER is not None:
        audio_prompt_table.select(select_audio_prompt_file, outputs=[custom_audio_input, speaker_audio_path_state])
        audio_prompt_search.change(filter_audio_prompts, inputs=[audio_prompt_search], outputs=[audio_prompt_table])

        # When users click the upload button, save selected files into audio_prompts/
        # and refresh the table with a clear status message.
        def _on_upload(files):
            table_update, status = upload_audio_prompts(files)
            return table_update, gr.update(value=status, visible=True)

        upload_btn.click(_on_upload, inputs=[audio_prompt_uploader], outputs=[audio_prompt_table, upload_status])

    # Keep speaker_audio_path_state in sync with manual uploads/mic recording
    # and with clearing the Audio component.
    def _sync_speaker_path(p):
        return p or ""

    custom_audio_input.change(_sync_speaker_path, inputs=[custom_audio_input], outputs=[speaker_audio_path_state])

    text_presets_table.select(select_text_preset, outputs=text_prompt)

    mode_selector.change(toggle_mode, inputs=[mode_selector], outputs=[advanced_mode_column])

    force_speaker.change(update_force_row, inputs=[force_speaker], outputs=[speaker_kv_row])

    def toggle_custom_shapes(enabled):
        return gr.update(visible=enabled)

    use_custom_shapes_checkbox.change(
        toggle_custom_shapes,
        inputs=[use_custom_shapes_checkbox],
        outputs=[custom_shapes_row],
    )

    def on_compile_change(compile_enabled):
        """When compile is enabled, force custom shapes to be enabled."""
        if compile_enabled:
            return (
                gr.update(value=True),   # use_custom_shapes_checkbox
                gr.update(visible=True), # custom_shapes_row
            )
        return (
            gr.update(),
            gr.update(),
        )

    compile_checkbox.change(
        on_compile_change,
        inputs=[compile_checkbox],
        outputs=[use_custom_shapes_checkbox, custom_shapes_row],
    )

    cfg_preset_higher_speaker.click(
        lambda: apply_cfg_preset("higher speaker"), outputs=[cfg_scale_text, cfg_scale_speaker, cfg_min_t, cfg_max_t, preset_dropdown]
    )
    cfg_preset_large_guidances.click(
        lambda: apply_cfg_preset("large guidances"), outputs=[cfg_scale_text, cfg_scale_speaker, cfg_min_t, cfg_max_t, preset_dropdown]
    )

    spk_kv_preset_enable.click(lambda: apply_speaker_kv_preset("enable"), outputs=[force_speaker, speaker_kv_row, preset_dropdown])
    spk_kv_preset_off.click(lambda: apply_speaker_kv_preset("off"), outputs=[force_speaker, speaker_kv_row, preset_dropdown])

    trunc_preset_flat.click(lambda: apply_truncation_preset("flat"), outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown])
    trunc_preset_sharp.click(lambda: apply_truncation_preset("sharp"), outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown])
    trunc_preset_baseline.click(
        lambda: apply_truncation_preset("baseline(sharp)"), outputs=[truncation_factor, rescale_k, rescale_sigma, preset_dropdown]
    )

    preset_dropdown.change(
        apply_sampler_preset,
        inputs=preset_dropdown,
        outputs=[
            num_steps,
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_min_t,
            cfg_max_t,
            truncation_factor,
            rescale_k,
            rescale_sigma,
            force_speaker,
            speaker_kv_row,
            speaker_kv_scale,
            speaker_kv_min_t,
            speaker_kv_max_layers,
        ],
    )

    # Persist Low VRAM mode selection; takes effect on next restart (models reload).
    def on_low_vram_change(enabled: bool):
        cfg = load_runtime_config()
        cfg["low_vram_mode"] = bool(enabled)
        save_runtime_config(cfg)
        # Inform the user that a restart is recommended to apply dtype changes.
        return gr.update(
            value=(
                "Low VRAM mode setting saved. Please restart the app to fully "
                "apply dtype changes to all models."
            )
        )

    low_vram_message = gr.Markdown("", visible=False)
    low_vram_checkbox.change(on_low_vram_change, inputs=[low_vram_checkbox], outputs=[low_vram_message])

    generate_btn.click(
        generate_audio,
        inputs=[
            text_prompt,
            speaker_audio_path_state,
            num_steps,
            rng_seed,
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_min_t,
            cfg_max_t,
            truncation_factor,
            rescale_k,
            rescale_sigma,
            force_speaker,
            speaker_kv_scale,
            speaker_kv_min_t,
            speaker_kv_max_layers,
            reconstruct_first_30_seconds,
            use_custom_shapes_checkbox,
            max_text_byte_length,
            max_speaker_latent_length,
            sample_latent_length,
            audio_format,
            compile_checkbox,
            show_original_audio,
            session_id_state,
        ],
        outputs=[
            generated_section,
            generated_audio,
            text_prompt_display,
            original_audio,
            generation_time_display,
            reference_audio,
            original_accordion,
            reference_accordion,
            reference_audio_header,
        ],
    )

    # When the device is changed, reinitialize all models on the new device and persist the choice.
    def on_device_change(new_device_label: str):
        save_device_config(new_device_label)
        _init_models_for_device(new_device_label)
        # No UI component needs updating immediately; future generations use the new device.
        return gr.update()

    device_dropdown.change(on_device_change, inputs=[device_dropdown], outputs=[])

    demo.load(init_session, outputs=[session_id_state]).then(
        lambda: apply_sampler_preset(list(load_sampler_presets().keys())[0]),
        outputs=[
            num_steps,
            cfg_scale_text,
            cfg_scale_speaker,
            cfg_min_t,
            cfg_max_t,
            truncation_factor,
            rescale_k,
            rescale_sigma,
            force_speaker,
            speaker_kv_row,
            speaker_kv_scale,
            speaker_kv_min_t,
            speaker_kv_max_layers,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(AUDIO_PROMPT_FOLDER)]
    )
