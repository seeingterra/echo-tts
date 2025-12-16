import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from gradio_app import (
    AUDIO_PROMPT_FOLDER,
    AUDIO_EXTS,
    DEFAULT_SAMPLE_LATENT_LENGTH,
    synthesize_to_file,
    _device_label_to_torch,
    _get_device_priority_list,
    _init_models_for_torch_device,
    INITIAL_DEVICE_LABEL,
)

app = FastAPI(title="Echo-TTS API")


class TTSRequest(BaseModel):
    text: str
    voice_mode: str = "predefined"
    predefined_voice_id: Optional[str] = None
    reference_audio_filename: Optional[str] = None
    output_format: str = "wav"  # "wav" | "mp3" | "pcm" (16-bit little-endian, 44.1kHz)
    split_text: bool = False
    chunk_size: int = 120
    temperature: float = 0.8
    exaggeration: float = 0.8
    cfg_weight: float = 0.5
    seed: int = 0
    speed_factor: float = 1.0
    culture: Optional[str] = None
    language: Optional[str] = None
    # Backward-compatible knob (some clients send this) for number of steps.
    num_steps: Optional[int] = None

    # --- Echo-TTS / inference parameters (complete set) ---
    # If a field is omitted, api_server uses conservative defaults tuned for Voxta-style utterances.
    # If a field is explicitly provided, it overrides the default.
    rng_seed: Optional[int] = None
    cfg_scale_text: Optional[float] = None
    cfg_scale_speaker: Optional[float] = None
    cfg_mode: Optional[str] = None  # independent | joint-unconditional | apg-independent
    cfg_min_t: Optional[float] = None
    cfg_max_t: Optional[float] = None
    truncation_factor: Optional[float] = None
    rescale_k: Optional[float] = None
    rescale_sigma: Optional[float] = None
    speaker_kv_scale: Optional[float] = None
    speaker_kv_max_layers: Optional[int] = None
    speaker_kv_min_t: Optional[float] = None
    sequence_length: Optional[int] = None


@app.get("/get_predefined_voices")
async def get_predefined_voices():
    """Return list of predefined voices from the audio_prompts folder."""
    voices = []
    if AUDIO_PROMPT_FOLDER.exists():
        for p in AUDIO_PROMPT_FOLDER.iterdir():
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                voices.append(
                    {
                        "label": p.stem,
                        "display_name": p.stem,
                        "filename": p.name,
                        "culture": "en-US",
                        "language": "en",
                    }
                )
    return voices


def _synthesize_to_path(
    text: str,
    speaker_audio_path: Optional[str],
    output_format: str,
    seed: int,
    num_steps: Optional[int] = None,
    *,
    rng_seed: Optional[int] = None,
    cfg_scale_text: Optional[float] = None,
    cfg_scale_speaker: Optional[float] = None,
    cfg_mode: Optional[str] = None,
    cfg_min_t: Optional[float] = None,
    cfg_max_t: Optional[float] = None,
    truncation_factor: Optional[float] = None,
    rescale_k: Optional[float] = None,
    rescale_sigma: Optional[float] = None,
    speaker_kv_scale: Optional[float] = None,
    speaker_kv_max_layers: Optional[int] = None,
    speaker_kv_min_t: Optional[float] = None,
    sequence_length: Optional[int] = None,
) -> Path:
    """Generate audio using the shared synthesize_to_file helper with OOM fallback.

    This mirrors the Gradio UI's device fallback logic but returns a plain Path,
    which is stable for HTTP API use.
    """

    # Conservative defaults tuned for short, Voxta-style utterances.
    # These are slightly lower than the UI defaults to reduce tail noise.
    # If the request provides num_steps, clamp it to a safe range.
    if num_steps is None:
        num_steps = 20
    num_steps = max(5, min(int(num_steps), 80))
    # Prefer rng_seed if provided (Echo-TTS naming), otherwise use seed.
    effective_seed = seed if rng_seed is None else int(rng_seed)

    # CFG settings
    cfg_scale_text = 3.0 if cfg_scale_text is None else float(cfg_scale_text)
    cfg_scale_speaker = cfg_scale_text if cfg_scale_speaker is None else float(cfg_scale_speaker)
    cfg_mode_norm = (cfg_mode or "independent").strip().lower()
    if cfg_mode_norm not in {"independent", "joint-unconditional", "apg-independent"}:
        cfg_mode_norm = "independent"
    cfg_min_t = 0.5 if cfg_min_t is None else float(cfg_min_t)
    cfg_max_t = 1.0 if cfg_max_t is None else float(cfg_max_t)

    # Noise truncation (initial noise scaling)
    truncation_factor = 0.8 if truncation_factor is None else float(truncation_factor)

    # Temporal score rescaling (optional)
    # synthesize_to_file treats rescale_k==1.0 as disabled.
    rescale_k = 1.2 if rescale_k is None else float(rescale_k)
    rescale_sigma = 3.0 if rescale_sigma is None else float(rescale_sigma)
    force_speaker = bool(speaker_audio_path)
    speaker_kv_scale = 1.2
    speaker_kv_min_t = 0.9
    speaker_kv_max_layers = 24
    reconstruct_first_30_seconds = False
    use_custom_shapes = False
    max_text_byte_length = "768"
    max_speaker_latent_length = "640, 2816, 6400"

    # Slightly shorter than UI default to keep clips tight and avoid noisy tails.
    base_len = int(DEFAULT_SAMPLE_LATENT_LENGTH)
    if sequence_length is not None:
        try:
            seq_len = int(sequence_length)
        except Exception:
            seq_len = max(384, base_len - 32)
        sample_latent_length = str(max(128, seq_len))
    else:
        sample_latent_length = str(max(384, base_len - 32))
    use_compile = False
    show_original_audio = False
    session_id = "api"

    # Device selection and OOM fallback (copied conceptually from gradio_app.generate_audio)
    from gradio_app import model_compiled, fish_ae_compiled  # type: ignore

    primary_torch_device = _device_label_to_torch(INITIAL_DEVICE_LABEL)
    device_candidates = _get_device_priority_list(primary_torch_device)

    last_oom: Exception | None = None

    for dev in device_candidates:
        try:
            if dev != primary_torch_device:
                _init_models_for_torch_device(dev)
                # Reset compiled variants when switching devices
                model_compiled = None
                fish_ae_compiled = None

            output_path, *_ = synthesize_to_file(
                text_prompt=text,
                speaker_audio_path=speaker_audio_path or "",
                num_steps=num_steps,
                rng_seed=effective_seed,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_mode=cfg_mode_norm,
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
                audio_format=output_format,
                use_compile=use_compile,
                show_original_audio=show_original_audio,
                session_id=session_id,
                fade_out_seconds=0.5,
            )

            return Path(output_path)

        except RuntimeError as e:
            msg = str(e)
            is_cuda_oom = "CUDA out of memory" in msg or "CUDA error: out of memory" in msg
            is_cuda_device = dev.startswith("cuda")
            if is_cuda_oom and is_cuda_device:
                last_oom = e
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise

    if last_oom is not None:
        raise last_oom

    raise RuntimeError("Failed to generate audio on any available device.")


@app.post("/tts")
async def tts(req: TTSRequest):
    """Voxta/ChatterBox-compatible TTS endpoint.

    Returns raw audio bytes in the requested format (wav, mp3, or pcm).
    """
    if not req.text:
        raise HTTPException(status_code=400, detail="'text' field is required")

    # Map predefined_voice_id -> path in AUDIO_PROMPT_FOLDER
    speaker_path: Optional[str] = None
    if req.predefined_voice_id:
        candidate = AUDIO_PROMPT_FOLDER / req.predefined_voice_id
        if candidate.exists():
            speaker_path = str(candidate)

    fmt = req.output_format.lower()
    if fmt not in {"wav", "mp3", "pcm"}:
        fmt = "wav"

    # synthesize_to_file currently produces container formats (wav/mp3).
    # For pcm output, generate a wav and convert to raw 16-bit PCM bytes.
    synth_fmt = "wav" if fmt == "pcm" else fmt

    # Field-set-aware compatibility behavior:
    # - If a client explicitly sends legacy fields (temperature/exaggeration/cfg_weight), we map them
    #   to the new parameter set unless the new fields are also provided.
    fields_set = getattr(req, "__fields_set__", set())

    # Legacy cfg_weight -> cfg_scale_text (when cfg_scale_text not provided)
    cfg_scale_text = req.cfg_scale_text
    if cfg_scale_text is None and "cfg_weight" in fields_set:
        # Preserve historical default mapping: cfg_weight=0.5 -> cfg_scale=3.0
        # Map [0..2] roughly into [1..9].
        w = float(req.cfg_weight)
        w = max(0.0, min(2.0, w))
        cfg_scale_text = 1.0 + (4.0 * w)

    # Legacy exaggeration -> speaker cfg boost (only when cfg_scale_speaker is not provided)
    cfg_scale_speaker = req.cfg_scale_speaker
    if cfg_scale_speaker is None and cfg_scale_text is not None and "exaggeration" in fields_set:
        ex = float(req.exaggeration)
        speaker_cfg_mult = max(0.1, min(3.0, 0.5 + ex))
        cfg_scale_speaker = float(cfg_scale_text) * speaker_cfg_mult

    # Legacy temperature -> truncation_factor (when truncation_factor not provided)
    truncation_factor = req.truncation_factor
    if truncation_factor is None and "temperature" in fields_set:
        truncation_factor = float(req.temperature)

    # If either rescale parameter is explicitly provided as null, disable rescaling.
    rescale_k = req.rescale_k
    rescale_sigma = req.rescale_sigma
    if ("rescale_k" in fields_set or "rescale_sigma" in fields_set) and (rescale_k is None or rescale_sigma is None):
        rescale_k = 1.0
        rescale_sigma = 1.0

    try:
        audio_path = _synthesize_to_path(
            text=req.text,
            speaker_audio_path=speaker_path,
            output_format=synth_fmt,
            seed=req.seed,
            num_steps=req.num_steps,
            rng_seed=req.rng_seed,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_mode=req.cfg_mode,
            cfg_min_t=req.cfg_min_t,
            cfg_max_t=req.cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            speaker_kv_scale=req.speaker_kv_scale,
            speaker_kv_max_layers=req.speaker_kv_max_layers,
            speaker_kv_min_t=req.speaker_kv_min_t,
            sequence_length=req.sequence_length,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

    if not audio_path.exists():
        raise HTTPException(status_code=500, detail="Generated audio file not found")

    if fmt == "pcm":
        try:
            from audio_io import read_wav_pcm16_frames

            data, sample_rate, channels = read_wav_pcm16_frames(audio_path)
            if int(sample_rate) != 44100 or int(channels) != 1:
                raise RuntimeError(
                    f"Expected 44.1kHz mono WAV for PCM output; got rate={sample_rate}, channels={channels}"
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PCM conversion failed: {e}")

        return Response(content=data, media_type="audio/L16; rate=44100; channels=1")

    data = audio_path.read_bytes()
    mime = "audio/wav" if fmt == "wav" else "audio/mpeg"
    return Response(content=data, media_type=mime)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
