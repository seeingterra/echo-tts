# Echo‑TTS (Windows + Voxta oriented fork)

> A Windows- and Voxta-friendly fork of the original Echo‑TTS project.

This fork of Echo‑TTS is optimized for Windows users and for integration with
[Voxta.ai](https://voxta.ai). It builds on the original Gradio UI and adds:

- A **Windows-friendly Gradio UI** with sensible defaults
- A **Voxta compatible HTTP TTS API**
- A **Voxta provider JSON generator** directly in the UI
- **GPU / CPU selection with persistence** across runs
- **Low-VRAM safeguards** tuned for ≈8 GB GPUs
- A **Windows-tested install path** (Python 3.13+, CUDA 12.4 wheels option)
- Added to ** audio prompt library** with drag-and-drop **multi-file upload**

All of this is layered on top of the original Echo‑TTS model behavior.

A multi-speaker text-to-speech model with speaker reference conditioning. See
the original [blog post](https://jordandarefsky.com/blog/2025/echo/) for
technical details.

**Model:** [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base)
| **Demo:** [echo-tts-preview](https://huggingface.co/spaces/jordand/echo-tts-preview)

> **Note**
> This repository is an **early test fork** focused on Windows and Voxta.ai
> integration.
> The **official Echo‑TTS API implementation** is maintained separately at:
> https://github.com/KevinAHM/echo-tts-api
>
> This is just a fun learning experiment from a fan of the original creators’
> work.

---

## Original README (upstream project)

> The following is the modified original README content from the upstream
> `jordand/echo-tts` repository, partially preserved here for reference.

## Responsible Use

Don't use this model to:
- Impersonate real people without their consent
- Generate deceptive audio (e.g., fraud, misinformation, deepfakes)

You are responsible for complying with local laws regarding biometric data and voice cloning.

## Installation

Tested and recommended with **Python 3.13+**.
You can download the latest Python 3.13 installer for Windows from:
https://www.python.org/downloads/windows/

The **recommended setup on Windows** is to use a dedicated virtual
environment for this project:

```bash
python -m venv venv
venv\Scripts\activate
```

Then install dependencies:

On Windows (CPU or generic install):

```bash
pip install -r requirements.txt
```

On Windows with NVIDIA GPU (CUDA 12.4 wheels):

```bash
pip install -r requirements-cuda.txt
```

> **Windows tip: long path support**
> Some Python packages (and virtual environments) can hit Windows' default
> path length limit and fail with odd file/path errors. Enabling "LongPaths"
> support in Windows and using a short base path like `C:\dev\echo-tts` can
> help. See:
> - Windows 10/11 Group Policy: `Local Computer Policy → Computer Configuration
>   → Administrative Templates → System → Filesystem → Enable Win32 long paths`
> - Or set `LongPathsEnabled = 1` under
>   `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`

> **VRAM**
> Echo-TTS is tuned here for GPUs with **≈8GB VRAM or more**. The fork adds a
> "Low VRAM Mode" that uses bfloat16 and conservative sequence lengths to
> reduce out-of-memory errors on smaller GPUs.

> **Windows audio backend / codec notes**
> This fork replaces the older TorchCodec-based audio path with a
> **torchaudio + FFmpeg** pipeline that has been tested on Windows. This
> avoids several Windows-specific codec issues and makes reference audio
> loading more robust (mono downmix + 44.1 kHz resampling).

## Quick Start

### Gradio UI (Voxta config in the bottom)

From an activated virtual environment in the repo root:

```bash
python gradio_app.py
```

Once the UI is open:

- **Add your voice samples** (WAV/MP3) into the `audio_prompts/` folder so
    they appear in the reference dropdown.
    - You can now do this directly in the UI:
        - Use the **Speaker Reference → Add Audio Prompts** panel.
        - Drag & drop or select multiple files.
        - Click **Upload Voice Files** to copy them into `audio_prompts/`.
- **Scroll to the bottom** of the Gradio interface to find the
    **"Voxta / HTTP API Integration"** section, where you can start/stop the
    built-in API server and generate a ready-to-paste Voxta provider config.

### Python API

```python
from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
    load_audio,
    sample_pipeline,
    sample_euler_cfg_independent_guidances,
)
from functools import partial
import torchaudio

# Load models (downloads from HuggingFace on first run)
model = load_model_from_hf(delete_blockwise_modules=True)
fish_ae = load_fish_ae_from_hf()
pca_state = load_pca_state_from_hf()

# Load speaker reference (or set to None for no reference)
speaker_audio = load_audio("speaker.wav").cuda()

# Configure sampler
sample_fn = partial(
    sample_euler_cfg_independent_guidances,
    num_steps=20,
    cfg_scale_text=3.0,
    cfg_scale_speaker=8.0,
    cfg_min_t=0.5,
    cfg_max_t=1.0,
    truncation_factor=None,
    rescale_k=None,
    rescale_sigma=None,
    speaker_kv_scale=None,
    speaker_kv_max_layers=None,
    speaker_kv_min_t=None,
    sequence_length=640, # (~30 seconds)
)

# Generate
text = "[S1] Hello, this is a test of the Echo TTS model."
audio_out, _ = sample_pipeline(
    model=model,
    fish_ae=fish_ae,
    pca_state=pca_state,
    sample_fn=sample_fn,
    text_prompt=text,
    speaker_audio=speaker_audio,
    rng_seed=0,
)

torchaudio.save("output.wav", audio_out[0].cpu(), 44100)
```

See also:
- `inference.py` -- lower-level usage example at the bottom of the file
- `inference_blockwise.py` -- examples of blockwise/continuation generation

## Low VRAM (8GB) and GPU selection

This fork adds several quality-of-life features for running on Windows with
limited VRAM:

- **Low VRAM Mode** toggle in the Gradio UI
    - Controls dtype (`bfloat16` vs `float32`) and safe defaults for sequence
        length and reference duration.
    - Persists to `runtime_config.json` so your preference is remembered.
- **GPU / CPU selector** in the Gradio UI
    - Lists `CPU` and all available CUDA devices as
        `GPU 0 (cuda:0) - <name>`, etc.
    - Persists to `device_config.json` and is used by both the UI and HTTP API.
- **OOM-aware multi-GPU fallback**
    - Generation first tries the selected primary device.
    - On CUDA out-of-memory errors it automatically falls back to other GPUs,
        and finally to CPU as a last resort.

## HTTP API + Voxta.ai integration

This fork exposes a lightweight HTTP API, designed to be compatible with
Voxta / ChatterBox-style TTS integrations.

### Endpoints

- `POST /tts`
    - Accepts JSON similar to:

        ```jsonc
        {
            "text": "Hi!",
            "voice_mode": "predefined",
            "predefined_voice_id": "EARS p004 freeform.mp3",
            "reference_audio_filename": "string",
            "output_format": "wav",
            "split_text": false,
            "chunk_size": 120,

            // Legacy knobs (still accepted; mapped to the full parameter set if you don't send the new fields)
            "temperature": 0.8,
            "exaggeration": 0.8,
            "cfg_weight": 0.5,
            "seed": 0,

            // Preferred Echo-TTS parameters
            "num_steps": 20,
            "rng_seed": 0,
            "cfg_mode": "independent", // independent | joint-unconditional | apg-independent
            "cfg_scale_text": 3.0,
            "cfg_scale_speaker": null,
            "cfg_min_t": 0.5,
            "cfg_max_t": 1.0,
            "truncation_factor": null,
            "rescale_k": null,
            "rescale_sigma": null,
            "speaker_kv_scale": null,
            "speaker_kv_min_t": null,
            "speaker_kv_max_layers": null,
            "sequence_length": null,

            "speed_factor": 1.0,
            "culture": "en-US",
            "language": "en"
        }
        ```

    - Returns raw audio bytes (`audio/wav` or `audio/mpeg`) on success.
    - Uses the same device / low-VRAM configuration and OOM-aware fallback
        as the Gradio UI.
    - `rng_seed` (if provided) takes precedence over `seed`.

- `GET /get_predefined_voices`
    - Scans the `audio_prompts/` folder for supported audio files and returns a
        list of voice entries suitable for Voxta (label, display_name, filename,
        culture, language).

### Running the HTTP API

You can start the API directly:

```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8004
```

Or use the **Voxta / HTTP API Integration** section in the Gradio UI to
start/stop the server from within the app.

### Voxta provider config

The Gradio UI includes a small helper panel that can generate a Voxta
provider JSON compatible with the Echo-TTS HTTP API. It:

- Lets you choose a label, host, port, and output format (WAV or MP3).
- Generates a JSON snippet with:
    - `UrlTemplate` pointing at `http://<host>:<port>/tts`
    - `VoicesUrl` pointing at `http://<host>:<port>/get_predefined_voices`
    - A `RequestBody` template patterned after a working ChatterBox config.
    - A `VoicesFormat` entry that maps each file in `audio_prompts/` to a
        Voxta voice.

You can paste this generated JSON directly into Voxta's provider
configuration to connect Echo-TTS as a TTS backend.

> **Official Echo-TTS API**
> This fork's HTTP API design is inspired by, but separate from, the
> official Echo-TTS API implementation:
> https://github.com/KevinAHM/echo-tts-api

---

## Changelog (fork highlights)

This is a hand-written summary of the most important changes in this fork
relative to the original Echo‑TTS repo.

### 2025-12-06 — num_steps for Voxta / HTTP API

- **Configurable diffusion steps from Voxta**
        - Extended the HTTP API request model (`TTSRequest`) to accept an
            optional `num_steps` field.
        - The FastAPI server now passes `num_steps` into the sampler and clamps
            it to a safe range (5–80 steps) per request.
        - Updated the generated Voxta provider JSON so it includes
            `"num_steps": 20` in the `RequestBody` by default.
        - This lets you tune quality vs. latency for different use cases
            directly from Voxta.

### 2025-12-06 — Windows / Voxta polish & prompt uploader

- **Multi-file prompt upload in the UI**
        - Added a **Speaker Reference → Add Audio Prompts** panel.
        - You can drag & drop or select multiple audio files and click
            **Upload Voice Files** to copy them into `audio_prompts/`.
        - The **Audio Library** table refreshes automatically and shows a clear
            status message (e.g. "Uploaded 3 files into audio_prompts/").
- **Improved Voxta / HTTP API panel**
        - The **"🔌 Echo-TTS API Server & Voxta Config"** accordion now opens by
            default so you can see it immediately.
        - Start/stop actions show clearer status text with indicators:
            `🟢` running/already running, `🟡` stopping, `🔴` stopped, `⚠️` errors.

### 2025-12-06 — Windows install & CUDA wheels

- **CUDA 12.4 requirements**
        - Added `requirements-cuda.txt` that pins `torch==2.6.0+cu124` and
            `torchaudio==2.6.0+cu124` and uses the official PyTorch CUDA index
            for those wheels.
        - All other packages (e.g. `safetensors`, `gradio`, `fastapi`) are
            pulled from PyPI to avoid resolution issues.
- **Windows-friendly Python recommendation**
        - README now recommends **Python 3.13+** on Windows and links directly to
            the official downloads page.
        - Example venv commands and a note about Windows long path support.

### 2025-12-06 — Audio pipeline & tail behavior

- **Audio backend / codec fix**
        - Removed the TorchCodec-based audio I/O path.
        - Switched to **torchaudio + FFmpeg** for decoding, mono downmixing, and
            44.1 kHz resampling to avoid Windows codec issues.
- **Tail noise and long-output tweaks**
        - Generation now adds a small silence tail and optional fade-out to
            reduce static/noise at the end of clips, especially noticeable when
            used as a streaming TTS backend.

### 2025-12-06 — Devices, low VRAM & API integration

- **Low VRAM mode and device persistence**
        - Added a **Low VRAM Mode (<= 8GB)** toggle that uses `bfloat16` and
            conservative sequence lengths, persisted in `runtime_config.json`.
        - Added a **GPU / CPU selector** whose choice is stored in
            `device_config.json` and used by both the UI and the HTTP API.
        - Implemented OOM-aware multi-GPU fallback (primary → other GPUs → CPU).
- **HTTP API + Voxta helper**
        - Added the FastAPI-based `api_server.py` with:
                - `POST /tts` for Voxta/ChatterBox-style TTS.
                - `GET /get_predefined_voices` reading from `audio_prompts/`.
        - Added a **Voxta provider JSON generator** panel in the Gradio UI to
            quickly produce a Voxta-compatible provider config.

---

## Updating an existing installation

If you already cloned this repo and want to pull in the latest changes
(Windows fixes, CUDA wheels, Voxta improvements, prompt uploader, etc.):

1. **Activate your virtual environment** (if not already):

    ```powershell
    cd J:\AITools\echo-tts
    venv\Scripts\activate
    ```

2. **Update from GitHub**:

    ```powershell
    git pull origin main
    ```

3. **(Recommended) Reinstall / refresh dependencies** in case
   `requirements.txt` or `requirements-cuda.txt` changed:

    ```powershell
    pip install -r requirements.txt
    ```

    Or, if you use the CUDA wheels on Windows:

    ```powershell
    pip install -r requirements-cuda.txt
    ```

4. **Restart the app** so the new code is loaded:

    ```powershell
    python gradio_app.py
    ```

If `git pull` reports local changes you don’t care about, you can discard
them with:

```powershell
git restore .
git pull origin main
```

## Tips

### Generation Length

Echo is trained to generate up to 30 seconds of audio (640 latents) given text and reference audio. Since the supplied text always corresponded to ≤30 seconds of audio during training, the model will attempt to fit any text prompt at inference into the 30 seconds of generated audio (and thus, e.g., long text prompts may result in faster speaking rates). On the other hand, shorter text prompts will work and will produce shorter outputs (as the model generates latent padding automatically).

If "Sample Latent Length" (in Custom Shapes in gradio)/sequence_length is set to less than 640, the model will attempt to generate the prefix corresponding to that length. I.e., if you set this to 320, and supply ~30 seconds worth of text, the model will likely generate the first half of the text (rather than try to fit the entirety of the text into the first 15 seconds).

### Reference Audio

You can condition on up to 5 minutes of reference audio, but shorter clips (e.g., 10 seconds or shorter) work well too.

### Force Speaker (KV Scaling)

Sometimes out-of-distribution text for a given reference speaker will cause the model to generate a different speaker entirely. Enabling "Force Speaker" (which scales speaker KV for a portion of timesteps, default scale 1.5) generally fixes this. However, high values may introduce artifacts or "overconditioning." Aim for the lowest scale that produces the correct speaker: 1.0 is baseline, 1.5 is the default when enabled and will usually force the speaker, but lower values (e.g., 1.3, 1.1) may suffice.

### Text Prompt Format

Text prompts use the format from [WhisperD](https://huggingface.co/jordand/whisper-d-v1a). Colons, semicolons, and emdashes are normalized to commas (see inference.py tokenizer_encode) by default, and "[S1] " will be added to the beginning of the prompt if not already present. Commas generally function as pauses. Exclamation points (and other non-bland punctuation) may lead to increased expressiveness but also potentially lower quality on occasion; improving controllability is an important direction for future work.

The included text presets are stylistically in-distribution with the WhisperD transcription style.

### Blockwise Generation

`inference_blockwise.py` includes blockwise sampling, which allows generating audio in smaller blocks as well as producing continuations of existing audio (where the prefix and continuation are up to 30 seconds combined). The model released on HF is a fully fine-tuned model (not the LoRA as described in the blog). Blockwise generation enables audio streaming (not included in current code) since the S1-DAC decoder is causal. Blockwise functionality hasn't been thoroughly tested and may benefit from different (e.g., smaller) CFG scales.

## License

Code in this repo is MIT‑licensed except where file headers specify otherwise (e.g., autoencoder.py is Apache‑2.0).

Regardless of our model license, audio outputs are CC-BY-NC-SA-4.0 due to the dependency on the Fish Speech S1-DAC autoencoder, which is CC-BY-NC-SA-4.0.

We have chosen to release the Echo-TTS weights under CC-BY-NC-SA-4.0.

For included audio prompts, see `audio_prompts/LICENSE`.

## Citation

```bibtex
@misc{darefsky2025echo,
    author = {Darefsky, Jordan},
    title = {Echo-TTS},
    year = {2025},
    url = {https://jordandarefsky.com/blog/2025/echo/}
}
```
