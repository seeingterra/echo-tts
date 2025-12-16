from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch


def _to_channels_first(audio: torch.Tensor) -> torch.Tensor:
    """Normalize input audio tensor to shape (channels, samples) on CPU."""
    audio = audio.detach().cpu()
    if audio.ndim == 1:
        return audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio tensor, got shape {tuple(audio.shape)}")
    return audio


def tensor_to_pcm16le(audio: torch.Tensor, *, downmix_to_mono: bool = True) -> tuple[bytes, int]:
    """Convert float audio tensor (C,T) or (T,) in [-1,1] to PCM16LE bytes.

    Returns (pcm_bytes, channels).
    """
    audio_cf = _to_channels_first(audio)

    if downmix_to_mono and audio_cf.shape[0] > 1:
        audio_cf = audio_cf.mean(dim=0, keepdim=True)

    audio_cf = audio_cf.clamp(-1.0, 1.0)
    pcm = (audio_cf * 32767.0).to(torch.int16)

    # Interleave channels for WAV/PCM bytes: (T, C)
    pcm_np = pcm.transpose(0, 1).contiguous().numpy().astype(np.int16, copy=False)
    return pcm_np.tobytes(), int(audio_cf.shape[0])


def save_wav_pcm16(path: str | Path, audio: torch.Tensor, sample_rate: int, *, downmix_to_mono: bool = True) -> Path:
    """Save float audio tensor to a PCM16LE WAV using Python stdlib (no torchaudio/ffmpeg)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pcm_bytes, channels = tensor_to_pcm16le(audio, downmix_to_mono=downmix_to_mono)

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)

    return out_path


def read_wav_pcm16_frames(path: str | Path) -> tuple[bytes, int, int]:
    """Read WAV PCM frames as bytes.

    Returns (pcm_bytes, sample_rate, channels). Only supports 16-bit PCM WAV.
    """
    in_path = Path(path)
    with wave.open(str(in_path), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_rate = int(wf.getframerate())
        sampwidth = int(wf.getsampwidth())
        if sampwidth != 2:
            raise RuntimeError(f"Unsupported WAV sample width {sampwidth * 8} bits; expected 16-bit PCM")
        pcm_bytes = wf.readframes(wf.getnframes())

    return pcm_bytes, sample_rate, channels
