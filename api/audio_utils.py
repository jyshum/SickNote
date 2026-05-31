"""Lightweight audio loading via ffmpeg — no pandas dependency."""

import subprocess

import numpy as np
import torch


def load_audio_ffmpeg(audio_path, target_sr=None):
    """Load audio using ffmpeg subprocess (handles .webm, .ogg, .wav).

    Returns (waveform_tensor, sample_rate).
    """
    sr = target_sr or 22050
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sr),
        "-v", "quiet",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {audio_path}: {result.stderr.decode(errors='replace')}"
        )
    pcm = np.frombuffer(result.stdout, dtype=np.float32).copy()
    waveform = torch.from_numpy(pcm).unsqueeze(0)
    return waveform, sr
