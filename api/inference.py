"""
Inference — handoff file.

P2 writes the mock version (below). P1 replaces the function body with
real model inference after training is complete. The signature never changes.
"""
import random
import base64
import io
import numpy as np
from PIL import Image


def predict(audio_path: str) -> dict:
    """
    Run prediction on an audio file.

    Args:
        audio_path: path to an audio file on disk (webm, wav, ogg, mp3)

    Returns:
        {
            "label": "healthy" or "abnormal",
            "confidence": float between 0.0 and 1.0,
            "spectrogram": base64-encoded PNG string ("data:image/png;base64,...")
        }
    """
    # ── MOCK — P1 replaces everything below this line ──
    label = random.choice(["healthy", "abnormal"])
    confidence = round(random.uniform(0.65, 0.97), 3)
    spectrogram = _mock_spectrogram()

    return {
        "label": label,
        "confidence": confidence,
        "spectrogram": spectrogram,
    }


def _mock_spectrogram() -> str:
    """Generate a random spectrogram-like image as base64 PNG."""
    data = np.random.rand(64, 173) * 255
    img = Image.fromarray(data.astype(np.uint8), mode="L")
    img = img.resize((400, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
