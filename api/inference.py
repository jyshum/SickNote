"""
Inference -- real model prediction.

Loads the trained SickNoteCNN model and preprocessing params once at module
level, then runs inference on uploaded audio files.

The function signature matches the contract in ARCHITECTURE.md:
    predict(audio_path: str) -> dict
"""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torchaudio

# Model and params are cached at module level -- loaded once, not per request.
_model = None
_params = None

# Resolve checkpoint paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "model", "checkpoints")


def _load_model():
    """Load model_final.pt and preprocessing_params.pt.

    Uses weights_only=True and map_location='cpu' per ARCHITECTURE.md.
    Caches result in module-level globals so it only runs once.
    """
    global _model, _params

    if _model is not None:
        return _model, _params

    from model.architecture import SickNoteCNN

    # Load preprocessing params
    params_path = os.path.join(_CHECKPOINT_DIR, "preprocessing_params.pt")
    params = torch.load(params_path, weights_only=True, map_location="cpu")

    # Build model with correct dimensions and load weights
    model = SickNoteCNN(n_mels=params["n_mels"], time_frames=params["time_frames"])
    state_dict = torch.load(
        os.path.join(_CHECKPOINT_DIR, "model_final.pt"),
        weights_only=True,
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.eval()

    _model = model
    _params = params
    return _model, _params


def _audio_to_spectrogram(audio_path, params):
    """Convert audio file to normalized log-mel spectrogram tensor.

    Same pipeline as preprocess.py but uses params from preprocessing_params.pt
    instead of config.py constants, ensuring consistency with training.

    Returns: torch.Tensor of shape (1, 1, n_mels, time_frames) -- batched for model input.
    """
    from model.explore import _load_audio_ffmpeg

    sample_rate = params["sample_rate"]
    clip_length = params["clip_length"]
    n_mels = params["n_mels"]
    n_fft = params["n_fft"]
    hop_length = params["hop_length"]
    mean = params["mean"]
    std = params["std"]

    # Load audio via ffmpeg (handles .webm, .ogg, .wav, .mp3)
    waveform, sr = _load_audio_ffmpeg(audio_path, target_sr=sample_rate)

    # Pad or trim to exactly clip_length seconds
    clip_samples = int(sample_rate * clip_length)
    if waveform.shape[1] < clip_samples:
        pad_size = clip_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        waveform = waveform[:, :clip_samples]

    # Compute mel spectrogram + log scale
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    mel = mel_transform(waveform)
    mel_db = amp_to_db(mel)

    # Normalize using train-set stats
    mel_normalized = (mel_db - mean) / std

    return mel_normalized


def _spectrogram_to_base64(spec_tensor):
    """Render spectrogram tensor as a matplotlib figure and return base64 PNG.

    Args:
        spec_tensor: tensor of shape (1, n_mels, time_frames) -- the raw dB
                     spectrogram (before normalization) for visualization.

    Returns:
        str: "data:image/png;base64,..." encoded PNG string.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
    ax.imshow(
        spec_tensor.squeeze().numpy(),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bin")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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
    model, params = _load_model()

    # Convert audio to spectrogram (normalized)
    spec_tensor = _audio_to_spectrogram(audio_path, params)

    # Also get the unnormalized version for visualization
    from model.explore import _load_audio_ffmpeg
    waveform, sr = _load_audio_ffmpeg(audio_path, target_sr=params["sample_rate"])
    clip_samples = int(params["sample_rate"] * params["clip_length"])
    if waveform.shape[1] < clip_samples:
        pad_size = clip_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        waveform = waveform[:, :clip_samples]
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=params["sample_rate"],
        n_mels=params["n_mels"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()
    raw_spec = amp_to_db(mel_transform(waveform))

    # Run inference: model.eval() already called in _load_model()
    with torch.no_grad():
        # Add batch dimension: (1, n_mels, time_frames) -> (1, 1, n_mels, time_frames)
        input_tensor = spec_tensor.unsqueeze(0) if spec_tensor.dim() == 3 else spec_tensor
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    # Determine label and confidence
    # prob = P(abnormal) since label 1 = abnormal
    if prob >= 0.5:
        label = "abnormal"
        confidence = prob
    else:
        label = "healthy"
        confidence = 1.0 - prob

    # Generate spectrogram image from raw (unnormalized) dB values
    spectrogram_b64 = _spectrogram_to_base64(raw_spec)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "spectrogram": spectrogram_b64,
    }
