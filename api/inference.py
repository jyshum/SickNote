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

# Models and params are cached at module level -- loaded once, not per request.
_models = None
_params = None

# Resolve checkpoint paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "model", "checkpoints")


def _load_models():
    """Load ensemble models (or single model_final.pt as fallback).

    Checks for model_final_0.pt through model_final_4.pt first (ensemble).
    Falls back to model_final.pt if no ensemble files exist.
    Caches result in module-level globals so it only runs once.
    """
    global _models, _params

    if _models is not None:
        return _models, _params

    from model.architecture import SickNoteCNN

    params_path = os.path.join(_CHECKPOINT_DIR, "preprocessing_params.pt")
    params = torch.load(params_path, weights_only=True, map_location="cpu")

    n_mels = params["n_mels"]
    time_frames = params["time_frames"]

    # Try loading ensemble members
    models = []
    for i in range(5):
        path = os.path.join(_CHECKPOINT_DIR, f"model_final_{i}.pt")
        if os.path.exists(path):
            model = SickNoteCNN(n_mels=n_mels, time_frames=time_frames)
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            model.eval()
            models.append(model)

    # Fallback to single model
    if not models:
        single_path = os.path.join(_CHECKPOINT_DIR, "model_final.pt")
        model = SickNoteCNN(n_mels=n_mels, time_frames=time_frames)
        model.load_state_dict(torch.load(single_path, weights_only=True, map_location="cpu"))
        model.eval()
        models.append(model)

    _models = models
    _params = params
    return _models, _params


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


def _compute_gradcam(model, input_tensor):
    """Compute Grad-CAM heatmap showing which spectrogram regions drove the prediction.

    Hooks into the last conv block, computes gradient-weighted activation maps,
    and returns a heatmap array resized to the input spectrogram dimensions.
    """
    activations = []
    gradients = []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = model.conv.register_forward_hook(fwd_hook)
    handle_bwd = model.conv.register_full_backward_hook(bwd_hook)

    input_copy = input_tensor.clone().requires_grad_(True)
    logits = model(input_copy)

    model.zero_grad()
    logits.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

    cam = cam.squeeze()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(input_tensor.shape[2], input_tensor.shape[3]),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    return cam.numpy()


def _gradcam_to_base64(spec_tensor, cam_array):
    """Overlay Grad-CAM heatmap on the spectrogram and return as base64 PNG."""
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.5))
    ax.imshow(
        spec_tensor.squeeze().numpy(),
        aspect="auto",
        origin="lower",
        cmap="gray",
    )
    ax.imshow(
        cam_array,
        aspect="auto",
        origin="lower",
        cmap="jet",
        alpha=0.5,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel bin")
    ax.set_title("Model Focus Areas (Grad-CAM)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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
            "spectrogram": base64-encoded PNG string ("data:image/png;base64,..."),
            "gradcam": base64-encoded PNG string — heatmap showing model focus areas
        }
    """
    models, params = _load_models()

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

    # Add batch dimension: (1, n_mels, time_frames) -> (1, 1, n_mels, time_frames)
    input_tensor = spec_tensor.unsqueeze(0) if spec_tensor.dim() == 3 else spec_tensor

    # Compute Grad-CAM from the first model (representative of ensemble focus)
    cam = _compute_gradcam(models[0], input_tensor)

    # Run ensemble inference — average probabilities across all models
    with torch.no_grad():
        probs = []
        for model in models:
            logits = model(input_tensor)
            probs.append(torch.sigmoid(logits).item())
        prob = sum(probs) / len(probs)

    from model.config import THRESHOLD

    if prob >= THRESHOLD:
        label = "abnormal"
        confidence = prob
    else:
        label = "healthy"
        confidence = 1.0 - prob

    # Generate visualizations
    spectrogram_b64 = _spectrogram_to_base64(raw_spec)
    gradcam_b64 = _gradcam_to_base64(raw_spec, cam)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "spectrogram": spectrogram_b64,
        "gradcam": gradcam_b64,
        "ensemble_size": len(models),
    }
