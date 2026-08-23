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
    """Load the production ensemble, or refuse to start.

    Every failure here is raised rather than worked around. A screening tool
    that quietly serves a different model than the one that was measured is
    worse than one that will not start: the old code fell back to a single
    model_final.pt, so a stale checkpoint or a half-copied deploy would run
    and answer confidently. Caches into module globals so this runs once.
    """
    global _models, _params

    if _models is not None:
        return _models, _params

    from model.architecture import SickNoteCNN
    from model.config import ENSEMBLE_SEEDS, HEAD as _HEAD, POOL as _POOL

    # Read from model.config, never from model.train: train.py imports
    # scikit-learn at module level, and commit da1ac74 deliberately dropped it
    # from requirements.txt to cut ~150MB off the API container. Importing
    # train.py here would pass locally, where sklearn happens to be in the dev
    # venv, and raise ModuleNotFoundError on every prediction in production.
    # config.py imports only torch, so it is safe from the serving path.
    expected = len(ENSEMBLE_SEEDS)

    params_path = os.path.join(_CHECKPOINT_DIR, "preprocessing_params.pt")
    params = torch.load(params_path, weights_only=True, map_location="cpu")

    n_mels = params["n_mels"]
    time_frames = params["time_frames"]

    found = [
        i for i in range(expected)
        if os.path.exists(os.path.join(_CHECKPOINT_DIR, f"model_final_{i}.pt"))
    ]
    if not found:
        raise RuntimeError(
            f"no ensemble checkpoints in {_CHECKPOINT_DIR} — expected "
            f"model_final_0.pt through model_final_{expected - 1}.pt. "
            f"Train them with: PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. "
            f"python -m model.train --ensemble"
        )

    # The count is deliberately checked AFTER the load loop. An architecture
    # mismatch on a checkpoint that is present is a more specific and more
    # actionable error than "wrong number of files", so when both are true the
    # mismatch wins.
    models = []
    for i in found:
        path = os.path.join(_CHECKPOINT_DIR, f"model_final_{i}.pt")
        model = SickNoteCNN(
            n_mels=n_mels, time_frames=time_frames, head=_HEAD, pool=_POOL
        )
        try:
            model.load_state_dict(
                torch.load(path, weights_only=True, map_location="cpu")
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"{os.path.basename(path)} does not match the "
                f"head={_HEAD!r} pool={_POOL!r} architecture this build "
                f"expects. It is probably a checkpoint from before the "
                f"pooling head landed — retrain with "
                f"`python -m model.train --ensemble`. Original error: {e}"
            ) from e
        model.eval()
        models.append(model)

    if len(models) != expected:
        raise RuntimeError(
            f"partial ensemble: loaded {len(models)} of {expected} members "
            f"from {_CHECKPOINT_DIR} (present: {found}). Averaging a partial "
            f"ensemble would not be the model that was measured — retrain "
            f"with `python -m model.train --ensemble`."
        )

    _models = models
    _params = params
    return _models, _params


def _audio_to_spectrogram(audio_path, params):
    """Convert audio file to normalized log-mel spectrogram tensor.

    Same pipeline as preprocess.py but uses params from preprocessing_params.pt
    instead of config.py constants, ensuring consistency with training.

    Returns: torch.Tensor of shape (1, 1, n_mels, time_frames) -- batched for model input.
    """
    from api.audio_utils import load_audio_ffmpeg

    sample_rate = params["sample_rate"]
    clip_length = params["clip_length"]
    n_mels = params["n_mels"]
    n_fft = params["n_fft"]
    hop_length = params["hop_length"]
    mean = params["mean"]
    std = params["std"]

    # Load audio via ffmpeg (handles .webm, .ogg, .wav, .mp3)
    waveform, sr = load_audio_ffmpeg(audio_path, target_sr=sample_rate)

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
    from api.audio_utils import load_audio_ffmpeg
    waveform, sr = load_audio_ffmpeg(audio_path, target_sr=params["sample_rate"])
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
