"""
Raw audio → mel spectrogram tensor pipeline.

Reads COUGHVID audio files, filters by expert labels and quality,
converts to normalized log-mel spectrograms, saves as .pt tensors,
and creates stratified train/val/test splits.

See ARCHITECTURE.md → Data Pipeline for the full processing flow.

Usage: python -m model.preprocess
"""


def load_and_filter_metadata():
    """Load CSV, filter to expert-labeled + cough_detected > 0.8 + quality.

    Returns a DataFrame with columns: uuid, label_int (0=healthy, 1=abnormal).
    Uses majority_label() and has_good_quality() from explore.py.
    """
    raise NotImplementedError


def audio_to_spectrogram(audio_path):
    """Load audio, resample to SAMPLE_RATE, pad/trim, convert to log-mel spectrogram.

    Returns: torch.Tensor of shape (1, N_MELS, TIME_FRAMES)
    """
    raise NotImplementedError


def main():
    """Full preprocessing pipeline. Saves:
    - .pt tensor per clip in PROCESSED_DIR
    - checkpoints/splits.pt (file lists + labels for train/val/test)
    - checkpoints/preprocessing_params.pt (mean, std, audio params)

    CRITICAL: compute normalization mean/std on train split ONLY.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
