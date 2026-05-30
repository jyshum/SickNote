"""
Raw audio → mel spectrogram tensor pipeline.

Reads COUGHVID audio files, filters by expert labels and quality,
converts to normalized log-mel spectrograms, saves as .pt tensors,
and creates stratified train/val/test splits.

See ARCHITECTURE.md → Data Pipeline for the full processing flow.

Usage: python -m model.preprocess
"""

import os

import pandas as pd
import torch
import torchaudio

from model.config import (
    SAMPLE_RATE,
    CLIP_LENGTH_S,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    RAW_DIR,
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    METADATA_CSV,
)
from model.explore import (
    majority_label,
    has_good_quality,
    find_audio_file,
    _load_audio_ffmpeg,
)


def load_and_filter_metadata(csv_path=None):
    """Load CSV, filter to expert-labeled + cough_detected > 0.8 + quality.

    Returns a DataFrame with columns: uuid, label, label_int (0=healthy, 1=abnormal).
    Uses majority_label() and has_good_quality() from explore.py.
    """
    if csv_path is None:
        csv_path = METADATA_CSV

    df = pd.read_csv(csv_path)

    # Filter: cough_detected > 0.8
    mask_cough = df["cough_detected"] > 0.8

    # Filter: at least one expert diagnosis
    has_expert = df[
        ["diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4"]
    ].notna().any(axis=1)

    # Compute label and quality
    df["label"] = df.apply(majority_label, axis=1)
    df["_good_quality"] = df.apply(has_good_quality, axis=1)

    # Combine all filters: cough detected, expert label, good quality, valid label
    mask = mask_cough & has_expert & df["_good_quality"] & df["label"].notna()

    filtered = df[mask].copy()

    # Map label to int: healthy=0, abnormal=1
    filtered["label_int"] = (filtered["label"] == "abnormal").astype(int)

    # Return only the columns we need
    result = filtered[["uuid", "label", "label_int"]].reset_index(drop=True)
    return result


def audio_to_spectrogram(audio_path):
    """Load audio, resample to SAMPLE_RATE, pad/trim, convert to log-mel spectrogram.

    Returns: torch.Tensor of shape (1, N_MELS, TIME_FRAMES)
    """
    # Load audio via ffmpeg (handles .webm, .ogg, .wav)
    waveform, sr = _load_audio_ffmpeg(audio_path, target_sr=SAMPLE_RATE)

    # Pad or trim to exactly CLIP_LENGTH_S seconds
    clip_samples = int(SAMPLE_RATE * CLIP_LENGTH_S)
    if waveform.shape[1] < clip_samples:
        pad_size = clip_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        waveform = waveform[:, :clip_samples]

    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    mel = mel_transform(waveform)
    mel_db = amp_to_db(mel)

    return mel_db


def main():
    """Full preprocessing pipeline. Saves:
    - .pt tensor per clip in PROCESSED_DIR
    - checkpoints/splits.pt (file lists + labels for train/val/test)
    - checkpoints/preprocessing_params.pt (mean, std, audio params)

    CRITICAL: compute normalization mean/std on train split ONLY.
    """
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("SickNote — Preprocessing Pipeline (preprocess.py)")
    print("=" * 70)

    # Create output directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Step 1: Load and filter metadata ──────────────────────────────
    print("\n[1/5] Loading and filtering metadata...")
    metadata = load_and_filter_metadata()
    print(f"  Rows after filtering: {len(metadata)}")
    print(f"  Healthy (0): {(metadata['label_int'] == 0).sum()}")
    print(f"  Abnormal (1): {(metadata['label_int'] == 1).sum()}")

    # ── Step 2: Convert each audio file to spectrogram tensor ─────────
    print(f"\n[2/5] Converting audio files to spectrograms...")
    saved_files = []
    saved_labels = []
    skipped = 0

    for idx, row in metadata.iterrows():
        uuid = row["uuid"]
        label_int = row["label_int"]

        audio_path = find_audio_file(uuid, RAW_DIR)
        if audio_path is None:
            skipped += 1
            continue

        try:
            tensor = audio_to_spectrogram(audio_path)
            out_path = os.path.join(PROCESSED_DIR, f"{uuid}.pt")
            torch.save(tensor, out_path)
            saved_files.append(out_path)
            saved_labels.append(label_int)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  SKIP {uuid}: {e}")

        # Progress indicator
        total_done = len(saved_files) + skipped
        if total_done % 200 == 0:
            print(f"  Processed {total_done}/{len(metadata)} "
                  f"(saved: {len(saved_files)}, skipped: {skipped})")

    print(f"  Done. Saved: {len(saved_files)}, Skipped: {skipped}")

    # ── Step 3: Stratified 70/15/15 split ─────────────────────────────
    print(f"\n[3/5] Creating stratified 70/15/15 splits...")

    # First split: 70% train, 30% temp
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        saved_files,
        saved_labels,
        test_size=0.30,
        random_state=42,
        stratify=saved_labels,
    )

    # Second split: 50/50 of the 30% → 15% val, 15% test
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels,
    )

    print(f"  Train: {len(train_files)} "
          f"(healthy={train_labels.count(0)}, abnormal={train_labels.count(1)})")
    print(f"  Val:   {len(val_files)} "
          f"(healthy={val_labels.count(0)}, abnormal={val_labels.count(1)})")
    print(f"  Test:  {len(test_files)} "
          f"(healthy={test_labels.count(0)}, abnormal={test_labels.count(1)})")

    # ── Step 4: Compute normalization mean/std on TRAIN ONLY ──────────
    print(f"\n[4/5] Computing normalization stats on train split only...")

    # Accumulate statistics using Welford's online algorithm for numerical stability
    n_pixels = 0
    running_sum = 0.0
    running_sq_sum = 0.0
    time_frames = None

    for f in train_files:
        t = torch.load(f, weights_only=True)
        if time_frames is None:
            time_frames = t.shape[2]
        n_pixels += t.numel()
        running_sum += t.sum().item()
        running_sq_sum += (t ** 2).sum().item()

    train_mean = running_sum / n_pixels
    train_std = ((running_sq_sum / n_pixels) - (train_mean ** 2)) ** 0.5

    print(f"  Train mean: {train_mean:.4f}")
    print(f"  Train std:  {train_std:.4f}")
    print(f"  Time frames: {time_frames}")

    # ── Step 5: Save splits.pt and preprocessing_params.pt ────────────
    print(f"\n[5/5] Saving splits.pt and preprocessing_params.pt...")

    splits = {
        "train_files": train_files,
        "train_labels": train_labels,
        "val_files": val_files,
        "val_labels": val_labels,
        "test_files": test_files,
        "test_labels": test_labels,
    }
    splits_path = os.path.join(CHECKPOINT_DIR, "splits.pt")
    torch.save(splits, splits_path)
    print(f"  Saved: {splits_path}")

    params = {
        "mean": train_mean,
        "std": train_std,
        "sample_rate": SAMPLE_RATE,
        "clip_length": CLIP_LENGTH_S,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "time_frames": time_frames,
    }
    params_path = os.path.join(CHECKPOINT_DIR, "preprocessing_params.pt")
    torch.save(params, params_path)
    print(f"  Saved: {params_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total spectrograms: {len(saved_files)}")
    print(f"  Skipped:            {skipped}")
    print(f"  Train/Val/Test:     {len(train_files)}/{len(val_files)}/{len(test_files)}")
    print(f"  Train mean:         {train_mean:.4f}")
    print(f"  Train std:          {train_std:.4f}")
    print(f"  Tensor shape:       (1, {N_MELS}, {time_frames})")
    print(f"  Output dir:         {PROCESSED_DIR}")
    print(f"  Splits file:        {splits_path}")
    print(f"  Params file:        {params_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
