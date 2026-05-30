"""
Run FIRST before writing any model code.
Answers dataset questions that feed into config.py.

See ARCHITECTURE.md → P1 Step 0 for the 6 questions this must answer.

Usage: python -m model.explore
"""

import os
import subprocess

import numpy as np
import pandas as pd


def majority_label(row):
    """Compute binary label from expert diagnosis_1-4 columns via majority vote.

    Returns 'healthy' or 'abnormal', or None if no expert labels exist.
    Label mapping: healthy_cough → healthy | everything else → abnormal.
    Ties default to abnormal.
    """
    diagnoses = []
    for i in range(1, 5):
        val = row.get(f"diagnosis_{i}")
        if pd.notna(val):
            diagnoses.append(val)

    if not diagnoses:
        return None

    healthy_count = sum(1 for d in diagnoses if d == "healthy_cough")
    abnormal_count = len(diagnoses) - healthy_count

    if healthy_count > abnormal_count:
        return "healthy"
    else:
        return "abnormal"


def has_good_quality(row):
    """Return False if majority of experts rate quality as 'poor' or 'no_cough'.

    Return True if no quality info exists.
    """
    qualities = []
    for i in range(1, 5):
        val = row.get(f"quality_{i}")
        if pd.notna(val):
            qualities.append(val)

    if not qualities:
        return True

    bad_count = sum(1 for q in qualities if q in ("poor", "no_cough"))
    # Majority means more than half
    if bad_count > len(qualities) / 2:
        return False
    return True


def find_audio_file(uuid, raw_dir):
    """Find audio file for a UUID. COUGHVID uses mixed extensions: .webm, .wav, .ogg.

    Returns the full path if found, None otherwise.
    """
    for ext in (".webm", ".wav", ".ogg"):
        path = os.path.join(raw_dir, uuid + ext)
        if os.path.exists(path):
            return path
    return None


def _load_audio_ffmpeg(audio_path, target_sr=None):
    """Load audio using ffmpeg subprocess (handles .webm, .ogg, .wav).

    Returns (waveform_tensor, sample_rate).
    Uses ffmpeg to decode to raw PCM, then wraps in a torch tensor.
    """
    import torch

    sr = target_sr or 22050
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-f", "f32le",       # raw 32-bit float PCM
        "-acodec", "pcm_f32le",
        "-ac", "1",          # mono
        "-ar", str(sr),      # target sample rate
        "-v", "quiet",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {audio_path}: {result.stderr.decode(errors='replace')}"
        )
    pcm = np.frombuffer(result.stdout, dtype=np.float32).copy()
    waveform = torch.from_numpy(pcm).unsqueeze(0)  # (1, num_samples)
    return waveform, sr


def _get_duration_ffmpeg(audio_path, sr=22050):
    """Get audio duration in seconds by decoding via ffmpeg.

    WebM/Opus containers often lack duration metadata, so we decode
    to raw PCM and count samples.
    """
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sr),
        "-v", "quiet",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {audio_path}")
    num_samples = len(result.stdout) // 4  # 4 bytes per float32
    return num_samples / sr


def main():
    """Print answers to all 6 dataset questions. See ARCHITECTURE.md."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torchaudio

    from model.config import (
        RAW_DIR, METADATA_CSV, SAMPLE_RATE,
        CLIP_LENGTH_S, N_MELS, N_FFT, HOP_LENGTH,
    )

    print("=" * 70)
    print("SickNote — Dataset Exploration (explore.py)")
    print("=" * 70)

    # ── Load metadata ─────────────────────────────────────────────────
    df = pd.read_csv(METADATA_CSV)
    print(f"\nTotal rows in metadata: {len(df)}")

    # ── Q1: How many rows survive filtering? ──────────────────────────
    print("\n" + "─" * 70)
    print("Q1: How many rows survive filtering?")
    print("    (cough_detected > 0.8 AND expert label AND good quality)")
    print("─" * 70)

    # Filter: cough_detected > 0.8
    mask_cough = df["cough_detected"] > 0.8
    print(f"  After cough_detected > 0.8: {mask_cough.sum()}")

    # Filter: at least one expert diagnosis
    has_expert = df[["diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4"]].notna().any(axis=1)
    mask_expert = mask_cough & has_expert
    print(f"  After + expert label: {mask_expert.sum()}")

    # Filter: good quality
    df["_label"] = df.apply(majority_label, axis=1)
    df["_good_quality"] = df.apply(has_good_quality, axis=1)

    mask_quality = mask_expert & df["_good_quality"]
    # Also require a valid label (not None)
    mask_label = mask_quality & df["_label"].notna()

    filtered = df[mask_label].copy()
    print(f"  After + good quality: {len(filtered)}")
    print(f"\n  >>> SURVIVING ROWS: {len(filtered)}")

    # ── Q2: Class distribution ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Q2: Class distribution after filtering")
    print("─" * 70)

    label_counts = filtered["_label"].value_counts()
    total = len(filtered)
    for label in ["healthy", "abnormal"]:
        count = label_counts.get(label, 0)
        pct = 100.0 * count / total
        print(f"  {label:>10}: {count:>5}  ({pct:5.1f}%)")

    healthy_count = label_counts.get("healthy", 0)
    abnormal_count = label_counts.get("abnormal", 0)
    if healthy_count > 0:
        pos_weight = abnormal_count / healthy_count
        print(f"\n  pos_weight (abnormal/healthy): {pos_weight:.2f}")
    print(f"  Total: {total}")

    # ── Q3: Duration distribution ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("Q3: Duration distribution of surviving clips")
    print("─" * 70)

    durations = []
    missing_audio = 0
    for _, row in filtered.iterrows():
        audio_path = find_audio_file(row["uuid"], RAW_DIR)
        if audio_path is None:
            missing_audio += 1
            continue
        try:
            dur = _get_duration_ffmpeg(audio_path, sr=SAMPLE_RATE)
            durations.append(dur)
        except Exception:
            missing_audio += 1

    durations = np.array(durations)
    print(f"  Clips with audio found: {len(durations)}")
    print(f"  Missing/unreadable audio: {missing_audio}")

    if len(durations) == 0:
        print("\n  No audio durations collected — skipping stats.")
    else:
        print(f"\n  Median:  {np.median(durations):.2f}s")
        print(f"  P10:     {np.percentile(durations, 10):.2f}s")
        print(f"  P90:     {np.percentile(durations, 90):.2f}s")
        print(f"  Min:     {np.min(durations):.2f}s")
        print(f"  Max:     {np.max(durations):.2f}s")

    # ── Q4: Plot spectrograms ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Q4: Plotting 5 healthy + 5 abnormal spectrograms")
    print("─" * 70)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    clip_samples = int(SAMPLE_RATE * CLIP_LENGTH_S)

    def load_and_transform(audio_path):
        """Load audio via ffmpeg, pad/trim, compute mel spectrogram."""
        waveform, sr = _load_audio_ffmpeg(audio_path, target_sr=SAMPLE_RATE)
        # Pad or trim
        if waveform.shape[1] < clip_samples:
            pad_size = clip_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            waveform = waveform[:, :clip_samples]
        # Mel spectrogram + log scale
        mel = mel_transform(waveform)
        mel_db = amp_to_db(mel)
        return mel_db

    # Gather samples
    healthy_rows = filtered[filtered["_label"] == "healthy"]
    abnormal_rows = filtered[filtered["_label"] == "abnormal"]

    def gather_spectrograms(subset, n=5):
        specs = []
        for _, row in subset.iterrows():
            if len(specs) >= n:
                break
            audio_path = find_audio_file(row["uuid"], RAW_DIR)
            if audio_path is None:
                continue
            try:
                spec = load_and_transform(audio_path)
                specs.append(spec)
            except Exception:
                continue
        return specs

    healthy_specs = gather_spectrograms(healthy_rows, 5)
    abnormal_specs = gather_spectrograms(abnormal_rows, 5)

    print(f"  Healthy spectrograms: {len(healthy_specs)}")
    print(f"  Abnormal spectrograms: {len(abnormal_specs)}")

    fig, axes = plt.subplots(2, 5, figsize=(20, 6))
    for i, spec in enumerate(healthy_specs):
        ax = axes[0, i]
        ax.imshow(spec.squeeze().numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"Healthy {i+1}")
        ax.set_ylabel("Mel bin" if i == 0 else "")
        ax.set_xlabel("Time frame")
    for i, spec in enumerate(abnormal_specs):
        ax = axes[1, i]
        ax.imshow(spec.squeeze().numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"Abnormal {i+1}")
        ax.set_ylabel("Mel bin" if i == 0 else "")
        ax.set_xlabel("Time frame")

    plt.suptitle("Sample Spectrograms: Healthy (top) vs Abnormal (bottom)", fontsize=14)
    plt.tight_layout()

    os.makedirs("data", exist_ok=True)
    out_path = "data/spectrograms_sample.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved to: {out_path}")

    # ── Q5: Tensor shape after MelSpectrogram ─────────────────────────
    print("\n" + "─" * 70)
    print("Q5: Tensor shape after MelSpectrogram transform")
    print("─" * 70)

    if healthy_specs:
        example = healthy_specs[0]
        print(f"  Shape: {example.shape}")
        print(f"  -> (channels={example.shape[0]}, n_mels={example.shape[1]}, time_frames={example.shape[2]})")
        flatten_dim = example.shape[1] * example.shape[2]
        print(f"  Flatten dim (before conv): {example.shape[1]} x {example.shape[2]} = {flatten_dim}")

    # ── Q6: Count corrupted/unreadable files ──────────────────────────
    print("\n" + "─" * 70)
    print("Q6: Corrupted/unreadable files")
    print("─" * 70)

    corrupted = []
    no_audio_file = []
    for _, row in filtered.iterrows():
        audio_path = find_audio_file(row["uuid"], RAW_DIR)
        if audio_path is None:
            no_audio_file.append(row["uuid"])
            continue
        try:
            _load_audio_ffmpeg(audio_path, target_sr=SAMPLE_RATE)
        except Exception as e:
            corrupted.append((row["uuid"], str(e)))

    print(f"  No audio file found: {len(no_audio_file)}")
    print(f"  Corrupted/unreadable: {len(corrupted)}")
    if corrupted:
        print("  First 10 corrupted:")
        for uuid, err in corrupted[:10]:
            print(f"    {uuid}: {err}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — Values for config.py")
    print("=" * 70)
    print(f"  Surviving rows:     {len(filtered)}")
    print(f"  Healthy:            {healthy_count} ({100*healthy_count/total:.1f}%)")
    print(f"  Abnormal:           {abnormal_count} ({100*abnormal_count/total:.1f}%)")
    print(f"  Median duration:    {np.median(durations):.2f}s")
    print(f"  P90 duration:       {np.percentile(durations, 90):.2f}s")
    if healthy_specs:
        print(f"  Tensor shape:       {healthy_specs[0].shape}")
    print(f"  Corrupted files:    {len(corrupted)}")
    print(f"  Missing audio:      {len(no_audio_file)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
