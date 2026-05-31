"""
Select 10 demo clips (5 healthy + 5 abnormal) from the test set.

Picks clips where the model makes confident CORRECT predictions.
Copies the original audio files (not .pt tensors) to data/demo_clips/.

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.select_demos
"""

import os
import shutil

import torch
from torch.utils.data import DataLoader

from model.config import BATCH_SIZE, CHECKPOINT_DIR, DEVICE
from model.dataset import CoughDataset
from model.architecture import SickNoteResNet
from model.explore import find_audio_file


DEMO_CLIPS_DIR = "data/demo_clips"
RAW_DIR = "data/raw/coughvid_20211012"


def main():
    print("=" * 70)
    print("SickNote -- Demo Clip Selection (select_demos.py)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load ensemble models and preprocessing params
    # ------------------------------------------------------------------
    params = torch.load(
        os.path.join(CHECKPOINT_DIR, "preprocessing_params.pt"),
        weights_only=True,
        map_location="cpu",
    )
    splits = torch.load(
        os.path.join(CHECKPOINT_DIR, "splits.pt"),
        weights_only=False,
    )

    mean = params["mean"]
    std = params["std"]

    models = []
    for i in range(3):
        path = os.path.join(CHECKPOINT_DIR, f"model_final_{i}.pt")
        if os.path.exists(path):
            m = SickNoteResNet()
            m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            m.to(DEVICE)
            m.eval()
            models.append(m)

    if not models:
        single_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
        m = SickNoteResNet()
        m.load_state_dict(torch.load(single_path, weights_only=True, map_location="cpu"))
        m.to(DEVICE)
        m.eval()
        models.append(m)

    print(f"Loaded {len(models)} model(s)")

    test_files = splits["test_files"]
    test_labels = splits["test_labels"]

    print(f"\nTest samples: {len(test_files)}")
    print(f"  Healthy (0): {test_labels.count(0)}")
    print(f"  Abnormal (1): {test_labels.count(1)}")

    # ------------------------------------------------------------------
    # 2. Run inference on all test samples
    # ------------------------------------------------------------------
    test_ds = CoughDataset(test_files, test_labels, mean=mean, std=std)

    # Collect per-sample predictions
    results = []  # list of (file_path, true_label, probability)

    with torch.no_grad():
        for i in range(len(test_ds)):
            spec, label = test_ds[i]
            spec = spec.unsqueeze(0).to(DEVICE)
            probs = []
            for m in models:
                logit = m(spec)
                probs.append(torch.sigmoid(logit).item())
            prob = sum(probs) / len(probs)
            results.append((test_files[i], int(label.item()), prob))

    # ------------------------------------------------------------------
    # 3. Find confident correct predictions
    # ------------------------------------------------------------------
    # Correct healthy: true_label=0, prob < 0.5 -> confidence = 1-prob
    # Correct abnormal: true_label=1, prob >= 0.5 -> confidence = prob
    correct_healthy = []
    correct_abnormal = []

    for filepath, true_label, prob in results:
        pred_label = 1 if prob >= 0.5 else 0
        if pred_label != true_label:
            continue  # skip incorrect predictions

        if true_label == 0:
            confidence = 1.0 - prob
            correct_healthy.append((filepath, confidence, prob))
        else:
            confidence = prob
            correct_abnormal.append((filepath, confidence, prob))

    # Sort by confidence (descending) to get most confident correct predictions
    correct_healthy.sort(key=lambda x: x[1], reverse=True)
    correct_abnormal.sort(key=lambda x: x[1], reverse=True)

    print(f"\nCorrect healthy predictions: {len(correct_healthy)}")
    print(f"Correct abnormal predictions: {len(correct_abnormal)}")

    # ------------------------------------------------------------------
    # 4. Select top 5 of each and copy audio files
    # ------------------------------------------------------------------
    os.makedirs(DEMO_CLIPS_DIR, exist_ok=True)

    # Clean existing demo clips
    for f in os.listdir(DEMO_CLIPS_DIR):
        fpath = os.path.join(DEMO_CLIPS_DIR, f)
        if os.path.isfile(fpath) and f != ".gitkeep":
            os.remove(fpath)

    selected_healthy = correct_healthy[:5]
    selected_abnormal = correct_abnormal[:5]

    print(f"\nSelected {len(selected_healthy)} healthy + {len(selected_abnormal)} abnormal clips")

    copied = 0

    print("\nHealthy clips:")
    for filepath, confidence, prob in selected_healthy:
        uuid = os.path.basename(filepath).replace(".pt", "")
        audio_path = find_audio_file(uuid, RAW_DIR)
        if audio_path is None:
            print(f"  SKIP {uuid}: audio file not found")
            continue
        ext = os.path.splitext(audio_path)[1]
        dest = os.path.join(DEMO_CLIPS_DIR, f"healthy_{copied + 1}_{uuid}{ext}")
        shutil.copy2(audio_path, dest)
        print(f"  {os.path.basename(dest)} (confidence={confidence:.4f})")
        copied += 1

    abnormal_copied = 0
    print("\nAbnormal clips:")
    for filepath, confidence, prob in selected_abnormal:
        uuid = os.path.basename(filepath).replace(".pt", "")
        audio_path = find_audio_file(uuid, RAW_DIR)
        if audio_path is None:
            print(f"  SKIP {uuid}: audio file not found")
            continue
        ext = os.path.splitext(audio_path)[1]
        dest = os.path.join(DEMO_CLIPS_DIR, f"abnormal_{abnormal_copied + 1}_{uuid}{ext}")
        shutil.copy2(audio_path, dest)
        print(f"  {os.path.basename(dest)} (confidence={confidence:.4f})")
        abnormal_copied += 1

    total = copied + abnormal_copied
    print(f"\n{'=' * 70}")
    print(f"DONE. Copied {total} demo clips to {DEMO_CLIPS_DIR}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
