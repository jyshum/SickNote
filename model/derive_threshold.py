"""
Select the shipped classification threshold on the validation split.

EXPERIMENTS.md section 3 found the previous THRESHOLD = 0.52 was the argmax of
Youden's J computed on the test set — a hyperparameter fitted to the data it
was reported on. Selecting honestly on validation picked 0.48, which performed
worse on test than leaving it at 0.50. The improvement was never real.

This script re-derives the threshold for whatever ensemble currently sits in
model/checkpoints/, using the policy stated in section 5: meet a sensitivity
floor, then take the best specificity available.

Usage: PYTHONPATH=. python -m model.derive_threshold
"""
import os

import torch

from model.architecture import SickNoteCNN
from model.config import CHECKPOINT_DIR, MIN_SENSITIVITY
from model.dataset import CoughDataset
from model.metrics import roc_auc, threshold_for_min_sensitivity
from model.train import ENSEMBLE_SEEDS


def ensemble_val_probabilities(head="pool_time", pool="sym"):
    """Average the ensemble's probabilities over the production val split."""
    splits = torch.load(
        os.path.join(CHECKPOINT_DIR, "splits.pt"), weights_only=False
    )
    params = torch.load(
        os.path.join(CHECKPOINT_DIR, "preprocessing_params.pt"), weights_only=True
    )

    val_ds = CoughDataset(
        splits["val_files"], splits["val_labels"],
        mean=params["mean"], std=params["std"],
    )
    x = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    y = [int(v) for v in splits["val_labels"]]

    summed = torch.zeros(len(val_ds))
    n_loaded = 0
    for i in range(len(ENSEMBLE_SEEDS)):
        path = os.path.join(CHECKPOINT_DIR, f"model_final_{i}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing ensemble member: {path}\n"
                f"train it first: PYTORCH_ENABLE_MPS_FALLBACK=1 "
                f"PYTHONPATH=. python -m model.train --ensemble"
            )
        model = SickNoteCNN(
            n_mels=params["n_mels"], time_frames=params["time_frames"],
            head=head, pool=pool,
        )
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            summed += torch.sigmoid(model(x)).squeeze(1)
        n_loaded += 1

    return y, (summed / n_loaded).tolist(), n_loaded


def main():
    y, probs, n_loaded = ensemble_val_probabilities()
    thr, row, met = threshold_for_min_sensitivity(y, probs, MIN_SENSITIVITY)

    print("=" * 66)
    print("Threshold selection — validation split only")
    print("=" * 66)
    print(f"  ensemble members : {n_loaded}")
    print(f"  val clips        : {len(y)}  ({sum(y)} abnormal)")
    print(f"  val AUC          : {roc_auc(y, probs):.4f}")
    print(f"  sensitivity floor: {MIN_SENSITIVITY}")
    print()
    if not met:
        print("  WARNING: no threshold met the floor. Reporting the most")
        print("           sensitive option available instead.")
    print(f"  THRESHOLD        = {thr:.2f}")
    print(f"  sensitivity      = {row['sensitivity']:.4f}")
    print(f"  specificity      = {row['specificity']:.4f}")
    print()
    print("  Paste into model/config.py, replacing the existing THRESHOLD,")
    print("  and update the date in the comment above it.")
    print("=" * 66)


if __name__ == "__main__":
    main()
