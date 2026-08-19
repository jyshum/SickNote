"""
Evaluate the shipped ensemble on the original held-out test split.

The threshold is selected on the VALIDATION split and then applied, unchanged,
to test. The previous version searched thresholds directly on the test set,
which fitted a hyperparameter to the data it was reporting on.

For the trustworthy number, prefer model/crossval.py — this script measures a
single 341-clip split, which we have measured to be worth roughly +-0.05 AUC.

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.evaluate
"""

import os

import numpy as np
import torch

from model import metrics
from model.architecture import SickNoteCNN
from model.config import CHECKPOINT_DIR, THRESHOLD, DEVICE, MIN_SENSITIVITY


def _load_models():
    """Load ensemble models (or single model_final.pt as fallback)."""
    params = torch.load(
        os.path.join(CHECKPOINT_DIR, "preprocessing_params.pt"), weights_only=True
    )
    n_mels = params["n_mels"]
    time_frames = params["time_frames"]

    models = []
    for i in range(5):
        path = os.path.join(CHECKPOINT_DIR, f"model_final_{i}.pt")
        if os.path.exists(path):
            model = SickNoteCNN(n_mels=n_mels, time_frames=time_frames)
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            model.to(DEVICE)
            model.eval()
            models.append(model)

    if not models:
        single_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
        model = SickNoteCNN(n_mels=n_mels, time_frames=time_frames)
        model.load_state_dict(torch.load(single_path, weights_only=True, map_location="cpu"))
        model.to(DEVICE)
        model.eval()
        models.append(model)

    return models, params


@torch.no_grad()
def _ensemble_probs(models, files, mean, std, batch_size=128):
    """Average the ensemble's sigmoid outputs over a list of .pt files."""
    out = []
    for start in range(0, len(files), batch_size):
        batch = torch.stack([
            torch.load(f, weights_only=True) for f in files[start:start + batch_size]
        ])
        x = ((batch - mean) / std).to(DEVICE)
        per_model = [torch.sigmoid(m(x)).squeeze(1).cpu() for m in models]
        out.append(torch.stack(per_model).mean(dim=0))
    return torch.cat(out).numpy()


def _report(name, y, probs, threshold):
    row = [r for r in metrics.sweep(y, probs, [threshold])][0]
    print(f"\n{name} (threshold={threshold:.2f})")
    print(f"  {'AUC-ROC':<14}{metrics.roc_auc(y, probs):.4f}")
    print(f"  {'Accuracy':<14}{row['accuracy']:.4f}")
    print(f"  {'Sensitivity':<14}{row['sensitivity']:.4f}   (caught {row['tp']} of {row['tp'] + row['fn']} abnormal)")
    print(f"  {'Specificity':<14}{row['specificity']:.4f}   (cleared {row['tn']} of {row['tn'] + row['fp']} healthy)")
    print(f"  confusion: TP={row['tp']} TN={row['tn']} FP={row['fp']} FN={row['fn']}")


def main():
    print("=" * 72)
    print("SickNote — single-split evaluation (threshold chosen on val)")
    print("=" * 72)

    models, params = _load_models()
    splits = torch.load(os.path.join(CHECKPOINT_DIR, "splits.pt"), weights_only=False)
    mean, std = params["mean"], params["std"]

    y_val = np.asarray(splits["val_labels"], dtype=int)
    y_test = np.asarray(splits["test_labels"], dtype=int)

    print(f"\nDevice: {DEVICE}   Ensemble size: {len(models)}")
    print(f"Val: {len(y_val)} clips   Test: {len(y_test)} clips")

    val_probs = _ensemble_probs(models, splits["val_files"], mean, std)
    test_probs = _ensemble_probs(models, splits["test_files"], mean, std)

    # ---- threshold selected on VAL only ----
    thr, row, met = metrics.threshold_for_min_sensitivity(y_val, val_probs, MIN_SENSITIVITY)
    print(f"\nThreshold chosen on val for min sensitivity {MIN_SENSITIVITY}: {thr:.2f}")
    print(f"  on val -> sensitivity {row['sensitivity']:.4f}, specificity {row['specificity']:.4f}")
    if not met:
        print(f"  WARNING: sensitivity floor not reachable on this split")

    yj_val, _ = metrics.youden_threshold(y_val, val_probs)
    print(f"  (Youden's J on val would pick {yj_val:.2f}; config currently ships {THRESHOLD:.2f})")

    # ---- applied unchanged to TEST ----
    _report("TEST — sensitivity-first threshold", y_test, test_probs, thr)
    _report("TEST — currently shipped threshold", y_test, test_probs, THRESHOLD)

    point, lo, hi, _ = metrics.bootstrap_auc_ci(y_test, test_probs)
    print(f"\nTest AUC with uncertainty: {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print("  This interval is wide because it rests on one 341-clip split.")
    print("  Run `python -m model.crossval` for the 2,267-clip estimate.")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
