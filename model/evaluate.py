"""
Evaluate trained model on held-out test set.

Reports: AUC-ROC, accuracy, sensitivity, specificity, confusion matrix.
See ARCHITECTURE.md -> P1 Step 4 for target thresholds.

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.evaluate
"""

import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from model.config import BATCH_SIZE, CHECKPOINT_DIR, THRESHOLD, DEVICE
from model.dataset import CoughDataset
from model.architecture import SickNoteCNN


def _load_models():
    """Load ensemble models (or single model_final.pt as fallback)."""
    params = torch.load(
        os.path.join(CHECKPOINT_DIR, "preprocessing_params.pt"),
        weights_only=True,
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


def main():
    """Load ensemble (or single model), run on test split, print metrics."""
    print("=" * 70)
    print("SickNote -- Test Set Evaluation (evaluate.py)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load models and preprocessing params
    # ------------------------------------------------------------------
    models, params = _load_models()

    splits = torch.load(
        os.path.join(CHECKPOINT_DIR, "splits.pt"),
        weights_only=False,
    )

    mean = params["mean"]
    std = params["std"]

    test_files = splits["test_files"]
    test_labels = splits["test_labels"]

    print(f"\nDevice: {DEVICE}")
    print(f"Ensemble size: {len(models)}")
    print(f"Test samples: {len(test_files)}")
    print(f"  Healthy (0): {test_labels.count(0)}")
    print(f"  Abnormal (1): {test_labels.count(1)}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Normalization: mean={mean:.4f}, std={std:.4f}")

    # ------------------------------------------------------------------
    # 3. Create dataset and dataloader
    # ------------------------------------------------------------------
    test_ds = CoughDataset(test_files, test_labels, mean=mean, std=std)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------
    # 4. Run ensemble inference — average probabilities across models
    # ------------------------------------------------------------------
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms = spectrograms.to(DEVICE)
            batch_probs = []
            for model in models:
                logits = model(spectrograms)
                batch_probs.append(torch.sigmoid(logits).squeeze(1).cpu())
            avg_probs = torch.stack(batch_probs).mean(dim=0).tolist()
            truth = labels.tolist()

            if isinstance(avg_probs, float):
                avg_probs = [avg_probs]
            if isinstance(truth, float):
                truth = [truth]

            all_probs.extend(avg_probs)
            all_labels.extend(truth)

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    all_labels_int = [int(l) for l in all_labels]
    all_preds = [1 if p >= THRESHOLD else 0 for p in all_probs]

    auc = roc_auc_score(all_labels_int, all_probs)
    acc = accuracy_score(all_labels_int, all_preds)

    cm = confusion_matrix(all_labels_int, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall on abnormal
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall on healthy

    # ------------------------------------------------------------------
    # 6. Report results with ARCHITECTURE.md targets
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<16} {'Actual':>8}  {'Minimum':>8}  {'Target':>8}  {'Status'}")
    print("-" * 65)

    def status(val, minimum, target):
        if val >= target:
            return "PASS (target)"
        elif val >= minimum:
            return "PASS (minimum)"
        else:
            return "BELOW MINIMUM"

    print(f"{'AUC-ROC':<16} {auc:>8.4f}  {0.75:>8.2f}  {0.82:>8.2f}  {status(auc, 0.75, 0.82)}")
    print(f"{'Accuracy':<16} {acc:>8.4f}  {0.70:>8.2f}  {0.78:>8.2f}  {status(acc, 0.70, 0.78)}")
    print(f"{'Sensitivity':<16} {sensitivity:>8.4f}  {0.65:>8.2f}  {0.75:>8.2f}  {status(sensitivity, 0.65, 0.75)}")
    print(f"{'Specificity':<16} {specificity:>8.4f}  {0.60:>8.2f}  {0.70:>8.2f}  {status(specificity, 0.60, 0.70)}")

    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred=0(H)  Pred=1(A)")
    print(f"  Actual=0(H)  {tn:>6}     {fp:>6}")
    print(f"  Actual=1(A)  {fn:>6}     {tp:>6}")
    print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # ------------------------------------------------------------------
    # 7. Debugging checklist if AUC < 0.75
    # ------------------------------------------------------------------
    if auc < 0.75:
        print("\n" + "!" * 70)
        print("WARNING: AUC < 0.75 -- Debugging Checklist")
        print("!" * 70)
        print("1. Are class weights computed from filtered expert-labeled data, not full dataset?")
        print("2. Is normalization fitted on train split only?")
        print("3. Is the model seeing enough healthy examples?")
        print("   (print batch labels -- healthy is minority ~22%)")
        print("4. Try LR = 1e-4, more epochs, lighter model")
        print("\nNote: AUC in the 0.60-0.70 range is expected with ~2200")
        print("expert-labeled samples. The small dataset limits performance.")

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
