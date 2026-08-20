"""
Nested 5-fold cross-validation — the honest measurement harness.

Every clip is scored exactly once, by the one model that never trained on it.
Pooling those out-of-fold predictions gives an AUC over the whole dataset
(2,267 clips) instead of a single 341-clip test split, which is what makes it
possible to tell a real improvement from split luck.

Per fold:
    test  = the held-out fold
    val   = CV_VAL_FRACTION of the remaining folds  (early stopping + threshold)
    train = the rest

Normalisation stats and pos_weight are recomputed from each fold's TRAIN split
only. Using global stats would leak test information into training.

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.crossval
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from model import metrics
from model.architecture import SickNoteCNN
from model.config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOPPING_PATIENCE,
    CHECKPOINT_DIR, CV_CHECKPOINT_DIR, DEVICE,
    N_FOLDS, CV_VAL_FRACTION, CV_SEED, MIN_SENSITIVITY,
)


SMOOTH_WINDOW = 3  # epochs averaged by the val_auc_smooth criterion

# SpecAugment masks, sized for a 64 x 345 spectrogram. Deliberately milder
# than the speech-recognition defaults — 1,542 training clips is not much to
# hide parts of.
FREQ_MASKS, FREQ_PARAM = 2, 12   # up to 2 bands of <=12 of 64 mel bins
TIME_MASKS, TIME_PARAM = 2, 40   # up to 2 bands of <=40 of 345 frames


def spec_augment(x, rng):
    """Black out random frequency and time bands. Training only.

    Applied AFTER normalisation, so a masked region is set to 0 — which is the
    dataset mean, i.e. "no information here" rather than "silence". Silence has
    its own meaning in a spectrogram and we do not want to fake it.

    Unlike augmenting the waveform (noise injection, time stretching, both
    tried and dropped earlier in this project), nothing here distorts the
    signal. Whatever survives the mask is untouched real audio; the model just
    cannot lean on any single frequency band or moment to make its decision.
    """
    x = x.clone()
    n, _, n_mels, n_frames = x.shape
    for i in range(n):
        for _ in range(FREQ_MASKS):
            f = int(rng.integers(0, FREQ_PARAM + 1))
            if f:
                f0 = int(rng.integers(0, n_mels - f + 1))
                x[i, :, f0:f0 + f, :] = 0.0
        for _ in range(TIME_MASKS):
            t = int(rng.integers(0, TIME_PARAM + 1))
            if t:
                t0 = int(rng.integers(0, n_frames - t + 1))
                x[i, :, :, t0:t0 + t] = 0.0
    return x


def selection_score(metric, val_auc_history, val_loss_history):
    """Higher is better, for every criterion.

    val_auc         what the project has always used. On a 272-clip val split
                    this is noisy enough to latch onto a single fluke epoch.
    val_loss        lower variance — every clip's probability contributes,
                    not just the ranking. Negated so higher is better.
    val_auc_smooth  mean AUC over the last SMOOTH_WINDOW epochs, so a peak has
                    to persist for several epochs before it is believed.
    """
    if metric == "val_auc":
        return val_auc_history[-1]
    if metric == "val_loss":
        return -val_loss_history[-1]
    if metric == "val_auc_smooth":
        window = val_auc_history[-SMOOTH_WINDOW:]
        return sum(window) / len(window)
    raise ValueError(f"unknown selection metric: {metric}")


def load_dataset():
    """Load every preprocessed spectrogram into RAM (~200 MB).

    Reads the file list from splits.pt but ignores its train/val/test
    boundaries — cross-validation makes its own. Returns raw (unnormalised)
    tensors; each fold normalises with its own training stats.
    """
    splits = torch.load(os.path.join(CHECKPOINT_DIR, "splits.pt"), weights_only=False)

    files, labels = [], []
    for part in ("train", "val", "test"):
        files.extend(splits[f"{part}_files"])
        labels.extend(splits[f"{part}_labels"])

    tensors = [torch.load(f, weights_only=True) for f in files]
    X = torch.stack(tensors)
    y = np.asarray(labels, dtype=int)
    uuids = [os.path.basename(f)[:-3] for f in files]
    return X, y, uuids


def make_folds(y, n_folds=N_FOLDS, seed=CV_SEED):
    """Stratified fold assignment. Returns an array giving each clip's fold.

    Each class is shuffled and dealt round-robin, so every fold carries the
    same healthy/abnormal ratio as the full dataset.
    """
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_folds
    return fold_of


def stratified_split(idx, y, fraction, seed):
    """Split idx into (larger, smaller) keeping the class ratio in both."""
    rng = np.random.default_rng(seed)
    small, large = [], []
    for cls in np.unique(y[idx]):
        members = idx[y[idx] == cls]
        members = members.copy()
        rng.shuffle(members)
        cut = max(1, int(round(len(members) * fraction)))
        small.extend(members[:cut])
        large.extend(members[cut:])
    return np.asarray(large), np.asarray(small)


def _batches(n, batch_size, shuffle, rng=None):
    order = np.arange(n)
    if shuffle:
        rng.shuffle(order)
    for s in range(0, n, batch_size):
        yield order[s:s + batch_size]


@torch.no_grad()
def predict(model, X, idx, mean, std, batch_size=256):
    """Probabilities for the given indices."""
    model.eval()
    out = []
    for b in _batches(len(idx), batch_size, shuffle=False):
        xb = ((X[idx[b]] - mean) / std).to(DEVICE)
        out.append(torch.sigmoid(model(xb)).squeeze(1).cpu())
    return torch.cat(out).numpy()


def train_fold(X, y, train_idx, val_idx, fold, seed, curve_rows, select="val_auc",
               head="flatten", specaug=False, max_epochs=EPOCHS):
    """Train one fold's model, returning the weights the criterion liked best.

    `select` chooses the early-stopping criterion — see selection_score().

    Returns (model, mean, std, best_val_auc, best_epoch, epochs_run).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ---- normalisation from THIS fold's training clips only ----
    train_x = X[train_idx]
    mean = float(train_x.mean())
    std = float(train_x.std())

    n_pos = int(y[train_idx].sum())
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)

    model = SickNoteCNN(n_mels=X.shape[2], time_frames=X.shape[3], head=head).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

    best_score, best_auc, best_epoch = float("-inf"), 0.0, 0
    best_state, patience, epochs_run = None, 0, 0
    auc_history, loss_history = [], []

    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        model.train()
        running = 0.0
        for b in _batches(len(train_idx), BATCH_SIZE, shuffle=True, rng=rng):
            xb = (train_x[b] - mean) / std
            if specaug:
                xb = spec_augment(xb, rng)
            xb = xb.to(DEVICE)
            yb = y_train[b].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(b)
        train_loss = running / len(train_idx)

        # ---- validation ----
        model.eval()
        val_running = 0.0
        probs = []
        with torch.no_grad():
            for b in _batches(len(val_idx), 256, shuffle=False):
                xb = ((X[val_idx[b]] - mean) / std).to(DEVICE)
                yb = y_val[b].to(DEVICE)
                logits = model(xb)
                val_running += criterion(logits, yb).item() * len(b)
                probs.append(torch.sigmoid(logits).squeeze(1).cpu())
        val_loss = val_running / len(val_idx)
        val_auc = metrics.roc_auc(y[val_idx], torch.cat(probs).numpy())

        scheduler.step(val_auc)
        curve_rows.append({
            "fold": fold, "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auc": round(val_auc, 6),
            "lr": optimizer.param_groups[0]["lr"],
        })

        auc_history.append(val_auc)
        loss_history.append(val_loss)
        score = selection_score(select, auc_history, loss_history)

        if score > best_score:
            best_score, best_auc, best_epoch = score, val_auc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, mean, std, best_auc, best_epoch, epochs_run


def main(select="val_auc", head="flatten", specaug=False, max_epochs=EPOCHS, seed=CV_SEED):
    print("=" * 72)
    print(f"SickNote — nested 5-fold cross-validation")
    print(f"  head={head}  select={select}  specaug={specaug}  max_epochs={max_epochs}  seed={seed}")
    print("=" * 72)

    # Each criterion writes to its own directory so ablation runs never
    # overwrite one another.
    tag = f"{head}__{select}__ep{max_epochs}" + ("__specaug" if specaug else "")
    if seed != CV_SEED:
        tag += f"__seed{seed}"
    out_dir = os.path.join(CV_CHECKPOINT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    X, y, uuids = load_dataset()
    probe = SickNoteCNN(n_mels=X.shape[2], time_frames=X.shape[3], head=head)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"\nDevice: {DEVICE}   head: {head}   parameters: {n_params:,}")
    print(f"Clips: {len(y)}  (abnormal={int(y.sum())}, healthy={len(y) - int(y.sum())})")
    print(f"Spectrogram: {tuple(X.shape[1:])}   in RAM: {X.numel() * 4 / 1e6:.0f} MB")

    fold_of = make_folds(y)
    oof = np.full(len(y), np.nan)          # one prediction per clip
    fold_thresholds, fold_aucs, curve_rows = [], [], []

    for fold in range(N_FOLDS):
        test_idx = np.flatnonzero(fold_of == fold)
        rest_idx = np.flatnonzero(fold_of != fold)
        train_idx, val_idx = stratified_split(rest_idx, y, CV_VAL_FRACTION, seed + fold)

        t0 = time.time()
        model, mean, std, best_val_auc, best_epoch, epochs = train_fold(
            X, y, train_idx, val_idx, fold, seed + fold, curve_rows,
            select=select, head=head, specaug=specaug, max_epochs=max_epochs
        )

        val_probs = predict(model, X, val_idx, mean, std)
        test_probs = predict(model, X, test_idx, mean, std)
        oof[test_idx] = test_probs

        thr, row, met = metrics.threshold_for_min_sensitivity(
            y[val_idx], val_probs, MIN_SENSITIVITY
        )
        fold_thresholds.append(thr)
        test_auc = metrics.roc_auc(y[test_idx], test_probs)
        fold_aucs.append(test_auc)

        torch.save(model.state_dict(), os.path.join(out_dir, f"fold_{fold}.pt"))
        torch.save({"mean": mean, "std": std, "threshold": thr},
                   os.path.join(out_dir, f"fold_{fold}_params.pt"))

        print(f"\nfold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
              f"  ({epochs} epochs, kept epoch {best_epoch}, {time.time() - t0:.0f}s)")
        print(f"  val AUC at kept epoch {best_val_auc:.4f} | test AUC {test_auc:.4f}")
        print(f"  threshold from val (min sens {MIN_SENSITIVITY}): {thr:.2f}"
              f"{'' if met else '  [FLOOR NOT MET]'}")

    assert not np.isnan(oof).any(), "every clip must receive exactly one prediction"

    # ---- pooled out-of-fold results ----
    point, lo, hi, n_boot = metrics.bootstrap_auc_ci(y, oof)
    print("\n" + "=" * 72)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 72)
    print(f"\nclips scored: {len(y)}  (vs 341 under the old single split)")
    print(f"AUC-ROC: {point:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   ({n_boot} bootstrap resamples)")
    print(f"per-fold test AUC: " + "  ".join(f"{a:.4f}" for a in fold_aucs))
    print(f"  spread: min {min(fold_aucs):.4f}  max {max(fold_aucs):.4f}  "
          f"mean {np.mean(fold_aucs):.4f}  std {np.std(fold_aucs):.4f}")
    print(f"per-fold thresholds: " + "  ".join(f"{t:.2f}" for t in fold_thresholds))

    print(f"\nsensitivity / specificity trade-off on the pooled predictions:")
    print(f"  {'thr':>5} {'sens':>8} {'spec':>8} {'youden':>8} {'acc':>8}   {'missed':>7} {'false alarm':>12}")
    for r in metrics.sweep(y, oof):
        if round(r["threshold"] * 100) % 5 == 0:
            print(f"  {r['threshold']:>5.2f} {r['sensitivity']:>8.4f} {r['specificity']:>8.4f} "
                  f"{r['youden_j']:>8.4f} {r['accuracy']:>8.4f}   {r['fn']:>7} {r['fp']:>12}")

    yj_thr, yj = metrics.youden_threshold(y, oof)
    print(f"\nfor reference, Youden's J would pick {yj_thr:.2f} "
          f"(sens {yj['sensitivity']:.4f}, spec {yj['specificity']:.4f})")

    # ---- artefacts ----
    with open(os.path.join(out_dir, "oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "fold", "label", "oof_probability"])
        for i, u in enumerate(uuids):
            w.writerow([u, int(fold_of[i]), int(y[i]), f"{oof[i]:.6f}"])

    with open(os.path.join(out_dir, "loss_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "epoch", "train_loss", "val_loss", "val_auc", "lr"])
        w.writeheader()
        w.writerows(curve_rows)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "n_clips": len(y), "n_folds": N_FOLDS,
            "pooled_auc": point, "ci_low": lo, "ci_high": hi,
            "fold_aucs": [float(a) for a in fold_aucs],
            "fold_thresholds": [float(t) for t in fold_thresholds],
            "min_sensitivity": MIN_SENSITIVITY,
            "youden_reference_threshold": yj_thr,
            "early_stopping_metric": select,
            "head": head,
            "specaug": specaug,
            "max_epochs": max_epochs,
            "n_parameters": n_params,
        }, f, indent=2)

    print(f"\nwrote oof_predictions.csv, loss_curves.csv, summary.json, fold_*.pt "
          f"-> {out_dir}")
    print(f"total time: {time.time() - t_start:.0f}s")
    return {"select": select, "head": head, "n_parameters": n_params,
            "auc": point, "ci": (lo, hi),
            "fold_aucs": [float(a) for a in fold_aucs]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", default="val_auc",
                    choices=["val_auc", "val_loss", "val_auc_smooth"],
                    help="early-stopping criterion (default: val_auc, the historical behaviour)")
    ap.add_argument("--head", default="flatten", choices=["flatten", "pool_time"],
                    help="classifier head (default: flatten, the historical behaviour)")
    ap.add_argument("--specaug", action="store_true",
                    help="mask random frequency and time bands during training")
    ap.add_argument("--seed", type=int, default=CV_SEED,
                    help="training seed: weight init, batch order, inner val split. "
                         "Fold assignment is NOT affected, so runs stay comparable.")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"epoch cap (default: {EPOCHS} from config.py)")
    args = ap.parse_args()
    main(select=args.select, head=args.head, specaug=args.specaug,
         max_epochs=args.epochs, seed=args.seed)
