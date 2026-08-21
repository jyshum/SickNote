"""
Nested-ensemble cross-validation — measuring what actually ships.

Every number in EXPERIMENTS.md so far comes from a single model, but the
deployed app averages five. Those are different systems, and the gap between
them has never been measured.

The problem is that plain cross-validation cannot measure an ensemble: each
clip sits in the training data of four of the five fold models, so there is no
clip all five are innocent of. The fix is to nest the ensemble INSIDE each
fold:

    for each fold k:
        train N models, all on folds != k, differing only in weight init
        average their probabilities on fold k

Every member of fold k's ensemble is innocent of fold k, so averaging them is
legitimate. Pool across folds and you get one honest out-of-fold ENSEMBLE
prediction per clip.

The inner train/val split is held FIXED within a fold, so members differ only
by initialisation — which is exactly what the shipped model does (five seeds,
identical data). Varying the split too would give more diverse members and
probably a better ensemble; that is a separate experiment, not this one.

Also reports the ensemble-size curve: how much is a 2nd, 3rd, 4th, 5th member
actually worth? Averaged over every subset of that size, so the curve does not
depend on which members happened to be listed first.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.ensemble_cv \
        --head pool_time --select val_auc_smooth --specaug --members 5
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

from model import metrics
from model.architecture import SickNoteCNN
from model.config import (
    CV_CHECKPOINT_DIR, DEVICE, EPOCHS,
    N_FOLDS, CV_VAL_FRACTION, CV_SEED, MIN_SENSITIVITY,
)
from model.crossval import (
    load_dataset, make_folds, stratified_split, train_fold, predict,
)


def ensemble_size_curve(y, member_probs):
    """AUC as a function of ensemble size, averaged over all subsets of each size.

    Returns a list of (size, mean_auc, std_auc, n_subsets).
    """
    n_members = member_probs.shape[0]
    rows = []
    for k in range(1, n_members + 1):
        aucs = [
            metrics.roc_auc(y, member_probs[list(combo)].mean(axis=0))
            for combo in itertools.combinations(range(n_members), k)
        ]
        rows.append((k, float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if k < n_members else 0.0,
                     len(aucs)))
    return rows


def main(head="pool_time", select="val_auc_smooth", specaug=False,
         pool="sym", n_members=5, max_epochs=EPOCHS, seed=CV_SEED):
    print("=" * 72)
    print("SickNote — nested-ensemble cross-validation")
    print(f"  head={head}  pool={pool}  select={select}  specaug={specaug}  members={n_members}")
    print("=" * 72)

    tag = f"ensemble{n_members}__{head}__{select}"
    if specaug:
        tag += "__specaug"
    if pool != "sym":
        tag += f"__{pool}"
    out_dir = os.path.join(CV_CHECKPOINT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    X, y, uuids = load_dataset()
    fold_of = make_folds(y)

    # member_oof[m, i] = member m's probability for clip i, from the fold model
    # that never trained on clip i.
    member_oof = np.full((n_members, len(y)), np.nan)
    fold_thresholds, single_aucs, ensemble_fold_aucs = [], [], []

    for fold in range(N_FOLDS):
        test_idx = np.flatnonzero(fold_of == fold)
        rest_idx = np.flatnonzero(fold_of != fold)
        # Fixed inner split: every member of this fold sees identical data.
        train_idx, val_idx = stratified_split(rest_idx, y, CV_VAL_FRACTION, seed + fold)

        t0 = time.time()
        val_stack, test_stack = [], []
        for m in range(n_members):
            model, mean, std, best_val_auc, best_epoch, epochs = train_fold(
                X, y, train_idx, val_idx, fold, seed + 1000 * m + fold, [],
                select=select, head=head, specaug=specaug,
                max_epochs=max_epochs, pool=pool,
            )
            test_probs = predict(model, X, test_idx, mean, std)
            member_oof[m, test_idx] = test_probs
            val_stack.append(predict(model, X, val_idx, mean, std))
            test_stack.append(test_probs)
            single_aucs.append(metrics.roc_auc(y[test_idx], test_probs))

        ens_val = np.mean(val_stack, axis=0)
        ens_test = np.mean(test_stack, axis=0)
        ens_auc = metrics.roc_auc(y[test_idx], ens_test)
        ensemble_fold_aucs.append(ens_auc)

        # Threshold from the ENSEMBLE's validation predictions, never from test.
        thr, row, met = metrics.threshold_for_min_sensitivity(
            y[val_idx], ens_val, MIN_SENSITIVITY
        )
        fold_thresholds.append(thr)

        mean_single = float(np.mean([metrics.roc_auc(y[test_idx], p) for p in test_stack]))
        print(f"fold {fold}: {n_members} members trained in {time.time() - t0:.0f}s")
        print(f"  mean single-model AUC {mean_single:.4f} | ensemble AUC {ens_auc:.4f} "
              f"({ens_auc - mean_single:+.4f}) | threshold {thr:.2f}")

    # ---------------- pooled results ----------------
    pooled_single = float(np.mean([metrics.roc_auc(y, member_oof[m]) for m in range(n_members)]))
    pooled_ens_probs = member_oof.mean(axis=0)
    auc, lo, hi, _ = metrics.bootstrap_auc_ci(y, pooled_ens_probs)

    print("\n" + "=" * 72)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 72)
    print(f"  mean single model : {pooled_single:.4f}")
    print(f"  {n_members}-model ensemble : {auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  ensemble gain     : {auc - pooled_single:+.4f}")

    print(f"\n  ensemble size curve (averaged over every subset of that size):")
    curve = ensemble_size_curve(y, member_oof)
    for k, m, s, n in curve:
        bar = "#" * int(round((m - curve[0][1]) * 2000))
        print(f"    {k} member{'s' if k > 1 else ' '}: {m:.4f}"
              f"{f' +/- {s:.4f}' if s else '         '}  ({n:>2} subsets)  {bar}")

    # Per-fold thresholds applied to their own fold, then pooled.
    preds = np.zeros(len(y), dtype=int)
    for fold in range(N_FOLDS):
        idx = np.flatnonzero(fold_of == fold)
        preds[idx] = (pooled_ens_probs[idx] >= fold_thresholds[fold]).astype(int)
    tp = int(((y == 1) & (preds == 1)).sum()); fn = int(((y == 1) & (preds == 0)).sum())
    tn = int(((y == 0) & (preds == 0)).sum()); fp = int(((y == 0) & (preds == 1)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    print(f"\n  operating point (per-fold thresholds from val, floor {MIN_SENSITIVITY}):")
    print(f"    sensitivity {sens:.4f}  specificity {spec:.4f}  "
          f"missed {fn}  false alarms {fp}")

    summary = {
        "n_clips": len(y), "n_folds": N_FOLDS, "n_members": n_members,
        "head": head, "pool": pool, "specaug": specaug, "select": select,
        "mean_single_auc": pooled_single,
        "ensemble_auc": auc, "ci_low": lo, "ci_high": hi,
        "ensemble_gain": auc - pooled_single,
        "size_curve": [{"size": k, "auc": m, "std": s, "n_subsets": n} for k, m, s, n in curve],
        "fold_thresholds": fold_thresholds,
        "ensemble_fold_aucs": ensemble_fold_aucs,
        "sensitivity": sens, "specificity": spec, "fn": fn, "fp": fp,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savetxt(os.path.join(out_dir, "member_oof.csv"), member_oof, delimiter=",")
    print(f"\n  wrote summary.json, member_oof.csv -> {out_dir}")
    print(f"  total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="pool_time", choices=["flatten", "pool_time"])
    ap.add_argument("--select", default="val_auc_smooth")
    ap.add_argument("--pool", default="sym", choices=["sym", "freq16", "freq32"])
    ap.add_argument("--specaug", action="store_true")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=CV_SEED)
    a = ap.parse_args()
    main(head=a.head, select=a.select, specaug=a.specaug, pool=a.pool,
         n_members=a.members, max_epochs=a.epochs, seed=a.seed)
