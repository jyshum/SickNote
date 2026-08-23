"""Paired comparison of cross-validation runs.

Every run scored the identical clips in the identical folds, so differences
are tested by resampling clips ONCE and scoring both runs on that same
resample. That is far sharper than checking whether two marginal confidence
intervals happen to overlap.

Usage: python compare_runs.py
"""

import csv
import sys

import numpy as np

from model import metrics

BASE = "model/checkpoints/cv"

RUNS = {
    "flatten @80":         "val_auc_smooth",
    "pool @80":            "pool_time__val_auc_smooth",
    "pool @250":           "pool_time__val_auc_smooth__ep250",
    "pool @250 +specaug":  "pool_time__val_auc_smooth__ep250__specaug",
}


def load(run_dir):
    rows = sorted(csv.DictReader(open(f"{BASE}/{run_dir}/oof_predictions.csv")),
                  key=lambda r: r["uuid"])
    return (np.array([int(r["label"]) for r in rows]),
            np.array([float(r["oof_probability"]) for r in rows]),
            np.array([int(r["fold"]) for r in rows]))


def paired_delta(y, folds, probs_a, probs_b, n_boot=5000, seed=0):
    """Bootstrap the AUC difference on shared resamples. Returns (mean, lo, hi, folds_improved)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        diffs.append(metrics.roc_auc(y[i], probs_b[i]) - metrics.roc_auc(y[i], probs_a[i]))
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    improved = sum(
        metrics.roc_auc(y[folds == f], probs_b[folds == f])
        > metrics.roc_auc(y[folds == f], probs_a[folds == f])
        for f in np.unique(folds)
    )
    return d, float(lo), float(hi), int(improved)


def main():
    data = {name: load(d) for name, d in RUNS.items()}
    y, _, folds = data["pool @80"]

    print(f"{'run':<22}{'AUC':>9}   {'95% CI':<20}{'params':>12}")
    print("-" * 66)
    for name, (yy, p, _) in data.items():
        auc, lo, hi, _ = metrics.bootstrap_auc_ci(yy, p)
        n = "1,415,281" if name.startswith("flatten") else "71,793"
        print(f"{name:<22}{auc:>9.4f}   [{lo:.4f}, {hi:.4f}]   {n:>12}")

    print("\n=== paired comparisons (identical clips, identical folds) ===")
    for a, b in [("flatten @80", "pool @80"),
                 ("pool @80", "pool @250"),
                 ("pool @250", "pool @250 +specaug"),
                 ("pool @80", "pool @250 +specaug")]:
        d, lo, hi, improved = paired_delta(y, folds, data[a][1], data[b][1])
        verdict = "REAL — CI excludes zero" if lo > 0 else "not distinguishable from noise"
        print(f"\n  {a}  ->  {b}")
        print(f"    delta {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"P(better) {np.mean(d > 0):.3f}   folds improved {improved}/5")
        print(f"    {verdict}")

    print("\n=== operating point at the 0.80 sensitivity floor ===")
    for name, (yy, p, _) in data.items():
        t, r, ok = metrics.threshold_for_min_sensitivity(yy, p, 0.80)
        print(f"  {name:<22} thr {t:.2f}  sens {r['sensitivity']:.4f}  spec {r['specificity']:.4f}"
              f"   missed {r['fn']:>4}  false-alarmed {r['fp']:>4}")


if __name__ == "__main__":
    main()
