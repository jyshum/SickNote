"""Deployed architecture vs proposed, both measured as 5-model ensembles.

This is the only apples-to-apples comparison in the project: what ships today
against what would ship after the change, each measured the way it actually
runs. Everything else in EXPERIMENTS.md compares single models.

Usage: python compare_ensembles.py
"""

import csv
import json
import os

import numpy as np

from model import metrics
from model.crossval import load_dataset, make_folds

BASE = "model/checkpoints/cv"
RUNS = {
    "deployed  (flatten)":        "ensemble5__flatten__val_auc_smooth",
    "proposed  (pool+specaug)":   "ensemble5__pool_time__val_auc_smooth__specaug",
}


def main():
    _, y, _ = load_dataset()

    ens = {}
    for name, d in RUNS.items():
        members = np.loadtxt(os.path.join(BASE, d, "member_oof.csv"), delimiter=",")
        ens[name] = members.mean(axis=0)
        s = json.load(open(os.path.join(BASE, d, "summary.json")))
        print(f"{name:<26} single {s['mean_single_auc']:.4f}   "
              f"ensemble {s['ensemble_auc']:.4f}   "
              f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]   gain {s['ensemble_gain']:+.4f}")

    a, b = list(RUNS)
    pa, pb = ens[a], ens[b]

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        diffs.append(metrics.roc_auc(y[i], pb[i]) - metrics.roc_auc(y[i], pa[i]))
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])

    fold_of = make_folds(y)
    improved = sum(
        metrics.roc_auc(y[fold_of == f], pb[fold_of == f])
        > metrics.roc_auc(y[fold_of == f], pa[fold_of == f])
        for f in range(5)
    )

    print(f"\n=== paired: {a}  ->  {b} ===")
    print(f"  delta {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"P(better) {np.mean(d > 0):.3f}   folds improved {improved}/5")
    print(f"  verdict: {'REAL — CI excludes zero' if lo > 0 else 'not distinguishable from noise'}")


if __name__ == "__main__":
    main()
