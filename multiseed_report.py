"""Aggregate the multi-seed cross-validation runs.

Two numbers matter and they answer different questions:

  mean of per-seed AUCs   how good is this CONFIGURATION, on average, if you
                          train it once? The spread across seeds is the noise
                          floor — anything smaller than it is unmeasurable.

  AUC of averaged probs   how good is an ENSEMBLE of these models? This is what
                          you would actually deploy, since the app averages
                          several models per prediction.

Usage: python multiseed_report.py
"""

import csv
import json

import numpy as np

from model import metrics

BASE = "model/checkpoints/cv"
SEEDS = [100, 200, 300]

CONFIGS = {
    "flatten":             "flatten__val_auc_smooth__ep80__seed{s}",
    "pool_time":           "pool_time__val_auc_smooth__ep80__seed{s}",
    "pool_time +specaug":  "pool_time__val_auc_smooth__ep80__specaug__seed{s}",
}


def load(run_dir):
    rows = sorted(csv.DictReader(open(f"{BASE}/{run_dir}/oof_predictions.csv")),
                  key=lambda r: r["uuid"])
    return (np.array([int(r["label"]) for r in rows]),
            np.array([float(r["oof_probability"]) for r in rows]))


def main():
    results = {}
    print(f"{'config':<22}{'seed':>6}{'AUC':>10}")
    print("-" * 38)
    for name, pattern in CONFIGS.items():
        aucs, probs = [], []
        for s in SEEDS:
            y, p = load(pattern.format(s=s))
            a = metrics.roc_auc(y, p)
            aucs.append(a)
            probs.append(p)
            print(f"{name:<22}{s:>6}{a:>10.4f}")
        results[name] = (y, np.array(aucs), np.stack(probs))

    print(f"\n{'config':<22}{'mean':>9}{'std':>9}{'ensemble':>11}{'gain':>8}")
    print("-" * 60)
    for name, (y, aucs, probs) in results.items():
        ens = metrics.roc_auc(y, probs.mean(axis=0))
        print(f"{name:<22}{aucs.mean():>9.4f}{aucs.std(ddof=1):>9.4f}"
              f"{ens:>11.4f}{ens - aucs.mean():>+8.4f}")

    a_name, b_name = list(CONFIGS)[-2:]
    y, a_aucs, a_probs = results[a_name]
    _, b_aucs, b_probs = results[b_name]

    print(f"\n=== does SpecAugment help? ({a_name} -> {b_name}) ===")
    print(f"  per-seed deltas: " + "  ".join(f"{d:+.4f}" for d in b_aucs - a_aucs))
    print(f"  mean delta {(b_aucs - a_aucs).mean():+.4f}   "
          f"seeds improved {(b_aucs > a_aucs).sum()}/{len(SEEDS)}")

    # Paired bootstrap on the seed-averaged predictions: resample clips once,
    # score both configurations on that same resample.
    rng = np.random.default_rng(0)
    pa, pb = a_probs.mean(axis=0), b_probs.mean(axis=0)
    diffs = []
    for _ in range(5000):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(i)):
            continue
        diffs.append(metrics.roc_auc(y[i], pb[i]) - metrics.roc_auc(y[i], pa[i]))
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  ensemble-vs-ensemble paired delta {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  verdict: {'REAL' if lo > 0 else 'still not distinguishable from noise'}")

    print(f"\n=== operating point, seed-averaged, at the {0.80:.2f} sensitivity floor ===")
    for name, (y, _, probs) in results.items():
        t, r, ok = metrics.threshold_for_min_sensitivity(y, probs.mean(axis=0), 0.80)
        print(f"  {name:<22} thr {t:.2f}  sens {r['sensitivity']:.4f}  "
              f"spec {r['specificity']:.4f}   missed {r['fn']:>4}  false alarms {r['fp']:>4}")


if __name__ == "__main__":
    main()
