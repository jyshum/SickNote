"""
Reproduces every number in EXPERIMENTS.md section 10.

Section 10 asks a question sections 1-9 never did: how much of the measured
AUC is about coughs, and how good are the labels we are fitting?

Usage:  PYTHONPATH=. python -m model.audit
Needs:  data/processed/*.pt and the COUGHVID metadata CSV.
"""
import glob
import itertools
import os

import numpy as np
import pandas as pd
import torch

from model.config import METADATA_CSV, PROCESSED_DIR
from model.explore import majority_label

DB_FLOOR = -100.0          # AmplitudeToDB maps digital silence here
PEAK_WINDOW_DB = 20.0      # "active" = within this many dB of the loudest frame
EXPERT_COLS = [f"diagnosis_{i}" for i in range(1, 5)]


def auc(y, score):
    """Rank-based AUC (Mann-Whitney U), tie-corrected."""
    y = np.asarray(y)
    r = pd.Series(score).rank().values
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def clip_features(path):
    """Three scalars obtainable without learning anything about coughs."""
    t = torch.load(path, weights_only=True)[0]        # (n_mels, frames)
    silent = t.max(0).values <= DB_FLOOR + 1e-3
    voiced = t[:, ~silent]
    return {
        "duration": float((~silent).sum()) * 512 / 22050,
        "peak": t.max().item(),
        "mean_db": voiced.mean().item() if voiced.numel() else DB_FLOOR,
        "pad_frac": silent.float().mean().item(),
        "active_frac": ((t.mean(0) - t.mean(0).max()) > -PEAK_WINDOW_DB).float().mean().item(),
    }


def shortcut_baseline(df):
    """Section 10.1 — what a model that never hears the cough can score."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    print("\n10.1  A model that hears nothing")
    for col in ("duration", "peak", "mean_db"):
        print(f"  {col:10s} alone           AUC {auc(df.y, df[col]):.4f}")

    X = df[["duration", "peak", "mean_db"]].values
    X = (X - X.mean(0)) / X.std(0)
    p = cross_val_predict(
        LogisticRegression(max_iter=1000), X, df.y, cv=5, method="predict_proba"
    )[:, 1]
    print(f"  all three, 5-fold CV       AUC {auc(df.y, p):.4f}")
    print(f"  (CNN ensemble, section 4b  AUC 0.7238)")


def emptiness(df):
    """Section 10.2 — how much of the 8s window carries no cough."""
    print("\n10.2  The input is mostly empty")
    q = df.pad_frac.quantile([.5, .9]).values
    print(f"  digital silence   p50 {q[0]:.3f}  p90 {q[1]:.3f}  "
          f"max {df.pad_frac.max():.3f}  mean {df.pad_frac.mean():.3f}")
    print(f"  clips with any padding            {(df.pad_frac > 0.01).mean():.3f}")
    print(f"  frames within {PEAK_WINDOW_DB:.0f} dB of peak    mean {df.active_frac.mean():.3f}")


def per_bin_scale(paths, n=400):
    """Section 10.2 — one global mean/std across bins that are not alike."""
    rng = np.random.default_rng(0)
    sel = rng.choice(paths, size=min(n, len(paths)), replace=False)
    print(f"  (per-bin scale measured on {len(sel)} clips, seed 0)")
    X = torch.stack([torch.load(f, weights_only=True)[0] for f in sel])
    m, s = X.mean(dim=(0, 2)), X.std(dim=(0, 2))
    print(f"  per-bin mean spread {float(m.max() - m.min()):.1f} dB   "
          f"std ratio {float(s.max() / s.min()):.2f}")


def label_audit(meta, used):
    """Section 10.3/10.4 — how good is the target we are fitting.

    Rater counts are reported over the clips we actually train on; the
    agreement statistics use every 4-rater clip in the dataset, since only
    126 exist and restricting further would leave nothing to measure.
    """
    print("\n10.3  Label quality")
    trained = meta[meta.uuid.isin(used)]
    n_exp = trained[EXPERT_COLS].notna().sum(axis=1)
    print("  raters per trained clip:", n_exp.value_counts().sort_index().to_dict())
    print(f"  single-rater share: {(n_exp == 1).mean():.3f}")

    full = meta[meta[EXPERT_COLS].notna().all(axis=1)]
    B = (full[EXPERT_COLS] != "healthy_cough").astype(int).values
    agree, kappas = [], []
    for i, j in itertools.combinations(range(4), 2):
        a, b = B[:, i], B[:, j]
        po = (a == b).mean()
        pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        agree.append(po)
        kappas.append((po - pe) / (1 - pe))

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        Bx = B[rng.integers(0, len(B), len(B))]
        ks = []
        for i, j in itertools.combinations(range(4), 2):
            a, b = Bx[:, i], Bx[:, j]
            po = (a == b).mean()
            pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
            ks.append((po - pe) / (1 - pe) if pe < 1 else 0.0)
        boot.append(np.mean(ks))

    print(f"  n with all 4 raters: {len(B)}")
    print(f"  two doctors agree            {np.mean(agree):.3f}")
    print(f"  chance agreement             {_chance(B):.3f}")
    print(f"  rubber stamp ('abnormal')    {np.mean([B[:, i].mean() for i in range(4)]):.3f}")
    print(f"  mean pairwise kappa          {np.mean(kappas):.3f}  "
          f"95% CI [{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}]")

    sens, spec = [], []
    for i in range(4):
        others = [j for j in range(4) if j != i]
        maj = (B[:, others].mean(1) > 0.5).astype(int)
        sens.append(((B[:, i] == 1) & (maj == 1)).sum() / max((maj == 1).sum(), 1))
        spec.append(((B[:, i] == 0) & (maj == 0)).sum() / max((maj == 0).sum(), 1))
    print(f"  one rater vs other three: sens {np.mean(sens):.3f}  spec {np.mean(spec):.3f}"
          f"  balanced acc {(np.mean(sens) + np.mean(spec)) / 2:.3f}")

    print("\n10.4  Base rate is a rater artifact")
    one = meta[n_exp == 1]
    print(f"  abnormal rate, 1 rater   {(one.diagnosis_1 != 'healthy_cough').mean():.3f}")
    print(f"  abnormal rate, 4 raters  {(B.mean(1) > 0.5).mean():.3f}")


def _chance(B):
    return np.mean([
        B[:, i].mean() * B[:, j].mean() + (1 - B[:, i].mean()) * (1 - B[:, j].mean())
        for i, j in itertools.combinations(range(4), 2)
    ])


def self_report_vs_expert(meta, used):
    """Section 10.5 — the two label sources are near-independent."""
    print("\n10.5  Self-report vs expert")
    d = meta[meta.uuid.isin(used)].copy()
    d["expert"] = (d.apply(majority_label, axis=1) == "abnormal").astype(int)
    sub = d[d.status.notna()].copy()
    sub["self_abn"] = (sub.status != "healthy").astype(int)
    po = (sub.self_abn == sub.expert).mean()
    pe = (sub.self_abn.mean() * sub.expert.mean()
          + (1 - sub.self_abn.mean()) * (1 - sub.expert.mean()))
    print(f"  n = {len(sub)}   agreement {po:.3f}   kappa {(po - pe) / (1 - pe):.3f}")
    print(pd.crosstab(sub.status, sub.expert).to_string())


def deployment_ppv(sens=0.8029, spec=0.5054):
    """Section 10.4 — the shipped operating point at realistic prevalence."""
    print("\n10.4  PPV at deployment prevalence")
    for prev in (0.797, 0.30, 0.15, 0.05):
        tp, fp = prev * sens, (1 - prev) * (1 - spec)
        print(f"  prevalence {prev:.3f} -> PPV {tp / (tp + fp):.3f}"
              f"   ({fp / (tp + fp) * 100:.0f}% of flags wrong)")


def main():
    paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.pt")))
    if not paths:
        raise SystemExit(f"no spectrograms in {PROCESSED_DIR} — run model.preprocess first")

    meta = pd.read_csv(METADATA_CSV)
    meta["label"] = meta.apply(majority_label, axis=1)
    label_of = dict(zip(meta.uuid, (meta.label == "abnormal").astype(int)))

    rows = []
    for f in paths:
        uuid = os.path.basename(f)[:-3]
        rows.append({"uuid": uuid, "y": label_of[uuid], **clip_features(f)})
    df = pd.DataFrame(rows)

    print("=" * 66)
    print(f"SickNote — data audit   ({len(df)} clips, "
          f"abnormal rate {df.y.mean():.3f})")
    print("=" * 66)

    shortcut_baseline(df)
    emptiness(df)
    per_bin_scale(paths)
    label_audit(meta, set(df.uuid))
    self_report_vs_expert(meta, set(df.uuid))
    deployment_ppv()
    print()


if __name__ == "__main__":
    main()
