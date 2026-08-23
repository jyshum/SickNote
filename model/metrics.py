"""
Evaluation metrics — AUC, confidence intervals, and threshold selection.

Deliberately dependency-light (numpy only) so this module can be imported
without pulling in the training stack. Every function takes plain sequences
of labels (0/1) and predicted probabilities.

Threshold policy: SickNote is a screening tool, so a missed abnormal cough
costs more than a false alarm. `threshold_for_min_sensitivity` encodes that
as a stated requirement rather than an argmax. `youden_threshold` is kept
for comparison against the historical 0.52.
"""

import numpy as np

# Thresholds are searched on this grid. 0.20-0.80 matches the range the
# original evaluate.py used, at the same 0.01 resolution.
DEFAULT_GRID = np.round(np.arange(0.20, 0.8001, 0.01), 2)


def roc_auc(y_true, y_score):
    """Area under the ROC curve, via the Mann-Whitney U identity.

    Handles ties by averaging ranks, matching sklearn's roc_auc_score.
    Returns nan if either class is absent (undefined, not an error).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = ((starts + ends + 1) / 2.0)[inv]

    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion(y_true, y_score, threshold):
    """Returns (tn, fp, fn, tp) at the given threshold. Predicts 1 when p >= threshold."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return tn, fp, fn, tp


def sensitivity_specificity(y_true, y_score, threshold):
    """Returns (sensitivity, specificity). Sensitivity = caught abnormals,
    specificity = correctly cleared healthy."""
    tn, fp, fn, tp = confusion(y_true, y_score, threshold)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


def sweep(y_true, y_score, grid=None):
    """Full threshold sweep. Returns a list of dicts, one per threshold."""
    grid = DEFAULT_GRID if grid is None else grid
    rows = []
    n = len(y_true)
    for t in grid:
        tn, fp, fn, tp = confusion(y_true, y_score, t)
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append({
            "threshold": float(t),
            "sensitivity": sens,
            "specificity": spec,
            "youden_j": sens + spec - 1.0,
            "accuracy": (tp + tn) / n if n else 0.0,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
    return rows


def youden_threshold(y_true, y_score, grid=None):
    """Threshold maximising sensitivity + specificity - 1.

    Weights the two equally. Retained so we can report what the old policy
    would have chosen, not because it is the policy we ship.
    """
    rows = sweep(y_true, y_score, grid)
    best = max(rows, key=lambda r: r["youden_j"])
    return best["threshold"], best


def threshold_for_min_sensitivity(y_true, y_score, min_sensitivity, grid=None):
    """Highest specificity among thresholds that meet a sensitivity floor.

    This is the shipped policy: state how many abnormal coughs the tool must
    catch, then pay as little specificity as possible for it.

    If no threshold on the grid reaches the floor, falls back to the most
    sensitive threshold available and flags it — better to say "we could not
    meet the requirement" than to silently ship something that misses it.
    """
    rows = sweep(y_true, y_score, grid)
    eligible = [r for r in rows if r["sensitivity"] >= min_sensitivity]
    if eligible:
        best = max(eligible, key=lambda r: (r["specificity"], -r["threshold"]))
        return best["threshold"], best, True
    best = max(rows, key=lambda r: (r["sensitivity"], r["specificity"]))
    return best["threshold"], best, False


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval for AUC.

    Resamples clips with replacement. Resamples that end up single-class are
    skipped rather than counted as 0.5 — they carry no information about
    ranking quality.

    Returns (point_estimate, low, high, n_used).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    point = roc_auc(y, s)

    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = roc_auc(y[idx], s[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return point, float("nan"), float("nan"), 0
    vals = np.asarray(vals)
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return point, lo, hi, len(vals)
