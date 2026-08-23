import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from stats_experiment.config import (
    RESULTS_CSV, GRAPHS_DIR, ANALYSIS_FILE, ALPHA,
    TRAINING_SIZES, SEEDS,
)


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    # ── Part A: Load and validate ─────────────────────────────────────
    df = pd.read_csv(RESULTS_CSV)
    assert len(df) == 50, f"Expected 50 rows, got {len(df)}"
    assert df["val_auc_roc"].notna().all(), "Found NaN values — check failed runs"

    n = len(df)

    # ── Part B: Descriptive statistics ────────────────────────────────
    auc = df["val_auc_roc"]
    mean = auc.mean()
    median = auc.median()
    std = auc.std(ddof=1)
    amin = auc.min()
    amax = auc.max()
    q1 = auc.quantile(0.25)
    q3 = auc.quantile(0.75)

    group = df.groupby("training_size")["val_auc_roc"].agg(["mean", "std"])

    # ── Part C: Linear regression (raw training_size) ─────────────────
    x = df["training_size"].values.astype(float)
    y = df["val_auc_roc"].values.astype(float)

    slope, intercept, r_value, p_value_two, std_err = scipy_stats.linregress(x, y)

    b1 = slope
    b0 = intercept
    t = b1 / std_err
    df_reg = n - 2
    p_one = p_value_two / 2 if slope > 0 else 1 - p_value_two / 2
    r = r_value
    r2 = r_value ** 2

    y_pred = b0 + b1 * x
    residuals = y - y_pred

    # ── Part D: Curvature check ───────────────────────────────────────
    coeffs = np.polyfit(x, y, 2)
    x_range = x.max() - x.min()
    curvature_detected = abs(coeffs[0]) * x_range > abs(slope) * 0.1

    if curvature_detected:
        x_log = np.log(x)
        slope_log, intercept_log, r_log, p_log_two, se_log = scipy_stats.linregress(x_log, y)
        p_log_one = p_log_two / 2 if slope_log > 0 else 1 - p_log_two / 2
        y_pred_log = intercept_log + slope_log * x_log
        residuals_log = y - y_pred_log

    # ── Part E: Generate all graphs ───────────────────────────────────
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
    })

    # Graph 1: scatterplot_raw.png
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="steelblue", alpha=0.7)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, b0 + b1 * x_line, color="red")
    ax.annotate(f"ŷ = {b0:.4f} + {b1:.6f}x\nr² = {r2:.4f}",
                xy=(0.05, 0.92), xycoords="axes fraction", va="top")
    ax.set_title("Training Set Size vs. Validation AUC-ROC")
    ax.set_xlabel("Training Set Size (recordings)")
    ax.set_ylabel("Validation AUC-ROC")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "scatterplot_raw.png"))
    plt.close()

    # Graph 2: residual_plot_raw.png
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, color="steelblue", alpha=0.7)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residual Plot — Training Size vs. AUC-ROC")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "residual_plot_raw.png"))
    plt.close()

    # Graph 3: histogram_residuals.png
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=10, color="steelblue", edgecolor="white")
    ax.set_title("Distribution of Residuals")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "histogram_residuals.png"))
    plt.close()

    # Graph 4: normal_probability_plot.png
    fig, ax = plt.subplots()
    scipy_stats.probplot(residuals, plot=ax)
    ax.set_title("Normal Probability Plot of Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "normal_probability_plot.png"))
    plt.close()

    # Graph 5: boxplots_by_size.png
    fig, ax = plt.subplots()
    groups = [df[df["training_size"] == s]["val_auc_roc"].values for s in TRAINING_SIZES]
    ax.boxplot(groups, labels=[str(s) for s in TRAINING_SIZES])
    ax.set_title("Distribution of AUC-ROC by Training Set Size")
    ax.set_xlabel("Training Set Size (recordings)")
    ax.set_ylabel("AUC-ROC")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "boxplots_by_size.png"))
    plt.close()

    # Graph 6: pvalue_diagram.png
    fig, ax = plt.subplots()
    x_td = np.linspace(-5, max(t + 2, 5), 1000)
    pdf = scipy_stats.t.pdf(x_td, df=df_reg)
    ax.plot(x_td, pdf, color="steelblue")
    ax.fill_between(x_td, 0, pdf, where=(x_td >= t), color="red", alpha=0.3)
    ax.axvline(t, color="red", linestyle="--")
    peak_y = scipy_stats.t.pdf(0, df=df_reg)
    ax.annotate(f"t = {t:.3f}", xy=(t, peak_y * 0.15), fontsize=10, color="red", ha="left")
    ax.annotate(f"p = {p_one:.4f}", xy=(t, peak_y * 0.08), fontsize=10, color="red", ha="left")
    ax.set_title("P-Value Diagram — t-Test for Regression Slope")
    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "pvalue_diagram.png"))
    plt.close()

    # Conditional Graphs 7 & 8 (only if curvature detected)
    if curvature_detected:
        # Graph 7: scatterplot_log.png
        fig, ax = plt.subplots()
        ax.scatter(x_log, y, color="steelblue", alpha=0.7)
        xl_line = np.linspace(x_log.min(), x_log.max(), 100)
        ax.plot(xl_line, intercept_log + slope_log * xl_line, color="red")
        ax.annotate(f"r² = {r_log**2:.4f}", xy=(0.05, 0.92),
                    xycoords="axes fraction", va="top")
        ax.set_title("ln(Training Size) vs. Validation AUC-ROC")
        ax.set_xlabel("ln(Training Set Size)")
        ax.set_ylabel("Validation AUC-ROC")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, "scatterplot_log.png"))
        plt.close()

        # Graph 8: residual_plot_log.png
        fig, ax = plt.subplots()
        ax.scatter(y_pred_log, residuals_log, color="steelblue", alpha=0.7)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Residual Plot — Log-Transformed Regression")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, "residual_plot_log.png"))
        plt.close()

    # ── Part F: Write analysis_summary.txt ────────────────────────────
    decision = "Reject" if p_one < ALPHA else "Fail to reject"
    is_or_not = "is" if p_one < ALPHA else "is not"

    lines = []
    lines.append("STATS EXPERIMENT — ANALYSIS SUMMARY")
    lines.append("====================================")
    lines.append("")
    lines.append("DESCRIPTIVE STATISTICS")
    lines.append(f"  n = {n}")
    lines.append(f"  Mean AUC-ROC = {mean:.4f}")
    lines.append(f"  Median AUC-ROC = {median:.4f}")
    lines.append(f"  Std Dev = {std:.4f}")
    lines.append(f"  Min = {amin:.4f}, Q1 = {q1:.4f}, Median = {median:.4f}, Q3 = {q3:.4f}, Max = {amax:.4f}")
    lines.append("")
    lines.append("GROUP MEANS (by training size)")
    for s in TRAINING_SIZES:
        gm = group.loc[s, "mean"]
        gs = group.loc[s, "std"]
        lines.append(f"  {s}: mean={gm:.4f}, std={gs:.4f}")
    lines.append("")
    lines.append("REGRESSION (raw training_size)")
    lines.append(f"  Equation: ŷ = {b0:.6f} + {b1:.8f}x")
    lines.append(f"  r = {r:.4f}")
    lines.append(f"  r² = {r2:.4f}")
    lines.append(f"  b₁ (slope) = {b1:.8f}")
    lines.append(f"  SE(b₁) = {std_err:.8f}")
    lines.append(f"  t = b₁ / SE(b₁) = {b1:.8f} / {std_err:.8f} = {t:.4f}")
    lines.append(f"  df = {df_reg}")
    lines.append(f"  p-value (one-sided) = {p_one:.4f}")
    lines.append("")
    lines.append("HYPOTHESIS TEST")
    lines.append("  H₀: β₁ = 0")
    lines.append("  Hₐ: β₁ > 0")
    lines.append("  α = 0.05")
    lines.append(f"  Decision: {decision} H₀")
    lines.append("")
    lines.append(f"  At the α = 0.05 significance level, there {is_or_not} sufficient")
    lines.append("  evidence to conclude that there is a positive linear relationship")
    lines.append("  between training set size and validation AUC-ROC for CNN models trained")
    lines.append("  on COUGHVID cough audio spectrograms.")

    if curvature_detected:
        lines.append("")
        lines.append("REGRESSION (log-transformed)")
        lines.append(f"  Equation: ŷ = {intercept_log:.6f} + {slope_log:.8f}·ln(x)")
        lines.append(f"  r = {r_log:.4f}")
        lines.append(f"  r² = {r_log**2:.4f}")
        lines.append(f"  b₁ = {slope_log:.8f}")
        lines.append(f"  SE(b₁) = {se_log:.8f}")
        lines.append(f"  t = {slope_log/se_log:.4f}")
        lines.append(f"  df = {df_reg}")
        lines.append(f"  p-value (one-sided) = {p_log_one:.4f}")

    with open(ANALYSIS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")

    # ── Part G: Print to console ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Graphs saved to: {GRAPHS_DIR}")
    print(f"Summary saved to: {ANALYSIS_FILE}")
    print(f"\nKey results:")
    print(f"  slope b1 = {b1:.8f}")
    print(f"  t = {t:.4f}")
    print(f"  p (one-sided) = {p_one:.4f}")
    print(f"  Decision: {'Reject' if p_one < ALPHA else 'Fail to reject'} H0")


if __name__ == "__main__":
    main()
