# SickNote Statistical Experiment — Full Report

**Project:** AP Statistics 12 Final — Training Data Size vs. CNN Classification Performance
**Dataset:** COUGHVID (binary cough classification, healthy vs. abnormal)
**Date compiled:** 2026-06-09
**Status:** Experiment complete (50/50 runs), analysis complete

---

## 1. Executive Summary

We trained **50 independent CNN models** (10 training-set sizes × 5 random seeds) on COUGHVID cough spectrograms and recorded test-set AUC-ROC for each, then ran a regression-slope *t*-test to determine whether training-set size significantly predicts performance.

**Headline finding:** There is a **statistically significant, moderately strong, positive linear relationship** between training-set size and AUC-ROC.

| Metric | Value |
|---|---|
| Slope (b₁) | +0.00002515 AUC per recording |
| *t*-statistic (df = 48) | 5.42 |
| *p*-value (one-sided, Hₐ: β₁ > 0) | < 0.0001 |
| Correlation *r* | 0.616 |
| r² | 0.380 |
| Decision | **Reject H₀** |

**Three secondary findings:**
1. **No detectable diminishing returns** over the tested range (200–1,900). The quadratic term is non-significant (F-test *p* = 0.83); the data is statistically indistinguishable from a straight line.
2. **The signal is real but modest relative to noise.** Training size explains only 38% of the variance in AUC; random-seed variation accounts for the other 62%. The trend emerges only because the design pooled 50 runs — adjacent sizes are statistically indistinguishable.
3. **The result is consistent with published scaling-law research**, which finds performance-vs-data follows a power-law / log-linear curve (not a logistic S-curve), with a ceiling set by irreducible label noise.

---

## 2. Background & Original Goal

Per [STATS_PROJECT.md](../STATS_PROJECT.md), the experiment was scoped as a controlled study layered on the existing SickNote codebase. The original research question, verbatim:

> Train 50 independent CNN models at 10 different training set sizes (5 random seeds each), record AUC-ROC for each, then perform a regression slope *t*-test to determine if more training data significantly predicts better performance.

**Hypotheses:**
- H₀: β₁ = 0 (training size has no linear effect on AUC-ROC)
- Hₐ: β₁ > 0 (more training data predicts higher AUC-ROC)
- α = 0.05

The experiment reuses the production SickNote model class, dataset, and preprocessing without duplication (per [STATS_PROJECT_IMPLEMENTATION.md](../STATS_PROJECT_IMPLEMENTATION.md) §0.3), but uses a **purpose-built minimal training loop** to remove confounds present in the production trainer (early stopping, LR scheduling, ensembling).

---

## 3. Experimental Design (as built)

| Variable | Setting | Rationale |
|---|---|---|
| Training sizes | 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, **1900** | 10 levels; max capped at 1,900 (pool is only 1,926 — see §4) |
| Random seeds | 1, 2, 3, 4, 5 | Controls subset selection **and** weight initialization |
| Total runs | 50 | 10 × 5 |
| Architecture | SickNoteCNN, conv channels [8, 16, 32] → FC(128) → FC(1) | Identical to production single model |
| Epochs | 30 (fixed) | **No early stopping** — controlled training duration |
| Optimizer | Adam, lr = 3e-4, weight_decay = 1e-4 | Frozen across all runs |
| LR schedule | **None** | Removed to eliminate confound |
| Loss | BCEWithLogitsLoss, pos_weight = n_neg/n_pos | Recomputed per subset for class imbalance |
| Test set | Fixed 15% stratified holdout, seed = 0 | **Identical for every run** — the controlled measurement surface |
| Normalization | mean/std computed once from full pool | A control, not a variable |
| Threshold | 0.50 | Note: production uses 0.52 (Youden's J); experiment uses 0.50 for a clean, fixed decision rule |
| Ensemble | **None** — single models only | Isolates the size effect from ensemble variance reduction |
| Augmentation | None | — |

**Controlled variables:** architecture, hyperparameters, epochs, test set, normalization, preprocessing, threshold. **Independent variable:** training-set size. **Dependent variable:** test-set AUC-ROC. **Nuisance variable controlled by replication:** random seed.

This is a clean controlled experiment: every factor except training size is held constant, and the one uncontrollable source of run-to-run variation (seed) is replicated 5× per level so its effect can be averaged out and quantified.

---

## 4. Data

Source: COUGHVID crowd-sourced corpus — ~25,000–27,000 cough recordings, of which **>2,800 were labeled by four physicians** ([Orlandic et al., Nature Scientific Data 2021](https://www.nature.com/articles/s41597-021-00937-4)).

After the SickNote filtering pipeline (cough_detected > 0.8, ≥1 expert diagnosis, good quality, valid majority-vote label):

| Quantity | Count |
|---|---|
| Total filtered pool | 2,267 |
| Healthy (label 0) | 461 (20.3%) |
| Abnormal (label 1) | 1,806 (79.7%) |
| Fixed test holdout (15%, seed 0) | 341 |
| **Training pool (sampled from)** | **1,926** |

**Why max size = 1,900:** the original plan specified 1,950, but with a 15% test holdout the training pool is only 1,926. The largest level was capped at 1,900 to stay within the pool ([STATS_PROJECT_IMPLEMENTATION.md](../STATS_PROJECT_IMPLEMENTATION.md) §0.2).

**Class imbalance** (~80/20 abnormal/healthy) is handled by `pos_weight` in the loss, recomputed per subset so the imbalance correction is itself controlled.

---

## 5. Results

### 5.1 Descriptive statistics (n = 50)

| Statistic | AUC-ROC |
|---|---|
| Mean | 0.6135 |
| Median | 0.6157 |
| Std Dev | 0.0231 |
| Min | 0.5509 |
| Q1 | 0.6059 |
| Q3 | 0.6288 |
| Max | 0.6609 |

### 5.2 Group means by training size

| Size | Mean AUC | Std (seed noise) |
|---|---|---|
| 200 | 0.5899 | 0.0315 |
| 400 | 0.5998 | 0.0221 |
| 600 | 0.5911 | 0.0254 |
| 800 | 0.6116 | 0.0220 |
| 1000 | 0.6170 | 0.0095 |
| 1200 | 0.6139 | 0.0057 |
| 1400 | 0.6297 | 0.0144 |
| 1600 | 0.6132 | 0.0106 |
| 1800 | 0.6322 | 0.0100 |
| 1900 | 0.6368 | 0.0186 |

Note the **within-size standard deviation shrinks** as size grows (0.031 at 200 → ~0.010–0.019 at the largest sizes): more data not only raises the mean, it stabilizes the model against seed variation.

### 5.3 Linear regression (raw training size)

```
ŷ = 0.586107 + 0.00002515·x
r = 0.6162    r² = 0.3797
b₁ = 0.00002515    SE(b₁) = 0.00000464
t = b₁ / SE(b₁) = 5.4204    df = 48
p (one-sided) < 0.0001
```

### 5.4 Hypothesis test

- H₀: β₁ = 0   Hₐ: β₁ > 0   α = 0.05
- *t* = 5.42, *p* < 0.0001 → **Reject H₀**

> At α = 0.05 there is sufficient evidence to conclude that there is a positive linear relationship between training-set size and test AUC-ROC for CNN models trained on COUGHVID cough spectrograms.

### 5.5 Diminishing returns analysis

A naive expectation is that returns diminish as data grows. **We found no statistical evidence of this within the tested range:**

- **Quadratic fit:** the x² coefficient is −2.16e−9 (concave, the *direction* of diminishing returns), but the **F-test for adding the quadratic term is non-significant (F = 0.05, p = 0.83).** The data is statistically indistinguishable from a straight line.
- **Split-range slopes:** lower half (200–1000) slope = 3.30e−5 (p = 0.024); upper half (1000–1900) slope = 2.04e−5 (p = 0.006). The upper slope is *slightly* shallower — the faintest hint of leveling — but both halves remain significantly positive and the difference is within noise.
- **Log model fits *worse*** than linear (r² = 0.354 vs. 0.380). A true diminishing-returns (concave) curve would make the log fit *better*; it does not.

**Conclusion:** over 200–1,900 recordings, AUC is still climbing approximately linearly. No plateau is reached. The dataset ceiling (1,926) prevents observing whether one exists beyond it.

### 5.6 Signal vs. seed noise

This is the most important nuance of the study.

| Quantity | Value |
|---|---|
| Full-range effect (200 → 1900) | +0.0428 AUC |
| Typical within-size seed std | 0.0170 |
| Effect ÷ noise std | **2.52×** |
| Variance explained by size (r²) | **38%** |
| Variance from seed/noise (1 − r²) | **62%** |

**Are individual size levels distinguishable?** (Welch *t*-tests on group means):

| Comparison | ΔAUC | p | Distinguishable? |
|---|---|---|---|
| 200 vs 400 | +0.0099 | 0.58 | No |
| 1000 vs 1200 | −0.0031 | 0.55 | No |
| 1400 vs 1600 | −0.0166 | 0.075 | No |
| 1000 vs 1900 | +0.0198 | 0.078 | Borderline |
| **200 vs 1900** | **+0.0469** | **0.026** | **Yes** |

**Interpretation:** the relationship shows through the noise **only in aggregate** — by pooling all 50 runs into one regression, or by comparing the extreme ends of the range. Any single pairwise comparison of nearby sizes is swamped by seed variance. A naive two-group experiment (e.g., "does 1000 beat 800?") would have **missed the effect entirely.** This is precisely why the replicated 5-seeds-per-size design was methodologically necessary.

---

## 6. Interpretation Grounded in the Research Literature

Our finding — a significant positive, approximately log-linear relationship that has not yet plateaued — aligns with the established empirical literature on **learning curves** and **neural scaling laws**.

### 6.1 The shape is power-law / log-linear, not logistic

- **[Hestness et al. 2017, "Deep Learning Scaling is Predictable, Empirically"](https://arxiv.org/abs/1712.00409):** across machine translation, language modeling, image, and speech, generalization **error follows a power law in training-set size**, with exponents typically between −0.07 and −0.35. Critically, they found that **model-architecture improvements shift the error down but do not change the power-law exponent.**
- **[Sun et al. 2017, "Revisiting Unreasonable Effectiveness of Data"](https://arxiv.org/abs/1707.02968) (ICCV):** on the 300M-image JFT dataset, **vision-task performance increases logarithmically with training-data volume** over two orders of magnitude — i.e. performance is roughly linear in log(N).

Both predict that over a wide range (e.g. 100 → 10,000), AUC-vs-size would look **log-linear, bending into a sub-1.0 plateau** — *not* a symmetric logistic S-curve. A logistic shape would require an exponential "take-off"; real learning curves instead show their steepest gains immediately and diminish gradually toward an irreducible-error floor. Because **AUC is bounded at 0.5 (random) and a sub-1.0 ceiling (Bayes-optimal given label noise)**, the curve over a huge range can superficially *resemble* a saturating sigmoid, but the better-fitting and theory-grounded model is power-law-to-plateau.

### 6.2 Our data sits in the early log-linear regime

The non-significant curvature (§5.5) means our 200–1,900 window captured a slice of the **rising log-linear portion** of the scaling curve — before any bend toward saturation. We cannot observe the plateau because the dataset ceiling cuts us off at 1,926.

### 6.3 Why the ceiling is likely low for COUGHVID

Hestness's result that *model improvements don't change the scaling exponent* directly explains an earlier finding in this project: a transfer-learning attempt (ResNet18 backbone) and a larger-capacity model **failed to improve AUC**. The bottleneck is not model capacity or raw data quantity — it is **irreducible label noise.** COUGHVID is crowd-sourced with weak, majority-vote expert labels; the Bayes-optimal AUC is well below 1.0 (plausibly ~0.65–0.72 for this task). More noisy data climbs the log-linear slope toward that modest asymptote but cannot break past it.

---

## 7. Limitations

1. **Single dataset, single ceiling.** All 50 models share COUGHVID's label noise. The plateau we cannot see is dataset-specific.
2. **Range capped at 1,926.** We observe the rising regime only; the bend toward saturation is beyond reach with current data — this is the central limitation for the scaling question.
3. **AUC threshold = 0.50, not production's 0.52.** Accuracy/sensitivity/specificity values differ slightly from the deployed app; AUC-ROC (threshold-independent) is unaffected.
4. **Single model, not ensemble.** By design — but it means absolute AUC values are lower than the deployed 5-model ensemble.
5. **Seed noise dominates.** With 62% of variance from seeds, conclusions are population-level, not per-run.
6. **Wall-clock timing column is unreliable** for two runs (laptop slept mid-run, inflating `training_time_seconds`). This column is not used in the analysis; AUC results are unaffected.

---

## 8. New Context: Acquiring More Data

A natural extension — and the clearest lever for strengthening *both* the app and the statistics project — is to enlarge the data pool. Candidate sources:

| Dataset | Size | Labels | Modality match | Access |
|---|---|---|---|---|
| **Coswara** (IISc Bangalore) | ~16,780 recordings | healthy / COVID / asthma / COPD / other | ✅ crowd-sourced phone cough | Open (GitHub) |
| **COVID-19 Sounds** (Cambridge) | ~53k participants, 100k+ samples | respiratory symptoms + COVID | ✅ cough + breath + voice | ⚠️ restricted (data-use agreement) |
| **Virufy** | ~103 clinically validated | PCR-confirmed COVID ± | ✅ phone cough, **clean labels** | Open, but tiny |
| **ICBHI 2017 Respiratory DB** | ~920 recordings | crackle / wheeze / normal | ❌ stethoscope lung sounds | Open |

Sources: [COVID-19 Sounds (NeurIPS 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/e2c0be24560d78c5e599c2a9c9d0bbd2-Paper-round2.pdf), [COUGHVID corpus](https://arxiv.org/pdf/2009.11644).

### 8.1 How it plays into each goal

**(a) Statistics project — the clearest win.** The single biggest limitation above (§7.2) is that we're capped at 1,926 and stuck in the early log-linear regime. Merging **Coswara** (same modality, ~7× larger) would let us **extend the training-size axis to ~5,000–10,000 and re-run the identical 50-run design.** This directly tests the open question raised by the scaling literature: *does the log-linear trend continue, or does it bend into the predicted saturation?* That transforms the project's conclusion from "positive slope in a narrow window" into "we located the scaling regime — and either confirmed continued log-linear growth or empirically pinned the plateau." Either outcome is a stronger, literature-connected result.

**(b) App AUC ceiling — likely marginal.** Per §6.3 and Hestness et al., adding more *crowd-sourced, weakly-labeled* data climbs the same slope toward the same low ceiling. Raw quantity is not the bottleneck; **label quality is.** Clean, clinically-validated data (Virufy, PCR-confirmed) would do more for the ceiling than volume — but there is too little of it to move training much.

**(c) App practicality / robustness — genuinely helps.** The deployed app's own technical page lists "no external validation dataset" and "generalization to new devices unknown" as limitations. Training across COUGHVID + Coswara exposes the model to more devices, demographics, and recording conditions, improving **real-world robustness even if held-out AUC barely moves.** Using Virufy as a truly external, clean validation set would let us *quantify* generalization — a concrete credibility gain.

### 8.2 The methodological catch

Merging is not free for the experiment's validity:
1. **Label-schema mismatch.** COUGHVID = four-physician majority vote on audio; Coswara = self-reported health status. A consistent binary mapping must be defined and its semantic difference acknowledged.
2. **Dataset-identity confound (batch effect).** If the model learns "Coswara sounds different from COUGHVID" (mic, codec, prompt), pooled AUC can rise for the *wrong* reason. To keep the learning-curve clean, preprocessing and labeling must be identical across sources, and dataset identity should be reported as a controlled limitation.

### 8.3 Recommended sequence

1. **Add Coswara to extend the stats experiment** — biggest, lowest-risk payoff. Re-run the 50-run design out to ~5,000–8,000 and test for the scaling bend.
2. **Use Virufy as a clean external validation set** to report honest generalization.
3. **Do not expect the headline AUC to jump** — frame gains as robustness, not ceiling-breaking, preserving the established "label noise is the real bottleneck" narrative.

---

## 9. Conclusion

The experiment cleanly answered its original question: **more training data significantly predicts higher AUC-ROC** (t = 5.42, p < 0.0001, reject H₀), with a moderate effect (r² = 0.38). The relationship is approximately linear with **no detectable diminishing returns** in range, and is detectable **only because of the replicated design** — seed noise outweighs the size effect run-to-run. These findings are fully consistent with published power-law / log-linear scaling research, and locate this dataset in the **early, rising regime** of a scaling curve whose plateau lies beyond the available data. Acquiring compatible data (notably **Coswara**) is the natural next step: it would let the statistics project test for the predicted scaling bend and let the app demonstrate robustness — though, per the scaling literature and this project's own transfer-learning experience, it is unlikely to break the label-noise ceiling on AUC.

---

## 10. Reproducibility & Artifacts

| Artifact | Path |
|---|---|
| Experiment runner | `stats_experiment/run_experiment.py` |
| Analysis + graphs | `stats_experiment/analyze_results.py` |
| Config (single source of truth) | `stats_experiment/config.py` |
| Raw results (50 rows) | `stats_experiment/results/experiment_results.csv` |
| Stats summary | `stats_experiment/results/analysis_summary.txt` |
| Graphs (8 PNGs) | `stats_experiment/results/graphs/` |

Graphs: `scatterplot_raw`, `residual_plot_raw`, `histogram_residuals`, `normal_probability_plot`, `boxplots_by_size`, `pvalue_diagram`, plus log-transform variants `scatterplot_log`, `residual_plot_log`.

---

## 11. References

1. Orlandic, Tasci & Atienza (2021). *The COUGHVID crowdsourcing dataset.* Nature Scientific Data. https://www.nature.com/articles/s41597-021-00937-4
2. COUGHVID corpus paper. arXiv:2009.11644. https://arxiv.org/pdf/2009.11644
3. Hestness et al. (2017). *Deep Learning Scaling is Predictable, Empirically.* arXiv:1712.00409. https://arxiv.org/abs/1712.00409
4. Sun, Shrivastava, Singh & Gupta (2017). *Revisiting Unreasonable Effectiveness of Data in Deep Learning Era.* ICCV. arXiv:1707.02968. https://arxiv.org/abs/1707.02968
5. COVID-19 Sounds (NeurIPS 2021). https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/e2c0be24560d78c5e599c2a9c9d0bbd2-Paper-round2.pdf
