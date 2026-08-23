# SickNote — Experiment Log

A running record of what was tried, what was measured, and what was ruled out.
Every number here is reproducible from the code in this repo.

**Status as of 2026-08-22.** Sections 1–9 are the architecture revamp
(branch `worktree-cv-harness-pooling-head`) and stand as written. Sections
10–12 are a later audit of the *data* rather than the model, and they change
how §4–§9 should be read — start there if you only read one part.

---

## 1. Why this log exists

The README reported **AUC-ROC 0.73**, measured on a single 341-clip test split.

Running the same five checkpoints against the equal-sized *validation* split
gave **0.645**. Same models, same-size splits, drawn from the same pool — a
**0.09 swing from split assignment alone.**

That number was not a measurement. It was a draw. Any architecture change
worth less than ±0.05 AUC was invisible, and every change we ended up testing
was worth less than that.

So the first work was not on the model. It was on the ruler.

---

## 2. The harness

### Nested 5-fold cross-validation — `model/crossval.py`

All 2,267 clips are dealt into 5 stratified folds. Five models are trained;
each scores only the fold it never trained on.

```
             fold 0   fold 1   fold 2   fold 3   fold 4
model 0      TEST     train    train    train    train
model 1      train    TEST     train    train    train
model 2      train    train    TEST     train    train
model 3      train    train    train    TEST     train
model 4      train    train    train    train    TEST
```

Read down any column: exactly one `TEST`. No clip is ever scored by a model
that trained on it. Pooling gives **2,267 out-of-fold predictions** instead of
341 — error bars roughly 2.6× tighter (`sqrt(2267/341)`).

Inside each fold:

```
test  = the held-out fold           ~453 clips
val   = 15% of the remaining folds   ~272 clips   early stopping + threshold
train = the rest                   ~1,542 clips
```

Two details that make it honest:

- **Normalisation stats and `pos_weight` are recomputed per fold**, from that
  fold's training clips only. Global stats would leak test information.
- **The threshold is selected on `val`, applied to `test`.** Never on test.

Runtime: ~60s for a full 5-fold run on MPS. All 2,267 spectrograms (~200 MB)
are loaded into RAM once rather than re-read from disk each epoch.

### Metrics — `model/metrics.py`

- `roc_auc` — rank-based (Mann-Whitney U), tie-corrected, validated exact
  against brute-force pair counting
- `bootstrap_auc_ci` — percentile bootstrap, 2,000 resamples
- `threshold_for_min_sensitivity` — the shipped threshold policy, see §5

### Paired comparison — `compare_runs.py`, `multiseed_report.py`

Every run scores the identical clips in the identical folds, so differences
are tested by resampling clips **once** and scoring both runs on that same
resample. Shared bad luck cancels.

This matters more than it sounds. The flatten and pooling confidence intervals
**overlap** — the naive read is "no difference." The paired difference is
`+0.0336 [+0.0131, +0.0542]`, which **excludes zero**. Overlapping marginal
CIs do not mean "no difference" when the measurements are paired.

---

## 3. Bug found and fixed

`model/evaluate.py` swept classification thresholds **over the test set** and
reported the best one. `THRESHOLD = 0.52` in `config.py` is exactly the argmax
of Youden's J on test — a hyperparameter fitted to the data it was reported on.

Selecting honestly on validation picks **0.48**, and 0.48 performs *worse on
test than leaving it at the default 0.50*. The "improvement" from 0.50 to 0.52
was never real:

```
threshold           sens     spec    Youden J
default    0.50   0.6949   0.6957     0.3905
tuned on test 0.52  0.6765   0.7681     0.4446   <- fitted to test
tuned on val  0.48  0.7243   0.6522     0.3764   <- honest, and worse
```

The peak was also a spike, not a hill (`0.50 -> 0.3905`, `0.52 -> 0.4446`,
`0.54 -> 0.4152`) — noise on 341 samples, not signal.

Fixed: the sweep now runs on validation.

---

## 4. Results

Three training seeds per configuration. Fold assignment held fixed, so every
run is scored on identical clips; only weight init, batch order, and the inner
val split vary.

| config | per-seed AUC | mean | std | ensemble | params |
|---|---|---|---|---|---|
| flatten (original) | 0.6745 / 0.6928 / 0.6840 | **0.6838** | 0.0091 | 0.6971 | 1,415,281 |
| pool over time | 0.7109 / 0.7146 / 0.6997 | **0.7084** | 0.0078 | 0.7233 | 71,793 |
| pool + SpecAugment | 0.7171 / 0.7179 / 0.7219 | **0.7190** | 0.0025 | **0.7266** | 71,793 |
| + freq16 pooling | 0.7070 / 0.7084 / 0.7134 | 0.7096 | 0.0034 | 0.7212 | 137,329 |
| + freq32 pooling | 0.7165 / 0.7119 / 0.7246 | 0.7177 | 0.0064 | 0.7287 | 268,401 |

**Headline: 0.6971 → 0.7266 ensemble AUC, at 1/20th the parameters.**

### What worked

**Pooling head** — replace `flatten` (11,008 features) with concatenated
average + max pooling over the **time axis only**, keeping all 8 frequency
bins (512 features).

```
                params      train/val loss gap
flatten      1,415,281            +0.151
pool_time       71,793            +0.054
```

`+0.0246` mean, improved on **3/3 seeds** and **5/5 folds**. Paired bootstrap
at matched seed: `+0.0336 [+0.0131, +0.0542]`, excludes zero.

Mechanism: flattening gives every `(channel, mel, frame)` coordinate its own
weight, so the classifier must learn "abnormal" independently at all 43 time
positions. With 1,542 clips it memorises coordinates instead. Pooling over
time makes the answer shift-invariant — which is correct, because a cough at
2s and at 5s are the same cough.

Note train loss got *worse* (0.144 → 0.193) while val loss improved
(0.295 → 0.247). That is the intended trade: less memorisation capacity.

**SpecAugment** — 2 frequency masks (≤12 of 64 mel bins), 2 time masks (≤40 of
345 frames), applied post-normalisation so masked regions read as the dataset
mean rather than as silence.

`+0.0106` mean, **3/3 seeds**. But the more striking result is variance:

```
seed-to-seed std:   0.0078  ->  0.0025     3x more stable
```

Caveat worth understanding: at the **ensemble** level the gain shrinks to
`+0.0034 [-0.0090, +0.0159]` — not significant. SpecAugment's main benefit is
variance reduction, and ensembling *also* reduces variance. The two overlap,
so they do not add. It is worth keeping (it is free, and it makes single
models much more reliable), but do not expect it to move the deployed number.

Unlike the waveform augmentation tried earlier in this project (noise
injection, time stretching — both dropped), masking does not distort the
signal. Whatever survives the mask is untouched real audio.

### What did not work

| change | delta | verdict |
|---|---|---|
| epoch cap 80 → 250 | `-0.0016 [-0.0031, -0.0002]`, 0/5 folds | no. Only fold 0 was hitting the cap; training it to 135 epochs made it *worse* |
| early-stopping criterion | `<0.004` across `val_auc` / `val_loss` / `val_auc_smooth` | no. Well inside the CI |
| asymmetric pooling | freq16 `-0.0094`, 0/3 seeds. freq32 `-0.0013` | no |
| widening trunk `[8,16,32]` → `[16,32,64]` | `-0.0060`, **0/3 seeds** | no |

**Trunk widening is the cleanest confirmation that capacity is not the
constraint.** The loss curves predicted this — the model was still mildly
*over*fitting (train 0.193 < val 0.247), and you do not fix overfitting by
adding parameters. Doubling every conv layer cost 2.2× the parameters
(71,793 → 154,849) and **23× the training time** (~90s → ~2,070s per 5-fold
run), and every seed came out worse.

The trunk has been 6,000 parameters since the beginning of the project. It
turns out that was never the problem.

**Asymmetric pooling deserves a note, because the reasoning was good and the
result still said no.** §6 argues that pooling over frequency destroys
diagnostic band information, and `MaxPool2d(2)` three times takes 64 mel bins
down to 8. Keeping more frequency resolution — `freq16` (64→16) and `freq32`
(64→32), pooling time just as hard — should have helped.

It did not. `freq16` is clearly worse; `freq32` is a wash. And the ordering is
**non-monotonic** (sym 0.7190, freq16 0.7096, freq32 0.7177), which is the
signature of three configurations that are actually equivalent, with ±0.01
noise on top.

Two caveats that keep this from being a clean refutation of §6:

- **The test was confounded with capacity.** Keeping frequency resolution
  widens the classifier input, so `freq16` has 1.9× and `freq32` has 3.7× the
  parameters. Any gain from resolution was paid for with more memorisation
  capacity. A clean test would hold parameters fixed by narrowing the
  classifier to compensate.
- `freq32` does have the best ensemble (0.7287) and the best operating point
  (specificity 0.5184, 222 false alarms vs 236). Both are within noise at 3
  seeds, but it is the one variant worth revisiting if it is ever param-matched.

The honest conclusion is that 8 mel bins may simply be enough resolution for
this task, or that label noise is masking the difference.

The two null results above were cheap to rule out (~2 min each) and would
otherwise have sat in the repo forever looking plausible.

---

## 4b. The ensemble, measured honestly

Everything above compares **single models**. The app ships an average of five.
Those are different systems, and plain cross-validation cannot measure the
second one: each clip sits in four of the five fold models' training data, so
no clip exists that all five are innocent of.

`model/ensemble_cv.py` nests the ensemble inside each fold — train N models on
folds ≠ k, all innocent of fold k, average their probabilities on fold k. The
inner train/val split is held fixed within a fold so members differ only by
initialisation, which is exactly what the shipped model does.

**Deployed architecture vs proposed, both as 5-model ensembles:**

| | mean single | 5-model ensemble | 95% CI | ensemble gain |
|---|---|---|---|---|
| deployed (flatten) | 0.6798 | 0.6931 | [0.6661, 0.7212] | +0.0133 |
| proposed (pool + SpecAugment) | 0.7161 | **0.7238** | [0.6987, 0.7504] | +0.0077 |

Paired: **`+0.0308 [+0.0137, +0.0494]`, 5/5 folds, P(better) 0.999.** CI
excludes zero. This is the only apples-to-apples number in the project — what
ships today against what would ship after the change, each measured the way it
actually runs.

**Ensembling is worth far less than the architecture change.** `+0.0077`
against `+0.0308`. Averaging five copies of a weak model does not rescue it.

**The size curve saturates by three members:**

```
                deployed        proposed
1 member         0.6798          0.7161
2 members        0.6879          0.7209
3 members        0.6908          0.7225
4 members        0.6922          0.7233
5 members        0.6931          0.7238
```

Three members capture ~83% of the ensemble benefit in both cases. Members 4
and 5 are worth `+0.0013` combined — deep inside noise. **Dropping from 5 to 3
models would cut inference cost and storage by 40% for no measurable loss.**

**Note which model benefits more from ensembling.** The flatten head gains
`+0.0133`, the regularised one only `+0.0077`. That is consistent and
expected: ensembling works by averaging away variance, and §4 showed
SpecAugment already cut seed-to-seed std from 0.0078 to 0.0025. There is
simply less variance left for the ensemble to remove. The two techniques
overlap rather than add — the same conclusion §4 reached from a different
direction.

**Operating point at the 0.80 sensitivity floor** (per-fold thresholds
selected on validation):

```
deployed   sens 0.8051   spec 0.4794   missed 352   false alarms 240
proposed   sens 0.8029   spec 0.5054   missed 356   false alarms 228
```

---

## 5. Threshold policy

SickNote is a screening tool. A missed abnormal cough costs more than a false
alarm, so maximising Youden's J — which weights the two equally — is the wrong
objective.

The shipped rule is a **stated sensitivity floor**:

> Catch at least `MIN_SENSITIVITY` of abnormal coughs. Among thresholds that
> do, take the best specificity.

At `MIN_SENSITIVITY = 0.80`, seed-averaged:

```
                    thr    sens     spec    missed   false alarms
flatten            0.42   0.8123   0.4534     339        252
pool_time          0.40   0.8106   0.4837     342        238
pool + specaug     0.44   0.8007   0.4881     360        236
```

The tradeoff is steep at this AUC — 80% sensitivity costs roughly half your
specificity. That is a product decision, not a math one, and it should be
revisited deliberately rather than inherited from an argmax.

---

## 6. Spectrograms are not photographs

The finding that shaped the head redesign, and the reason to be careful about
importing computer-vision defaults.

Convolution has one baked-in assumption: **a pattern means the same thing
wherever it appears.** In a photo that holds on both axes. In a spectrogram it
holds on one and breaks on the other:

```
TIME (horizontal)      holds.  a cough at 2s = a cough at 5s
FREQUENCY (vertical)   BREAKS. 200 Hz and 2 kHz are different sounds
```

Low frequencies carry chest resonance and voicing; high frequencies carry
turbulence and wetness. A filter that detects "wet crackle" at 4 kHz is
meaningless at 100 Hz. This is a known problem with named solutions —
limited weight sharing, frequency-dynamic convolution, frequency-aware
convolution — all variants of "convolve, but tell the network which band it
is in."

The honest counter-argument: in a log-frequency spectrogram, vertical
translation *is* pitch shift, so frequency convolution buys pitch invariance
across speakers. It is a trade, not a mistake.

Practical consequences for this model:

1. Pooling over **time** is safe. Pooling over **frequency** destroys
   diagnostic band information. The current head pools only time — deliberate.
2. `MaxPool2d(2)` halves **both** axes three times, so frequency goes
   `64 → 32 → 16 → 8`. **87% of frequency resolution is destroyed** before the
   classifier ever sees it. Asymmetric pooling is the obvious next experiment.
3. `3x3` kernels treat 3 mel bins and 3 frames (~70 ms) as comparable units.
   They are not. Rectangular kernels are common in audio CNNs for this reason.

References: [Frequency-aware convolution for sound event
detection](https://arxiv.org/html/2403.13252v1) ·
[Frequency Dependent Sound Event Detection, DCASE 2022](https://arxiv.org/pdf/2206.11645)

---

## 7. Limitations

**Subject leakage cannot be ruled out.** COUGHVID has no participant
identifier — every clip has its own uuid and there is no way to link
recordings from the same person. Using `(latitude, longitude, age, gender)` as
a rough fingerprint, **202 of the 1,185 clips with all four fields populated
sit in multi-clip groups** (largest group: 3). If those are repeat
submissions, some AUC is inflated. Grouped CV on that fingerprint would
measure it; not done yet.

**The ceiling is probably label noise, not architecture.** COUGHVID labels are
crowd-sourced physician majority votes. `stats_experiment/REPORT.md` §6.3
estimates a Bayes-optimal AUC around 0.65–0.72 for this task and notes that a
ResNet18 transfer-learning attempt and a larger-capacity model both failed to
improve on it. We are plausibly near that ceiling already.

**The ensemble is now measured** — see §4b. One caveat remains: fold ensembles
train on 4/5 of the data while the deployed ensemble trains on all of it, so
the real deployed number should be slightly better than 0.7238.

**Ensemble members differ only by initialisation.** The inner train/val split
is held fixed within each fold, matching current deployment. Varying the split
per member would produce more diverse members and probably a better ensemble
(closer to bagging). Untested.

---

## 8. Reproducing

```bash
# reproduce every number in section 10 (~40s)
PYTHONPATH=. python -m model.audit

# full 5-fold run, ~60s on MPS
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.crossval --head pool_time --select val_auc_smooth

# with SpecAugment
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.crossval --head pool_time --select val_auc_smooth --specaug

# three seeds per config
bash run_multiseed.sh

# aggregate and compare
python multiseed_report.py
python compare_runs.py
```

Fold checkpoints (~110 MB per config) are gitignored. The `oof_predictions.csv`,
`loss_curves.csv`, and `summary.json` files under `model/checkpoints/cv/` are
committed — those are the evidence.

`data/processed/` is gitignored, so a fresh worktree needs the spectrograms
symlinked or regenerated with `python -m model.preprocess`.

---

## 9. Next

> **Superseded by §10–§12 (2026-08-22).** Item 1 below recommends shipping the
> pooling head. That recommendation is still correct *relative to what ships
> today*, but §10 shows a large share of the AUC it improves is recording
> duration and gain rather than pathology, and §11 identifies a dataset with a
> laboratory label and 31× the volume. Read §12 before acting on this list.

1. **Ship it.** §4b closes the last measurement gap: `0.6931 -> 0.7238`
   ensemble-to-ensemble, 5/5 folds, CI excludes zero. Swapping the head to
   `pool_time`, enabling SpecAugment, and cutting the ensemble from 5 members
   to 3 is now supported by measurement rather than by argument. Requires
   retraining production checkpoints and updating `api/inference.py` and
   `config.py` (`THRESHOLD` should come from the validation-selected value,
   not the current 0.52).
2. **Param-matched frequency resolution** — `freq32` with the classifier
   narrowed (128 → 48 units) so it lands at ~72K parameters, matching the
   baseline. That is the clean version of the §4 test and the only way to know
   whether frequency resolution or capacity was responsible.
3. **Accept the ceiling.** Five of seven ideas came back null, all of them
   capacity- or training-related. `stats_experiment/REPORT.md` §6.3 estimates
   Bayes-optimal AUC at 0.65–0.72 for COUGHVID given its label noise, and
   §4b measures 0.7238. Further architecture work is very likely wasted; the
   remaining levers are better labels or more data, not a better model.

A note on hit rate: **two ideas out of seven worked** (the pooling head, and
SpecAugment for stability). That is normal, and the only reason it is knowable
is that the harness came first.

---

## 10. The measurement that reframes everything above

**Status: 2026-08-22.** Sections 1–9 asked "which architecture is best?" and
answered it carefully. This section asks a question nobody had asked: **how
much of the 0.7238 is about coughs at all?**

### 10.1 A model that hears nothing gets 0.6556

Three scalars per clip, derived from the spectrogram with no learning:

- `duration` — non-silent frames × hop ÷ sample rate
- `peak` — loudest cell in the clip, dB
- `mean_db` — mean dB over non-silent frames

Fed to a logistic regression, 5-fold cross-validated on the same 2,267 clips:

| feature | AUC |
|---|---|
| duration alone | 0.5655 |
| peak dB alone | 0.5546 |
| mean dB alone | 0.6300 |
| **all three** | **0.6556** |
| CNN ensemble (§4b) | 0.7238 |

Above chance, the CNN's margin over "how long and how loud was the recording"
is **+0.068**. That is the number this project actually earned.

`mean_db` at 0.6300 is the one that should worry us. `preprocess.py` normalises
with a **single global mean/std** — there is no per-clip normalisation — so
recording gain reaches the network untouched. Microphone distance, browser AGC
and device differences are all in that feature. None of them transfer to a
deployment where the user records on their own phone.

This is not a hypothesis about leakage. It is a measurement of it.

### 10.2 The input is mostly empty

Padding to 8 seconds happens **before** the mel transform, so short clips carry
a block of constant −100 dB. Measured over all 2,267 clips:

```
fraction of frames that are digital silence   p50 0.122   p90 0.638   max 0.965
                                              mean 0.236
clips containing any padding                  0.690
frames within 20 dB of the clip's peak        mean 0.151
```

A quarter of the average input is padding, and only ~15% of frames carry the
cough. §4 justified max-pooling partly because average pooling "would leak how
much of the clip is zero-padding" — §10.1 shows that leak is worth real AUC.

Also: the pipeline normalises all 64 mel bins with one scalar, but the bins
differ by **29.9 dB in mean** and 1.99× in std (400 clips, seed 0). The top
bins — where turbulence and wetness live — get squashed into a narrow range
before the first conv.

### 10.3 The labels are close to noise, and it is measurable

§7 called the labels "crowd-sourced physician majority votes." That is wrong
for almost all of them:

```
clips labelled by exactly 1 expert    2,161  (95.3%)
clips labelled by 3 experts               1
clips labelled by 4 experts             105
```

(Rater counts are over the 2,267 clips we train on. The agreement statistics
below use all 126 four-rater clips in the dataset — only 126 exist, and
restricting to our filtered set would leave too few to measure.)

There is no majority. There is one physician. On the 126 clips in the full
dataset where all four rated, agreement between raters is:

```
two real doctors agree                74.2%
two random guessers at that base rate 67.7%     <- chance
mean pairwise Cohen's kappa           0.181     95% CI [0.103, 0.268]
```

κ = 0.181 is "slight" on the Landis–Koch scale. Diagnostic work wants 0.6+.
The intuition that lands it:

```
a rubber stamp that answers "abnormal" to everything,
with no audio at all, agrees with a real doctor       80.4%
two real doctors agree with each other                74.2%
```

**A rubber stamp matches a physician better than another physician does.**

A single expert scored against the majority of the other three gets sensitivity
0.838, specificity 0.424 — **balanced accuracy 0.631**. The proposed model at
its shipped operating point (sens 0.8029, spec 0.5054) gets **0.654**. The model
is already at or just past the humans who wrote its labels. There is no
architecture that fixes this.

Caveat: n=126. The 4-expert subset is not obviously cherry-picked
(cough_detected 0.878 vs 0.899 for single-rater clips; self-reported-healthy
0.270 in both), but the abnormal rate differs sharply — see §10.4.

Two related checks, in case a better-posed target existed:

| target | kappa | verdict |
|---|---|---|
| diagnosis → abnormal | 0.181 | the current target |
| cough_type → wet | 0.264 | better, still poor |
| quality → good | 0.116 | experts cannot agree what a good recording is |
| dyspnea / wheezing | — | not populated on multi-rater clips |

### 10.4 The 80% abnormal base rate is a rater artifact

```
abnormal rate, 1-rater clips   0.918
abnormal rate, 4-rater clips   0.762
single-rater specificity       0.424
```

Single physicians call nearly everything abnormal, and 95% of the labels come
from single physicians. Meanwhile 27% of these participants self-report as
healthy. The class balance is a property of the raters, not the population.

Consequence at the shipped threshold (sens 0.803 / spec 0.505):

```
prevalence 0.797 (as trained) -> PPV 0.864
prevalence 0.30               -> PPV 0.410
prevalence 0.15               -> PPV 0.223   78% of "abnormal" flags are wrong
prevalence 0.05               -> PPV 0.079   92% wrong
```

SickNote is deployed to the general public, not to a cohort that is 80% ill.

### 10.5 Self-report and expert diagnosis are independent

COUGHVID also carries a self-reported `status`. On the 2,048 training clips
that have both:

```
expert says →           healthy   abnormal
self-reported healthy       155        498      76% called abnormal
self-reported symptomatic   138        726      84%
self-reported COVID-19      127        404      76%

raw agreement 0.627     kappa 0.052
```

The expert abnormal rate barely moves with what the participant reports about
their own health. **The two label sources are all but statistically
independent.** Whatever the physicians were responding to, it was not illness.

### 10.6 What is unused

```
clips with cough_detected > 0.8 and no expert label   16,091
  ...of which carry a self-reported status            11,327
  ...with no label of any kind                         4,764
```

Seven times the labelled set, same recording distribution, entirely untouched.

---

## 11. Dataset survey — what is actually obtainable

Every row below was verified against the primary source, not a description of
it. Licences were read from the licence file; availability was confirmed by
requesting bytes.

| dataset | licence | access | usable cough clips | label type |
|---|---|---|---|---|
| **UK COVID-19 Vocal Audio** | Open Government Licence v3.0 | open, download verified | **71,194** | PCR test |
| **Coswara** (IISc) | CC BY 4.0 | open, download verified | ~2,100 heavy (+~2,100 shallow) | self-report, PCR subset |
| Cambridge COVID-19 Sounds | academic-only agreement | institutional signatory | 459 | self-report |
| Virufy | CC BY-NC-SA + addendum | open | 121 clips, **16 patients** | PCR |

Both open datasets were confirmed downloadable with an unauthenticated ranged
request (`HTTP 206`); the Coswara chunk verifies as a real gzip archive.

**Cambridge is closed in practice.** The shareable subset is 459 samples from
378 users — smaller than what we already have — and requires a Data Access
Agreement signed by an institutionally authorised signatory, for academic
research only. Its host also serves on an **expired TLS certificate**, which
suggests the project is unmaintained.

**Virufy is 16 patients.** PCR-referenced and therefore high quality, but far
too small to train on. Possible use: a tiny gold-standard sanity check.

**Do not use the Kaggle mirrors.** `himanshu007121/coswara-dataset` is listed
as CC0 public domain; the real Coswara is CC BY 4.0. Third-party re-uploads
misstate licences. Go to the primary source.

### 11.1 The UK COVID-19 Vocal Audio Dataset

Verified from the metadata files, not the description:

```
participants                72,999          PCR positive        25,706
usable audio                71,194          PCR negative        47,036
RT-qPCR                     68,068          exhalations @48kHz  72,999

pre-built splits    train 20,717 · val 5,180 · test 11,121 · long 30,824
matched splits      train  4,974 · test  1,968   (confounder-controlled)
```

**31× the labelled set we have, with a laboratory test in place of a
physician's guess.** It additionally carries viral load and CT values, a
14-symptom panel, asthma and smoking status, and influenza PCR results. Age and
gender are populated for every participant — which makes the grouped-CV
leakage check that §7 could not run on COUGHVID straightforward here.

Licence is OGL v3.0: attribution only, commercial use permitted, no application
and no account.

### 11.2 The result that ships with it

The dataset's companion paper is titled *"Audio-based AI classifiers show no
evidence of improved COVID-19 screening over simple symptoms checkers"*
([arXiv:2212.08570](https://arxiv.org/abs/2212.08570)).

```
unadjusted (naive splits)                    ROC-AUC 0.846 [0.838, 0.854]
after matching age, gender, symptoms         ROC-AUC 0.619 [0.594, 0.644]
```

and the classifiers are beaten by a simple score over self-reported symptoms.

**This is §10.1 at national scale.** UKHSA and the Alan Turing Institute, with
67,842 participants, measured the same collapse we measured on 2,267 — apparent
performance that is mostly confounding. That the two findings were reached
independently, on different datasets, by different methods, is the strongest
evidence in this document. It also means the `in_matched_*` splits are the only
honest way to evaluate on this data, and they are provided.

The unavoidable reading: **cough audio alone is a weak signal for infection
status.** Not zero — 0.619 excludes 0.5 — but weak, and weaker than asking
someone what hurts.

---

## 12. Where this leaves the project

§9 said "accept the ceiling; the remaining levers are better labels or more
data." §10 and §11 sharpen that considerably.

The ceiling was never architectural, and it is not really label noise either.
It is that **the question was badly posed.** "Does a physician think this cough
sounds abnormal" is a target with κ=0.181 among physicians, independent of
whether the participant is ill (§10.5), on a class balance that is an artifact
of how raters behave (§10.4), scored partly by recording duration and gain
(§10.1). No head, pooling shape, or trunk width addresses any of that.

### 12.1 Direction

1. **Move to the UK dataset and report the matched-split number.** Landing near
   0.62 would *reproduce a government study* — a far stronger result than 0.72
   on a benchmark we now know is confounded. Use `in_matched_rebalanced_*`.

2. **Strip the shortcuts regardless of dataset.** Cough-burst segmentation,
   per-clip normalisation, per-mel-bin normalisation. Without this we will
   relearn duration and gain against a new target. §10.1 and §10.2 are the
   specification.

3. **Build what the evidence supports: symptoms plus audio.** §11.2 says the
   symptom score is the strong component. The UK data ships a 14-symptom panel
   alongside every recording. This is also the honest product — a screening aid
   for someone who cannot see a doctor for weeks, not a diagnosis.

### 12.2 Adventitious sound detection — the one architectural idea left

Detect **crackles and wheezes** rather than classifying illness directly.

The case for it is that it inverts every problem in §10. Crackle and wheeze are
*acoustic events*, not inferences about a person — an annotator marks a sound
that is either present in the waveform or is not. That makes the label far
better defined than a diagnosis, and makes the detector much harder to confound
with age, gender, or recording gain. The output is also interpretable: "a wheeze
was detected at 2.1 s" is actionable in a way that "0.73 abnormal" is not.

The obvious objection is that ICBHI — the only dataset with these labels — is
stethoscope audio, and SickNote's entire premise is a phone and no equipment.
But crackles and wheezes occur during **breathing**, and the UK dataset recorded
**72,999 exhalations at 48 kHz** alongside the coughs. That is the bridge: the
same physiological event ICBHI annotates, captured on the target hardware.

And it can be validated without any crackle/wheeze labels on the UK audio:

```
self-reported asthma        8,249 yes / 64,567 no
shortness of breath         7,813
other respiratory condition 1,892
```

Wheeze is the defining sound of asthma. Train the detector on ICBHI, run it on
UK exhalations, and test whether its wheeze score separates those 8,249. If it
does, we have an interpretable acoustic feature that §10's confounds cannot
explain. If it does not, we have learned the domain gap is fatal — in one
experiment.

**The risk is physical, not statistical.** A crackle is a 5–15 ms,
low-amplitude transient. ICBHI captures it with a contact microphone pressed to
the chest wall. Through air, at arm's length, into a phone, it may be *absent
from the signal* rather than merely hard to detect. No model fixes a signal
that is not there. That is precisely why the asthma-separation test should be
run before any effort goes into the detector itself.

Note on the audio metadata: `cough_snr` and `exhalation_snr` are provided but
undefined in the open data dictionary, and sit at 0.2–0.4 across all three
modalities. They are not decibels and should not be read as a noise estimate —
compute SNR from the waveforms instead.

### 12.3 What is now ruled out

- **More head, pooling, or capacity work on COUGHVID.** Five of seven ideas
  came back null in §4, and §10 explains why the remaining two were worth so
  little.
- **Soft or agreement-weighted labels.** Only 106 clips have more than one
  rater (§10.3).
- **Pretraining on COUGHVID's self-reported status.** κ=0.052 against the
  target (§10.5) — it is a different task, not a related one.
- **Cambridge and Virufy** as training data (§11).
