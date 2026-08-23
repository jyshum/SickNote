---
license: mit
tags:
  - audio-classification
  - audio
  - health
  - cough
  - pytorch
datasets:
  - coughvid
metrics:
  - roc_auc
library_name: pytorch
pipeline_tag: audio-classification
---

# SickNote — cough classifier (healthy vs. abnormal)

An ensemble of three small CNNs (71,793 parameters each) that scores a log-mel
spectrogram of a cough recording. Trained on the expert-labelled subset of
[COUGHVID](https://coughvid.epfl.ch/).

**Read the Limitations section before using this.** The headline number is not
the interesting part of this model card.

## Intended use

Screening support: a rough signal about whether a cough is worth showing to a
clinician. Not a diagnosis, not a COVID detector, not a triage system, not
medical software. It has no regulatory clearance of any kind.

## How it works

```
Input (1, 64, T) log-mel spectrogram   22.05 kHz, 64 mel bins, 8 s window
  → Conv2d(8)  + BatchNorm + ReLU + MaxPool
  → Conv2d(16) + BatchNorm + ReLU + MaxPool
  → Conv2d(32) + BatchNorm + ReLU + MaxPool
  → mean and max pooled over TIME only, concatenated
  → Linear(128) + ReLU + Dropout(0.5)
  → Linear(1) → logit
```

Pooling is over time only, never frequency. In a spectrogram the axes are not
interchangeable: a cough at 2 s and a cough at 5 s are the same event, but
200 Hz and 2 kHz are different sounds, so collapsing frequency would destroy the
band information that distinguishes a wheeze.

Three members, different seeds, probabilities averaged. Three rather than five
because the ensemble size curve was measured: members four and five contribute
+0.0013 combined, inside noise, for 40% more cost.

## Evaluation

Nested 5-fold cross-validation. Every one of the 2,267 clips is scored exactly
once, by a model that never trained on it.

| | mean single model | 3-model ensemble |
|---|---|---|
| previous architecture (flatten head) | 0.6798 | 0.6931 |
| this model (pooling head + SpecAugment) | 0.7161 | **0.7238** |

Paired bootstrap on identical folds: **+0.0308, 95% CI [+0.0137, +0.0494],
better on 5 of 5 folds.**

Operating point at a stated 80% sensitivity floor, threshold selected on
validation and never on test: **sensitivity 0.8029, specificity 0.5054.**

## Limitations

This section carries the numbers that decide whether the model is useful.

**Much of the measured performance is recording conditions, not pathology.**
Three values requiring no learning — clip duration, peak dB, mean dB — reach
**AUC 0.6556** under the same cross-validation. Above chance, this model's
margin over "how long and how loud was the recording" is **+0.068**. Recording
gain reaches the network untouched, and it will not transfer to a different
microphone distribution.

**The labels are close to noise.** 2,161 of the 2,267 clips carry a single
physician's opinion rather than a consensus. On the 126 COUGHVID clips rated by
four physicians, pairwise Cohen's kappa is **0.181 (95% CI [0.103, 0.268])** —
"slight" on the Landis–Koch scale, where diagnostic work expects 0.6+.

```
a rubber stamp answering "abnormal" to everything   matches a physician 80.4%
two real physicians                                 match each other    74.2%
```

A single physician scored against the majority of the other three achieves
balanced accuracy 0.631. This model at its operating point achieves 0.654. It is
already at the level of the humans who produced its training labels, which is
the ceiling, not an achievement.

**The class balance is a rater artifact.** 79.7% of clips are labelled abnormal,
but single raters call 91.8% abnormal against a four-rater consensus of 76.2%,
while 27% of participants self-reported as healthy.

**Positive predictions are unreliable at realistic prevalence.** At 15%
prevalence the operating point gives PPV **0.223** — roughly 78% of "abnormal"
outputs would be false alarms. The model was calibrated where 79.7% of clips
were abnormal.

**Other constraints.** All recordings are voluntary, intentional coughs. No
external validation set, so generalisation to new devices and populations is
untested. COUGHVID has no participant identifier, so subject leakage between
folds cannot be ruled out. Data was collected during the COVID-19 pandemic and
the label distribution reflects that.

## Reproducing

Everything above regenerates from the repository:

```bash
python -m model.audit             # every limitation figure, ~40s
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.crossval \
    --head pool_time --select val_auc_smooth --specaug --seed 100
```

Full experiment log, including the five ideas that did not work:
https://github.com/jyshum/SickNote/blob/main/EXPERIMENTS.md

## Citation

COUGHVID: Orlandic, L., Teijeiro, T., Atienza, D. *The COUGHVID crowdsourcing
dataset: A corpus for the study of large-scale cough analysis algorithms.*
Scientific Data 8, 156 (2021).
