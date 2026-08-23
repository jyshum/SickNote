---
title: SickNote
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Binary cough classifier with measured, honest limitations
---

# SickNote — API

Binary cough classifier (healthy vs. abnormal) served over HTTP. Audio is
converted to a log-mel spectrogram and scored by an ensemble of three small
CNNs, 71,793 parameters each.

```
POST /api/predict     multipart/form-data, field name: file
                      accepts .webm .wav .ogg .mp3

{
  "label": "healthy" | "abnormal",
  "confidence": 0.87,
  "spectrogram": "data:image/png;base64,...",
  "gradcam": "data:image/png;base64,...",
  "ensemble_size": 3
}
```

## Read this before trusting a prediction

Measured by nested 5-fold cross-validation over all 2,267 clips:

| | |
|---|---|
| AUC-ROC | **0.7238** |
| the same harness, using only clip duration and loudness | **0.6556** |
| agreement between the physicians who wrote the labels | **kappa 0.181** |

The second row is the one that matters. A model given three values that require
no learning — how long the recording is, how loud its peak is, how loud it is on
average — reaches 0.6556. Above chance, this model's genuine margin over
recording conditions is about **+0.068**.

The third row is why that ceiling exists. 2,161 of the 2,267 clips carry a
single physician's opinion rather than a consensus, and where four physicians
rated the same audio they agreed at kappa 0.181 — "slight" on the standard
scale. A rubber stamp answering "abnormal" to everything matches a physician
80.4% of the time; two physicians match each other 74.2%.

At a realistic prevalence of 15%, the shipped operating point yields a positive
predictive value of 0.223 — roughly **78% of "abnormal" results would be false
alarms.**

**Screening support only. Not a diagnostic tool, not a COVID detector, not
medical software.**

Full experiment log, including the five ideas that did not work:
https://github.com/jyshum/SickNote/blob/main/EXPERIMENTS.md
