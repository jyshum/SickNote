# SickNote — Architecture Plan
> Binary cough classifier: healthy vs. abnormal
> Two-partner project | Next.js + FastAPI + PyTorch

---

## Project Structure

```
sicknote/
├── ARCHITECTURE.md
├── requirements.txt              # Python deps (FastAPI, torch, etc.)
├── .env.example
├── .gitignore
│
├── model/                        # [P1 — ML]
│   ├── __init__.py
│   ├── config.py                 # all hyperparams — edit after explore.py
│   ├── explore.py                # run FIRST — answers dataset questions
│   ├── dataset.py                # CoughDataset (torch Dataset)
│   ├── architecture.py           # SickNoteCNN definition
│   ├── train.py                  # training loop + checkpointing
│   ├── evaluate.py               # AUC, accuracy, confusion matrix
│   ├── preprocess.py             # raw audio → tensor pipeline
│   └── checkpoints/              # saved .pt files (gitignored)
│
├── api/                          # [Shared — see ownership rules below]
│   ├── __init__.py
│   ├── main.py                   # FastAPI app + POST /api/predict  [P2 owns]
│   └── inference.py              # predict() function               [handoff file]
│
├── frontend/                     # [P2 — Frontend]
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── tsconfig.json
│   ├── public/
│   └── src/
│       └── app/
│           ├── layout.tsx        # root layout + metadata
│           ├── page.tsx          # main UI — record/upload, results, spectrogram
│           └── globals.css       # Tailwind base
│
├── data/                         # [P1 — ML]
│   ├── raw/                      # COUGHVID extracted here (gitignored)
│   ├── processed/                # mel spectrogram tensors (gitignored)
│   └── demo_clips/               # 10 held-out labeled clips
│
└── tests/
    ├── test_api.py               # [P2]
    └── test_model.py             # [P1]
```

---

## Partner Boundaries

| File/Folder | Owner | Never Touched By | Notes |
|---|---|---|---|
| `model/*` | **P1** | P2 | All ML code — explore, preprocess, train, evaluate |
| `data/*` | **P1** | P2 | Raw data, processed tensors, demo clips |
| `api/main.py` | **P2** | P1 | FastAPI route, file validation, error handling |
| `api/inference.py` | **Handoff** | — | P2 writes mock → P1 replaces body (see below) |
| `frontend/*` | **P2** | P1 | Entire Next.js + Tailwind app |
| `tests/test_api.py` | **P2** | P1 | API endpoint tests |
| `tests/test_model.py` | **P1** | P2 | Model/dataset tests |
| `requirements.txt` | Both | — | Python deps only |
| `ARCHITECTURE.md` | Both | — | Source of truth |

### The Handoff File: `api/inference.py`

This is the ONLY file both partners touch, and they do so **sequentially**:

1. **P2 writes the mock version first.** It returns random labels, random
   confidence, and a placeholder spectrogram image. P2 builds the entire
   frontend and API against this mock.

2. **P1 replaces the function body later.** After training is complete, P1
   fills in the real model loading, preprocessing, and inference logic.
   The function signature does not change — P2's code is unaffected.

They never edit this file at the same time. No merge conflicts.

---

## The Contract (API Endpoint)

The contract between P1 and P2 is an HTTP endpoint — neither partner
imports the other's code.

```
POST /api/predict
Content-Type: multipart/form-data

Request body:
  file: audio file (webm, wav, ogg, mp3)

Response (200 OK):
{
    "label": "healthy" | "abnormal",
    "confidence": 0.87,
    "spectrogram": "data:image/png;base64,..."
}
```

Under the hood, `api/main.py` receives the upload, saves it to a temp file,
and calls `predict(audio_path)` from `api/inference.py`. The `predict`
function signature is the internal contract:

```python
def predict(audio_path: str) -> dict:
    """
    Args:
        audio_path: path to an audio file on disk

    Returns:
        {
            "label": "healthy" or "abnormal",
            "confidence": float between 0.0 and 1.0,
            "spectrogram": base64-encoded PNG string
        }
    """
```

P2 writes the mock. P1 replaces the body. The signature never changes.

---

## P1 Path — ML Backend

P1 works exclusively in `model/` and `data/`. P1 never touches `frontend/`
or `api/main.py`. At the end, P1 implements the real `predict()` body in
`api/inference.py`.

### P1 Step 0 — Run explore.py Before Any Model Code

This is mandatory. Every CNN parameter depends on answers from the dataset.
Do not write dataset.py or architecture.py until explore.py has run.

explore.py must answer these questions and print results:

```
1. How many rows survive: cough_detected > 0.8 AND at least one diagnosis_N not null?
   → determines if dataset is large enough, affects batch size
   → ALSO check quality_N columns — exclude rows with majority poor/no_cough

2. Class distribution after filtering (using majority vote across diagnosis_1-4)?
   → exact healthy vs abnormal counts → compute real class weights
   → EXPECTED: ~22% healthy / ~78% abnormal (healthy is the MINORITY class)

3. Duration distribution of surviving clips (median, p10, p90)?
   → determines CLIP_LENGTH_S — do not pad more than necessary

4. Plot 5 healthy spectrograms + 5 abnormal spectrograms with current params
   → are they visually distinguishable?
   → do the frequency bands show clear differences?

5. Print tensor shape after MelSpectrogram transform
   → confirms flatten dimension for Linear layer
   → example: torch.Size([1, 64, 173]) with 4s clip at 22050Hz

6. Check for corrupted or unreadable files
   → log any files that fail to load
```

Output of explore.py feeds directly into config.py. Nothing in config.py
is final until explore.py has run on the real dataset.

### P1 Step 1 — Update config.py

| explore.py finding | config.py change |
|---|---|
| Median clip duration is 2.1s | `CLIP_LENGTH_S = 3` |
| Only 1,200 samples survive filter | `BATCH_SIZE = 16`, drop to 2 conv blocks |
| Spectrograms look identical between classes | Try `N_MELS = 128`, smaller `HOP_LENGTH` |
| Class ratio is ~78/22 abnormal/healthy | pos_weight < 1.0 or flip label convention |
| Tensor shape is `(1, 64, 173)` | Flatten dim = `64 * 21 * 5` = computed precisely |
| Many corrupted files | Add try/except in dataset.py, skip bad files |

### P1 Step 2 — Preprocess Data

Run `python -m model.preprocess` to:
- Filter metadata to expert-labeled rows with cough_detected > 0.8
- Apply quality filter (exclude poor/no_cough majority)
- Assign binary labels via majority vote across diagnosis_1-4
- Convert each audio file to a normalized log-mel spectrogram tensor
- Save .pt files to `data/processed/`
- Stratified 70/15/15 split → save `splits.pt`
- Compute train-only normalization stats → save `preprocessing_params.pt`

**Critical data leakage rule:** Compute normalization mean/std on the train
split only. Apply those same values to val and test.

### P1 Step 3 — Train

Run `python -m model.train`:

```
Per epoch:
  - Forward pass on train set
  - BCEWithLogitsLoss with pos_weight (numerically stable — no Sigmoid in model)
  - Adam optimizer step
  - Forward pass on val set (no grad, apply sigmoid to get probabilities for AUC)
  - Log: train_loss, val_loss, val_AUC

After each epoch:
  - Save checkpoint if val_AUC improved
  - Early stopping if no improvement for PATIENCE epochs

After training completes:
  - Load best checkpoint (not last epoch)
  - Run evaluate.py on test set
  - Print: AUC, accuracy, sensitivity, specificity, confusion matrix
  - Save model_final.pt + preprocessing_params.pt
```

Use `val_AUC` as the checkpoint criterion, not `val_loss`.
AUC is threshold-independent and more meaningful for imbalanced classes.

### P1 Step 4 — Evaluate

Run `python -m model.evaluate`. Report all four metrics on the held-out test set:

| Metric | Minimum | Target | Why |
|---|---|---|---|
| AUC-ROC | 0.75 | > 0.82 | Primary — threshold-independent |
| Accuracy | 70% | > 0.78 | Secondary — post class-balancing |
| Sensitivity | 0.65 | > 0.75 | Recall on abnormal (majority class) — missing sick is worse |
| Specificity | 0.60 | > 0.70 | Recall on healthy (minority class, ~22%) — harder to achieve |

**If AUC < 0.75 after first run, check in this order:**
1. Are class weights computed from filtered expert-labeled data, not full dataset?
2. Is normalization fitted on train split only?
3. Is the model seeing enough healthy examples? (print batch labels — healthy is minority ~22%)
4. Try LR = 1e-4, more epochs, lighter model

### P1 Step 5 — Implement Real Inference

Replace the mock body in `api/inference.py` with real model loading and
prediction. The function signature stays the same:

```python
def predict(audio_path: str) -> dict:
```

Must include:
- `torch.load()` with `map_location='cpu'` and `weights_only=True`
- `model.eval()` before inference (switches BatchNorm to running stats, disables Dropout)
- `torch.no_grad()` around the forward pass
- `torch.sigmoid()` on the raw logits to get probabilities
- Load preprocessing params from `preprocessing_params.pt` (never hardcode)
- Return base64-encoded spectrogram PNG, not a raw tensor

### P1 Step 6 — Select Demo Clips

Select 10 clips from the test set (5 healthy, 5 abnormal) where the model
makes confident correct predictions. Save to `data/demo_clips/`.

---

## P2 Path — Frontend + API

P2 works in `frontend/` and `api/main.py`. P2 writes the mock version of
`api/inference.py`. P2 never touches `model/` or `data/`.

### P2 Step 0 — Write Mock Inference

Create `api/inference.py` with a `predict()` function that returns random
results. This lets you build and test the entire frontend without waiting
for the model.

```python
def predict(audio_path: str) -> dict:
    # MOCK — returns random results for frontend development
    # P1 replaces this body with real inference later
    ...
```

### P2 Step 1 — Set Up FastAPI

Create `api/main.py` with a single endpoint:

```
POST /api/predict
- Accept multipart audio file upload
- Validate file type (webm, wav, ogg, mp3)
- Save to temp file
- Call predict(temp_path)
- Return JSON response
- Clean up temp file
```

Add CORS middleware so the Next.js dev server (port 3000) can call the
API server (port 8000).

### P2 Step 2 — Build Next.js Frontend

Minimal MVP — single page with:
- Audio upload button OR microphone recording
- Submit button → calls `POST /api/predict`
- Result display: label (HEALTHY / ABNORMAL) + confidence percentage
- Spectrogram image display (base64 from API response)
- Medical disclaimer footer

### P2 Step 3 — Integration

When P1 delivers the real `api/inference.py`:
- No frontend changes needed
- No API route changes needed
- Just restart the FastAPI server — it picks up the new inference code

---

## Key Architectural Decisions

### 1. Binary classification only
Collapse COUGHVID labels to healthy vs. abnormal.
**Why:** Multi-class (COVID/URTI/LRTI) accuracy drops sharply — pathological
coughs occupy overlapping feature space. Binary exceeds 84% AUC. Medically
honest — "something sounds off" not "you have X."

### 2. Expert labels only + cough_detected filter
Use `diagnosis_1` through `diagnosis_4` columns (expert physician annotations)
with majority vote. Filter `cough_detected > 0.8`.
**Why:** Raw `status` is self-reported and noisy. The `status_SSL` column is
semi-supervised learning predictions — NOT expert labels — and is wrong 66%
of the time vs expert consensus. The four physician diagnosis columns are the
only trustworthy signal (~2,841 recordings). Majority vote across experts
resolves disagreements. The cough_detected filter removes non-cough audio
that pollutes the spectrogram distribution.
**Label mapping:** `healthy_cough` → healthy (0) | `COVID-19`, `upper_infection`,
`lower_infection`, `obstructive_disease` → abnormal (1).
**Quality filter:** Exclude rows where majority of experts rate quality as
`poor` or `no_cough`.

### 3. Fixed-length clips via pad/trim
Trim or zero-pad all audio to CLIP_LENGTH_S before spectrogram.
**Why:** CNN requires fixed input size. Value set after explore.py reveals
the median clip duration — do not pad more than 1.5x the median or you
are adding more silence than signal.

### 4. Log-mel spectrogram as input (not raw waveform)
`torchaudio.transforms.MelSpectrogram` → `AmplitudeToDB()` → 2D tensor.
**Why:** Mel scale matches human auditory perception. Log compression handles
dynamic range. Treats audio as an image — lets standard 2D CNN apply
without custom architecture. Established approach in audio classification.

### 5. Small CNN trained from scratch
Three conv-batchnorm-relu-maxpool blocks → flatten → two FC layers.
Model outputs raw logits — no Sigmoid in the network. Use `BCEWithLogitsLoss`
for training (numerically stable). Apply `torch.sigmoid()` at inference only.
**If usable samples after filtering < 1,500:** drop to two conv blocks.

### 6. Class weighting over oversampling
Pass `pos_weight` to `BCEWithLogitsLoss` based on inverse class frequency
computed from the FILTERED expert-labeled dataset.
**Why:** After expert-label filtering, the dataset is ~78% abnormal / ~22%
healthy — healthy is the MINORITY class. Oversampling duplicates data and
risks minority class overfitting. Class weights rebalance the loss function
without touching the data.

### 7. MPS backend on M2 Mac
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```
Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in shell environment before training.
Use `num_workers=0` in DataLoader to avoid MPS + multiprocessing conflicts.
**Why:** MPS gives 3-5x speedup over CPU on M2 for conv operations.

### 8. Next.js + Tailwind frontend with FastAPI backend
Frontend and ML backend communicate via a single HTTP endpoint.
**Why:** Clean separation — P2 never needs PyTorch, P1 never needs Node.js.
Both can develop in parallel with the mock API. Industry-standard pattern
for ML-powered web apps.

### 9. Spectrogram visualization in the UI
Display the mel spectrogram the model analyzed alongside the prediction.
**Why:** Judges see visually different frequency patterns between healthy
and abnormal. Makes the AI legible to non-technical people.

---

## Config (P1) — defaults before explore.py

All values in model/config.py. Every value marked * must be verified
or updated after running explore.py on the real dataset.

```python
# Audio
SAMPLE_RATE = 22050
CLIP_LENGTH_S = 4          # * verify against duration distribution
N_MELS = 64                # * adjust after visualizing spectrograms
N_FFT = 1024               # * adjust if time resolution looks wrong
HOP_LENGTH = 512           # * adjust if frequency resolution looks wrong

# Training
BATCH_SIZE = 32            # * lower to 16 if <1500 usable samples
EPOCHS = 30                # ceiling — early stopping will fire before this
LR = 1e-3
WEIGHT_DECAY = 1e-4        # L2 regularization — helps with small dataset
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 5

# Model
CONV_CHANNELS = [16, 32, 64]   # * reduce to [8, 16, 32] if overfitting
KERNEL_SIZE = 3

# Evaluation
THRESHOLD = 0.5
DEVICE = "mps"             # auto-detected at runtime with cpu fallback

# Paths
RAW_DIR = "data/raw/coughvid_20211012"
PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "model/checkpoints"
DEMO_CLIPS_DIR = "data/demo_clips"
METADATA_CSV = "data/raw/coughvid_20211012/metadata_compiled.csv"
```

---

## Model Architecture (P1)

The flatten dimension MUST be computed from actual tensor shape — do not
hardcode from an estimate. Model outputs raw logits (no Sigmoid layer).

```python
class SickNoteCNN(nn.Module):
    def __init__(self, n_mels, time_frames):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self._flatten_dim = self._get_flatten_dim(n_mels, time_frames)

        self.classifier = nn.Sequential(
            nn.Linear(self._flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            # NO Sigmoid — use BCEWithLogitsLoss for numerical stability
            # Apply torch.sigmoid() at inference time only
        )

    def _get_flatten_dim(self, n_mels, time_frames):
        dummy = torch.zeros(1, 1, n_mels, time_frames)
        out = self.conv(dummy)
        return int(torch.prod(torch.tensor(out.shape[1:])))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)  # returns raw logits
```

**If overfitting** (val loss rises while train loss falls):
- Add Dropout after each conv block
- Reduce CONV_CHANNELS to [8, 16, 32]
- Add data augmentation: time masking, frequency masking (SpecAugment)

**If underfitting** (both losses plateau high):
- Increase CONV_CHANNELS to [32, 64, 128]
- Lower LR to 1e-4
- Check that class weights are being applied correctly

---

## Data Pipeline (P1)

```
COUGHVID raw WEBM/WAV/OGG (2.1GB, ~34,400 recordings)
    ↓
FILTER: cough_detected > 0.8 AND at least one diagnosis_N not null
    ↓
QUALITY: exclude rows where majority of experts rate quality as poor/no_cough
    ↓ (expected: ~2,200–2,450 rows survive)
    ↓
LABEL: majority vote across diagnosis_1–diagnosis_4
       healthy_cough → 0   |   COVID-19/upper_infection/lower_infection/obstructive_disease → 1
    ↓ (expected: ~22% healthy / ~78% abnormal)
    ↓
LOAD: torchaudio.load() → (waveform, sample_rate)
    ↓
RESAMPLE: torchaudio.functional.resample() → SAMPLE_RATE Hz mono
    ↓
PAD/TRIM: to exactly CLIP_LENGTH_S seconds
    ↓
SPECTROGRAM: MelSpectrogram(n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    ↓
LOG SCALE: AmplitudeToDB()
    ↓
NORMALIZE: (x - train_mean) / train_std
    ↓ (mean/std computed on train split ONLY — never full dataset)
    ↓
SAVE: torch.save(tensor, path)  → .pt file per clip
    ↓
SPLIT: 70/15/15 stratified by label
    → train / val / test
```

**Save alongside model.pt:**
```python
torch.save({
    'mean': train_mean,
    'std': train_std,
    'sample_rate': SAMPLE_RATE,
    'clip_length': CLIP_LENGTH_S,
    'n_mels': N_MELS,
    'n_fft': N_FFT,
    'hop_length': HOP_LENGTH,
}, 'checkpoints/preprocessing_params.pt')
```

---

## Integration Checklist

Before merging P1 and P2 work:

**P1 must complete:**
- [ ] explore.py has run, config.py updated with real values
- [ ] Labeling uses diagnosis_1-4 majority vote (NOT status_SSL)
- [ ] model_final.pt saved in checkpoints/
- [ ] preprocessing_params.pt saved alongside model_final.pt
- [ ] Test set AUC reported and above 0.75 minimum
- [ ] 10 demo clips selected from test set, saved to data/demo_clips/
- [ ] Real predict() implemented in api/inference.py
- [ ] predict() calls model.eval() and uses torch.no_grad()
- [ ] predict() applies torch.sigmoid() to raw logits
- [ ] predict() uses torch.load() with weights_only=True and map_location='cpu'
- [ ] predict() returns base64 spectrogram PNG (not raw tensor)

**P2 must complete:**
- [ ] Mock predict() working in api/inference.py
- [ ] FastAPI POST /api/predict endpoint working in api/main.py
- [ ] CORS configured for frontend dev server
- [ ] Next.js frontend: audio upload/record working
- [ ] Next.js frontend: result + confidence display working
- [ ] Next.js frontend: spectrogram image rendering from base64
- [ ] Medical disclaimer visible

**Integration (both):**
- [ ] Swap mock inference for real — restart API server
- [ ] End-to-end test on one healthy + one abnormal clip
- [ ] Demo clips load and classify in expected direction
- [ ] Spectrogram visualization renders without error

---

## Common Integration Bugs

1. **P1 hardcodes transform params** instead of loading from
   preprocessing_params.pt. If N_MELS changed after explore.py, inference
   silently produces wrong tensor shapes.

2. **P1 forgets `model.eval()`** — BatchNorm uses batch stats from a single
   sample and Dropout randomly zeros elements. Predictions are wrong and
   non-deterministic.

3. **P1 forgets `torch.sigmoid()`** — model outputs raw logits, not
   probabilities. Confidence values will be nonsensical.

4. **P2 forgets CORS** — Next.js dev server on port 3000 can't reach
   FastAPI on port 8000 without CORS middleware.

5. **P2 sends wrong Content-Type** — must be `multipart/form-data` for
   file upload, not `application/json`.

---

## Known Limitations (own these in Q&A)

- All COUGHVID recordings are voluntary intentional coughs
- Pathological signal still present in voluntary sick coughs — but boundary
  is cleaner than real-world spontaneous coughs
- Trained on ~2,200-2,450 expert-labeled samples — small by production ML standards
- Class imbalance: ~78% abnormal / ~22% healthy after expert filtering
- No external validation dataset — generalization to new devices unknown
- Binary only — does not distinguish COVID vs URTI vs LRTI vs other
- COUGHVID collected during COVID pandemic — label distribution reflects
  that specific epidemiological context
- Screening tool only — not a diagnostic

---

## What This Is Not

- Not a COVID detector
- Not a stethoscope replacement
- Not production medical software
- Not validated on spontaneous real-world coughs outside COUGHVID
