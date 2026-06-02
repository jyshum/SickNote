# SickNote

**Binary cough classifier: healthy vs. abnormal**

SickNote is a screening tool that analyzes cough audio recordings to distinguish healthy coughs from potentially abnormal ones. It converts audio into mel spectrograms and classifies them using an ensemble of five convolutional neural networks, returning a prediction alongside a Grad-CAM heatmap showing which spectrogram regions influenced the decision.

> **Screening support only. Does not replace medical diagnosis.**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16 + React 19 + TypeScript + Tailwind CSS 4 + GSAP |
| Backend | FastAPI + uvicorn (Python) |
| ML | PyTorch + torchaudio |
| Audio | ffmpeg (handles .webm, .wav, .ogg, .mp3) |
| Deployment | Docker on Railway (API), Vercel (frontend) |

---

## Features

- **Dual audio input** — record via microphone or upload an audio file
- **Real-time recording** — live elapsed timer with pulsing indicator
- **Ensemble inference** — five independently trained CNNs, predictions averaged
- **Grad-CAM explainability** — heatmap overlay showing which spectrogram regions drove the prediction
- **Confidence display** — tiered (High / Moderate / Low) with visual bar
- **Prediction history** — carousel of up to 10 recent results per session
- **Technical deep-dive page** — full transparency on dataset, architecture, metrics, and limitations

---

## Architecture

```
┌──────────────────────────────┐
│     Browser                  │
│  (Next.js on Vercel)         │
└──────────┬───────────────────┘
           │ POST /api/predict
           │ multipart/form-data
           ▼
┌──────────────────────────────┐
│   Railway (Docker)           │
│  FastAPI                     │
│   ├─ Audio → Mel Spectrogram │
│   ├─ 5-model ensemble        │
│   ├─ Grad-CAM computation    │
│   └─ JSON + base64 PNG       │
└──────────────────────────────┘
```

The frontend and backend communicate through a single HTTP endpoint. Neither side imports the other's code.

### API Contract

```
POST /api/predict
Content-Type: multipart/form-data

Request:
  file: audio file (.webm, .wav, .ogg, .mp3)

Response:
{
    "label": "healthy" | "abnormal",
    "confidence": 0.87,
    "spectrogram": "data:image/png;base64,...",
    "gradcam": "data:image/png;base64,...",
    "ensemble_size": 5
}
```

---

## The Model

### Dataset

Trained on [COUGHVID](https://coughvid.epfl.ch/), a crowdsourced dataset of 34,400 cough recordings. After filtering to expert-labeled samples with cough detection confidence > 0.8 and acceptable quality ratings, approximately **2,300 samples** survive. Labels are assigned by majority vote across four physician annotations.

**Class distribution:** ~78% abnormal / ~22% healthy

### Architecture: SickNoteCNN

A small, purpose-built CNN trained from scratch:

```
Input (1, 64, T) mel spectrogram
  → Conv2d(8) + BatchNorm + ReLU + MaxPool
  → Conv2d(16) + BatchNorm + ReLU + MaxPool
  → Conv2d(32) + BatchNorm + ReLU + MaxPool
  → Flatten
  → Linear(128) + ReLU + Dropout(0.5)
  → Linear(1) → raw logit
```

Five models are trained independently with different random seeds. At inference, their sigmoid outputs are averaged to produce the final confidence score.

### Training Configuration

| Parameter | Value |
|---|---|
| Sample rate | 22,050 Hz |
| Clip length | 8 seconds (pad/trim) |
| Mel bins | 64 |
| Optimizer | Adam (lr=3e-4, wd=1e-4) |
| Loss | BCEWithLogitsLoss + pos_weight |
| Early stopping | Patience 15, on val AUC |
| Threshold | 0.52 (Youden's J optimized) |

### Evaluation (held-out test set)

| Metric | Result | Target |
|---|---|---|
| AUC-ROC | 0.73 | > 0.82 |
| Accuracy | 76% | > 78% |
| Sensitivity | 0.68 | > 0.75 |
| Specificity | 0.77 | > 0.70 |

---

## Key Design Decisions

**Binary classification only.** Multi-class (COVID / URTI / LRTI) dropped accuracy sharply. Binary is medically honest — "something sounds off" rather than "you have X."

**CNN from scratch, not transfer learning.** ~2,300 samples favors a small 3-layer CNN with aggressive dropout. ResNet18 was attempted but overfitted — pretrained ImageNet features were too general for narrow spectrogram patterns. See `TRANSFER_LEARNING_PLAN.md` for the full experiment log.

**Class weighting over oversampling.** `pos_weight` in the loss function rebalances without duplicating data or risking minority class overfitting.

**Ensemble + threshold tuning.** Five models with different seeds smooth variance on a small dataset. Classification threshold optimized from 0.50 to 0.52 via Youden's J statistic to balance sensitivity and specificity.

**Data augmentation removed.** Standard augmentation (noise injection, time stretching) amplified noise rather than adding signal on this dataset.

---

## Project Structure

```
sicknote/
├── frontend/                   # Next.js app (P2)
│   └── src/
│       ├── app/
│       │   ├── page.tsx        # Main analyzer UI
│       │   ├── technical/      # How It Works page
│       │   └── globals.css     # Design tokens
│       ├── components/         # AudioInput, RecordButton, ResultCard, Navbar
│       └── lib/                # API client, prediction history
│
├── api/                        # FastAPI backend
│   ├── main.py                 # POST /api/predict endpoint
│   ├── inference.py            # Ensemble prediction + Grad-CAM
│   └── audio_utils.py          # ffmpeg-based audio loading
│
├── model/                      # ML pipeline (P1)
│   ├── config.py               # Hyperparameters (verified by explore.py)
│   ├── architecture.py         # SickNoteCNN definition
│   ├── explore.py              # Dataset analysis (run first)
│   ├── preprocess.py           # Audio → spectrogram tensors
│   ├── dataset.py              # PyTorch Dataset
│   ├── train.py                # Training loop + early stopping
│   ├── evaluate.py             # Test set metrics + threshold tuning
│   └── checkpoints/            # Model weights + preprocessing params
│
├── data/                       # COUGHVID dataset + processed tensors
├── tests/                      # test_api.py, test_model.py
├── ARCHITECTURE.md             # Full specification + partner contract
├── Dockerfile                  # CPU-only PyTorch for Railway
└── railway.toml                # Deployment config
```

---

## Development

### Prerequisites

- Python 3.12+
- Node.js 20+
- ffmpeg installed (`brew install ffmpeg` on macOS)

### Backend

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend dev server runs on port 3000 and expects the API at `http://localhost:8000` (configurable via `NEXT_PUBLIC_API_URL`).

### ML Pipeline (training from scratch)

```bash
python -m model.explore      # Analyze dataset, verify config
python -m model.preprocess   # Audio → spectrogram tensors
python -m model.train        # Train 5-model ensemble
python -m model.evaluate     # Test set metrics + threshold
```

---

## Deployment

**Backend** is containerized with Docker and deployed on Railway. Uses CPU-only PyTorch to keep the image small. Models are lazy-loaded on the first prediction request to reduce startup memory.

**Frontend** is deployed on Vercel. Set `NEXT_PUBLIC_API_URL` to the Railway backend URL.

---

## Known Limitations

- All COUGHVID recordings are voluntary, intentional coughs
- Trained on ~2,300 expert-labeled samples — small by production standards
- ~78% abnormal / ~22% healthy class imbalance
- No external validation dataset — generalization to new devices/populations unknown
- Binary only — does not distinguish COVID vs. URTI vs. LRTI
- Collected during the COVID pandemic — label distribution reflects that epidemiological context
- Not a COVID detector, not a stethoscope replacement, not production medical software

---

## Team

Two-person project for a Hackathon:

- **Migael** — ML pipeline (`model/`, `data/`, real `api/inference.py`)
- **Jared** — Frontend + API routing (`frontend/`, `api/main.py`)

The handoff file `api/inference.py` is the only shared file, edited sequentially: Jared wrote the mock, Migael replaced the body. The function signature never changed. See `ARCHITECTURE.md` for the full partner contract.
