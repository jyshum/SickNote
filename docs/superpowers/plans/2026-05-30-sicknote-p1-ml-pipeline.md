# SickNote P1 ML Pipeline — Implementation Plan

> **Spec:** `ARCHITECTURE.md` is the source of truth for all code, signatures, and architectural decisions. This plan covers execution order, decision gates, and verification — not implementation details.

**Goal:** Implement the P1 ML pipeline from raw COUGHVID audio to a trained binary cough classifier with real inference.

**Device:** M4 Mac, 16GB RAM, MPS backend. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` before any PyTorch step.

**Prerequisite:** `ffmpeg` installed (`brew install ffmpeg`) — torchaudio needs it for .webm files.

---

## Task 1: explore.py — Understand the Data

**Why first:** Every config value marked `*` is a guess until this runs. Nothing else can be finalized.

- [ ] Implement `majority_label()` and `has_good_quality()` with tests (see ARCHITECTURE.md § P1 Step 0)
- [ ] Implement `main()` answering all 6 dataset questions
- [ ] Run `python -m model.explore`, capture output
- [ ] Review spectrogram plot — are healthy vs abnormal visually distinguishable?

**Decision gate:** Record these values before proceeding:
- Surviving sample count → determines batch size and conv block count
- Class distribution → determines pos_weight
- Median duration → determines CLIP_LENGTH_S
- Tensor shape → determines flatten dimension
- Corrupted file count → determines error handling needs

**Not in ARCHITECTURE.md:** Need a `find_audio_file(uuid)` helper since COUGHVID files have mixed extensions (.webm/.wav/.ogg). Also: ties in `majority_label` default to abnormal (cautious).

**Verify:** All 6 questions printed with concrete numbers. Spectrogram image saved.

---

## Task 2: config.py — Lock In Real Values

**Why now:** Blocks all downstream code — every module imports from config.

- [ ] Update every `*`-marked value using the decision table in ARCHITECTURE.md § Config
- [ ] Verify with a quick import: `python -c "from model.config import *; print(...)"`

**Decision gate:** If sample count < 1500, drop to 2 conv blocks and BATCH_SIZE=16. This changes architecture.py scope.

**Verify:** All values reflect real data, not estimates.

---

## Task 3: preprocess.py — Build the Tensor Dataset

**Why now:** Training needs .pt files and normalization stats.

- [ ] Implement `load_and_filter_metadata()` with tests — reuses helpers from explore.py
- [ ] Implement `audio_to_spectrogram()` with tests (synthetic wav files)
- [ ] Implement `main()` — full pipeline: convert → split → normalize → save
- [ ] Run `python -m model.preprocess`

**Critical rule:** Normalization mean/std computed on TRAIN SPLIT ONLY. This is the #1 data leakage risk.

**Verify:**
- `.pt` files in `data/processed/`
- `splits.pt` with correct train/val/test counts (70/15/15)
- `preprocessing_params.pt` with mean, std, and time_frames

---

## Task 4: dataset.py + architecture.py — Model Components

**Why together:** Both are small, independent, and needed before training.

- [ ] Implement `CoughDataset` with tests — thin wrapper, loads .pt + applies normalization
- [ ] Implement `SickNoteCNN` with tests — verify output shape (batch, 1) with dummy tensor
- [ ] Use time_frames from `preprocessing_params.pt`, never hardcode flatten dim

**Verify:** Forward pass with dummy input returns correct shape. DataLoader produces valid batches.

---

## Task 5: train.py — Train the Model

**Why now:** All dependencies ready.

- [ ] Implement training loop per ARCHITECTURE.md § P1 Step 3
- [ ] Run `python -m model.train`
- [ ] Watch for overfitting (val loss rising while train loss falls)

**Key details:** Checkpoint on val_AUC (not val_loss). BCEWithLogitsLoss with pos_weight. num_workers=0 for MPS. ReduceLROnPlateau monitoring val_AUC.

**Decision gate:** If val_AUC plateaus below 0.70, check the 4-item debugging list in ARCHITECTURE.md § P1 Step 4 before re-training.

**Verify:** `model_final.pt` exists. Best val_AUC printed.

---

## Task 6: evaluate.py — Validate on Test Set

- [ ] Implement per ARCHITECTURE.md § P1 Step 4
- [ ] Run `python -m model.evaluate`

**Minimum thresholds:** AUC > 0.75, Accuracy > 70%, Sensitivity > 0.65, Specificity > 0.60

**Decision gate:** If below minimums, do NOT proceed to inference. Debug using the checklist in ARCHITECTURE.md.

**Verify:** All 4 metrics printed with confusion matrix.

---

## Task 7: inference.py — Replace the Mock

- [ ] Replace mock `predict()` body with real model loading + inference
- [ ] Must include: `model.eval()`, `torch.no_grad()`, `torch.sigmoid()`, `weights_only=True`, `map_location='cpu'`
- [ ] Load preprocessing params from file (never hardcode)
- [ ] Generate base64 spectrogram PNG for visualization
- [ ] Cache model at module level (don't reload per request)
- [ ] Test with a demo clip

**Contract:** Function signature `predict(audio_path: str) -> dict` does NOT change. P2's code must work without modification.

**Verify:** `predict()` returns valid `{label, confidence, spectrogram}` on a real audio file.

---

## Task 8: Demo Clips

- [ ] Select 10 clips from test set (5 healthy, 5 abnormal) with confident correct predictions
- [ ] Copy audio files to `data/demo_clips/`

**Verify:** 10 files in `data/demo_clips/`. Each classifies correctly when passed through `predict()`.

---

## Final Integration Checklist

From ARCHITECTURE.md § Integration Checklist — all P1 items must pass:

- [ ] explore.py ran, config.py updated
- [ ] Labeling uses diagnosis_1-4 majority vote (NOT status_SSL)
- [ ] model_final.pt + preprocessing_params.pt saved
- [ ] Test set AUC > 0.75
- [ ] 10 demo clips in data/demo_clips/
- [ ] Real predict() in api/inference.py
- [ ] predict() calls model.eval() + torch.no_grad()
- [ ] predict() applies torch.sigmoid() to logits
- [ ] predict() uses weights_only=True + map_location='cpu'
- [ ] predict() returns base64 PNG spectrogram
