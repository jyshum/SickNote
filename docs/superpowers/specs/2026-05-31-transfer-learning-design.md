# Transfer Learning Design — ResNet18 Backbone for SickNote

## Goal

Replace the from-scratch SickNoteCNN with a pretrained ResNet18 backbone to improve AUC-ROC beyond the current 0.7363 ceiling. The small dataset (~1,600 train samples) is insufficient for the custom CNN to learn strong spectrogram features from zero.

## Approach

Use `torchvision.models.resnet18(weights="IMAGENET1K_V1")` with two modifications:
- Adapt input from 3-channel to 1-channel (our spectrograms)
- Replace 1000-class output with binary logit

Two-phase training: frozen backbone → full fine-tuning with differential learning rates.

## Changes

### 1. `model/architecture.py` — full rewrite

New class `SickNoteResNet`:

- Loads `resnet18(weights="IMAGENET1K_V1")`
- Replaces `conv1`: new `Conv2d(1, 64, 7, stride=2, padding=3, bias=False)` initialized by averaging the pretrained 3-channel weights across the channel dimension
- Replaces `fc`: new `Linear(512, 1)` for binary classification
- Exposes `self.layer4` as the Grad-CAM target (replaces `model.conv`)
- No `n_mels`/`time_frames` constructor args — ResNet's `AdaptiveAvgPool2d` handles arbitrary spatial dimensions
- Provides `backbone_params()` and `head_params()` methods returning parameter iterators for differential LR in training
- `forward()` returns raw logits of shape `(batch, 1)` — same convention as before

### 2. `model/config.py` — add transfer learning hyperparams

New values:
- `LR_BACKBONE = 1e-5` — learning rate for pretrained layers
- `LR_HEAD = 1e-3` — learning rate for classifier head
- `FREEZE_EPOCHS = 5` — epochs with backbone frozen before unfreezing

Modified values:
- `EPOCHS`: 80 → 40 (transfer learning converges faster)
- `EARLY_STOPPING_PATIENCE`: 15 → 10
- `ENSEMBLE_SEEDS`: `[42, 123, 456, 789, 1024]` → `[42, 123, 456]` (3 models — stronger per-model, diminishing ensemble returns, ~135MB total vs ~225MB)

Unchanged: `CONV_CHANNELS` and `KERNEL_SIZE` remain for reference but are no longer used.

### 3. `model/train.py` — two-phase training

**Phase 1 (epochs 1 through FREEZE_EPOCHS):**
- Freeze all backbone parameters (everything except `fc`)
- Single optimizer with only head parameters at `LR_HEAD`
- Lets the randomly-initialized head learn reasonable weights without destructive gradients through the backbone

**Phase 2 (epochs FREEZE_EPOCHS+1 through EPOCHS):**
- Unfreeze all parameters
- New optimizer with two parameter groups:
  - Backbone params at `LR_BACKBONE`
  - Head params at `LR_HEAD`
- New scheduler on the new optimizer

Both phases: same `BCEWithLogitsLoss` with `pos_weight`, same `ReduceLROnPlateau` on `val_AUC`, same early stopping logic, same checkpoint saving.

Ensemble: 3 models with seeds `[42, 123, 456]`.

### 4. `model/evaluate.py` — minimal change

In `_load_models()`:
- Import `SickNoteResNet` instead of `SickNoteCNN`
- Remove `n_mels`/`time_frames` constructor args
- Ensemble loop changes from `range(5)` to `range(3)`

Everything else unchanged: metric computation, Youden's J threshold tuning, confusion matrix, reporting.

### 5. `api/inference.py` — two changes

**Model loading (`_load_models()`):**
- Import `SickNoteResNet` instead of `SickNoteCNN`
- Remove `n_mels`/`time_frames` constructor args
- Ensemble loop changes from `range(5)` to `range(3)`

**Grad-CAM (`_compute_gradcam()`):**
- Change hook registration from `model.conv` to `model.layer4`
- Rest of Grad-CAM computation is architecture-agnostic

### 6. `model/select_demos.py` — minimal change

Same pattern as evaluate.py: swap class, remove constructor args, loop `range(3)`.

## What Does NOT Change

- `model/preprocess.py` — same audio-to-spectrogram pipeline
- `model/explore.py` — same dataset analysis
- `model/dataset.py` — same dataset class, 1-channel tensors work directly
- `data/processed/*.pt` — existing spectrogram tensors stay as-is
- `model/checkpoints/splits.pt` — same train/val/test split
- `model/checkpoints/preprocessing_params.pt` — still used by inference
- `api/main.py` — same FastAPI endpoint
- `frontend/*` — completely untouched
- `predict()` function signature and return format — identical

## Rollback Plan

Keep current `model_final_*.pt` checkpoints backed up before retraining. If ResNet18 produces worse results, restore originals and revert `architecture.py` to `SickNoteCNN`. The pipeline is backwards-compatible.

## Expected Outcome

AUC improvement from ~0.74 to 0.75-0.85 range. Checkpoint size increases from ~5.7MB to ~45MB per model (~135MB total for 3-model ensemble).
