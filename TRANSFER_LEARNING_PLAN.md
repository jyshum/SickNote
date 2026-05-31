# Transfer Learning Plan — SickNoteCNN to ResNet18 Backbone

## Goal

Replace the from-scratch 3-layer CNN with a pretrained ResNet18 (ImageNet) feature extractor to improve AUC-ROC, accuracy, sensitivity, and specificity. The current model plateaus due to insufficient training data (~1,600 train samples) to learn discriminative spectrogram features from zero.

---

## Why ResNet18

- Ships with torchvision — no new dependencies, no external checkpoint downloads
- 11.7M parameters but most are frozen — effective trainable params will be similar to current model
- ImageNet features (edges, textures, spatial patterns) transfer well to spectrograms
- Small enough for fast training on MPS
- Well-understood architecture — easy to explain to hackathon judges

Do NOT use PANNs, AST, or other audio-specific pretrained models. They require specific spectrogram formats that would force full re-preprocessing of all data. ResNet18 accepts our existing spectrograms as-is.

---

## What Does NOT Change

These files and artifacts are untouched:

- `model/preprocess.py` — same audio-to-spectrogram pipeline
- `model/explore.py` — same dataset analysis
- `data/processed/*.pt` — existing spectrogram tensors stay as-is
- `model/checkpoints/splits.pt` — same train/val/test split
- `model/checkpoints/preprocessing_params.pt` — still used by inference for audio conversion
- `api/main.py` — same FastAPI endpoint
- `frontend/*` — completely untouched, API contract unchanged
- The `predict()` function signature and return format — identical

---

## What Changes

### 1. `model/architecture.py` — full rewrite

Replace `SickNoteCNN` with a new class that wraps `torchvision.models.resnet18(weights="IMAGENET1K_V1")`.

Two modifications to the stock ResNet18:

**Input adaptation:** ResNet18 expects 3-channel input. Our spectrograms are 1-channel `(1, 64, T)`. Replace `resnet.conv1` (which is `Conv2d(3, 64, 7, stride=2, padding=3)`) with `Conv2d(1, 64, 7, stride=2, padding=3)`. Initialize by averaging the pretrained conv1 weights across the 3 input channels — this preserves most of the pretrained knowledge in a single-channel filter.

**Output adaptation:** Replace `resnet.fc` (which is `Linear(512, 1000)` for ImageNet classes) with `Linear(512, 1)` for binary classification. Same convention as before: raw logits, no sigmoid in the model.

Keep the `forward()` method returning raw logits of shape `(batch, 1)` so `BCEWithLogitsLoss` and all downstream code works identically.

Expose the last residual block as an attribute (or provide a method to access it) so Grad-CAM can hook into it. The equivalent of the current `model.conv` for Grad-CAM purposes is `resnet.layer4`.

The constructor should NOT take `n_mels` and `time_frames` as arguments anymore — ResNet handles arbitrary spatial dimensions through adaptive average pooling before the FC layer. Remove `_get_flatten_dim`. The flatten dimension is always 512 for ResNet18.

### 2. `model/config.py` — add transfer learning hyperparameters

Add these new values alongside existing ones:

- `LR_BACKBONE` — learning rate for pretrained layers, should be ~10-20x lower than classifier LR (e.g., `1e-5`)
- `LR_HEAD` — learning rate for the new classifier head (e.g., `1e-3`)
- `FREEZE_EPOCHS` — number of epochs to train with backbone frozen before unfreezing (e.g., `5`)

Reduce `EPOCHS` — transfer learning converges much faster. 30-40 is likely sufficient (down from 80).

Reduce `EARLY_STOPPING_PATIENCE` accordingly — 8-10 instead of 15.

Keep `CONV_CHANNELS` and `KERNEL_SIZE` in the file for reference but they are no longer used.

### 3. `model/dataset.py` — one-line change in `__getitem__`

After loading the tensor `(1, 64, T)`, do NOT repeat to 3 channels. The modified `conv1` in the architecture accepts 1-channel input directly. No change needed here if the architecture handles 1-channel.

However, if for any reason the conv1 weight-averaging approach causes issues, the fallback is to `.repeat(3, 1, 1)` here instead. Try the conv1 approach first.

### 4. `model/train.py` — moderate changes

**Two-phase training with differential learning rates:**

**Phase 1 (frozen backbone):** For the first `FREEZE_EPOCHS` epochs, freeze all parameters except the classifier head (`resnet.fc`). Use `LR_HEAD` only. This lets the randomly-initialized head learn reasonable weights without sending destructive gradients back through the pretrained backbone.

**Phase 2 (full fine-tuning):** Unfreeze all parameters. Use two optimizer parameter groups:
- Backbone parameters (everything except `fc`) at `LR_BACKBONE`
- Classifier head (`fc`) at `LR_HEAD`

Both phases use the same `BCEWithLogitsLoss` with `pos_weight` (unchanged). Same `ReduceLROnPlateau` on `val_AUC`. Same early stopping logic. Same ensemble training with 5 seeds.

The checkpoint files change name from `model_final_0.pt` through `model_final_4.pt` — same naming convention, but the state dicts will be larger (~45 MB each instead of ~5.7 MB). Total ensemble disk usage goes from ~28 MB to ~225 MB. If this is a concern, drop to a 3-model ensemble — each model is stronger so the ensemble benefit has diminishing returns.

### 5. `model/evaluate.py` — minimal change

The only change: instantiate the new architecture class instead of `SickNoteCNN` in `_load_models()`. Remove the `n_mels` and `time_frames` arguments from the constructor call. Everything else — metric computation, Youden's J threshold tuning, confusion matrix, reporting — stays identical.

### 6. `api/inference.py` — two small changes

**Model loading:** In `_load_models()`, instantiate the new architecture class instead of `SickNoteCNN`. Remove `n_mels`/`time_frames` constructor args.

**Grad-CAM target:** In `_compute_gradcam()`, change the hook registration from `model.conv` to `model.layer4` (or whatever the last residual block is exposed as). This is a one-line change. The rest of the Grad-CAM computation (gradient weighting, ReLU, interpolation, overlay) is architecture-agnostic and stays identical.

---

## Training Strategy

### Recommended approach

1. Retrain with `--ensemble` flag, same 5 seeds
2. Run evaluate.py to compare metrics against the from-scratch baseline
3. If AUC improves (expected: 0.75-0.85 range), ship it
4. If AUC doesn't improve meaningfully, revert to the current checkpoints — the old model files still work

### Rollback plan

Keep the current `model_final_*.pt` checkpoints backed up. If transfer learning produces worse results (unlikely but possible with very small datasets), restore the originals. The rest of the pipeline is backwards-compatible — just swap the architecture class back.

---

## Hackathon Narrative

"We built a CNN from scratch to validate our full pipeline — preprocessing, training, evaluation, and inference. Once we confirmed the pipeline worked end-to-end, we applied transfer learning with a pretrained ResNet18 backbone to overcome our small dataset limitation (~2,300 expert-labeled samples). This improved our AUC from X to Y. We froze the pretrained feature extractor and only trained a lightweight classifier head on our cough data, then fine-tuned the full network with a lower learning rate."

This demonstrates: pipeline-first engineering, understanding of data limitations, and knowledge of when and how to apply transfer learning. Stronger than either approach alone.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Worse metrics than from-scratch | Low | Keep backup of current checkpoints |
| Overfitting with powerful backbone | Medium | Freeze phases, high dropout on head, early stopping |
| Grad-CAM breaks with new architecture | Low | Just change hook target to `layer4` |
| Larger model slows inference | Low | ResNet18 forward pass is still <100ms on MPS |
| Breaking the API contract | None | `predict()` signature and return format are unchanged |
| Frontend impact | None | Frontend has zero awareness of model architecture |
