# Transfer Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the from-scratch SickNoteCNN with a pretrained ResNet18 backbone to improve AUC-ROC beyond the current 0.7363 ceiling.

**Architecture:** Wrap `torchvision.models.resnet18(weights="IMAGENET1K_V1")` with a 1-channel input adapter and binary classifier head. Two-phase training: frozen backbone for 5 epochs, then full fine-tuning with differential learning rates. 3-model ensemble.

**Tech Stack:** PyTorch, torchvision (already installed)

---

### Task 1: Backup current checkpoints

**Files:**
- Read: `model/checkpoints/model_final_*.pt`

- [ ] **Step 1: Copy current checkpoints to a backup directory**

```bash
mkdir -p model/checkpoints/backup_cnn
cp model/checkpoints/model_final.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_final_0.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_final_1.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_final_2.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_final_3.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_final_4.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_best.pt model/checkpoints/backup_cnn/ 2>/dev/null || true
cp model/checkpoints/model_best_0.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_best_1.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_best_2.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_best_3.pt model/checkpoints/backup_cnn/
cp model/checkpoints/model_best_4.pt model/checkpoints/backup_cnn/
```

- [ ] **Step 2: Verify backup**

```bash
ls -la model/checkpoints/backup_cnn/
```

Expected: All `model_final_*.pt` and `model_best_*.pt` files present.

---

### Task 2: Rewrite `model/architecture.py`

**Files:**
- Modify: `model/architecture.py` (full rewrite)
- Modify: `tests/test_model.py:392-454` (update architecture tests)

- [ ] **Step 1: Update the test class for the new architecture**

Replace the `TestSickNoteCNN` class in `tests/test_model.py` (lines 392-454) with:

```python
# ── SickNoteResNet tests ─────────────────────────────────────────────


class TestSickNoteResNet:
    """Test SickNoteResNet: output shape, various input sizes, Grad-CAM target."""

    def test_output_shape_batch_4(self):
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        x = torch.randn(4, 1, 64, 173)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_shape_batch_1(self):
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        x = torch.randn(1, 1, 64, 173)
        out = model(x)
        assert out.shape == (1, 1)

    def test_output_shape_batch_16(self):
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        x = torch.randn(16, 1, 64, 173)
        out = model(x)
        assert out.shape == (16, 1)

    def test_different_input_dimensions(self):
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        x = torch.randn(2, 1, 128, 87)
        out = model(x)
        assert out.shape == (2, 1)

    def test_outputs_raw_logits(self):
        """Model should output raw logits, not probabilities (no sigmoid)."""
        import torch
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        model.eval()
        x = torch.randn(4, 1, 64, 173)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_has_layer4_for_gradcam(self):
        """model.layer4 must exist for Grad-CAM hooks."""
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        assert hasattr(model, "layer4")

    def test_backbone_and_head_params(self):
        """backbone_params() and head_params() must return non-empty iterators."""
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        backbone = list(model.backbone_params())
        head = list(model.head_params())
        assert len(backbone) > 0
        assert len(head) > 0
        # head should be much smaller than backbone
        assert len(head) < len(backbone)

    def test_conv1_is_single_channel(self):
        """conv1 must accept 1-channel input, not 3-channel."""
        from model.architecture import SickNoteResNet

        model = SickNoteResNet()
        assert model.conv1.in_channels == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_model.py::TestSickNoteResNet -v`
Expected: FAIL — `SickNoteResNet` not defined.

- [ ] **Step 3: Rewrite architecture.py**

Replace the entire contents of `model/architecture.py` with:

```python
"""
SickNoteResNet — ResNet18 backbone for binary cough classification.

Model outputs raw logits (no Sigmoid). Use BCEWithLogitsLoss for training.
Apply torch.sigmoid() at inference time only.

Pretrained on ImageNet. Input adapted from 3-channel to 1-channel.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class SickNoteResNet(nn.Module):
    """ResNet18 with 1-channel input and binary output.

    conv1 weights initialized by averaging pretrained 3-channel weights.
    fc replaced with Linear(512, 1) for binary classification.
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")

        # Adapt conv1: 3-channel → 1-channel
        old_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Replace classifier: 1000-class → binary
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        """Returns raw logits, shape (batch_size, 1)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def backbone_params(self):
        """All parameters except the classifier head."""
        for name, param in self.named_parameters():
            if not name.startswith("fc."):
                yield param

    def head_params(self):
        """Classifier head parameters only."""
        return self.fc.parameters()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model.py::TestSickNoteResNet -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add model/architecture.py tests/test_model.py
git commit -m "feat: replace SickNoteCNN with ResNet18 backbone

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Update `model/config.py`

**Files:**
- Modify: `model/config.py`

- [ ] **Step 1: Update config with transfer learning hyperparams**

Replace the full contents of `model/config.py` with:

```python
"""
All hyperparameters — verified by explore.py on 2026-05-30.
See ARCHITECTURE.md → Config section for full documentation.
"""
import torch

# Audio
SAMPLE_RATE = 22050
CLIP_LENGTH_S = 8
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# Training
BATCH_SIZE = 32
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
EARLY_STOPPING_PATIENCE = 10

# Transfer learning
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3
FREEZE_EPOCHS = 5

# Model (legacy — not used by SickNoteResNet)
CONV_CHANNELS = [8, 16, 32]
KERNEL_SIZE = 3

# Evaluation
THRESHOLD = 0.52

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Paths
RAW_DIR = "data/raw/coughvid_20211012"
PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "model/checkpoints"
DEMO_CLIPS_DIR = "data/demo_clips"
METADATA_CSV = "data/raw/coughvid_20211012/metadata_compiled.csv"
```

- [ ] **Step 2: Commit**

```bash
git add model/config.py
git commit -m "feat: add transfer learning hyperparams to config

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Rewrite `model/train.py`

**Files:**
- Modify: `model/train.py`

- [ ] **Step 1: Rewrite train.py with two-phase training**

Replace the full contents of `model/train.py` with:

```python
"""
Training loop with early stopping on val_AUC.

Two-phase transfer learning:
  Phase 1 (frozen backbone): train classifier head only
  Phase 2 (fine-tuning): unfreeze all, differential learning rates

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.train
"""
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model.config import (
    BATCH_SIZE, EPOCHS, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, CHECKPOINT_DIR, DEVICE,
    LR_BACKBONE, LR_HEAD, FREEZE_EPOCHS,
)
from model.dataset import CoughDataset
from model.architecture import SickNoteResNet


ENSEMBLE_SEEDS = [42, 123, 456]


def _validate(model, val_loader, criterion, val_ds, device):
    """Run validation pass, return (val_loss, val_auc)."""
    model.eval()
    val_running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = spectrograms.to(device)
            labels_dev = labels.unsqueeze(1).to(device)

            logits = model(spectrograms)
            loss = criterion(logits, labels_dev)
            val_running_loss += loss.item() * spectrograms.size(0)

            probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
            truth = labels.tolist()

            if isinstance(probs, float):
                probs = [probs]
            if isinstance(truth, float):
                truth = [truth]

            all_probs.extend(probs)
            all_labels.extend(truth)

    val_loss = val_running_loss / len(val_ds)
    val_auc = roc_auc_score(all_labels, all_probs)
    return val_loss, val_auc


def train_single(seed=42, suffix=""):
    """Train one SickNoteResNet model with a given seed.

    Phase 1: frozen backbone, train head only at LR_HEAD.
    Phase 2: unfreeze all, backbone at LR_BACKBONE, head at LR_HEAD.
    """
    torch.manual_seed(seed)
    print(f"\n{'='*70}")
    print(f"Training with seed={seed}{f' (ensemble member {suffix})' if suffix else ''}")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")

    # ------------------------------------------------------------------
    # 1. Load splits and preprocessing params
    # ------------------------------------------------------------------
    splits = torch.load(
        os.path.join(CHECKPOINT_DIR, "splits.pt"), weights_only=False
    )
    params = torch.load(
        os.path.join(CHECKPOINT_DIR, "preprocessing_params.pt"), weights_only=True
    )

    train_files = splits["train_files"]
    train_labels = splits["train_labels"]
    val_files = splits["val_files"]
    val_labels = splits["val_labels"]

    mean = params["mean"]
    std = params["std"]

    print(f"Train: {len(train_files)} samples | Val: {len(val_files)} samples")

    # ------------------------------------------------------------------
    # 2. Create datasets and dataloaders
    # ------------------------------------------------------------------
    train_ds = CoughDataset(train_files, train_labels, mean=mean, std=std)
    val_ds = CoughDataset(val_files, val_labels, mean=mean, std=std)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------
    # 3. Compute pos_weight for class imbalance
    # ------------------------------------------------------------------
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"Class balance — healthy(0): {n_neg}, abnormal(1): {n_pos}, pos_weight: {pos_weight.item():.4f}")

    # ------------------------------------------------------------------
    # 4. Model, loss
    # ------------------------------------------------------------------
    model = SickNoteResNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_head = sum(p.numel() for p in model.head_params())
    print(f"Total parameters: {total_params:,}")
    print(f"Head parameters: {trainable_head:,}")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_val_auc = 0.0
    patience_counter = 0
    best_path = os.path.join(CHECKPOINT_DIR, f"model_best{suffix}.pt")
    final_path = os.path.join(CHECKPOINT_DIR, f"model_final{suffix}.pt")

    # Phase 1: freeze backbone
    for param in model.backbone_params():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model.head_params(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    phase = 1
    print(f"\n--- Phase 1: Frozen backbone (epochs 1-{FREEZE_EPOCHS}) ---")

    for epoch in range(1, EPOCHS + 1):
        # Switch to phase 2 after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            phase = 2
            print(f"\n--- Phase 2: Fine-tuning all layers (epochs {epoch}-{EPOCHS}) ---")
            for param in model.backbone_params():
                param.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": model.backbone_params(), "lr": LR_BACKBONE},
                {"params": model.head_params(), "lr": LR_HEAD},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=3, factor=0.5
            )
            patience_counter = 0

        # ---- Train ----
        model.train()
        running_loss = 0.0
        for spectrograms, labels in train_loader:
            spectrograms = spectrograms.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            logits = model(spectrograms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spectrograms.size(0)

        train_loss = running_loss / len(train_ds)

        # ---- Validate ----
        val_loss, val_auc = _validate(model, val_loader, criterion, val_ds, DEVICE)
        current_lr = optimizer.param_groups[-1]["lr"]

        scheduler.step(val_auc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} [P{phase}] | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.1e}"
        )

        # ---- Checkpoint on best val_AUC ----
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best val_AUC — checkpoint saved")
        else:
            patience_counter += 1
            if phase == 2 and patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement")
                break

    # ------------------------------------------------------------------
    # 6. Copy model_best.pt → model_final.pt
    # ------------------------------------------------------------------
    shutil.copy2(best_path, final_path)
    print(f"\nTraining complete. Best val_AUC: {best_val_auc:.4f}")
    print(f"Saved: {final_path}")
    return best_val_auc


def main():
    """Train a single model or an ensemble of 3 models.

    Usage:
        python -m model.train              # single model (seed=42)
        python -m model.train --ensemble   # 3 models with different seeds
    """
    import sys

    if "--ensemble" in sys.argv:
        print("Training ensemble of 3 models...")
        aucs = []
        for i, seed in enumerate(ENSEMBLE_SEEDS):
            auc = train_single(seed=seed, suffix=f"_{i}")
            aucs.append(auc)
        print(f"\n{'='*70}")
        print(f"ENSEMBLE COMPLETE")
        print(f"{'='*70}")
        for i, auc in enumerate(aucs):
            print(f"  Model {i} (seed={ENSEMBLE_SEEDS[i]}): val_AUC={auc:.4f}")
        print(f"  Mean val_AUC: {sum(aucs)/len(aucs):.4f}")
        print(f"  Saved: model_final_0.pt through model_final_2.pt")
    else:
        train_single(seed=42)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add model/train.py
git commit -m "feat: two-phase transfer learning training loop

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Update `model/evaluate.py`

**Files:**
- Modify: `model/evaluate.py:18,27-28,31-34,40-42`

- [ ] **Step 1: Update evaluate.py to use SickNoteResNet**

Three changes in `model/evaluate.py`:

1. Change the import (line 18):
```python
# OLD:
from model.architecture import SickNoteCNN
# NEW:
from model.architecture import SickNoteResNet
```

2. In `_load_models()`, remove `n_mels`/`time_frames` usage and change ensemble range. Replace the entire `_load_models` function body with:
```python
def _load_models():
    """Load ensemble models (or single model_final.pt as fallback)."""
    params = torch.load(
        os.path.join(CHECKPOINT_DIR, "preprocessing_params.pt"),
        weights_only=True,
    )

    models = []
    for i in range(3):
        path = os.path.join(CHECKPOINT_DIR, f"model_final_{i}.pt")
        if os.path.exists(path):
            model = SickNoteResNet()
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            model.to(DEVICE)
            model.eval()
            models.append(model)

    if not models:
        single_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
        model = SickNoteResNet()
        model.load_state_dict(torch.load(single_path, weights_only=True, map_location="cpu"))
        model.to(DEVICE)
        model.eval()
        models.append(model)

    return models, params
```

- [ ] **Step 2: Commit**

```bash
git add model/evaluate.py
git commit -m "feat: update evaluate.py for ResNet18 architecture

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Update `api/inference.py`

**Files:**
- Modify: `api/inference.py:43,49-59,62-67,136-137`

- [ ] **Step 1: Update model loading in _load_models()**

In `api/inference.py`, in the `_load_models()` function:

1. Change the import (line 43):
```python
# OLD:
from model.architecture import SickNoteCNN
# NEW:
from model.architecture import SickNoteResNet
```

2. Remove `n_mels`/`time_frames` variables (delete lines that read them from params).

3. Change ensemble loop from `range(5)` to `range(3)`.

4. Replace all `SickNoteCNN(n_mels=n_mels, time_frames=time_frames)` with `SickNoteResNet()`.

The full replacement for `_load_models()`:
```python
def _load_models():
    """Load ensemble models (or single model_final.pt as fallback).

    Checks for model_final_0.pt through model_final_2.pt first (ensemble).
    Falls back to model_final.pt if no ensemble files exist.
    Caches result in module-level globals so it only runs once.
    """
    global _models, _params

    if _models is not None:
        return _models, _params

    from model.architecture import SickNoteResNet

    params_path = os.path.join(_CHECKPOINT_DIR, "preprocessing_params.pt")
    params = torch.load(params_path, weights_only=True, map_location="cpu")

    models = []
    for i in range(3):
        path = os.path.join(_CHECKPOINT_DIR, f"model_final_{i}.pt")
        if os.path.exists(path):
            model = SickNoteResNet()
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            model.eval()
            models.append(model)

    if not models:
        single_path = os.path.join(_CHECKPOINT_DIR, "model_final.pt")
        model = SickNoteResNet()
        model.load_state_dict(torch.load(single_path, weights_only=True, map_location="cpu"))
        model.eval()
        models.append(model)

    _models = models
    _params = params
    return _models, _params
```

- [ ] **Step 2: Update Grad-CAM hook target**

In `_compute_gradcam()` function (around line 136-137), change the hook registrations:

```python
# OLD:
handle_fwd = model.conv.register_forward_hook(fwd_hook)
handle_bwd = model.conv.register_full_backward_hook(bwd_hook)
# NEW:
handle_fwd = model.layer4.register_forward_hook(fwd_hook)
handle_bwd = model.layer4.register_full_backward_hook(bwd_hook)
```

- [ ] **Step 3: Commit**

```bash
git add api/inference.py
git commit -m "feat: update inference for ResNet18 and Grad-CAM layer4

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Update `model/select_demos.py`

**Files:**
- Modify: `model/select_demos.py`

- [ ] **Step 1: Update select_demos.py to use SickNoteResNet**

Three changes:

1. Change the import (line 16):
```python
# OLD:
from model.architecture import SickNoteCNN
# NEW:
from model.architecture import SickNoteResNet
```

2. Replace the model loading block (the `models = []` loop and fallback) with the same pattern as evaluate.py — `range(3)`, `SickNoteResNet()`, no constructor args. Remove `n_mels` and `time_frames` variable reads (but keep `mean`, `std` which are still used for the dataset).

Full replacement for the model loading section:
```python
    models = []
    for i in range(3):
        path = os.path.join(CHECKPOINT_DIR, f"model_final_{i}.pt")
        if os.path.exists(path):
            m = SickNoteResNet()
            m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            m.to(DEVICE)
            m.eval()
            models.append(m)

    if not models:
        single_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
        m = SickNoteResNet()
        m.load_state_dict(torch.load(single_path, weights_only=True, map_location="cpu"))
        m.to(DEVICE)
        m.eval()
        models.append(m)

    print(f"Loaded {len(models)} model(s)")
```

- [ ] **Step 2: Commit**

```bash
git add model/select_demos.py
git commit -m "feat: update select_demos.py for ResNet18

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Train ensemble and evaluate

**Files:**
- Read: `model/checkpoints/model_final_*.pt` (generated)

- [ ] **Step 1: Run all tests before training**

```bash
python3 -m pytest tests/test_model.py -v
```

Expected: All tests pass (the `TestSickNoteResNet` tests replace the old `TestSickNoteCNN` tests).

- [ ] **Step 2: Train 3-model ensemble**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m model.train --ensemble
```

Expected: 3 models train with two-phase approach. Each should show Phase 1 (frozen) then Phase 2 (fine-tuning). Checkpoints saved as `model_final_0.pt` through `model_final_2.pt`.

- [ ] **Step 3: Evaluate on test set**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m model.evaluate
```

Expected: AUC-ROC improved from 0.7363 baseline. If AUC is worse, restore backups from `model/checkpoints/backup_cnn/` and revert architecture changes.

- [ ] **Step 4: Commit checkpoints and push**

```bash
git add model/checkpoints/model_final_0.pt model/checkpoints/model_final_1.pt model/checkpoints/model_final_2.pt
git add model/checkpoints/model_best_0.pt model/checkpoints/model_best_1.pt model/checkpoints/model_best_2.pt
git commit -m "feat: trained ResNet18 ensemble checkpoints

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push
```

---

### Task 9: Clean up old checkpoints

**Files:**
- Delete: `model/checkpoints/model_final_3.pt`, `model/checkpoints/model_final_4.pt`
- Delete: `model/checkpoints/model_best_3.pt`, `model/checkpoints/model_best_4.pt`
- Delete: `model/checkpoints/backup_cnn/` (after confirming new model is better)

- [ ] **Step 1: Remove old ensemble members 3 and 4 from git**

Only do this if evaluate.py showed improved metrics. Otherwise skip this task and restore from backup.

```bash
git rm model/checkpoints/model_final_3.pt model/checkpoints/model_final_4.pt
git rm model/checkpoints/model_best_3.pt model/checkpoints/model_best_4.pt
```

- [ ] **Step 2: Remove old single model if present**

```bash
git rm model/checkpoints/model_final.pt model/checkpoints/model_best.pt 2>/dev/null || true
```

- [ ] **Step 3: Commit and push**

```bash
git commit -m "chore: remove old CNN checkpoints (replaced by ResNet18)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push
```
