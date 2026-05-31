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
