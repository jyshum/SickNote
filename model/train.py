"""
Training loop with early stopping on val_AUC.

See ARCHITECTURE.md → P1 Step 3 for requirements.

Usage: PYTORCH_ENABLE_MPS_FALLBACK=1 python -m model.train
"""
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model.config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, CHECKPOINT_DIR, DEVICE,
)
from model.dataset import CoughDataset
from model.architecture import SickNoteCNN


def main():
    """Train SickNoteCNN with BCEWithLogitsLoss + pos_weight.

    - Checkpoint on best val_AUC (not val_loss)
    - Early stopping after EARLY_STOPPING_PATIENCE epochs
    - ReduceLROnPlateau scheduler monitoring val_AUC
    - Save model_best.pt → model_final.pt when done
    - Use num_workers=0 in DataLoader (MPS compatibility)
    """
    torch.manual_seed(42)
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
    n_mels = params["n_mels"]
    time_frames = params["time_frames"]

    print(f"Train: {len(train_files)} samples | Val: {len(val_files)} samples")
    print(f"n_mels={n_mels}, time_frames={time_frames}, mean={mean:.2f}, std={std:.2f}")

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
    n_pos = sum(train_labels)           # abnormal = 1
    n_neg = len(train_labels) - n_pos   # healthy = 0
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    print(f"Class balance — healthy(0): {n_neg}, abnormal(1): {n_pos}, pos_weight: {pos_weight.item():.4f}")

    # ------------------------------------------------------------------
    # 4. Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    model = SickNoteCNN(n_mels=n_mels, time_frames=time_frames).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    best_val_auc = 0.0
    patience_counter = 0
    best_path = os.path.join(CHECKPOINT_DIR, "model_best.pt")
    final_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        for spectrograms, labels in train_loader:
            spectrograms = spectrograms.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)  # (batch,) → (batch, 1)

            optimizer.zero_grad()
            logits = model(spectrograms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * spectrograms.size(0)

        train_loss = running_loss / len(train_ds)

        # ---- Validate ----
        model.eval()
        val_running_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms = spectrograms.to(DEVICE)
                labels_dev = labels.unsqueeze(1).to(DEVICE)

                logits = model(spectrograms)
                loss = criterion(logits, labels_dev)
                val_running_loss += loss.item() * spectrograms.size(0)

                probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
                truth = labels.tolist()

                # Handle single-sample batch: .tolist() returns float, not list
                if isinstance(probs, float):
                    probs = [probs]
                if isinstance(truth, float):
                    truth = [truth]

                all_probs.extend(probs)
                all_labels.extend(truth)

        val_loss = val_running_loss / len(val_ds)
        val_auc = roc_auc_score(all_labels, all_probs)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_auc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
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
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement")
                break

    # ------------------------------------------------------------------
    # 6. Copy model_best.pt → model_final.pt
    # ------------------------------------------------------------------
    shutil.copy2(best_path, final_path)
    print(f"\nTraining complete. Best val_AUC: {best_val_auc:.4f}")
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
