"""
Training loop with early stopping on val_AUC.

See ARCHITECTURE.md → P1 Step 3 for requirements.

Usage: python -m model.train
"""


def main():
    """Train SickNoteCNN with BCEWithLogitsLoss + pos_weight.

    - Checkpoint on best val_AUC (not val_loss)
    - Early stopping after EARLY_STOPPING_PATIENCE epochs
    - ReduceLROnPlateau scheduler monitoring val_AUC
    - Save model_best.pt → model_final.pt when done
    - Use num_workers=0 in DataLoader (MPS compatibility)
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
