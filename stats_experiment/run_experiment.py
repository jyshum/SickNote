import os, sys, time, csv, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from model.architecture import SickNoteCNN
from model.dataset import CoughDataset

from stats_experiment.config import (
    TRAINING_SIZES, SEEDS, TEST_SPLIT_SEED, TEST_SPLIT_RATIO,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
    N_MELS, TIME_FRAMES, RESULTS_DIR, RESULTS_CSV, GRAPHS_DIR,
)


def seed_everything(seed):
    """Seed all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Seeds CPU + MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_all_data():
    """Load all preprocessed spectrograms and labels from splits.pt."""
    splits = torch.load("model/checkpoints/splits.pt", weights_only=False)

    # Combine all splits into one pool
    all_files = splits["train_files"] + splits["val_files"] + splits["test_files"]
    all_labels = splits["train_labels"] + splits["val_labels"] + splits["test_labels"]

    return all_files, all_labels


def create_fixed_test_split(all_files, all_labels):
    """Split into training pool + fixed test set using constant seed."""
    pool_files, test_files, pool_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=TEST_SPLIT_RATIO,
        random_state=TEST_SPLIT_SEED,  # Always 0
        stratify=all_labels
    )
    return pool_files, pool_labels, test_files, test_labels


def compute_normalization(pool_files):
    """Compute mean and std from the full training pool (once)."""
    all_tensors = []
    for f in pool_files:
        t = torch.load(f, weights_only=True)
        all_tensors.append(t)
    stacked = torch.stack(all_tensors)
    return stacked.mean().item(), stacked.std().item()


def sample_stratified_subset(pool_files, pool_labels, size, seed):
    """Sample a stratified subset of `size` from the pool."""
    if size >= len(pool_files):
        return list(pool_files), list(pool_labels)

    subset_files, _, subset_labels, _ = train_test_split(
        pool_files, pool_labels,
        train_size=size,
        random_state=seed,
        stratify=pool_labels
    )
    return subset_files, subset_labels


def train_model(model, train_loader, optimizer, criterion, device):
    """Train for exactly NUM_EPOCHS epochs. No early stopping."""
    model.train()
    final_loss = 0.0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x).view(-1)  # Use .view(-1) not .squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        final_loss = epoch_loss / len(train_loader.dataset)

    return final_loss


def evaluate_model(model, test_loader, device):
    """Evaluate on the fixed test set. Returns dict of metrics."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch_y.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels, dtype=int)

    auc = roc_auc_score(all_labels, all_probs)

    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc_roc": auc,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # Load all data
    all_files, all_labels = load_all_data()
    print(f"Total recordings loaded: {len(all_files)}")
    print(f"Sample path: {all_files[0]}")  # verify relative paths resolve

    # Fixed test split
    pool_files, pool_labels, test_files, test_labels = create_fixed_test_split(all_files, all_labels)
    print(f"Training pool: {len(pool_files)}, Test set: {len(test_files)}")

    # Normalization stats (computed once from full pool)
    print("Computing normalization stats from full training pool...")
    norm_mean, norm_std = compute_normalization(pool_files)
    print(f"Normalization — mean: {norm_mean:.2f}, std: {norm_std:.2f}")

    # Test dataset (fixed for all runs)
    test_dataset = CoughDataset(test_files, test_labels, mean=norm_mean, std=norm_std)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup output directory and CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    csv_path = RESULTS_CSV
    write_header = not os.path.exists(csv_path)

    total_runs = len(TRAINING_SIZES) * len(SEEDS)
    run_num = 0
    times = []

    for training_size in TRAINING_SIZES:
        for seed in SEEDS:
            run_num += 1
            start_time = time.time()

            try:
                # Sample stratified training subset
                subset_files, subset_labels = sample_stratified_subset(
                    pool_files, pool_labels, training_size, seed
                )

                # Compute pos_weight for this subset
                n_pos = sum(subset_labels)
                n_neg = len(subset_labels) - n_pos
                if n_pos == 0 or n_neg == 0:
                    raise ValueError(f"Degenerate subset at size={training_size}, seed={seed}: n_pos={n_pos}, n_neg={n_neg}")

                # Create training dataset and loader
                train_dataset = CoughDataset(subset_files, subset_labels, mean=norm_mean, std=norm_std)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

                # Initialize fresh model (seed controls weight init)
                seed_everything(seed)
                model = SickNoteCNN(n_mels=N_MELS, time_frames=TIME_FRAMES).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )
                criterion = nn.BCEWithLogitsLoss()

                # Train for exactly 30 epochs
                final_loss = train_model(model, train_loader, optimizer, criterion, device)

                # Evaluate on fixed test set
                metrics = evaluate_model(model, test_loader, device)

                elapsed = time.time() - start_time
                times.append(elapsed)

                # Append to CSV immediately
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([
                            "training_size", "seed", "val_auc_roc", "val_accuracy",
                            "val_sensitivity", "val_specificity", "final_train_loss",
                            "training_time_seconds"
                        ])
                        write_header = False
                    writer.writerow([
                        training_size, seed,
                        f"{metrics['auc_roc']:.6f}",
                        f"{metrics['accuracy']:.6f}",
                        f"{metrics['sensitivity']:.6f}",
                        f"{metrics['specificity']:.6f}",
                        f"{final_loss:.6f}",
                        f"{elapsed:.1f}"
                    ])

                # Progress
                avg_time = sum(times) / len(times)
                remaining = avg_time * (total_runs - run_num)
                print(f"[Run {run_num}/{total_runs}] size={training_size}, seed={seed} "
                      f"→ AUC={metrics['auc_roc']:.4f}, Acc={metrics['accuracy']:.4f}, "
                      f"time={elapsed:.1f}s, ETA={remaining/60:.1f}min")

                # Free memory
                del model, optimizer, criterion, train_dataset, train_loader
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[Run {run_num}/{total_runs}] size={training_size}, seed={seed} "
                      f"→ FAILED: {e}")
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([
                            "training_size", "seed", "val_auc_roc", "val_accuracy",
                            "val_sensitivity", "val_specificity", "final_train_loss",
                            "training_time_seconds"
                        ])
                        write_header = False
                    writer.writerow([training_size, seed, "NaN","NaN","NaN","NaN","NaN", f"{elapsed:.1f}"])

    print(f"\nExperiment complete. Results saved to {csv_path}")
    print(f"Total time: {sum(times)/60:.1f} minutes")


if __name__ == "__main__":
    main()
