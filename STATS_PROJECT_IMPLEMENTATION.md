# STATS_PROJECT — Phase 2: Implementation Spec

> **What this document is**: Precise implementation instructions for Claude Code, written against the actual SickNote codebase structure discovered in Phase 1. Every import path, class name, and function signature references real code. Execute in order.

---

## 0. CRITICAL ISSUES FROM DISCOVERY (Address First)

### 0.1 Processed spectrograms are missing
`data/processed/` is empty. The `CoughDataset` loads `.pt` tensor files from this directory. Before ANY experiment runs, preprocessing must be executed:

```bash
python -m model.preprocess
```

This generates `.pt` spectrogram files in `data/processed/` and creates `model/checkpoints/splits.pt` and `model/checkpoints/preprocessing_params.pt`. Verify `data/processed/` is populated before proceeding.

### 0.2 Max training size must be capped
Total recordings: 2,267. With 15% test holdout: test ≈ 340, training pool ≈ 1,927. The original spec had max size 1,950 — **change to 1,900**.

Updated training sizes: `[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1900]`

### 0.3 train_single() cannot be reused
`train_single()` in `model/train.py` has early stopping (patience=15) and LR scheduling (ReduceLROnPlateau) baked in. It also loads from `splits.pt` and uses a 3-way split (train/val/test). The experiment requires:
- Fixed 30 epochs, no early stopping
- No LR scheduling
- 2-way split only (train subset + fixed test)
- Custom training subsets per run

**Solution**: Write a new minimal training loop in `stats_experiment/run_experiment.py`. Import the model class and config values, but NOT `train_single()`.

### 0.4 Missing dependencies
Install before running:
```bash
pip install scipy --break-system-packages
```
Confirm `sklearn` and `pandas` are already installed (used by existing code but not in requirements.txt).

### 0.5 Normalization strategy
`CoughDataset` normalizes tensors using `(tensor - mean) / std`. For the experiment, compute mean and std from the FULL training pool once (after the fixed test split), and use the same values for every run. This makes normalization a controlled variable.

---

## 1. FILES TO CREATE

```
stats_experiment/
├── __init__.py              (empty)
├── config.py                (experiment constants)
├── run_experiment.py        (main experiment runner — 50 training runs)
├── analyze_results.py       (statistical analysis + graph generation)
└── results/                 (created at runtime)
    ├── experiment_results.csv
    ├── analysis_summary.txt
    └── graphs/
```

---

## 2. FILE SPECS

### 2.1 `stats_experiment/__init__.py`
Empty file. Needed so `stats_experiment` is a package.

### 2.2 `stats_experiment/config.py`

```python
"""Experiment configuration — single source of truth."""
import os

# --- Experimental Design ---
TRAINING_SIZES = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 1900]
SEEDS = [1, 2, 3, 4, 5]
TEST_SPLIT_SEED = 0        # Fixed seed for test set split — NEVER changes
TEST_SPLIT_RATIO = 0.15    # 15% held out for test
NUM_EPOCHS = 30            # Fixed training duration — no early stopping

# --- Inherited from SickNote (model/config.py) ---
# These are duplicated here for clarity. If model/config.py changes, update these.
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
DROPOUT = 0.5
N_MELS = 64
TIME_FRAMES = 345

# --- Output Paths ---
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "experiment_results.csv")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
ANALYSIS_FILE = os.path.join(RESULTS_DIR, "analysis_summary.txt")

# --- Stats ---
ALPHA = 0.05  # Significance level
```

### 2.3 `stats_experiment/run_experiment.py`

**Imports needed:**
```python
import os, sys, time, csv, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# From existing SickNote codebase
from model.architecture import SickNoteCNN
from model.dataset import CoughDataset

# Experiment config
from stats_experiment.config import *
```

**Implementation — all functions in order:**

#### Function 1: Seeding
```python
def seed_everything(seed):
    """Seed all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Seeds CPU + MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

#### Function 2: Load all preprocessed data
Load ALL spectrograms and labels by combining train + val + test from the existing `splits.pt`:

```python
def load_all_data():
    """Load all preprocessed spectrograms and labels from splits.pt."""
    splits = torch.load("model/checkpoints/splits.pt", weights_only=False)
    
    # Combine all splits into one pool
    all_files = splits["train_files"] + splits["val_files"] + splits["test_files"]
    all_labels = splits["train_labels"] + splits["val_labels"] + splits["test_labels"]
    
    return all_files, all_labels
```

Verify: `len(all_files)` should be 2,267.

#### Function 3: Create fixed test split
```python
def create_fixed_test_split(all_files, all_labels):
    """Split into training pool + fixed test set using constant seed."""
    pool_files, test_files, pool_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=TEST_SPLIT_RATIO,
        random_state=TEST_SPLIT_SEED,  # Always 0
        stratify=all_labels
    )
    return pool_files, pool_labels, test_files, test_labels
```

#### Function 4: Compute normalization stats from full pool
```python
def compute_normalization(pool_files):
    """Compute mean and std from the full training pool (once)."""
    all_tensors = []
    for f in pool_files:
        t = torch.load(f, weights_only=True)
        all_tensors.append(t)
    stacked = torch.stack(all_tensors)
    return stacked.mean().item(), stacked.std().item()
```

These values are used for EVERY run — they are a control, not a variable.

#### Function 5: Stratified subset sampling
```python
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
```

#### Function 6: Training loop (NO early stopping, NO LR scheduling)
```python
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
```

**Note**: Use `.view(-1)` instead of `.squeeze()` to handle single-item batches safely. `.squeeze()` on shape `(1, 1)` removes ALL dimensions; `.view(-1)` always gives `(batch_size,)`.

#### Function 7: Evaluation
```python
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
```

#### Function 8: Main experiment loop
```python
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() 
                          else "cpu")
    print(f"Using device: {device}")
    
    # Load all data
    all_files, all_labels = load_all_data()
    print(f"Total recordings loaded: {len(all_files)}")
    
    # Fixed test split
    pool_files, pool_labels, test_files, test_labels = create_fixed_test_split(all_files, all_labels)
    print(f"Training pool: {len(pool_files)}, Test set: {len(test_files)}")
    
    # Normalization stats (computed once from full pool)
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
                pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
                
                # Create training dataset and loader
                train_dataset = CoughDataset(subset_files, subset_labels, mean=norm_mean, std=norm_std)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                
                # Initialize fresh model (seed controls weight init)
                seed_everything(seed)
                model = SickNoteCNN(n_mels=N_MELS, time_frames=TIME_FRAMES).to(device)
                
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
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
```

**Critical implementation notes for Claude Code:**

1. `CoughDataset.__getitem__` loads `.pt` files from disk. Verify it takes `mean` and `std` as constructor kwargs. If the signature differs, adapt accordingly.

2. `pool_labels` from `train_test_split` will be a list. `sum(pool_labels)` works if labels are ints or floats. If they're tensors, convert with `int(label)` first.

3. After subset sampling, call `seed_everything(seed)` AGAIN before `SickNoteCNN()` initialization. Subset sampling consumes random state, so re-seeding ensures weight init is reproducible for a given seed regardless of training_size.

4. MPS memory: call `torch.mps.empty_cache()` after each run to prevent accumulation over 50 runs on 8GB M2.

5. File paths in `splits.pt` may be absolute or relative. Print a sample path before iterating to verify they resolve correctly from the project root. Adjust if needed.

### 2.4 `stats_experiment/analyze_results.py`

**Imports:**
```python
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from stats_experiment.config import *
```

**This script produces ALL statistical outputs and ALL graphs.**

#### Part A: Load and validate data
```python
df = pd.read_csv(RESULTS_CSV)
assert len(df) == 50, f"Expected 50 rows, got {len(df)}"
assert df["val_auc_roc"].notna().all(), "Found NaN values — check failed runs"
```

#### Part B: Descriptive statistics
Compute and print to console + save to `analysis_summary.txt`:
- n
- For `val_auc_roc`: mean, median, std, min, max, Q1, Q3
- Group means: mean AUC at each of the 10 training sizes
- Group standard deviations: std AUC at each of the 10 training sizes
- r (correlation between training_size and val_auc_roc)
- r²

#### Part C: Linear regression (raw x first)
Use `scipy.stats.linregress(x, y)` where x = training_size, y = val_auc_roc.

`linregress` returns: slope, intercept, r_value, p_value, std_err

Compute:
- b₁ = slope, b₀ = intercept
- SE(b₁) = std_err
- t = slope / std_err
- **p_value adjustment**: `linregress` returns TWO-SIDED p-value. For one-sided (Hₐ: β₁ > 0): `p_one = p_two / 2` if slope > 0, else `p_one = 1 - p_two / 2`
- df = n - 2 = 48
- r = r_value, r² = r_value²

#### Part D: Check residuals for curvature
```python
y_pred = b0 + b1 * x
residuals = y - y_pred
```

Programmatic curvature check: fit quadratic `np.polyfit(x, y, 2)`, test if quadratic coefficient is large relative to the linear effect. If curvature detected → also run log-transformed analysis with `x_log = np.log(x)`.

#### Part E: Generate all graphs

**Graph styling:**
```python
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})
```

**Graph 1: `scatterplot_raw.png`**
- Scatter: training_size (x) vs. val_auc_roc (y), all 50 points
- Overlay: least-squares regression line
- Annotate: r² value and regression equation
- Title: "Training Set Size vs. Validation AUC-ROC"
- X-label: "Training Set Size (recordings)"
- Y-label: "Validation AUC-ROC"

**Graph 2: `residual_plot_raw.png`**
- Scatter: fitted values (ŷ) vs. residuals
- Horizontal reference line at y=0
- Title: "Residual Plot — Training Size vs. AUC-ROC"
- X-label: "Fitted Values"
- Y-label: "Residuals"

**Graph 3: `histogram_residuals.png`**
- Histogram of 50 residuals, ~8-10 bins
- Title: "Distribution of Residuals"
- X-label: "Residual"
- Y-label: "Frequency"

**Graph 4: `normal_probability_plot.png`**
- `scipy.stats.probplot(residuals, plot=plt)`
- Title: "Normal Probability Plot of Residuals"

**Graph 5: `boxplots_by_size.png`**
- 10 side-by-side boxplots, one per training size
- X-axis: training sizes as categorical labels
- Y-axis: AUC-ROC
- Title: "Distribution of AUC-ROC by Training Set Size"

**Graph 6: `pvalue_diagram.png`**
- t-distribution PDF curve with df=48
- Vertical line at test statistic t
- Shade area to the RIGHT of t (one-sided p-value)
- Annotate: "t = {value:.3f}" and "p = {value:.4f}"
- Title: "P-Value Diagram — t-Test for Regression Slope"
- X-label: "t"
- Y-label: "Density"

**Conditional (only if log transform needed):**

**Graph 7: `scatterplot_log.png`**
- ln(training_size) vs. val_auc_roc with regression line

**Graph 8: `residual_plot_log.png`**
- Residual plot for log-transformed regression

#### Part F: Save analysis_summary.txt

```
STATS EXPERIMENT — ANALYSIS SUMMARY
====================================

DESCRIPTIVE STATISTICS
  n = 50
  Mean AUC-ROC = ___
  Median AUC-ROC = ___
  Std Dev = ___
  Min = ___, Q1 = ___, Median = ___, Q3 = ___, Max = ___

GROUP MEANS (by training size)
  200:  mean=___, std=___
  400:  mean=___, std=___
  ...
  1900: mean=___, std=___

REGRESSION (raw training_size)
  Equation: ŷ = ___ + ___x
  r = ___
  r² = ___
  b₁ (slope) = ___
  SE(b₁) = ___
  t = b₁ / SE(b₁) = ___ / ___ = ___
  df = 48
  p-value (one-sided) = ___

HYPOTHESIS TEST
  H₀: β₁ = 0
  Hₐ: β₁ > 0
  α = 0.05
  Decision: [Reject / Fail to reject] H₀
  
  At the α = 0.05 significance level, there [is / is not] sufficient 
  evidence to conclude that there is a positive linear relationship 
  between training set size and validation AUC-ROC for CNN models trained 
  on COUGHVID cough audio spectrograms.

[IF LOG TRANSFORM APPLIED]
REGRESSION (log-transformed)
  Equation: ŷ = ___ + ___·ln(x)
  r = ___
  r² = ___
  b₁ = ___
  SE(b₁) = ___
  t = ___
  df = 48
  p-value (one-sided) = ___
```

---

## 3. EXECUTION ORDER

```
Step 1:  Ensure data/processed/ is populated (run python -m model.preprocess if empty)
Step 2:  Install scipy: pip install scipy --break-system-packages
Step 3:  Create stats_experiment/ directory and all files
Step 4:  Run a SINGLE test: size=200, seed=1 only. Verify:
           - CSV row is written correctly
           - AUC is a reasonable float between 0.5 and 0.8
           - No errors, no NaN
           - Print output looks correct
Step 5:  After single-run verification, run the full 50-run experiment
Step 6:  Verify CSV has 50 rows, all values present, no NaN
Step 7:  Run analyze_results.py
Step 8:  Verify all 6+ graphs are saved to stats_experiment/results/graphs/
Step 9:  Verify analysis_summary.txt contains all required statistics
Step 10: Print key results: slope, t-statistic, p-value, decision
```

---

## 4. KNOWN EDGE CASES

1. **Batch size vs. dataset size**: At training_size=200 with batch_size=32, last batch has < 32 samples. DataLoader handles this. But use `.view(-1)` not `.squeeze()` on model output — `.squeeze()` on shape `(1,1)` removes all dimensions.

2. **Stratified sampling at small sizes**: training_size=200 with 20.3% healthy ≈ 40 healthy, 160 abnormal. `train_test_split` can handle this. If any class has < 2 samples in a split, it errors — shouldn't happen at size 200 but add a check.

3. **MPS memory**: 8GB M2. Model is tiny (~50K params). Call `torch.mps.empty_cache()` between runs as precaution.

4. **File paths in splits.pt**: May be absolute or relative. Print a sample path first to verify. Adjust if they don't resolve from project root.

5. **Label types**: Labels from `splits.pt` may be ints, floats, or tensors. `CoughDataset` and `train_test_split` need array-like. Convert as needed.

6. **linregress p-value**: Returns two-sided. For one-sided Hₐ: β₁ > 0: `p_one = p_two / 2` if slope > 0.

---

## 5. DO NOT

- Do NOT modify any existing files in `model/`, `api/`, or `tests/`
- Do NOT use `train_single()` from `model/train.py`
- Do NOT use early stopping or LR scheduling
- Do NOT use the ensemble — train single models only
- Do NOT use data augmentation
- Do NOT tune hyperparameters between runs
- Do NOT use the existing train/val/test split from `splits.pt` — create a new one with `TEST_SPLIT_SEED=0`
- Do NOT cache or reuse models between runs — each run starts fresh
