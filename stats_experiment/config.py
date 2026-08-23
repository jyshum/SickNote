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
RESULTS_CSV = os.path.join(RESULTS_DIR, "experiment_results_no_weight.csv")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
ANALYSIS_FILE = os.path.join(RESULTS_DIR, "analysis_summary.txt")

# --- Stats ---
ALPHA = 0.05  # Significance level
