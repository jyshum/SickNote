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
EPOCHS = 80
LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.5
EARLY_STOPPING_PATIENCE = 15

# Model
CONV_CHANNELS = [8, 16, 32]
KERNEL_SIZE = 3

# Production architecture — the single source of truth. model/train.py,
# model/derive_threshold.py and api/inference.py all read these rather than
# each keeping a copy. EXPERIMENTS.md section 4b measured this configuration
# at 0.7238 against 0.6931 for the flatten head it replaced.
HEAD = "pool_time"
POOL = "sym"

# Three members, not five. EXPERIMENTS.md section 4b measured the ensemble size
# curve: members 4 and 5 are worth +0.0013 combined, well inside noise, for 40%
# more inference cost and storage. These are the first three of the original
# five, so the shipped ensemble is a strict subset of what came before.
#
# This lives in config rather than train.py because api/inference.py needs the
# ensemble size, and model/train.py imports scikit-learn at module level --
# which commit da1ac74 deliberately dropped from requirements.txt to cut ~150MB
# off the API container. Importing train.py from the serving path would pass
# locally and raise ModuleNotFoundError in production.
ENSEMBLE_SEEDS = [42, 123, 456]

# Evaluation
#
# Selected on the VALIDATION split under the MIN_SENSITIVITY policy below,
# never on test. The previous value (0.52) was the argmax of Youden's J
# computed on the test set — see EXPERIMENTS.md section 3.
#
# Regenerate with:  PYTHONPATH=. python -m model.derive_threshold
# Derived: PENDING — Task 5 replaces this after the ensemble is trained.
THRESHOLD = 0.50

# Cross-validation (model/crossval.py)
N_FOLDS = 5
CV_VAL_FRACTION = 0.15      # inner val split, carved from the 4 training folds
CV_SEED = 0
CV_CHECKPOINT_DIR = "model/checkpoints/cv"

# Threshold policy — SickNote is a screening tool, so a missed abnormal cough
# costs more than a false alarm. Provisional until we read the measured curve.
MIN_SENSITIVITY = 0.80

# Device
DEVICE = torch.device(
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
)

# Paths
RAW_DIR = "data/raw/coughvid_20211012"
PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "model/checkpoints"
DEMO_CLIPS_DIR = "data/demo_clips"
METADATA_CSV = "data/raw/coughvid_20211012/metadata_compiled.csv"
