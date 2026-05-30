"""
All hyperparameters. Values marked * MUST be verified after running explore.py.
See ARCHITECTURE.md → Config section for full documentation.
"""

# Audio
SAMPLE_RATE = 22050
CLIP_LENGTH_S = 4          # * verify against duration distribution
N_MELS = 64                # * adjust after visualizing spectrograms
N_FFT = 1024               # * adjust if time resolution looks wrong
HOP_LENGTH = 512           # * adjust if frequency resolution looks wrong

# Training
BATCH_SIZE = 32            # * lower to 16 if <1500 usable samples
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 5

# Model
CONV_CHANNELS = [16, 32, 64]   # * reduce to [8, 16, 32] if overfitting
KERNEL_SIZE = 3

# Evaluation
THRESHOLD = 0.5

# Paths
RAW_DIR = "data/raw/coughvid_20211012"
PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "model/checkpoints"
DEMO_CLIPS_DIR = "data/demo_clips"
METADATA_CSV = "data/raw/coughvid_20211012/metadata_compiled.csv"
