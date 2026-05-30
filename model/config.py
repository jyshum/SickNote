"""
All hyperparameters — verified by explore.py on 2026-05-30.
See ARCHITECTURE.md → Config section for full documentation.
"""
import torch

# Audio
SAMPLE_RATE = 22050
CLIP_LENGTH_S = 4
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# Training
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 5

# Model
CONV_CHANNELS = [16, 32, 64]
KERNEL_SIZE = 3

# Evaluation
THRESHOLD = 0.5

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Paths
RAW_DIR = "data/raw/coughvid_20211012"
PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "model/checkpoints"
DEMO_CLIPS_DIR = "data/demo_clips"
METADATA_CSV = "data/raw/coughvid_20211012/metadata_compiled.csv"
