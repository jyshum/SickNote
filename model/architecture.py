"""
SickNoteCNN — small CNN for binary cough classification.

Model outputs raw logits (no Sigmoid). Use BCEWithLogitsLoss for training.
Apply torch.sigmoid() at inference time only.

See ARCHITECTURE.md → Model Architecture for the full reference implementation.
"""
import torch.nn as nn


class SickNoteCNN(nn.Module):
    """
    Three conv-batchnorm-relu-maxpool blocks → flatten → two FC layers.

    Args:
        n_mels: number of mel frequency bins (from config.py, verified by explore.py)
        time_frames: number of time frames in spectrogram (derived from config.py)
    """

    def __init__(self, n_mels, time_frames):
        raise NotImplementedError

    def _get_flatten_dim(self, n_mels, time_frames):
        """Pass dummy tensor through conv layers to compute flatten dimension.
        Never hardcode this value."""
        raise NotImplementedError

    def forward(self, x):
        """Returns raw logits, shape (batch_size, 1)."""
        raise NotImplementedError
