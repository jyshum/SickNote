"""
SickNoteCNN — small CNN for binary cough classification.

Model outputs raw logits (no Sigmoid). Use BCEWithLogitsLoss for training.
Apply torch.sigmoid() at inference time only.

See ARCHITECTURE.md → Model Architecture for the full reference implementation.
"""
import torch
import torch.nn as nn

from model import config


class SickNoteCNN(nn.Module):
    """
    Three conv-batchnorm-relu-maxpool blocks → flatten → two FC layers.

    Args:
        n_mels: number of mel frequency bins (from config.py, verified by explore.py)
        time_frames: number of time frames in spectrogram (derived from config.py)
    """

    def __init__(self, n_mels, time_frames):
        super().__init__()
        channels = config.CONV_CHANNELS  # [16, 32, 64]
        k = config.KERNEL_SIZE           # 3

        self.conv = nn.Sequential(
            nn.Conv2d(1, channels[0], k, padding=1), nn.BatchNorm2d(channels[0]),
            nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(channels[0], channels[1], k, padding=1), nn.BatchNorm2d(channels[1]),
            nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(channels[1], channels[2], k, padding=1), nn.BatchNorm2d(channels[2]),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self._flatten_dim = self._get_flatten_dim(n_mels, time_frames)

        self.classifier = nn.Sequential(
            nn.Linear(self._flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 1),
            # NO Sigmoid — use BCEWithLogitsLoss for numerical stability
            # Apply torch.sigmoid() at inference time only
        )

    def _get_flatten_dim(self, n_mels, time_frames):
        """Pass dummy tensor through conv layers to compute flatten dimension.
        Never hardcode this value."""
        dummy = torch.zeros(1, 1, n_mels, time_frames)
        out = self.conv(dummy)
        return int(torch.prod(torch.tensor(out.shape[1:])))

    def forward(self, x):
        """Returns raw logits, shape (batch_size, 1)."""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
