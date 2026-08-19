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
        head: how the conv output is reduced before the classifier.
            "flatten"   every (channel, mel, frame) position gets its own weight.
                        The historical behaviour, and 99.6% of the parameters.
            "pool_time" average AND max over the time axis, keeping mel bins.
                        Time-shift invariant, and ~20x smaller.
            Default is "flatten" so existing checkpoints keep loading.
    """

    def __init__(self, n_mels, time_frames, head="flatten"):
        super().__init__()
        channels = config.CONV_CHANNELS  # [8, 16, 32]
        k = config.KERNEL_SIZE           # 3
        self.head = head

        self.conv = nn.Sequential(
            nn.Conv2d(1, channels[0], k, padding=1), nn.BatchNorm2d(channels[0]),
            nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(channels[0], channels[1], k, padding=1), nn.BatchNorm2d(channels[1]),
            nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(channels[1], channels[2], k, padding=1), nn.BatchNorm2d(channels[2]),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self._flatten_dim = self._get_feature_dim(n_mels, time_frames)

        self.classifier = nn.Sequential(
            nn.Linear(self._flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 1),
            # NO Sigmoid — use BCEWithLogitsLoss for numerical stability
            # Apply torch.sigmoid() at inference time only
        )

    def _get_feature_dim(self, n_mels, time_frames):
        """Pass a dummy tensor through the conv layers to size the classifier.
        Never hardcode this value."""
        dummy = torch.zeros(1, 1, n_mels, time_frames)
        out = self.conv(dummy)
        if self.head == "flatten":
            return int(torch.prod(torch.tensor(out.shape[1:])))
        if self.head == "pool_time":
            # channels x mel bins, doubled because avg and max are concatenated
            return int(out.shape[1] * out.shape[2] * 2)
        raise ValueError(f"unknown head: {self.head}")

    def _reduce(self, x):
        """Collapse the conv output (batch, C, mel, frames) into a flat vector."""
        if self.head == "flatten":
            return x.view(x.size(0), -1)
        # Average says "how much of this feature is present overall"; max says
        # "was it ever strongly present". Max matters here because a cough is a
        # short event inside a mostly-quiet 8s window — averaging alone would
        # dilute it, and would also leak how much of the clip is zero-padding.
        return torch.cat([x.mean(dim=3), x.amax(dim=3)], dim=1).flatten(1)

    def forward(self, x):
        """Returns raw logits, shape (batch_size, 1)."""
        x = self.conv(x)
        return self.classifier(self._reduce(x))
