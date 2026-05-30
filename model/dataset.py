"""
CoughDataset — PyTorch Dataset for preprocessed spectrogram tensors.

Usage: instantiate with lists of .pt file paths and integer labels.
Use torch.load() with weights_only=True when loading tensors.
"""
from torch.utils.data import Dataset


class CoughDataset(Dataset):
    """
    Args:
        file_list: list of paths to .pt spectrogram tensors
        label_list: list of int labels (0=healthy, 1=abnormal)
    """

    def __init__(self, file_list, label_list):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
