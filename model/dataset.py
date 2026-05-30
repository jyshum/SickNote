"""
CoughDataset — PyTorch Dataset for preprocessed spectrogram tensors.

Usage: instantiate with lists of .pt file paths and integer labels.
Use torch.load() with weights_only=True when loading tensors.
"""
import torch
from torch.utils.data import Dataset


class CoughDataset(Dataset):
    """
    Args:
        file_list: list of paths to .pt spectrogram tensors
        label_list: list of int labels (0=healthy, 1=abnormal)
        mean: optional float, train-set mean for normalization
        std: optional float, train-set std for normalization
    """

    def __init__(self, file_list, label_list, mean=None, std=None):
        self.file_list = file_list
        self.label_list = label_list
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        tensor = torch.load(self.file_list[idx], weights_only=True)
        if self.mean is not None and self.std is not None:
            tensor = (tensor - self.mean) / self.std
        label = torch.tensor(self.label_list[idx], dtype=torch.float32)
        return tensor, label
