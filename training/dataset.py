"""
PyTorch dataset for ADNI 3D brain MRI classification.

Expects a CSV with at least two columns:
    Path   - absolute path to a preprocessed .nii.gz file
    Group  - diagnostic label string ('CN', 'AD', or 'MCI')

The dataset applies a z-score normalisation before any MONAI transforms.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


class ADNIDataset3D(Dataset):
    """
    Load preprocessed NIfTI volumes for binary CN/AD classification.

    Args:
        csv_path: Path to a CSV file produced by training/splits.py.
        transform: Optional MONAI Compose transform applied to each volume.
    """

    def __init__(self, csv_path: str, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {"CN": 0, "AD": 1}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        img = nib.load(row["Path"]).get_fdata().astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-8)

        if self.transform:
            img = self.transform(img)

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()

        if img.ndim == 3:
            img = img.unsqueeze(0)

        label = torch.tensor(self.label_map[row["Group"]], dtype=torch.long)
        return img, label
