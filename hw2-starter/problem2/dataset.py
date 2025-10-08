"""
EE641 HW2 - Problem 2
Dataset loader for drum patterns (16x9 binary matrices).

This version matches your npz file keys:
  ['train_patterns', 'val_patterns', 'train_styles', 'val_styles']

Styles (mapped to ids):
  rock=0, jazz=1, hiphop=2, electronic=3, latin=4
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


STYLE_TO_ID = {"rock": 0, "jazz": 1, "hiphop": 2, "electronic": 3, "latin": 4}


class DrumPatternDataset(Dataset):
    """
    Loads from:
      data/drums/patterns.npz
        - train_patterns [Ntr,16,9], val_patterns [Nval,16,9]
        - train_styles   [Ntr],       val_styles   [Nval]
    Returns (pattern_tensor, style_id)
    """

    def __init__(self, root_dir="data/drums", split="train"):
        super().__init__()
        self.root_dir = root_dir
        npz_path = os.path.join(root_dir, "patterns.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing {npz_path}. Run setup_data.py first.")

        data = np.load(npz_path, allow_pickle=True)

        if split == "train":
            self.patterns = data["train_patterns"].astype(np.float32)  # [N,16,9]
            styles = data["train_styles"]
        elif split == "val":
            self.patterns = data["val_patterns"].astype(np.float32)    # [N,16,9]
            styles = data["val_styles"]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Convert style strings (if any) to ids
        self.style_ids = [
            STYLE_TO_ID.get(s.lower(), 0) if isinstance(s, str) else int(s)
            for s in styles
        ]

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patterns[idx])               # (16,9) float32 in {0,1}
        y = torch.tensor(self.style_ids[idx], dtype=torch.long)
        return x, y


def get_dataloader(root_dir="data/drums", split="train",
                   batch_size=128, num_workers=2, shuffle=True, drop_last=True):
    ds = DrumPatternDataset(root_dir=root_dir, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last)