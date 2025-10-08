"""
Dataset loader for font generation task.
Loads 28x28 grayscale letter images normalized to [-1, 1].
Assumes data under: {root_dir}/{split}/*.png  and a metadata file:
    {root_dir}/metadata.json  (optional; we best-effort infer label from filename)
"""

import os, glob, re, json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FontDataset(Dataset):
    def __init__(self, root_dir="data/fonts", split="train"):
        self.root_dir = root_dir
        self.split = split
        self.paths = sorted(glob.glob(os.path.join(root_dir, split, "*.png")))
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found in {root_dir}/{split}. "
                "Run `python setup_data.py --seed 641` at repo root or check your paths."
            )
        # optional metadata (not strictly needed)
        self.meta = None
        meta_path = os.path.join(root_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    self.meta = json.load(f)
            except Exception:
                self.meta = None

    def __len__(self):
        return len(self.paths)

    def _infer_letter_id(self, path):
        # Try to infer a single uppercase A-Z from filename (e.g., "A_font3_001.png")
        name = os.path.basename(path)
        m = re.search(r"([A-Z])", name)
        if m:
            letter = m.group(1)
            idx = ord(letter) - ord("A")
            if 0 <= idx < 26:
                return idx
        # Fallback -> 0 ('A')
        return 0

    def __getitem__(self, idx):
        fp = self.paths[idx]
        img = Image.open(fp).convert("L").resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0      # [0,1]
        arr = arr * 2.0 - 1.0                              # [-1,1]
        x = torch.from_numpy(arr)[None, ...]               # (1,28,28)

        letter_id = self._infer_letter_id(fp)
        return x, letter_id


def get_loader(root_dir="data/fonts", split="train", batch_size=128, num_workers=2):
    ds = FontDataset(root_dir=root_dir, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, drop_last=True)