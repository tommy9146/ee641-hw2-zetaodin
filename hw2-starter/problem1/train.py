"""
Main training script for GAN experiments.
- Robustly finds data/fonts
- Runs vanilla GAN (apply_fix=False) or with simple fix (True)
- Writes results to problem1/results/
"""

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .dataset import get_loader, FontDataset
from .training_dynamics import train_gan

def _find_data_dir():
    candidates = [Path("data/fonts"), Path("../data/fonts"), Path("../../data/fonts")]
    for p in candidates:
        if (p / "train").exists():
            return str(p)
    raise FileNotFoundError(
        "Could not find data/fonts. From repo root, run: `python setup_data.py --seed 641`."
    )

def main():
    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 128,
        "epochs": 50,
        "z_dim": 100,
        "results_dir": "results",
        "apply_fix": False,      # set True to enable label smoothing + instance noise
    }

    data_dir = _find_data_dir()
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(f'{cfg["results_dir"]}/visualizations').mkdir(parents=True, exist_ok=True)

    ds = FontDataset(root_dir=data_dir, split="train")
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, drop_last=True)

    print(f"Training GAN on {len(ds)} images. apply_fix={cfg['apply_fix']} device={cfg['device']}")
    train_gan(loader, epochs=cfg["epochs"], z_dim=cfg["z_dim"],
              device=cfg["device"], out_dir=cfg["results_dir"], apply_fix=cfg["apply_fix"])

    print(f"Done. See '{cfg['results_dir']}'. For analysis run: `python evaluate.py` or `python -m problem1.evaluate`")

if __name__ == "__main__":
    main()