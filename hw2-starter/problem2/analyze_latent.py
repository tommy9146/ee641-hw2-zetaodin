"""
Post-training analysis:
  - Build style prototypes in latent_high space (mean mu_h per style)
  - Style-conditioned sampling using prototypes
  - Style interpolation
Outputs to: results/generated_patterns/ and results/latent_analysis/
"""

import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from .dataset import DrumPatternDataset
from .hierarchical_vae import HierarchicalVAE

STYLE_NAMES = ["rock", "jazz", "hiphop", "electronic", "latin"]


def _vis_pattern(pattern, path, title=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.imshow(pattern, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.xlabel("Instruments (9)")
    plt.ylabel("Time (16)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_style_prototypes(model, dataset, device):
    """Return [5, z_high_dim] mean(mu_high) per style."""
    model.eval()
    zdim = model.z_high_dim
    sums = np.zeros((5, zdim), dtype=np.float64)
    counts = np.zeros(5, dtype=np.int64)
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            mu_h, lv_h, _, _ = model.encode(x)
            sums[y.item()] += mu_h.squeeze(0).cpu().numpy().astype(np.float64)
            counts[y.item()] += 1
    protos = [(sums[k] / counts[k] if counts[k] > 0 else np.zeros(zdim)) for k in range(5)]
    return np.stack(protos, 0).astype(np.float32)


def analyze():
    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "results_dir": "results",
        "z_high_dim": 8,
        "z_low_dim": 16,
        "hidden": 256,
        "num_samples_per_style": 6,
        "interp_steps": 8,
    }

    Path(os.path.join(cfg["results_dir"], "generated_patterns")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(cfg["results_dir"], "latent_analysis")).mkdir(parents=True, exist_ok=True)

    # Load trained model
    model = HierarchicalVAE(cfg["z_high_dim"], cfg["z_low_dim"], hidden=cfg["hidden"]).to(cfg["device"])
    state = torch.load(os.path.join(cfg["results_dir"], "best_model.pth"), map_location=cfg["device"])
    model.load_state_dict(state)
    model.eval()

    # Build style prototypes from full dataset (train split is enough)
    ds = DrumPatternDataset(split="train")
    protos = build_style_prototypes(model, ds, cfg["device"])  # [5,z_high_dim]
    np.save(os.path.join(cfg["results_dir"], "latent_analysis", "style_prototypes.npy"), protos)

    # Style-conditioned sampling
    for sid, name in enumerate(STYLE_NAMES):
        z_high = torch.from_numpy(np.tile(protos[sid], (cfg["num_samples_per_style"], 1))).to(cfg["device"])
        z_low = torch.randn(cfg["num_samples_per_style"], cfg["z_low_dim"], device=cfg["device"])
        with torch.no_grad():
            logits = model.decode(z_high, z_low)
            samples = torch.sigmoid(logits).cpu().numpy()
        for i in range(samples.shape[0]):
            _vis_pattern(samples[i],
                         os.path.join(cfg["results_dir"], "generated_patterns", f"style_{name}_{i}.png"),
                         title=f"Style: {name}")

    # Style interpolation (chain)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for a, b in pairs:
        name = f"{STYLE_NAMES[a]}_to_{STYLE_NAMES[b]}"
        grids = []
        for t in np.linspace(0, 1, cfg["interp_steps"]):
            z_high = torch.from_numpy(((1 - t) * protos[a] + t * protos[b])[None, :]).to(cfg["device"])
            z_low = torch.randn(1, cfg["z_low_dim"], device=cfg["device"])
            with torch.no_grad():
                logits = model.decode(z_high, z_low)
                samp = torch.sigmoid(logits).cpu().numpy()[0]
            grids.append(samp)
        fig, axes = plt.subplots(1, cfg["interp_steps"], figsize=(cfg["interp_steps"] * 2.2, 2.2))
        for i, samp in enumerate(grids):
            axes[i].imshow(samp, cmap="Greys", aspect="auto", vmin=0, vmax=1)
            axes[i].axis("off")
            axes[i].set_title(f"t={i/(cfg['interp_steps']-1):.2f}")
        plt.tight_layout()
        path = os.path.join(cfg["results_dir"], "generated_patterns", f"interp_{name}.png")
        plt.savefig(path, dpi=200)
        plt.close()

    print("Analysis done. See results/generated_patterns/ and results/latent_analysis/")


if __name__ == "__main__":
    analyze()