"""
Training script for Hierarchical VAE on drum patterns.

Outputs under problem2/results/:
  - training_log.json
  - best_model.pth
  - generated_patterns/*.png
"""

import os, json
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from .dataset import get_dataloader, DrumPatternDataset
from .hierarchical_vae import HierarchicalVAE
from .training_utils import compute_hierarchical_elbo, kl_anneal_linear, kl_anneal_cyclical


def _vis_pattern(pattern, path, title=""):
    """Save a 16x9 matrix as a heatmap image for quick inspection."""
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


def main():
    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 128,
        "epochs": 100,      # for quick test, set 20
        "z_high_dim": 8,
        "z_low_dim": 16,
        "hidden": 256,
        "lr": 1e-3,
        "anneal": "linear",   # "linear" or "cyclical"
        "results_dir": "results",
    }

    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(cfg["results_dir"], "generated_patterns")).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader = get_dataloader(split="train", batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = get_dataloader(split="val",   batch_size=cfg["batch_size"], shuffle=False, drop_last=False)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["epochs"]

    # Model/opt
    model = HierarchicalVAE(cfg["z_high_dim"], cfg["z_low_dim"], hidden=cfg["hidden"]).to(cfg["device"])
    opt = optim.Adam(model.parameters(), lr=cfg["lr"])

    log = {"loss": [], "recon": [], "kl_high": [], "kl_low": [], "beta": [],
           "val_loss": [], "val_recon": [], "val_kl_high": [], "val_kl_low": []}
    best_val = float("inf")
    global_step = 0

    for ep in range(1, cfg["epochs"] + 1):
        # -------- Train --------
        model.train()
        for x, _style in train_loader:
            global_step += 1
            x = x.to(cfg["device"])  # [B,16,9]

            if cfg["anneal"] == "linear":
                beta = kl_anneal_linear(global_step, total_steps, start=0.0, end=1.0)
            else:
                beta = kl_anneal_cyclical(global_step, total_steps, cycles=4, ratio=0.5)

            logits, (mu_h, lv_h, mu_l, lv_l), _ = model(x)
            loss, recon, kl_h, kl_l = compute_hierarchical_elbo(
                x, logits, mu_h, lv_h, mu_l, lv_l, beta_h=beta, beta_l=beta, reduction="mean"
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            log["loss"].append(float(loss.item()))
            log["recon"].append(float(recon.item()))
            log["kl_high"].append(float(kl_h.item()))
            log["kl_low"].append(float(kl_l.item()))
            log["beta"].append(float(beta))

        # -------- Validate --------
        model.eval()
        val_losses, val_recons, val_khs, val_kls = [], [], [], []
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(cfg["device"])
                logits, (mu_h, lv_h, mu_l, lv_l), _ = model(x)
                # use final beta=1.0 for reporting
                vloss, vrecon, vklh, vkll = compute_hierarchical_elbo(
                    x, logits, mu_h, lv_h, mu_l, lv_l, beta_h=1.0, beta_l=1.0, reduction="mean"
                )
                val_losses.append(vloss.item())
                val_recons.append(vrecon.item())
                val_khs.append(vklh.item())
                val_kls.append(vkll.item())

        vloss = float(np.mean(val_losses)) if val_losses else float("inf")
        vrecon = float(np.mean(val_recons)) if val_recons else float("inf")
        vkh = float(np.mean(val_khs)) if val_khs else float("inf")
        vkl = float(np.mean(val_kls)) if val_kls else float("inf")
        log["val_loss"].append(vloss)
        log["val_recon"].append(vrecon)
        log["val_kl_high"].append(vkh)
        log["val_kl_low"].append(vkl)

        # --- sample at epoch end ---
        with torch.no_grad():
            z_h = torch.randn(8, cfg["z_high_dim"], device=cfg["device"])
            z_l = torch.randn(8, cfg["z_low_dim"], device=cfg["device"])
            logits = model.decode(z_h, z_l)
            samples = torch.sigmoid(logits).cpu().numpy()  # [8,16,9] in [0,1]
        for i in range(min(4, samples.shape[0])):
            _vis_pattern(samples[i],
                         os.path.join(cfg["results_dir"], "generated_patterns", f"epoch_{ep}_sample_{i}.png"),
                         title=f"Epoch {ep}")

        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), os.path.join(cfg["results_dir"], "best_model.pth"))

        print(f"[Epoch {ep:03d}] train_loss={np.mean(log['loss'][-steps_per_epoch:]):.4f} "
              f"val_loss={vloss:.4f} recon={vrecon:.4f} kl_h={vkh:.4f} kl_l={vkl:.4f} "
              f"beta_last={log['beta'][-1]:.3f}")

    # save log
    with open(os.path.join(cfg["results_dir"], "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"Done. Results saved to {cfg['results_dir']}/")


if __name__ == "__main__":
    main()