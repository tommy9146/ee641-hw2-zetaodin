"""
GAN training implementation with:
- Standard BCE GAN training loop
- Mode collapse proxy logging
- Visualizations at epochs 10/30/50/100
"""

import os, json
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from .models import Generator, Discriminator
from .fixes import Stabilizer


def _save_grid_matplotlib(imgs, path, ncol=13):
    # imgs: [N,1,28,28] in [-1,1]
    N = imgs.size(0)
    rows = (N + ncol - 1) // ncol
    plt.figure(figsize=(ncol*0.8, rows*0.8))
    for i in range(N):
        plt.subplot(rows, ncol, i+1)
        plt.axis("off")
        plt.imshow((imgs[i,0].cpu().numpy()+1)/2, cmap="gray")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _edge_density_proxy(batch):
    # A crude structural diversity proxy: mean gradient magnitude
    dx = batch[:, :, :, 1:] - batch[:, :, :, :-1]
    dy = batch[:, :, 1:, :] - batch[:, :, :-1, :]
    dx = torch.nn.functional.pad(dx, (0,1,0,0))
    dy = torch.nn.functional.pad(dy, (0,0,0,1))
    mag = torch.sqrt(dx*dx + dy*dy)
    return mag.abs().mean().item()


def train_gan(train_loader, epochs=50, z_dim=100, lr=2e-4, betas=(0.5, 0.999),
              device=None, out_dir="results", apply_fix=False,
              save_epochs=(10, 30, 50, 100)):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "visualizations"), exist_ok=True)

    G, D = Generator(z_dim=z_dim).to(device), Discriminator().to(device)
    optG, optD = Adam(G.parameters(), lr=lr, betas=betas), Adam(D.parameters(), lr=lr, betas=betas)
    bce = nn.BCELoss()

    total_steps = epochs * len(train_loader)
    stabil = Stabilizer(total_steps) if apply_fix else None

    log = {"d_loss": [], "g_loss": [], "mode_proxy": []}
    step = 0
    best_g_loss, best_state = float("inf"), None

    fixed_z = torch.randn(26*4, z_dim, 1, 1, device=device)

    for ep in range(1, epochs+1):
        for x, _ in train_loader:
            step += 1
            x = x.to(device)
            bs = x.size(0)

            # ===== Train D =====
            if stabil: x = stabil.instance_noise(x, step)
            z = torch.randn(bs, z_dim, 1, 1, device=device)
            fake = G(z).detach()
            if stabil: fake = stabil.instance_noise(fake, step)

            real_lbl = torch.ones(bs, 1, device=device)
            fake_lbl = torch.zeros(bs, 1, device=device)
            if stabil: real_lbl = stabil.smooth_real_labels(real_lbl)

            real_pred, _ = D(x)
            fake_pred, _ = D(fake)
            d_loss = bce(real_pred, real_lbl) + bce(fake_pred, fake_lbl)
            optD.zero_grad(set_to_none=True); d_loss.backward(); optD.step()

            # ===== Train G =====
            z = torch.randn(bs, z_dim, 1, 1, device=device)
            gen = G(z)
            if stabil: gen = stabil.instance_noise(gen, step)
            pred, _ = D(gen)
            g_loss = bce(pred, torch.ones(bs, 1, device=device))
            optG.zero_grad(set_to_none=True); g_loss.backward(); optG.step()

            # logging
            log["d_loss"].append(float(d_loss.item()))
            log["g_loss"].append(float(g_loss.item()))
            if step % 200 == 0:
                with torch.no_grad():
                    grid_imgs = G(fixed_z).clamp(-1, 1)
                    proxy = _edge_density_proxy(grid_imgs)
                log["mode_proxy"].append({"step": step, "edge_density": proxy})

            if g_loss.item() < best_g_loss:
                best_g_loss = g_loss.item()
                best_state = G.state_dict()

        # save visualizations
        if ep in set(save_epochs):
            with torch.no_grad():
                viz = G(fixed_z).clamp(-1, 1)
            _save_grid_matplotlib(viz, os.path.join(out_dir, "visualizations", f"epoch_{ep}_grid.png"))

    # save best G and logs
    torch.save(best_state or G.state_dict(), os.path.join(out_dir, "best_generator.pth"))
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # proxy curve
    xs = [p["step"] for p in log["mode_proxy"]]
    ys = [p["edge_density"] for p in log["mode_proxy"]]
    if xs:
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("step"); plt.ylabel("edge_density proxy")
        plt.title("Mode Collapse Proxy Over Training")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mode_collapse_analysis.png"), dpi=200)
        plt.close()

    return os.path.join(out_dir, "training_log.json")