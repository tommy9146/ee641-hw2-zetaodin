"""
Analysis and evaluation for Problem 1 (Font GAN):
- Load best_generator.pth
- Final grid
- Latent interpolation
- Mode-coverage histogram (nearest-prototype, if data available)
- Proxy curve from training_log.json
"""

import os, json
import numpy as np                  
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from .models import Generator       
from .dataset import FontDataset


def _save_grid(imgs, path, ncol=13):
    """imgs: [N,1,28,28] in [-1,1]"""
    N = imgs.size(0)
    rows = (N + ncol - 1) // ncol
    plt.figure(figsize=(ncol*0.8, rows*0.8))
    for i in range(N):
        plt.subplot(rows, ncol, i+1)
        plt.axis("off")
        plt.imshow((imgs[i,0].cpu().numpy()+1)/2, cmap="gray")
    plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def _build_letter_prototypes(data_root="data/fonts", k_per_class=40):
    """从真实数据构建每个字母的平均原型，用于近原型分类（无需额外模型）。"""
    ds = FontDataset(root_dir=data_root, split="train")
    buckets = [[] for _ in range(26)]
    for i in range(len(ds)):
        x, y = ds[i]                       # x in [-1,1], shape [1,28,28]
        if len(buckets[y]) < k_per_class:
            buckets[y].append(x)
        if all(len(b) >= k_per_class for b in buckets):
            break
    protos = []
    for c in range(26):
        if buckets[c]:
            protos.append(torch.stack(buckets[c], 0).mean(0))
        else:
            protos.append(torch.zeros(1,28,28))
    return torch.stack(protos, 0)          # [26,1,28,28]


def _nearest_proto_labels(gen_imgs, protos):
    """把生成图片按余弦相似度匹配到最近的字母原型，得到“存活字母”的近似分布。"""
    import torch.nn.functional as F
    N = gen_imgs.size(0)
    gen_flat  = gen_imgs.view(N, -1)
    prot_flat = protos.view(26, -1).to(gen_imgs.device)
    gen_norm  = F.normalize(gen_flat,  dim=1)
    prot_norm = F.normalize(prot_flat, dim=1)
    sims = gen_norm @ prot_norm.t()        # [N,26]
    return sims.argmax(dim=1).cpu().numpy()


def analyze(results_dir="results", z_dim=100,
            data_root_candidates=("data/fonts","../data/fonts","../../data/fonts")):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = Generator(z_dim=z_dim).to(device)
    state = torch.load(os.path.join(results_dir, "best_generator.pth"), map_location=device)
    G.load_state_dict(state)
    G.eval()

    os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)

    z = torch.randn(26*4, z_dim, 1, 1, device=device)
    with torch.no_grad():
        imgs = G(z).clamp(-1,1).cpu()
    _save_grid(imgs, os.path.join(results_dir, "visualizations", "epoch_final_grid.png"))

    n_steps = 16
    z0, z1 = torch.randn(1, z_dim, 1, 1, device=device), torch.randn(1, z_dim, 1, 1, device=device)
    zs = torch.cat([(1-t)*z0 + t*z1 for t in np.linspace(0, 1, n_steps)], dim=0)  # 这里用到了 numpy
    with torch.no_grad():
        inter = G(zs).clamp(-1,1).cpu()
    _save_grid(inter, os.path.join(results_dir, "visualizations", "interpolation.png"), ncol=16)

    data_root = None
    for c in data_root_candidates:
        if (Path(c) / "train").exists():
            data_root = c; break
    if data_root is not None:
        protos = _build_letter_prototypes(data_root)
        with torch.no_grad():
            big = G(torch.randn(2000, z_dim, 1, 1, device=device)).clamp(-1,1).cpu()
        preds = _nearest_proto_labels(big, protos)
        #import numpy as np
        counts = np.bincount(preds, minlength=26)
        letters = [chr(ord('A')+i) for i in range(26)]
        plt.figure(figsize=(12,4))
        plt.bar(letters, counts)
        plt.title("Mode coverage (approx via nearest prototypes)")
        plt.xlabel("Letter"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "visualizations", "mode_coverage_histogram.png"), dpi=200)
        plt.close()

    log_path = os.path.join(results_dir, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f: log = json.load(f)
        xs = [p["step"] for p in log.get("mode_proxy",[])]
        ys = [p["edge_density"] for p in log.get("mode_proxy",[])]
        if xs:
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("step"); plt.ylabel("edge_density proxy")
            plt.title("Mode Collapse Proxy Over Training")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "mode_collapse_analysis.png"), dpi=200)
            plt.close()


if __name__ == "__main__":
    analyze()