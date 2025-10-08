"""
Loss utilities and annealing schedules for Hierarchical VAE.
"""

import torch
import torch.nn.functional as F


def kl_standard_normal(mu, logvar):
    # per-sample KL reduced over latent dims
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def kl_anneal_linear(step, total_steps, start=0.0, end=1.0):
    t = min(max(step / float(total_steps), 0.0), 1.0)
    return start + (end - start) * t


def kl_anneal_cyclical(step, total_steps, cycles=4, ratio=0.5):
    cycle_len = total_steps / float(cycles)
    step_in_cycle = step % cycle_len
    ramp = ratio * cycle_len
    if step_in_cycle <= ramp:
        return step_in_cycle / ramp
    return 1.0


def compute_hierarchical_elbo(x, recon_logits,
                              mu_h, logvar_h, mu_l, logvar_l,
                              beta_h=1.0, beta_l=1.0,
                              reduction="mean"):
    """
    Negative ELBO (minimize):
      recon_bce + beta_h * KL(z_h) + beta_l * KL(z_l)
    """
    recon_bce = F.binary_cross_entropy_with_logits(
        recon_logits, x, reduction="none"
    ).sum(dim=(1, 2))  # per-sample sum over 16*9

    kl_h = kl_standard_normal(mu_h, logvar_h)  # per-sample
    kl_l = kl_standard_normal(mu_l, logvar_l)

    loss = recon_bce + beta_h * kl_h + beta_l * kl_l

    if reduction == "mean":
        return loss.mean(), recon_bce.mean(), kl_h.mean(), kl_l.mean()
    elif reduction == "sum":
        return loss.sum(), recon_bce.sum(), kl_h.sum(), kl_l.sum()
    else:
        return loss, recon_bce, kl_h, kl_l