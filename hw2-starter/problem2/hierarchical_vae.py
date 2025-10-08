"""
Hierarchical VAE for drum patterns (16x9):
  - Two latents: z_high (style/global), z_low (detail/local)
  - Encoders output (mu, logvar), decoder outputs logits
"""

import torch
import torch.nn as nn


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16 * 9, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

    def forward(self, x):  # x: [B,16,9]
        h = self.net(x.view(x.size(0), -1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_high_dim, z_low_dim, hidden=256):
        super().__init__()
        in_dim = z_high_dim + z_low_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 16 * 9),  # logits
        )

    def forward(self, z_high, z_low):
        z = torch.cat([z_high, z_low], dim=1)
        logits = self.net(z)
        return logits.view(-1, 16, 9)


class HierarchicalVAE(nn.Module):
    def __init__(self, z_high_dim=8, z_low_dim=16, hidden=256):
        super().__init__()
        self.z_high_dim = z_high_dim
        self.z_low_dim = z_low_dim
        self.enc_high = Encoder(z_high_dim, hidden)
        self.enc_low = Encoder(z_low_dim, hidden)
        self.dec = Decoder(z_high_dim, z_low_dim, hidden)

    def encode(self, x):
        mu_h, logvar_h = self.enc_high(x)
        mu_l, logvar_l = self.enc_low(x)
        return mu_h, logvar_h, mu_l, logvar_l

    def decode(self, z_high, z_low):
        return self.dec(z_high, z_low)

    def forward(self, x):
        mu_h, lv_h, mu_l, lv_l = self.encode(x)
        z_h = reparameterize(mu_h, lv_h)
        z_l = reparameterize(mu_l, lv_l)
        logits = self.decode(z_h, z_l)
        return logits, (mu_h, lv_h, mu_l, lv_l), (z_h, z_l)