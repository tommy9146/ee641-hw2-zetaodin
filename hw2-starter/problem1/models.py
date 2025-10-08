"""
GAN models for font generation (28x28 grayscale).
Lightweight DCGAN-style Generator/Discriminator.
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, g_feat=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_feat*4, 7, 1, 0, bias=False),   # 7x7
            nn.BatchNorm2d(g_feat*4), nn.ReLU(True),

            nn.ConvTranspose2d(g_feat*4, g_feat*2, 4, 2, 1, bias=False),# 14x14
            nn.BatchNorm2d(g_feat*2), nn.ReLU(True),

            nn.ConvTranspose2d(g_feat*2, g_feat, 4, 2, 1, bias=False),  # 28x28
            nn.BatchNorm2d(g_feat), nn.ReLU(True),

            nn.Conv2d(g_feat, img_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        # z: [B, z_dim, 1, 1]
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, d_feat=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, d_feat, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_feat, d_feat*2, 4, 2, 1), nn.BatchNorm2d(d_feat*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_feat*2, d_feat*4, 3, 2, 1), nn.BatchNorm2d(d_feat*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_feat*4, 1, 3, 1, 1),
        )
        self.out = nn.Sigmoid()

    def forward(self, x):
        # x: [B,1,28,28]
        logits_map = self.features(x)                 # [B,1,H',W']
        logits = logits_map.mean(dim=(2,3))           # global avg -> [B,1]
        prob = self.out(logits)
        return prob, logits