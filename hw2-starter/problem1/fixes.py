"""
GAN stabilization techniques to combat mode collapse.
Here we implement ONE simple technique: label smoothing + instance noise.
"""

import torch

class Stabilizer:
    def __init__(self, total_steps, noise_init=0.1):
        self.total = max(1, int(total_steps))
        self.noise_init = float(noise_init)

    def smooth_real_labels(self, y_real):
        # Smooth real labels from 1.0 -> [0.8, 1.0]
        return torch.empty_like(y_real).uniform_(0.8, 1.0)

    def instance_noise(self, x, step):
        # Add small Gaussian noise to inputs to D; decay to 0 linearly
        sigma = self.noise_init * (1.0 - min(step, self.total) / self.total)
        if sigma <= 0:
            return x
        return x + sigma * torch.randn_like(x)