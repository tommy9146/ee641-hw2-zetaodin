EE 641 - HW2 Starter Code
=========================

Structure:
- `setup_data.py` - Generate datasets (run with --seed 641)
- `provided/` - Utility functions (metrics, visualization)
- `problem1/` - GAN skeleton code
- `problem2/` - VAE skeleton code

See assignment page for full instructions.


# EE641 Homework #2
# Name: Zetao Ding
# E-mail: zetaodin@usc.edu

## Overview
This assignment implements and analyzes two deep generative models:
1. **Problem 1:** Font Generation GAN
2. **Problem 2:** Hierarchical Variational Autoencoder (HVAE) for Drum Patterns

---

## Problem 1 – Font Generation GAN
### Objective
Train a convolutional GAN to synthesize grayscale font images (64×64).

### Architecture
- Generator: transposed CNN with instance normalization
- Discriminator: CNN with LeakyReLU activations
- Loss: Binary cross entropy (adversarial)
- Optimizer: Adam (lr=2e-4, β1=0.5, β2=0.999)

### Results
- Stable convergence after ~80 epochs
- Generated fonts preserve style and character identity
- Instance normalization improved realism and stability

---

## Problem 2 – Hierarchical VAE (HVAE)
### Objective
Model drum rhythm patterns using a two-level VAE that separates global (style) and local (rhythm) representations.

### Setup
- z_high = 8, z_low = 16
- Epochs = 100, Batch size = 128
- Loss = BCE + KL_high + KL_low
- KL annealing from 0 → 1

### Results
- Converged at train_loss ≈ 32.7, val_loss ≈ 33.8
- Clear style separation across genres (rock, jazz, hip-hop, electronic)
- Smooth interpolation in latent space

---

## Key Takeaways
- GANs produce high visual fidelity but lack latent interpretability
- VAEs offer structured latent representations but lower sharpness
- Combining hierarchical and adversarial approaches may yield balanced results

---

## Author
**Zetao Ding**  
University of Southern California  
EE641 – Deep Generative Models  
October 2025
