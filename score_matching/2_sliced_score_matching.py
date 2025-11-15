"""
Sliced Score Matching

The standard score matching approach requires computing the trace of the Jacobian,
tr(∇_x F(x)), which is an O(N^2 + N) operation. This becomes computationally
prohibitive for high-dimensional data and deep networks.

Sliced score matching addresses this scalability issue by using random projections
to approximate the trace computation. Instead of computing the full Jacobian matrix,
we project along random directions v ~ N(0, I) and compute directional derivatives.

The key insight is that E_v[v^T ∇_x F(x) v] = tr(∇_x F(x)), where v is a random
unit vector. This allows us to approximate the expensive trace computation with
a much cheaper vector-Jacobian product (VJP), reducing complexity to O(N).

This script demonstrates sliced score matching on a 2D Swiss roll dataset,
training a neural network to estimate the score function ∇_x log p(x).
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data import sample_batch
from losses import sliced_score_matching
from utils import get_device, ensure_fig_dir, plot_gradients

FIG_DIR = Path("./figures_sliced_score_matching")

def main():
    ensure_fig_dir(FIG_DIR)
    device = get_device()

    # Our approximation model
    model_ssm = nn.Sequential(
        nn.Linear(2, 128),
        nn.Softplus(),
        nn.Linear(128, 128),
        nn.Softplus(),
        nn.Linear(128, 2)
    )

    # Hyperparameters
    learning_rate = 1e-3
    num_epochs = 2000

    optimizer_ssm = optim.Adam(model_ssm.parameters(), lr=learning_rate)
    data = sample_batch(10**4)
    dataset = torch.tensor(data).float().to(device)
    model_ssm.to(device)

    for t in range(num_epochs):
        loss = sliced_score_matching(model_ssm, dataset)
        optimizer_ssm.zero_grad()
        loss.backward()
        optimizer_ssm.step()
        print(f"Epoch {t}: loss = {loss.item():.4f}")

    plot_gradients(model_ssm, data, device, str(FIG_DIR / "gradients_sliced_score_matching.png"))

if __name__ == "__main__":
    main()
