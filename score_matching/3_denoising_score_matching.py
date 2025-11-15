from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data import sample_batch
from losses import denoising_score_matching
from utils import get_device, ensure_fig_dir, plot_gradients

FIG_DIR = Path("./figures_denoising_score_matching")

def main():
    ensure_fig_dir(FIG_DIR)
    device = get_device()

    # Our approximation model
    model_dsm = nn.Sequential(
        nn.Linear(2, 128),
        nn.Softplus(),
        nn.Linear(128, 128),
        nn.Softplus(),
        nn.Linear(128, 2)
    )

    # Hyperparameters
    learning_rate = 1e-3
    num_epochs = 10000

    optimizer_dsm = optim.Adam(model_dsm.parameters(), lr=learning_rate)
    data = sample_batch(10**4)
    dataset = torch.tensor(data).float().to(device)
    model_dsm.to(device)

    for t in range(num_epochs):
        loss = denoising_score_matching(model_dsm, dataset)
        optimizer_dsm.zero_grad()
        loss.backward()
        optimizer_dsm.step()
        print(f"Epoch {t}: loss = {loss.item():.4f}")

    plot_gradients(model_dsm, data, device, str(FIG_DIR / "gradients_denoising_score_matching.png"))

if __name__ == "__main__":
    main()
