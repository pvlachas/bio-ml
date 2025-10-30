import torch
import torch.nn as nn
import torch.optim as optim

from data import sample_batch
from losses import sliced_score_matching
from utils import get_device, ensure_fig_dir, plot_gradients

def main():
    ensure_fig_dir()
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

    plot_gradients(model_ssm, data, device, "./figures/gradients_sliced_score_matching.png")

if __name__ == "__main__":
    main()
