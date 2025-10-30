"""
The idea of score matching was originally proposed by Hyvarinen et al.
Instead of learning directly the probability of the data log p(x),
we rather aim to learn the gradients of log p(x) with respect to x.
In this case, the gradients Nabla_x log p(x)
are termed the score of the density, hence the name score matching.
This can be understood as learning the direction of highest probability at each point in the input space.
Therefore, when the model is trained, we can improve a sample x
by moving it along the directions of highest probability.
"""

"""
In order to learn the score of the data distribution,
we can use a neural network s_theta(x) with parameters theta
that takes as input a data point x and outputs the score Nabla_x log p(x).
To train the neural network, we can minimize the expected squared error between the true score and the estimated score:
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data import sample_batch, plot_swiss_roll
from models import ScoreNet
from losses import score_matching
from utils import get_device, ensure_fig_dir, plot_all_trajectories, plot_gradients
from sampling import sample_simple

def main():
    ensure_fig_dir()
    plot_swiss_roll()

    # Hyperparameters
    input_dim = 2
    hidden_dim = 128
    learning_rate = 1e-3
    num_epochs = 2000
    batch_size = 256

    # Generate training data
    data = sample_batch(10**4)
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Initialize the model and optimizer
    model = ScoreNet(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = get_device()
    model.to(device)
    data_tensor = data_tensor.to(device)

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for t in range(num_epochs):
        # Compute the loss.
        loss = score_matching(model, data_tensor)
        # Before the backward pass, zero all of the network gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        optimizer.step()
        print(f"Epoch {t}: loss = {loss.item():.4f}")

    # Visualizations
    plot_gradients(model, data, device, "./figures/gradients.png")

    # Simple gradient ascent sampling from one point
    x0 = torch.Tensor([1.5, -1.5]).to(device)
    from utils import run_sampling, plot_trajectory
    samples = run_sampling(model, x0, device, sample_simple)

    # Plot gradients and the above trajectory
    # Reuse the multi-trajectory helper for consistency
    starting_points = [
        [1.5, -1.5],
        [-1.5, 1.5],
        [1.2, 1.2],
        [-1.0, -1.0],
        [0.5, -1.2],
    ]
    plot_all_trajectories(
        model,
        data,
        starting_points,
        device,
        "./figures/dynamics_simple_all.png",
        fn_run_sampling=sample_simple,
    )

if __name__ == "__main__":
    main()
