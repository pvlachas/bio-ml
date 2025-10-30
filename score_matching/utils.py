import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd

def get_device():
    # check device availability
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    return device

def ensure_fig_dir(path="./figures"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

"""
Next, we need to define the loss function for the score matching objective.
First to compute the Jacobian, we need a specific (and differentiable) function.
(This efficient implementation is based on a discussion that can be found here)
"""
def jacobian(f, x):
    """Computes the Jacobian of f w.r.t x.
    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """
    B, N = x.shape
    y = f(x)
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = autograd.grad(
            y, x,
            grad_outputs=v,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=2).requires_grad_()
    return jacobian

def plot_gradients(model, data, device, fpath):
    ensure_fig_dir()
    xgrid = np.linspace(-1.5, 2.0, 50)
    ygrid = np.linspace(-1.5, 2.0, 50)
    xx = np.stack(np.meshgrid(xgrid, ygrid), axis=-1).reshape(-1, 2)
    scores = model(torch.tensor(xx).float().to(device)).cpu().detach().numpy()
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    # Perform the plots
    plt.figure(figsize=(16, 12))
    x, y = (data.T if data.shape[1] == 2 else data)
    plt.scatter(x, y, alpha=0.3, color='red', edgecolor='white', s=40)
    plt.quiver(*xx.T, *scores_log1p.T, width=0.002, color='white')
    plt.xlim(-1.5, 2.0)
    plt.ylim(-1.5, 2.0)
    plt.savefig(fpath)
    plt.close()

# ---- Modular helpers ----
import numpy as np
import matplotlib.pyplot as plt
import torch

def run_sampling(model, x0, device, fn_sample):
    """Generate samples from a starting point"""
    x = torch.Tensor(x0).to(device)
    samples = fn_sample(model, x)
    return samples.cpu().numpy()

def plot_trajectory(samples, color):
    """Plot one trajectory with arrows"""
    plt.scatter(samples[:, 0], samples[:, 1], color=color, edgecolor='white', s=150)
    # Draw arrows between steps
    deltas = samples[1:] - samples[:-1]
    deltas = deltas - deltas / np.linalg.norm(deltas, keepdims=True, axis=-1) * 0.04
    for i, arrow in enumerate(deltas):
        plt.arrow(
            samples[i,0],
            samples[i,1],
            arrow[0],
            arrow[1],
            width=1e-4,
            head_width=2e-2,
            color=color,
            linewidth=3,
        )

def plot_all_trajectories(
    model,
    data,
    starting_points,
    device,
    figname,
    fn_run_sampling,
):
    """Plot gradients and multiple trajectories"""
    ensure_fig_dir()
    # Plot gradient field first
    plot_gradients(model, data, device, figname.replace(".png", "_gradients_tmp.png"))
    # Overlay trajectories on the same field
    # Recreate the gradient field to keep consistent background:
    xgrid = np.linspace(-1.5, 2.0, 50)
    ygrid = np.linspace(-1.5, 2.0, 50)
    xx = np.stack(np.meshgrid(xgrid, ygrid), axis=-1).reshape(-1, 2)
    scores = model(torch.tensor(xx).float().to(device)).cpu().detach().numpy()
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    plt.figure(figsize=(16, 12))
    x, y = (data.T if data.shape[1] == 2 else data)
    plt.scatter(x, y, alpha=0.3, color='red', edgecolor='white', s=40)
    plt.quiver(*xx.T, *scores_log1p.T, width=0.002, color='white')
    plt.xlim(-1.5, 2.0)
    plt.ylim(-1.5, 2.0)

    colors = ['green', 'blue', 'red', 'purple', 'orange']
    for i, x0 in enumerate(starting_points):
        samples = run_sampling(model, x0, device, fn_run_sampling)
        plot_trajectory(samples, colors[i % len(colors)])

    plt.savefig(figname)
    plt.close()
