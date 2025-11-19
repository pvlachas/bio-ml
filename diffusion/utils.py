import shutil

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_swiss_roll


def clean_figure_directories(*dirs):
    """Remove and recreate figure directories to start fresh."""
    for dir_path in dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Removed existing directory: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS GPU:", device)
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")
    return device

def sample_batch(n=20000, noise=0.5):
    """Generate 2-D Swiss roll points and return as a torch tensor."""
    x, _ = make_swiss_roll(n_samples=n, noise=noise)
    x = x[:, [0, 2]] / 10.0        # keep only 2 dimensions & rescale
    return torch.from_numpy(x).float()


def save_model(model, path):
    torch.save(model.state_dict(), path)

def plot_samples(samples, fig_dir, iteration):
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
    plt.title("Generated samples (mean-parameterization DDPM)")
    plt.savefig(fig_dir / f"ddpm_meanvar_samples_iter_{iteration}.png")
    plt.close()


def make_beta_schedule(
    schedule='linear',
    n_steps=1000,
    start=1e-5,
    end=1e-2,
    device='cpu',
):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_steps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_steps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_steps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas.to(device)


def extract(values, t, x_like):
    """
    Utility: given a 1-D tensor 'values' of shape [T],
    extract values[t] for each batch item and reshape so it broadcasts to x_like.
    """
    out = values[t].to(x_like.device)
    return out.view(-1, *([1] * (x_like.ndim - 1)))

