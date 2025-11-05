"""
Diffusion probabilistic models

A compact and hands on tutorial that builds diffusion models from scratch in PyTorch.
We follow the thermodynamic view of Sohl Dickstein et al. [2] and the denoising formulation of Ho et al. [1].
Everything runs on two dimensional swiss roll points for clarity.
You can later swap the toy data with images or audio.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.optim as optim
import os

# swiss roll data
from sklearn.datasets import make_swiss_roll

from models import MeanLogVarNet

# Reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


# Small helper to make plots look nice
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Sample swiss roll points
def sample_batch(n=10000, noise=0.5):
    x, _ = make_swiss_roll(n_samples=n, noise=noise)
    # keep x and z then rescale
    x = x[:, [0, 2]] / 10.0
    return torch.from_numpy(x).float()

fig_dir = Path("./figures")
os.makedirs(fig_dir, exist_ok=True)
data = sample_batch(20000)
plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.4)
plt.title("Swiss roll points")
plt.savefig(fig_dir / "swiss_roll.png")
plt.close()

"""
Forward diffusion process q:
We define a Markov chain that gradually adds Gaussian noise to the data.
Let b_t be a variance schedule with b_t \in (0, 1).
Define a_t = 1 - b_t and \bar{a}_t = \prod_{s=1}^t a_s.
The forward process is defined as:
q(x_t | x_{t-1}) = N(x_t; sqrt(a_t) * x_{t-1}, b_t * I)
Closed form expression for any time t:
q(x_t | x_{0}) = N[ x_t; sqrt(\bar{a}_t) * x_0, (1-\bar{a}_t) * I ]
"""


def make_beta_schedule(kind="sigmoid", T=1000, start=1e-4, end=2e-2):
    """ Create a beta schedule that goes from start to end in T steps"""
    if kind == "linear":
        betas = torch.linspace(start, end, T)
    elif kind == "quad":
        betas = torch.linspace(start**0.5, end**0.5, T) ** 2
    elif kind == "sigmoid":
        x = torch.linspace(-6, 6, T)
        betas = torch.sigmoid(x) * (end - start) + start
    else:
        raise ValueError("unknown schedule kind")
    return betas


T = 200 # keep small for speed in the tutorial
betas = make_beta_schedule("sigmoid", T=T, start=1e-4, end=2e-2)
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

def extract_timestep_values(values, tt, dims=1):
    """
    Extract values[t] for each element in the batch and reshape for broadcasting.

    Args:
        values: Tensor of shape [T] containing per-timestep values
        tt: Tensor of shape [N] with time step indices for each batch element
        x: Tensor to match shape for broadcasting (e.g. data or model input)

    Returns:
        Tensor of shape [N, 1, ..., 1] broadcastable to x
    """
    assert values.ndim == 1, f"values must be 1D, got shape {values.shape}"
    assert tt.ndim == 1, f"tt must be 1D, got shape {tt.shape}"
    assert tt.max() < values.size(0), "time index out of range"
    assert tt.min() >= 0, "negative time index"
    out = values[tt] # shape [N]
    return out.view(-1, *([1] * (dims - 1)))



@torch.no_grad()
def q_sample(x0, tt, noise=None):
    # # q(x_t | x_{0}) = N[ x_t; sqrt(\bar{a}_t) * x_0, (1-\bar{a}_t) * I ]
    # x0 size [20k, 2]
    # t size [20k, 2]
    if noise is None:
        noise = torch.randn_like(x0)
    alphas_bar_t = extract_timestep_values(alphas_bar, tt, dims=x0.ndim)
    one_minus_bar_a_t = 1. - alphas_bar_t
    sqrt_bar_a_t = alphas_bar_t.sqrt()
    sqrt_one_minus_bar_a_t = one_minus_bar_a_t.sqrt()
    # print(x0.size())
    # print(sqrt_bar_a_t.size())
    # print(sqrt_one_minus_bar_a_t.size())
    # print(noise.size())
    return x0 * sqrt_bar_a_t + sqrt_one_minus_bar_a_t * noise

T = 100
batch_size = data.size(0)
tt = torch.ones((batch_size,)) * T
tt = tt.long()
print(tt.size())
print(data.size())


def plot_forward_diffusion(x0, steps=10):
    idxs = torch.linspace(0, T - 1, steps).long()
    fig, axs = plt.subplots(1, steps, figsize=(3*steps, 3))
    # x0 is the initial state of the data
    for i, t_i in enumerate(idxs):
        tt = torch.full((x0.shape[0],), t_i)
        xt = q_sample(x0, tt=tt)
        axs[i].scatter(xt[:, 0], xt[:, 1], s=3)
        axs[i].set_title(f"t={int(t_i)}")
        axs[i].set_axis_off()

    plt.suptitle("Forward diffusion process q")
    plt.savefig(fig_dir / f"forward_diffusion_{steps}.png")
    plt.close()

"""
In order to train the model, we need to define the reverse process p:
q(x_{t-1} | x_{t}, x_0) = N( mu_q(x_t, x_0, t), sigma_q^2(t) * I )
with closed forms:
mu_q(x_t, x_0, t) = 
sqrt(a_t) * (1 - \bar{a}_{t-1}) / (1 - \bar{a}_t) * x_t
+ sqrt(\bar{a}_{t-1}) * b_t / (1 - \bar{a}_t) * x_0
sigma_q^2(t) = 
b_t * (1 - \bar{a}_{t-1}) / (1 - \bar{a}_{t})
"""

@torch.no_grad()
def q_posterior_mean_variance(x0, xt, tt):
    assert x0.size() == xt.size()
    batch_size = x0.size(0)
    assert tt.size()[0] == batch_size
    alphas_bar_t = extract_timestep_values(alphas_bar, tt, dims=x0.ndim)
    alphas_bar_t_prev = extract_timestep_values(alphas_bar, tt - 1, dims=x0.ndim)
    betas_t = extract_timestep_values(betas, tt, dims=x0.ndim)
    a_t = extract_timestep_values(alphas, tt, dims=x0.ndim)
    mu_q = a_t.sqrt() * (1 - alphas_bar_t_prev) / (1 - alphas_bar_t) * xt + \
        alphas_bar_t_prev.sqrt() * betas_t / (1 - alphas_bar_t) * x0
    sigma2_q = betas_t * (1 - alphas_bar_t_prev) / (1 - alphas_bar_t)
    sigma2_q = sigma2_q.clamp(min=1e-20)
    return mu_q, sigma2_q

def extract(a, t, x):
    """Extract a[t] and reshape to x shape for broadcasting."""
    out = a.gather(0, t)
    return out.view(-1, *([1] * (x.dim() - 1)))

def reference_q_posterior_mean_variance(x0, xt, t):
    beta_t = extract(betas, t, xt)
    alpha_t = extract(alphas, t, xt)
    alpha_bar_t = extract(alphas_bar, t, xt)
    alpha_bar_tm1 = extract(torch.cat([torch.tensor([1.0]), alphas_bar[:-1]]), t, xt)
    coef1 = (beta_t * alpha_bar_tm1.sqrt()) / (1.0 - alpha_bar_t)
    coef2 = ((1.0 - alpha_bar_tm1) * alpha_t.sqrt()) / (1.0 - alpha_bar_t)
    mean = coef1 * x0 + coef2 * xt
    var = ((1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t)) * beta_t
    log_var = var.clamp(min=1e-20).log()
    return mean, log_var

steps = 10
x0 = data
# test the q_posterior_mean_variance function
idxs = torch.linspace(0, T - 1, steps).long()[1:]  # skip t=0
# x0 is the initial state of the data
for i, t_i in enumerate(idxs):
    tt = torch.full((x0.shape[0],), t_i)
    xt = q_sample(x0, tt=tt)
    mu_q, sigma2_q = q_posterior_mean_variance(x0, xt, tt)
    print("mu_q.size(), sigma2_q.size()")
    print(mu_q.size(), sigma2_q.size())

    mu_q, sigma2_q = reference_q_posterior_mean_variance(x0, xt, tt)
    print("REFERENCE mu_q.size(), sigma2_q.size()")
    print(mu_q.size(), sigma2_q.size())

    # assert close
    assert torch.allclose(mu_q, mu_q, atol=1e-5)
    assert torch.allclose(sigma2_q, sigma2_q, atol=1e-5)


"""
Reverse diffusion process p
We want a parametric chain that inverts the forward chain
p(x_{t-1} | x_{t}) = N( mu_p(x_t, t), sigma_p^2(t) * I )
with learned mean mu_p and fixed variance schedule sigma_p^2(t).
A simple time conditioned MLP is enough for this case.
We will first implement the original mean and variance parameterization, then switch to the noise prediction form used in DDPM since it simplifies learning.
"""

net_mv = MeanLogVarNet(T, x0.size(-1), hidden_dim=128)
optimizer = optim.Adam(net_mv.parameters(), lr=1e-3)

# sample one reverse step
@torch.no_grad()
def p_sample_mv(model, xt, tt):
    mu, logvar = model(xt, tt)
    noise = torch.randn_like(xt)
    # std = exp(0.5 * logvar)
    xtm1 = mu + (0.5 * logvar).exp() * noise
    return xtm1






"""
References:
[1] Ho, J., Jain, A., Abbeel, P. Denoising diffusion probabilistic models. arXiv 2006.11239.
[2] Sohl Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. arXiv 1503.03585.
[4] Song, J., Meng, C., Ermon, S. Denoising Diffusion Implicit Models. arXiv 2010.02502.
[5] Chen, N., Zhang, Y., Zen, H., Weiss, R., Norouzi, M., Chan, W. WaveGrad. arXiv 2009.00713.
"""
