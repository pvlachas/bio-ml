"""
Noise-Conditional Score Networks (NCSN)
--------------------------------------

Score-based generative modeling originally relied on *score matching* to estimate
the gradient of the log-density ∇ₓ log p(x). While classical score matching and
denoising score matching work well in low-dimensional settings, Song and Ermon
(2019) highlighted two fundamental issues when applying them to real datasets:

    1) **Manifold inconsistency**
       Real high-dimensional data (images, speech, audio etc.) often lie on a
       lower-dimensional manifold. Classical score matching methods assume the
       data distribution has full support in ℝⁿ, which is not true in practice.
       As a consequence, score estimates become inconsistent.

    2) **Low-density instability**
       In regions where the data density is low, the score can be poorly
       defined and Langevin sampling becomes unstable or slow to mix.

To address both issues, NCSN introduces *noise-conditional* score learning:

    • Instead of learning scores only on the empirical data distribution,
      we *corrupt the data with Gaussian noise* at multiple scales.

    • A single network learns *scores for many noise levels* simultaneously.

    • This produces a family of smoothed densities, each easier to learn.

    • During sampling, we gradually *anneal the noise from large → small*,
      similar to diffusion models, moving from broad to sharp distributions.

This idea laid the foundation for modern diffusion models (DDPMs, score SDEs).

This script implements the key idea step by step on a toy 2D Swiss Roll dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll


# -------------------------------------------------------------------
# 1. Dataset utilities
# -------------------------------------------------------------------

def sample_swiss_roll(n=10_000, noise=1.0):
    """Sample 2D Swiss Roll points for toy score-based experiments."""
    x, _ = make_swiss_roll(n, noise=noise)
    return x[:, [0, 2]] / 10.0  # keep 2 dims for visualization


data = sample_swiss_roll()
data_t = torch.tensor(data, dtype=torch.float32)

plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,1], alpha=0.4, s=10)
plt.title("Swiss Roll Training Data")
plt.show()


# -------------------------------------------------------------------
# 2. NCSN loss function
# -------------------------------------------------------------------
"""
Recall denoising score matching (Vincent 2011):
    We perturb each sample x with Gaussian noise σ,
    and the optimal score estimator satisfies:

      s*(x̃) = - 1/σ² ⋅ (x̃ - x)

NCSN generalizes this by training *one model over many noise levels*.
We sample a noise level σ_k from a geometric ladder
σ₁ > σ₂ > ... > σ_K and train:

    L = E_k E_{qσ_k(x̃|x)} [ σ_k^α * || sθ(x̃, k) + (x̃ − x)/σ_k² ||² ]

The σ_k^α term (: anneal_power) balances contributions across scales.
"""

def anneal_dsm_score_estimation(model, samples, labels, sigmas, anneal_power=2.0):
    """
    Compute the NCSN training loss (annealed denoising score matching).

    model(x, y): conditional score network
    samples:     clean data samples
    labels:      index selecting noise level σ_k
    sigmas:      tensor of noise levels
    """
    # pick σ indexed by labels
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * (samples.dim()-1)))

    # Gaussian perturbation: x̃ = x + σ z
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas

    # analytical score of the Gaussian-perturbed density
    target = - (perturbed_samples - samples) / (used_sigmas ** 2)

    # model score prediction
    scores = model(perturbed_samples, labels)

    # flatten for per-point squared error
    scores = scores.view(scores.shape[0], -1)
    target = target.view(target.shape[0], -1)

    # annealed loss: σᵅ ||s - target||²
    loss = 0.5 * ((scores - target) ** 2).sum(dim=1) * (used_sigmas.squeeze() ** anneal_power)
    return loss.mean()


# -------------------------------------------------------------------
# 3. Conditional Score Network (NCSN)
# -------------------------------------------------------------------

"""
We require the model to understand the *noise scale*. We pass an embedding of the
noise level index, and each layer multiplies activations by a learned scale.

This "FiLM"-style conditioning is essential in original NCSN.
"""

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, num_classes):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(num_classes, num_out)

        # initialize embeddings well to encourage scale diversity
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)                     # base linear
        gamma = self.embed(y)                 # noise embedding
        out = gamma.view(-1, self.num_out) * out  # FiLM-style modulation
        return out


class ConditionalModel(nn.Module):
    """
    A simple MLP conditioned on noise level index.
    In high-dim NCSN, U-Nets and dilated convnets are used instead.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.lin1 = ConditionalLinear(2, 128, num_classes)
        self.lin2 = ConditionalLinear(128, 128, num_classes)
        self.lin3 = nn.Linear(128, 2)

    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        return self.lin3(x)


# -------------------------------------------------------------------
# 4. Train the NCSN
# -------------------------------------------------------------------

# Geometric noise schedule: large → small
sigma_begin = 1.0
sigma_end   = 0.01
num_classes = 5

sigmas = torch.tensor(
    np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_classes)),
    dtype=torch.float32
)

model = ConditionalModel(num_classes)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dataset = data_t

for t in range(5000):
    labels = torch.randint(0, num_classes, (dataset.shape[0],))
    loss = anneal_dsm_score_estimation(model, dataset, labels, sigmas)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 1000 == 0:
        print(f"[step {t}] loss = {loss.item():.4f}")


# -------------------------------------------------------------------
# 5. Visualize learned scores
# -------------------------------------------------------------------
"""
We visualize the learned score field at:

    • random noise levels (mixture)
    • a specific noise level (e.g., σ₁ or σ_last)

This lets us see how the field changes across noise scales.
"""

def plot_scores(label_choice):
    xx = np.stack(np.meshgrid(np.linspace(-1.5,2.0,50),
                              np.linspace(-1.5,2.0,50)), axis=-1).reshape(-1,2)
    X = torch.tensor(xx, dtype=torch.float32)

    if label_choice == "random":
        labels = torch.randint(0, num_classes, (X.shape[0],))
    else:
        labels = torch.ones(X.shape[0]).long() * label_choice

    scores = model(X, labels).detach().numpy()
    scores_norm = np.linalg.norm(scores, axis=-1, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)

    plt.figure(figsize=(8,6))
    plt.scatter(data[:,0], data[:,1], alpha=0.3, s=10, color='red')
    plt.quiver(*xx.T, *scores_log1p.T, width=0.002)
    plt.title(f"Learned score field for noise index = {label_choice}")
    plt.xlim(-1.5, 2.0)
    plt.ylim(-1.5, 2.0)
    plt.show()


plot_scores("random")  # mixture over σ
plot_scores(0)         # highest noise
plot_scores(num_classes-1)  # smallest noise
