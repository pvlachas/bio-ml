"""
Diagnostic tools for monitoring diffusion model training.

Provides metrics and visualizations to detect issues like:
- Mode collapse
- Variance collapse
- Gradient issues
- Per-timestep learning difficulties
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class TrainingMonitor:
    """Track training metrics over time."""

    def __init__(self):
        self.history = {
            'epoch': [],
            'loss': [],
            'lr': [],
            'grad_norm': [],
            'mean_logvar': [],
            'std_logvar': [],
            'kl_per_timestep': [],
        }

    def update(self, epoch, loss, lr, grad_norm=None, mean_logvar=None, std_logvar=None):
        """Update training history."""
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['lr'].append(lr)
        if grad_norm is not None:
            self.history['grad_norm'].append(grad_norm)
        if mean_logvar is not None:
            self.history['mean_logvar'].append(mean_logvar)
        if std_logvar is not None:
            self.history['std_logvar'].append(std_logvar)

    def plot_training_curves(self, fig_dir, window=50):
        """Plot training metrics over time."""
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        epochs = self.history['epoch']

        # Loss curve
        axs[0, 0].plot(epochs, self.history['loss'], alpha=0.3, label='Raw')
        if len(self.history['loss']) > window:
            smoothed = np.convolve(self.history['loss'],
                                  np.ones(window)/window, mode='valid')
            axs[0, 0].plot(epochs[window-1:], smoothed, label=f'Smoothed ({window})')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].legend()
        axs[0, 0].set_yscale('log')

        # Learning rate
        axs[0, 1].plot(epochs, self.history['lr'])
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Learning Rate')
        axs[0, 1].set_title('Learning Rate Schedule')

        # Gradient norm
        if self.history['grad_norm']:
            axs[1, 0].plot(epochs, self.history['grad_norm'], alpha=0.5)
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].set_ylabel('Gradient Norm')
            axs[1, 0].set_title('Gradient Magnitude')
            axs[1, 0].set_yscale('log')

        # Log-variance statistics
        if self.history['mean_logvar']:
            axs[1, 1].plot(epochs, self.history['mean_logvar'], label='Mean')
            axs[1, 1].plot(epochs, self.history['std_logvar'], label='Std', alpha=0.7)
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('Log-Variance')
            axs[1, 1].set_title('Model Log-Variance Statistics')
            axs[1, 1].legend()
            axs[1, 1].axhline(y=-20, color='r', linestyle='--', alpha=0.3, label='Clamp min')
            axs[1, 1].axhline(y=2, color='r', linestyle='--', alpha=0.3, label='Clamp max')

        plt.tight_layout()
        plt.savefig(fig_dir / "training_curves.png", dpi=150)
        plt.close()


@torch.no_grad()
def compute_sample_diversity_metrics(samples):
    """
    Compute diversity metrics for generated samples.

    Returns
    -------
    dict with:
        - mean: Sample mean
        - std: Sample standard deviation
        - pairwise_dist_mean: Average pairwise distance (detects collapse)
        - pairwise_dist_std: Std of pairwise distances
    """
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)

    # Pairwise distances (sample subset for efficiency)
    n_subset = min(500, len(samples))
    indices = torch.randperm(len(samples))[:n_subset]
    subset = samples[indices]

    # Compute pairwise L2 distances
    dists = torch.cdist(subset, subset, p=2)
    # Take upper triangle (excluding diagonal)
    triu_indices = torch.triu_indices(n_subset, n_subset, offset=1)
    pairwise_dists = dists[triu_indices[0], triu_indices[1]]

    return {
        'mean': mean.cpu().numpy(),
        'std': std.cpu().numpy(),
        'pairwise_dist_mean': pairwise_dists.mean().item(),
        'pairwise_dist_std': pairwise_dists.std().item(),
    }


@torch.no_grad()
def plot_sample_quality_diagnostics(
    real_data,
    generated_samples,
    epoch,
    fig_dir,
):
    """
    Comprehensive visualization of sample quality.

    Includes:
    - Side-by-side comparison of real vs generated
    - Marginal distributions
    - Pairwise distance histograms
    """
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    real_np = real_data.cpu().numpy()
    gen_np = generated_samples.cpu().numpy()

    # 1. Real data scatter
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.scatter(real_np[:, 0], real_np[:, 1], s=2, alpha=0.5, c='blue')
    ax1.set_title('Real Data')
    ax1.set_xlabel('x₀')
    ax1.set_ylabel('x₁')
    ax1.set_aspect('equal')

    # 2. Generated data scatter
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.scatter(gen_np[:, 0], gen_np[:, 1], s=2, alpha=0.5, c='orange')
    ax2.set_title(f'Generated (Epoch {epoch})')
    ax2.set_xlabel('x₀')
    ax2.set_ylabel('x₁')
    ax2.set_aspect('equal')

    # Match axis limits
    all_data = np.concatenate([real_np, gen_np], axis=0)
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min)
    ax1.set_xlim(x_min - margin, x_max + margin)
    ax1.set_ylim(y_min - margin, y_max + margin)
    ax2.set_xlim(x_min - margin, x_max + margin)
    ax2.set_ylim(y_min - margin, y_max + margin)

    # 3. Marginal distribution - dimension 0
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(real_np[:, 0], bins=50, alpha=0.5, label='Real', density=True, color='blue')
    ax3.hist(gen_np[:, 0], bins=50, alpha=0.5, label='Generated', density=True, color='orange')
    ax3.set_xlabel('x₀')
    ax3.set_ylabel('Density')
    ax3.set_title('Marginal Distribution (x₀)')
    ax3.legend()

    # 4. Marginal distribution - dimension 1
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(real_np[:, 1], bins=50, alpha=0.5, label='Real', density=True, color='blue')
    ax4.hist(gen_np[:, 1], bins=50, alpha=0.5, label='Generated', density=True, color='orange')
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('Density')
    ax4.set_title('Marginal Distribution (x₁)')
    ax4.legend()

    plt.savefig(fig_dir / f"epoch_{epoch}_sample_quality.png", dpi=150,
                bbox_inches='tight')
    plt.close()

    # Compute and print metrics
    metrics = compute_sample_diversity_metrics(generated_samples)
    print(f"\n  Sample Quality Metrics:")
    print(f"    Mean: [{metrics['mean'][0]:.3f}, {metrics['mean'][1]:.3f}]")
    print(f"    Std:  [{metrics['std'][0]:.3f}, {metrics['std'][1]:.3f}]")
    print(f"    Avg pairwise distance: {metrics['pairwise_dist_mean']:.3f} ± {metrics['pairwise_dist_std']:.3f}")

    return metrics


@torch.no_grad()
def plot_model_predictions_analysis(
    model,
    x_t,
    t,
    epoch,
    fig_dir,
    n_samples,
    mean_variance_fn=None,
):
    """
    Analyze model predictions at different timesteps.

    Visualizes:
    - Predicted means
    - Predicted log-variances
    - Distribution of predictions

    Parameters
    ----------
    mean_variance_fn : callable, optional
        Custom function to extract mean and variance from model.
        Should have signature: fn(model, x, t) -> (mean, logvar)
        If None, uses default denoising_model_mean_variance
    """
    if mean_variance_fn is None:
        from sampling import denoising_model_mean_variance
        mean_variance_fn = denoising_model_mean_variance

    # Sample a subset for visualization
    indices = torch.randperm(len(x_t))[:n_samples]
    x_subset = x_t[indices]
    t_subset = t[indices]

    # Get model predictions
    pred_mean, pred_logvar = mean_variance_fn(model, x_subset, t_subset)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # 1. Input points colored by timestep with varying alpha
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Normalize timesteps to [0, 1] for colormap
    norm = mcolors.Normalize(vmin=t_subset.min().item(), vmax=t_subset.max().item())
    # Red to white colormap: red for low timesteps, white for high timesteps
    cmap = cm.get_cmap('Reds_r')

    # Get RGBA colors from colormap
    colors = cmap(norm(t_subset.cpu().numpy()))

    # Modify alpha channel based on timestep (higher t = more opaque)
    alpha_values = norm(t_subset.cpu().numpy())  # Normalized to [0, 1]
    alpha_values = 0.3 + 0.6 * alpha_values  # Scale to [0.3, 0.9]
    colors[:, 3] = alpha_values

    scatter = axs[0, 0].scatter(
        x_subset[:, 0].cpu(),
        x_subset[:, 1].cpu(),
        c=colors,
        s=10,
    )
    axs[0, 0].set_title('Input x_t (colored by timestep, alpha by t)')
    axs[0, 0].set_xlabel('x₀')
    axs[0, 0].set_ylabel('x₁')

    # Create a ScalarMappable for colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axs[0, 0], label='timestep')

    # 2. Predicted means with varying alpha
    colors_pred = cmap(norm(t_subset.cpu().numpy()))
    colors_pred[:, 3] = alpha_values

    scatter = axs[0, 1].scatter(
        pred_mean[:, 0].cpu(),
        pred_mean[:, 1].cpu(),
        c=colors_pred,
        s=10,
    )
    axs[0, 1].set_title('Predicted Mean (alpha by timestep)')
    axs[0, 1].set_xlabel('μ₀')
    axs[0, 1].set_ylabel('μ₁')
    plt.colorbar(sm, ax=axs[0, 1], label='timestep')

    # 3. Log-variance distribution (dimension 0)
    axs[0, 2].hist(pred_logvar[:, 0].cpu().numpy(), bins=50, alpha=0.7)
    axs[0, 2].axvline(-20, color='r', linestyle='--', label='Clamp min')
    axs[0, 2].axvline(2, color='r', linestyle='--', label='Clamp max')
    axs[0, 2].set_title('Predicted Log-Var (dim 0)')
    axs[0, 2].set_xlabel('log σ²')
    axs[0, 2].legend()

    # 4. Log-variance distribution (dimension 1)
    axs[1, 0].hist(pred_logvar[:, 1].cpu().numpy(), bins=50, alpha=0.7)
    axs[1, 0].axvline(-20, color='r', linestyle='--', label='Clamp min')
    axs[1, 0].axvline(2, color='r', linestyle='--', label='Clamp max')
    axs[1, 0].set_title('Predicted Log-Var (dim 1)')
    axs[1, 0].set_xlabel('log σ²')
    axs[1, 0].legend()

    # 5. Log-variance vs timestep
    axs[1, 1].scatter(t_subset.cpu(), pred_logvar[:, 0].cpu(), s=5, alpha=0.3, label='dim 0')
    axs[1, 1].scatter(t_subset.cpu(), pred_logvar[:, 1].cpu(), s=5, alpha=0.3, label='dim 1')
    axs[1, 1].axhline(-20, color='r', linestyle='--', alpha=0.3)
    axs[1, 1].axhline(2, color='r', linestyle='--', alpha=0.3)
    axs[1, 1].set_xlabel('Timestep t')
    axs[1, 1].set_ylabel('Predicted log σ²')
    axs[1, 1].set_title('Log-Variance vs Timestep')
    axs[1, 1].legend()

    # 6. Mean magnitude vs timestep
    mean_norm = torch.norm(pred_mean, dim=1)
    axs[1, 2].scatter(t_subset.cpu(), mean_norm.cpu(), s=5, alpha=0.3)
    axs[1, 2].set_xlabel('Timestep t')
    axs[1, 2].set_ylabel('||μ||₂')
    axs[1, 2].set_title('Predicted Mean Magnitude vs Timestep')

    plt.suptitle(f'Model Predictions Analysis (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / f"epoch_{epoch}_model_predictions_epoch.png", dpi=150)
    plt.close()

    # Print statistics
    print(f"\n  Model Prediction Statistics:")
    print(f"    Log-var range: [{pred_logvar.min():.2f}, {pred_logvar.max():.2f}]")
    print(f"    Log-var mean: {pred_logvar.mean():.2f} ± {pred_logvar.std():.2f}")
    print(f"    Mean norm: {mean_norm.mean():.3f} ± {mean_norm.std():.3f}")


@torch.no_grad()
def analyze_loss_per_timestep(
    model,
    data,
    n_steps,
    device,
    forward_posterior_fn,
    q_sample_fn,
    fig_dir,
    epoch,
    n_samples=1000,
    mean_variance_fn=None,
):
    """
    Compute and visualize loss for each timestep.

    This helps identify which timesteps are difficult to learn.

    Parameters
    ----------
    mean_variance_fn : callable, optional
        Custom function to extract mean and variance from model.
        Should have signature: fn(model, x, t) -> (mean, logvar)
        If None, uses default denoising_model_mean_variance
    """
    if mean_variance_fn is None:
        from sampling import denoising_model_mean_variance
        mean_variance_fn = denoising_model_mean_variance

    from loss import normal_kl

    # Sample data
    indices = torch.randperm(len(data))[:n_samples]
    x0 = data[indices]

    losses_per_t = []

    for t_val in range(n_steps):
        t = torch.full((len(x0),), t_val, device=device, dtype=torch.long)

        # Forward diffusion
        x_t = q_sample_fn(x0, t)

        # Get true and predicted distributions
        true_mean, true_logvar = forward_posterior_fn(x0, x_t, t)
        model_mean, model_logvar = mean_variance_fn(model, x_t, t)

        # Clamp as in training
        model_logvar_clamped = torch.clamp(model_logvar, min=-20, max=2)

        # Compute KL
        kl = normal_kl(true_mean, true_logvar, model_mean, model_logvar_clamped)
        loss_t = kl.mean().item()
        losses_per_t.append(loss_t)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    timesteps = np.arange(n_steps)

    # Linear scale
    axs[0].plot(timesteps, losses_per_t)
    axs[0].set_xlabel('Timestep t')
    axs[0].set_ylabel('KL Divergence')
    axs[0].set_title(f'Loss per Timestep (Epoch {epoch})')
    axs[0].grid(alpha=0.3)

    # Log scale
    axs[1].plot(timesteps, losses_per_t)
    axs[1].set_xlabel('Timestep t')
    axs[1].set_ylabel('KL Divergence (log scale)')
    axs[1].set_title(f'Loss per Timestep - Log Scale (Epoch {epoch})')
    axs[1].set_yscale('log')
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / f"epoch_{epoch}_loss_per_timestep_epoch.png", dpi=150)
    plt.close()

    # Print statistics
    losses_arr = np.array(losses_per_t)
    worst_timesteps = np.argsort(losses_arr)[-10:][::-1]

    print(f"\n  Loss per Timestep Analysis:")
    print(f"    Mean loss: {losses_arr.mean():.4f}")
    print(f"    Max loss: {losses_arr.max():.4f} at t={losses_arr.argmax()}")
    print(f"    Top 10 worst timesteps: {worst_timesteps.tolist()}")

    return losses_per_t