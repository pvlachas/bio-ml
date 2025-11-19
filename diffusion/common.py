"""
Common utilities for DDPM experiments.

This module contains shared functionality used across all experiment types:
- Diffusion schedule setup
- Forward diffusion process
- Posterior computations
- Checkpoint management
- Training utilities
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from utils import make_beta_schedule, extract


def setup_seed(seed=0):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_diffusion_schedule(n_steps, beta_start, beta_end, device):
    """
    Set up diffusion schedule and precompute all necessary coefficients.

    Parameters
    ----------
    n_steps : int
        Number of diffusion timesteps
    beta_start : float
        Starting beta value
    beta_end : float
        Ending beta value
    device : torch.device
        Device to store tensors on

    Returns
    -------
    dict
        Dictionary containing all diffusion coefficients
    """
    # Diffusion schedule
    betas = make_beta_schedule(
        schedule='linear',
        n_steps=n_steps,
        start=beta_start,
        end=beta_end,
        device=device,
    )

    # Precompute coefficients
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_prev = torch.cat(
        [torch.tensor([1.]).to(device), alphas_prod[:-1]], dim=0
    )
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    # Posterior coefficients (for training)
    posterior_mean_coef_x0 = (
        betas * torch.sqrt(alphas_prod_prev) / (1.0 - alphas_prod)
    )
    posterior_mean_coef_xt = (
        (1.0 - alphas_prod_prev) * torch.sqrt(alphas) / (1.0 - alphas_prod)
    )
    posterior_variance = (
        betas * (1.0 - alphas_prod_prev) / (1.0 - alphas_prod)
    )

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_prod': alphas_prod,
        'alphas_prod_prev': alphas_prod_prev,
        'alphas_bar_sqrt': alphas_bar_sqrt,
        'one_minus_alphas_bar_log': one_minus_alphas_bar_log,
        'one_minus_alphas_bar_sqrt': one_minus_alphas_bar_sqrt,
        'posterior_mean_coef_x0': posterior_mean_coef_x0,
        'posterior_mean_coef_xt': posterior_mean_coef_xt,
        'posterior_variance': posterior_variance,
    }


def create_q_sample_fn(alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    """
    Create forward diffusion sampling function q(x_t | x_0).

    Parameters
    ----------
    alphas_bar_sqrt : torch.Tensor
        Precomputed sqrt(alpha_bar_t)
    one_minus_alphas_bar_sqrt : torch.Tensor
        Precomputed sqrt(1 - alpha_bar_t)

    Returns
    -------
    callable
        Function that samples from q(x_t | x_0)
    """
    @torch.no_grad()
    def q_sample(x0, t, noise=None):
        """
        Sample x_t from q(x_t | x0) = N(sqrt(ᾱ_t) x0, (1-ᾱ_t) I)
        using the closed-form expression.
        """
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)
        coef_1 = extract(alphas_bar_sqrt, t, x0)
        coef_2 = extract(one_minus_alphas_bar_sqrt, t, x0)
        return coef_1 * x0 + coef_2 * noise

    return q_sample


def create_forward_posterior_fn(posterior_mean_coef_x0, posterior_mean_coef_xt, posterior_variance):
    """
    Create forward posterior distribution function q(x_{t-1} | x_t, x_0).

    Parameters
    ----------
    posterior_mean_coef_x0 : torch.Tensor
        Coefficient for x_0 in posterior mean
    posterior_mean_coef_xt : torch.Tensor
        Coefficient for x_t in posterior mean
    posterior_variance : torch.Tensor
        Posterior variance schedule

    Returns
    -------
    callable
        Function that computes posterior mean and log-variance
    """
    def forward_posterior_mean_logvar(x0, xt, t):
        """
        Compute the mean and variance of the forward posterior q(x_{t-1} | x_t, x0).
        """
        coef1 = extract(posterior_mean_coef_x0, t, x0)
        coef2 = extract(posterior_mean_coef_xt, t, x0)
        mean = coef1 * x0 + coef2 * xt
        var = extract(posterior_variance, t, x0)
        # Get the log variance
        var = torch.clamp(var, min=1e-20)
        logvar = torch.log(var)
        return mean, logvar

    return forward_posterior_mean_logvar


def get_lr_scheduler_fn(warmup_iters):
    """
    Create learning rate scheduler function with warmup.

    Parameters
    ----------
    warmup_iters : int
        Number of warmup iterations

    Returns
    -------
    callable
        Function that returns LR multiplier for given epoch
    """
    def get_lr_multiplier(epoch):
        """Linear warmup then constant learning rate"""
        if epoch < warmup_iters:
            return (epoch + 1) / warmup_iters
        return 1.0

    return get_lr_multiplier


def save_checkpoint(epoch, model, optimizer, ema, monitor, checkpoint_dir, experiment_name=""):
    """
    Save training checkpoint.

    Parameters
    ----------
    epoch : int
        Current epoch number
    model : nn.Module
        Model to save
    optimizer : Optimizer
        Optimizer to save
    ema : EMA
        EMA object to save
    monitor : TrainingMonitor
        Training monitor to save
    checkpoint_dir : Path
        Directory to save checkpoint
    experiment_name : str, optional
        Experiment name for checkpoint filename
    """
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:05d}.pt"
    latest_path = checkpoint_dir / "checkpoint_latest.pt"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_shadow': ema.shadow,
        'monitor_history': monitor.history,
    }

    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, latest_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, ema, monitor, device):
    """
    Load training checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file
    model : nn.Module
        Model to load state into
    optimizer : Optimizer
        Optimizer to load state into
    ema : EMA
        EMA object to load state into
    monitor : TrainingMonitor
        Training monitor to load state into
    device : torch.device
        Device to load tensors to

    Returns
    -------
    int
        Epoch to resume from (checkpoint epoch + 1)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ema.shadow = checkpoint['ema_shadow']
    monitor.history = checkpoint['monitor_history']

    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return start_epoch


def get_experiment_config(experiment_name):
    """
    Get configuration for a specific experiment type.

    Parameters
    ----------
    experiment_name : str
        Name of experiment ('ddpm_mean', 'ddpm_mean_fixed_variance', 'ddpm_noise_pred')

    Returns
    -------
    dict
        Configuration dictionary with model class, loss function, etc.
    """
    if experiment_name == 'ddpm_mean':
        from models import ConditionalModel
        from loss import loss_likelihood_bound
        from sampling import denoising_model_mean_variance, denoising_process_sampling_trajectory

        return {
            'model_class': ConditionalModel,
            'loss_fn': loss_likelihood_bound,
            'denoising_fn': denoising_model_mean_variance,
            'sampling_fn': denoising_process_sampling_trajectory,
            'needs_fixed_variance': False,
            'display_name': 'Mean+Variance',
        }

    elif experiment_name == 'ddpm_mean_fixed_variance':
        from models import ConditionalModelMeanOnly
        from loss import loss_mse_mean_only
        from sampling import denoising_model_mean_fixed_variance, denoising_process_sampling_trajectory_fixed_variance

        return {
            'model_class': ConditionalModelMeanOnly,
            'loss_fn': loss_mse_mean_only,
            'denoising_fn': denoising_model_mean_fixed_variance,
            'sampling_fn': denoising_process_sampling_trajectory_fixed_variance,
            'needs_fixed_variance': True,
            'display_name': 'Fixed Variance',
        }

    elif experiment_name == 'ddpm_noise_pred':
        from models import ConditionalModelNoisePredictor
        from loss import loss_noise_prediction
        from sampling import denoising_process_sampling_trajectory_noise_prediction

        return {
            'model_class': ConditionalModelNoisePredictor,
            'loss_fn': loss_noise_prediction,
            'denoising_fn': None,  # Noise prediction doesn't use standard denoising fn
            'sampling_fn': denoising_process_sampling_trajectory_noise_prediction,
            'needs_fixed_variance': True,  # For sampling
            'needs_noise_schedule': True,  # Special for noise prediction
            'display_name': 'Noise Prediction',
        }

    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")
