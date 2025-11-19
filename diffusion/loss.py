import torch
from sampling import denoising_model_mean_variance


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence KL( N(mean1, var1) || N(mean2, var2) )
    for diagonal (1-D or elementwise) Gaussian distributions,
    where logvar = log(variance).

    Parameters
    ----------
    mean1 : torch.Tensor
        Mean of the "true" or "target" distribution p(x).
        This distribution is the first argument of KL(p || q).

    logvar1 : torch.Tensor
        Log-variance of the true distribution p(x).

    mean2 : torch.Tensor
        Mean of the "model" or "approximate" distribution q(x).

    logvar2 : torch.Tensor
        Log-variance of the model distribution q(x).

    Returns
    -------
    kl : torch.Tensor
        The KL divergence KL(p || q), computed elementwise:

            KL = 0.5 * [
                log(var2/var1)
                + var1/var2
                + (mean1 - mean2)^2 / var2
                - 1
            ]

        Using log-variances avoids numerical instability and ensures
        gradients flow correctly.
    """

    # --- Convert log-variance to variance ratios efficiently ---
    # log(var2 / var1) = logvar2 - logvar1
    log_var_ratio = logvar2 - logvar1

    # var1 / var2 = exp(logvar1 - logvar2)
    var_ratio = torch.exp(logvar1 - logvar2)

    # (mean1 - mean2)^2 / var2 = (mean diff)^2 * exp(-logvar2)
    mean_diff_sq_scaled = (mean1 - mean2) ** 2 * torch.exp(-logvar2)

    # --- Full KL formula for univariate Gaussians ---
    kl = 0.5 * (
        -1.0
        + log_var_ratio
        + var_ratio
        + mean_diff_sq_scaled
    )

    return kl


def compute_loss(
    true_mean,
    true_logvar,
    model_mean,
    model_logvar,
    loss_scale_factor=1.0,
):
    """
    Compute the scaled KL divergence loss between true posterior and model prediction.

    This is the core training objective for DDPM: matching the model's predicted
    denoising distribution to the true posterior distribution q(x_{t-1} | x_t, x0).

    Parameters
    ----------
    true_mean : torch.Tensor
        Mean of the true posterior distribution q(x_{t-1} | x_t, x0)
    true_logvar : torch.Tensor
        Log-variance of the true posterior distribution
    model_mean : torch.Tensor
        Model's predicted mean for the denoising step
    model_logvar : torch.Tensor
        Model's predicted log-variance for the denoising step
    loss_scale_factor : float, optional
        Scaling factor for the loss (default: 1.0)

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value (mean KL divergence across batch)
    """
    # Clamp model log-variance to prevent KL divergence explosion
    # This matches the clamping used during sampling
    model_logvar = torch.clamp(model_logvar, min=-20, max=2)

    # Compute KL divergence between true posterior and model prediction
    KL = normal_kl(
        true_mean, true_logvar, model_mean, model_logvar
    )

    # Use configurable loss scaling instead of always multiplying by n_steps
    # This prevents extreme gradients when n_steps is large
    loss = loss_scale_factor * KL.mean()
    return loss


def loss_likelihood_bound(
    model,
    x0,
    n_steps,
    device,
    q_sample_fn,
    forward_posterior_fn,
    loss_scale_factor=1.0,
):
    """
    Compute the variational lower bound loss for DDPM training.

    This function implements the simplified training objective from the DDPM paper,
    which randomly samples a timestep t and trains the model to denoise x_t back to x_{t-1}.

    The loss measures how well the model's predicted denoising distribution matches
    the true posterior q(x_{t-1} | x_t, x0).

    Timestep Sampling Strategy:
    ---------------------------
    Instead of purely random sampling, this implements a variance reduction technique:

    1. Generate batch_size//2 + 1 random timesteps
    2. Create complementary timesteps using: n_steps - t - 1
    3. Concatenate both and take first batch_size elements

    This ensures balanced sampling across the diffusion timeline:
    - For every early timestep (small noise), we also sample a late timestep (large noise)
    - Reduces variance in gradient estimates during training
    - Improves training stability by avoiding accidental oversampling of one region

    Example:
        If batch_size=8, n_steps=1000:
        - Generate 5 random timesteps: [42, 156, 789, 23, 500]
        - Create complements: [957, 843, 210, 976, 499]
        - Final batch: [42, 156, 789, 23, 500, 957, 843, 210]
        - Result: balanced mix of early and late timesteps

    Parameters
    ----------
    model : nn.Module
        The denoising model (typically a U-Net) that predicts mean and variance
    x0 : torch.Tensor
        Clean data samples from the dataset, shape (batch_size, ...)
    n_steps : int
        Total number of diffusion timesteps
    device : torch.device
        Device to run computations on (CPU or CUDA)
    q_sample_fn : callable
        Forward diffusion function: q_sample(x0, t) -> x_t
        Samples from q(x_t | x0) using the closed-form expression
    forward_posterior_fn : callable
        Computes true posterior: forward_posterior_fn(x0, xt, t) -> (mean, logvar)
        Returns mean and log-variance of q(x_{t-1} | x_t, x0)
    loss_scale_factor : float, optional
        Scaling factor for the loss (default: 1.0)

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value for the batch
    """
    batch_size = x0.shape[0]

    # ============================================================================
    # STEP 1: Sample timesteps with variance reduction
    # ============================================================================
    # Generate slightly more than half the batch worth of random timesteps
    # The +1 handles odd batch sizes gracefully
    # NOTE: Start from 1, not 0, because t=0 has degenerate posterior variance
    # (we would be denoising x_0 -> x_{-1}, which doesn't exist)
    t = torch.randint(
        1, n_steps, size=(batch_size // 2 + 1,), device=device
    )

    # Create complementary timesteps and concatenate
    # This ensures we sample both early and late timesteps in each batch
    # Example: if t=5, complement is (n_steps-5-1) = 994 (for n_steps=1000)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()

    # ============================================================================
    # STEP 2: Forward diffusion - add noise to clean data
    # ============================================================================
    # Sample x_t ~ q(x_t | x0) for each timestep t
    x_t = q_sample_fn(x0, t)

    # ============================================================================
    # STEP 3: Compute true posterior distribution
    # ============================================================================
    # Get the mean and log-variance of q(x_{t-1} | x_t, x0)
    # This is the "ground truth" we want our model to match
    true_mean, true_logvar = forward_posterior_fn(x0, x_t, t)

    # ============================================================================
    # STEP 4: Get model's predicted distribution
    # ============================================================================
    # Model predicts the mean and log-variance for denoising
    # This is p_θ(x_{t-1} | x_t), our learned approximation
    model_mean, model_logvar = denoising_model_mean_variance(model, x_t, t)

    # ============================================================================
    # STEP 5: Compute loss as KL divergence between distributions
    # ============================================================================
    # Train the model to minimize KL(q(x_{t-1}|x_t,x0) || p_θ(x_{t-1}|x_t))
    return compute_loss(true_mean, true_logvar, model_mean, model_logvar, loss_scale_factor)


def loss_mse_mean_only(
    model,
    x0,
    n_steps,
    device,
    q_sample_fn,
    forward_posterior_fn,
    fixed_log_var,
):
    """
    MSE loss for training with fixed variance (mean-only prediction).

    This is a simplified version of the DDPM loss where the model only predicts
    the mean of the denoising distribution, and the variance is fixed according
    to a predetermined schedule.

    The loss is simply the MSE between the predicted mean and the true posterior mean:
        L = ||μ_true - μ_model||^2

    This is equivalent to the KL divergence loss when the variance is fixed,
    because the KL divergence between two Gaussians with the same variance
    reduces to the MSE between their means (up to a constant).

    Parameters
    ----------
    model : nn.Module
        Model that outputs only the mean (2D output for 2D data)
    x0 : torch.Tensor
        Clean data samples from the dataset
    n_steps : int
        Total number of diffusion timesteps
    device : torch.device
        Device to run computations on
    q_sample_fn : callable
        Forward diffusion function: q_sample(x0, t) -> x_t
    forward_posterior_fn : callable
        Computes true posterior mean: forward_posterior_fn(x0, xt, t) -> (mean, logvar)
    fixed_log_var : torch.Tensor
        Fixed log-variance schedule (not used in loss, but kept for consistency)

    Returns
    -------
    loss : torch.Tensor
        Scalar MSE loss value
    """
    from sampling import denoising_model_mean_fixed_variance

    batch_size = x0.shape[0]

    # Sample timesteps with variance reduction (same as KL version)
    # NOTE: Start from 1, not 0, because t=0 has degenerate posterior variance
    t = torch.randint(1, n_steps, size=(batch_size // 2 + 1,), device=device)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()

    # Forward diffusion: add noise
    x_t = q_sample_fn(x0, t)

    # Get true posterior mean (we ignore the variance since it's fixed)
    true_mean, _ = forward_posterior_fn(x0, x_t, t)

    # Get model's predicted mean (variance is fixed)
    model_mean, _ = denoising_model_mean_fixed_variance(model, x_t, t, fixed_log_var)

    # MSE loss between predicted and true mean
    loss = torch.nn.functional.mse_loss(model_mean, true_mean)

    return loss


def loss_noise_prediction(
    model,
    x0,
    n_steps,
    device,
    q_sample_fn,
):
    """
    Noise prediction loss for DDPM training.

    This is the most common DDPM training objective. The model learns to predict
    the noise ε that was added during the forward diffusion process:

        x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε

    The loss is simply:
        L = ||ε - ε_θ(x_t, t)||²

    This formulation has several advantages:
    - Simple MSE loss on noise
    - Often better empirical performance than mean/variance prediction
    - Connection to score-based generative models (predicting noise ≈ predicting
    score ≈ the negative of the gradient of log density)
    - Original DDPM paper's primary formulation

    Parameters
    ----------
    model : nn.Module
        Model that predicts noise ε_θ(x_t, t)
    x0 : torch.Tensor
        Clean data samples from the dataset
    n_steps : int
        Total number of diffusion timesteps
    device : torch.device
        Device to run computations on
    q_sample_fn : callable
        Forward diffusion function: q_sample(x0, t, noise) -> x_t
        Must accept a noise parameter to use the same noise for training

    Returns
    -------
    loss : torch.Tensor
        Scalar MSE loss value between true and predicted noise
    """
    batch_size = x0.shape[0]

    # Sample timesteps with variance reduction (same as other losses)
    # NOTE: Start from 1, not 0, because t=0 has degenerate posterior variance
    t = torch.randint(1, n_steps, size=(batch_size // 2 + 1,), device=device)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()

    # Sample random noise
    noise = torch.randn_like(x0, device=device)

    # Forward diffusion: add noise to clean data
    # x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
    x_t = q_sample_fn(x0, t, noise=noise)

    # Model predicts the noise
    predicted_noise = model(x_t, t)

    # MSE loss between true and predicted noise
    loss = torch.nn.functional.mse_loss(predicted_noise, noise)

    return loss


def gaussian_log_likelihood(x, mean, logvar):
    """
    Compute Gaussian log likelihood: log N(x | mean, var).

    For a diagonal Gaussian:
        log p(x) = -0.5 * [D*log(2π) + sum(logvar) + sum((x-mean)^2 / var)]

    Parameters
    ----------
    x : torch.Tensor
        Data points
    mean : torch.Tensor
        Mean of the Gaussian
    logvar : torch.Tensor
        Log variance of the Gaussian

    Returns
    -------
    log_likelihood : torch.Tensor
        Log likelihood for each sample (shape: batch_size)
    """
    import math

    # Compute squared error scaled by variance
    squared_error = (x - mean) ** 2 * torch.exp(-logvar)

    # Sum over dimensions
    dim = x.shape[1]  # Dimensionality of data
    log_2pi = math.log(2 * math.pi)

    # log p(x) = -0.5 * [D*log(2π) + sum(logvar) + sum((x-mean)^2/var)]
    log_likelihood = -0.5 * (dim * log_2pi + logvar.sum(dim=1) + squared_error.sum(dim=1))

    return log_likelihood


@torch.no_grad()
def compute_vlb(
    model,
    x0,
    n_steps,
    device,
    q_sample_fn,
    forward_posterior_fn,
    denoising_fn,
    alphas_prod,
):
    """
    Compute the Variational Lower Bound (VLB) for a batch of data.

    The VLB for DDPM consists of three terms:
        VLB = L_0 + L_{1:T-1} + L_T

    Where:
    - L_0: Reconstruction term (negative log likelihood of x_0 given x_1)
    - L_{1:T-1}: Denoising matching terms (KL divergences at each timestep)
    - L_T: Prior matching term (KL between q(x_T|x_0) and prior N(0,I))

    Lower VLB (more negative) = worse model
    Higher VLB (less negative) = better model

    Parameters
    ----------
    model : nn.Module
        The trained denoising model
    x0 : torch.Tensor
        Clean data samples to evaluate on, shape (batch_size, D)
    n_steps : int
        Number of diffusion timesteps
    device : torch.device
        Device to run on
    q_sample_fn : callable
        Forward diffusion function q(x_t | x_0)
    forward_posterior_fn : callable
        Computes q(x_{t-1} | x_t, x_0) -> (mean, logvar)
    denoising_fn : callable
        Model's denoising function p_θ(x_{t-1} | x_t) -> (mean, logvar)
    alphas_prod : torch.Tensor
        Cumulative product of alphas (ᾱ_t)

    Returns
    -------
    vlb : torch.Tensor
        VLB for each sample in the batch (shape: batch_size)
        Units: nats (natural logarithm)
    """
    batch_size = x0.shape[0]
    dim = x0.shape[1]

    # Initialize VLB accumulator
    vlb = torch.zeros(batch_size, device=device)

    # ========================================================================
    # L_T: Prior matching term
    # ========================================================================
    # KL( q(x_T | x_0) || p(x_T) ) where p(x_T) = N(0, I)
    # q(x_T | x_0) = N(sqrt(ᾱ_T) * x_0, (1 - ᾱ_T) * I)

    alpha_T = alphas_prod[-1]
    mean_qxT = torch.sqrt(alpha_T) * x0
    var_qxT = 1 - alpha_T
    logvar_qxT = torch.log(torch.clamp(var_qxT, min=1e-20))

    # KL with standard normal prior N(0, I)
    mean_prior = torch.zeros_like(mean_qxT)
    logvar_prior = torch.zeros_like(logvar_qxT)

    kl_T = normal_kl(mean_qxT, logvar_qxT, mean_prior, logvar_prior)
    L_T = kl_T.sum(dim=1)  # Sum over dimensions

    vlb -= L_T  # Negative because we want log p(x)

    # ========================================================================
    # L_{1:T-1}: Denoising matching terms
    # ========================================================================
    # For each timestep t, compute KL( q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t) )

    for t_val in range(1, n_steps):
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

        # Sample x_t ~ q(x_t | x_0)
        x_t = q_sample_fn(x0, t)

        # Get true posterior q(x_{t-1} | x_t, x_0)
        true_mean, true_logvar = forward_posterior_fn(x0, x_t, t)

        # Get model prediction p_θ(x_{t-1} | x_t)
        model_mean, model_logvar = denoising_fn(model, x_t, t)

        # Clamp model logvar for numerical stability
        model_logvar = torch.clamp(model_logvar, min=-20, max=2)

        # Compute KL divergence
        kl_t = normal_kl(true_mean, true_logvar, model_mean, model_logvar)
        L_t = kl_t.sum(dim=1)  # Sum over dimensions

        vlb -= L_t  # Negative because we want log p(x)

    # ========================================================================
    # L_0: Reconstruction term
    # ========================================================================
    # Negative log likelihood: -log p_θ(x_0 | x_1)
    # Model predicts mean and variance for x_0 given x_1

    t = torch.full((batch_size,), 0, device=device, dtype=torch.long)
    x_1 = q_sample_fn(x0, torch.full((batch_size,), 1, device=device, dtype=torch.long))

    # Get model prediction for x_0 from x_1
    model_mean, model_logvar = denoising_fn(model, x_1, t)
    model_logvar = torch.clamp(model_logvar, min=-20, max=2)

    # Compute Gaussian log likelihood
    log_likelihood = gaussian_log_likelihood(x0, model_mean, model_logvar)

    vlb += log_likelihood  # Positive contribution to log p(x)

    return vlb


def compute_bpd(vlb, dim):
    """
    Convert VLB (in nats) to bits per dimension (bpd).

    bpd = -VLB / (D * log(2))

    Lower bpd = better model

    Parameters
    ----------
    vlb : torch.Tensor
        Variational lower bound in nats
    dim : int
        Dimensionality of the data

    Returns
    -------
    bpd : torch.Tensor
        Bits per dimension
    """
    import math
    return -vlb / (dim * math.log(2))
