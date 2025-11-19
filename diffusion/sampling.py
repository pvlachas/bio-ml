import torch

def denoising_model_mean_variance(model, x, t):
    # Go through model
    out = model(x, t)
    # Extract the mean and variance
    mean, log_var = torch.split(out, 2, dim=-1)
    return mean, log_var


def denoising_model_mean_fixed_variance(model, x, t, fixed_log_var):
    """
    Get model mean with fixed variance schedule.

    Parameters
    ----------
    model : nn.Module
        Model that outputs only the mean (2D output)
    x : torch.Tensor
        Current noisy sample x_t
    t : torch.Tensor
        Timestep indices
    fixed_log_var : torch.Tensor
        Pre-computed fixed log-variance schedule of shape (n_steps,) or (n_steps, 1)

    Returns
    -------
    mean : torch.Tensor
        Predicted mean from the model
    log_var : torch.Tensor
        Fixed log-variance extracted from the schedule, broadcasted to match mean shape
    """
    from utils import extract

    # Model outputs only the mean
    mean = model(x, t)

    # Extract fixed log-variance for this timestep
    # extract returns shape (batch_size, 1), we need to broadcast to (batch_size, 2)
    log_var = extract(fixed_log_var, t, x)

    # Broadcast to match the dimensionality of the mean
    # If mean is (batch_size, 2), log_var should also be (batch_size, 2)
    if log_var.shape != mean.shape:
        log_var = log_var.expand_as(mean)

    return mean, log_var


@torch.no_grad()
def denoising_process_samples(model, x, t):
    mean, log_var = denoising_model_mean_variance(model, x, torch.tensor(t, device=x.device))
    log_var = torch.clamp(log_var, min=-20, max=2)
    noise = torch.randn_like(x)
    sample = mean + torch.exp(0.5 * log_var) * noise
    return (sample)


@torch.no_grad()
def denoising_process_samples_fixed_variance(model, x, t, fixed_log_var):
    """
    Sample from denoising process with fixed variance schedule.

    Parameters
    ----------
    model : nn.Module
        Model that outputs only the mean
    x : torch.Tensor
        Current noisy sample x_t
    t : torch.Tensor
        Timestep (can be scalar or batch)
    fixed_log_var : torch.Tensor
        Fixed log-variance schedule

    Returns
    -------
    sample : torch.Tensor
        Denoised sample x_{t-1}
    """
    mean, log_var = denoising_model_mean_fixed_variance(model, x, t, fixed_log_var)
    log_var = torch.clamp(log_var, min=-20, max=2)
    noise = torch.randn_like(x)
    sample = mean + torch.exp(0.5 * log_var) * noise
    return sample


@torch.no_grad()
def denoising_process_sampling_trajectory(
    model,
    shape,
    n_steps,
    device,
):
    """
    Generate samples from the denoising process p(x_{t-1} | x_t)
    starting from pure Gaussian noise x_T ~ N(0, I).
    """
    x_t = torch.randn(shape, device=device)
    x_traj = [x_t.detach().cpu()]
    for t in reversed(range(n_steps)):
        t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
        x_t = denoising_process_samples(model, x_t, t_batch)
        x_traj.append(x_t.detach().cpu())
    return x_traj, x_t


@torch.no_grad()
def denoising_process_sampling_trajectory_fixed_variance(
    model,
    shape,
    n_steps,
    device,
    fixed_log_var,
):
    """
    Generate samples from the denoising process with fixed variance.

    Parameters
    ----------
    model : nn.Module
        Model that outputs only the mean
    shape : tuple
        Shape of samples to generate
    n_steps : int
        Number of diffusion timesteps
    device : torch.device
        Device to run on
    fixed_log_var : torch.Tensor
        Fixed log-variance schedule

    Returns
    -------
    x_traj : list of torch.Tensor
        Sampling trajectory at each timestep
    x_t : torch.Tensor
        Final generated samples
    """
    x_t = torch.randn(shape, device=device)
    x_traj = [x_t.detach().cpu()]
    for t in reversed(range(n_steps)):
        t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
        x_t = denoising_process_samples_fixed_variance(model, x_t, t_batch, fixed_log_var)
        x_traj.append(x_t.detach().cpu())
    return x_traj, x_t


def predict_x0_from_noise(x_t, t, predicted_noise, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    """
    Predict x0 from noisy sample x_t and predicted noise.

    Given: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
    Solve for: x_0 = (x_t - √(1-ᾱ_t) ε) / √(ᾱ_t)

    Parameters
    ----------
    x_t : torch.Tensor
        Noisy sample at timestep t
    t : torch.Tensor
        Timestep indices
    predicted_noise : torch.Tensor
        Predicted noise from model
    alphas_bar_sqrt : torch.Tensor
        √(ᾱ_t) schedule
    one_minus_alphas_bar_sqrt : torch.Tensor
        √(1-ᾱ_t) schedule

    Returns
    -------
    x0_pred : torch.Tensor
        Predicted clean sample
    """
    from utils import extract

    coef_xt = extract(alphas_bar_sqrt, t, x_t)
    coef_noise = extract(one_minus_alphas_bar_sqrt, t, x_t)

    x0_pred = (x_t - coef_noise * predicted_noise) / coef_xt
    return x0_pred


def denoising_mean_from_noise_prediction(
    model,
    x_t,
    t,
    alphas,
    alphas_bar_sqrt,
    one_minus_alphas_bar_sqrt,
    betas,
):
    """
    Compute denoising mean from noise prediction.

    The model predicts noise ε_θ(x_t, t), and we compute the mean as:
        μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))

    This is equivalent to predicting x0 first, then computing the posterior mean.

    Parameters
    ----------
    model : nn.Module
        Model that predicts noise
    x_t : torch.Tensor
        Noisy sample
    t : torch.Tensor
        Timestep indices
    alphas : torch.Tensor
        α_t = 1 - β_t schedule
    alphas_bar_sqrt : torch.Tensor
        √(ᾱ_t) schedule
    one_minus_alphas_bar_sqrt : torch.Tensor
        √(1-ᾱ_t) schedule
    betas : torch.Tensor
        β_t schedule

    Returns
    -------
    mean : torch.Tensor
        Predicted mean for denoising step
    predicted_noise : torch.Tensor
        The predicted noise (for diagnostics)
    """
    from utils import extract

    # Model predicts the noise
    predicted_noise = model(x_t, t)

    # Extract coefficients for this timestep
    alpha_t = extract(alphas, t, x_t)
    beta_t = extract(betas, t, x_t)
    one_minus_alpha_bar_sqrt_t = extract(one_minus_alphas_bar_sqrt, t, x_t)

    # Compute mean: μ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε)
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / one_minus_alpha_bar_sqrt_t) * predicted_noise
    )

    return mean, predicted_noise


@torch.no_grad()
def denoising_process_samples_noise_prediction(
    model,
    x_t,
    t,
    alphas,
    alphas_bar_sqrt,
    one_minus_alphas_bar_sqrt,
    betas,
    fixed_log_var,
):
    """
    Sample from denoising process using noise prediction.

    Parameters
    ----------
    model : nn.Module
        Model that predicts noise
    x_t : torch.Tensor
        Current noisy sample
    t : torch.Tensor
        Timestep (can be scalar or batch)
    alphas : torch.Tensor
        α_t schedule
    alphas_bar_sqrt : torch.Tensor
        √(ᾱ_t) schedule
    one_minus_alphas_bar_sqrt : torch.Tensor
        √(1-ᾱ_t) schedule
    betas : torch.Tensor
        β_t schedule
    fixed_log_var : torch.Tensor
        Fixed log-variance schedule

    Returns
    -------
    sample : torch.Tensor
        Denoised sample x_{t-1}
    """
    from utils import extract

    # Get mean from noise prediction
    mean, _ = denoising_mean_from_noise_prediction(
        model, x_t, t, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, betas
    )

    # Get fixed variance
    log_var = extract(fixed_log_var, t, x_t)
    if log_var.shape != mean.shape:
        log_var = log_var.expand_as(mean)
    log_var = torch.clamp(log_var, min=-20, max=2)

    # Sample: x_{t-1} = μ + σ * z
    noise = torch.randn_like(x_t)
    sample = mean + torch.exp(0.5 * log_var) * noise

    return sample


@torch.no_grad()
def denoising_process_sampling_trajectory_noise_prediction(
    model,
    shape,
    n_steps,
    device,
    alphas,
    alphas_bar_sqrt,
    one_minus_alphas_bar_sqrt,
    betas,
    fixed_log_var,
):
    """
    Generate samples using noise prediction model.

    Parameters
    ----------
    model : nn.Module
        Model that predicts noise
    shape : tuple
        Shape of samples to generate
    n_steps : int
        Number of diffusion timesteps
    device : torch.device
        Device to run on
    alphas : torch.Tensor
        α_t schedule
    alphas_bar_sqrt : torch.Tensor
        √(ᾱ_t) schedule
    one_minus_alphas_bar_sqrt : torch.Tensor
        √(1-ᾱ_t) schedule
    betas : torch.Tensor
        β_t schedule
    fixed_log_var : torch.Tensor
        Fixed log-variance schedule

    Returns
    -------
    x_traj : list of torch.Tensor
        Sampling trajectory at each timestep
    x_t : torch.Tensor
        Final generated samples
    """
    x_t = torch.randn(shape, device=device)
    x_traj = [x_t.detach().cpu()]
    for t in reversed(range(n_steps)):
        t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
        x_t = denoising_process_samples_noise_prediction(
            model, x_t, t_batch, alphas, alphas_bar_sqrt,
            one_minus_alphas_bar_sqrt, betas, fixed_log_var
        )
        x_traj.append(x_t.detach().cpu())
    return x_traj, x_t


