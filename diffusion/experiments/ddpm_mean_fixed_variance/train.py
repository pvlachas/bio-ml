"""
Training script for DDPM with fixed variance.

This experiment trains a diffusion model that only learns the mean,
while using a fixed variance schedule.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.optim as optim
from tqdm import tqdm

from ema import EMA
from utils import sample_batch, get_device, clean_figure_directories
from plotting import plot_data_samples, plot_forward_diffusion, plot_sampling_trajectory
from diagnostics import (
    TrainingMonitor,
    plot_sample_quality_diagnostics,
    plot_model_predictions_analysis,
    analyze_loss_per_timestep,
)
from sampling import denoising_model_mean_fixed_variance
from common import (
    setup_seed,
    setup_diffusion_schedule,
    create_q_sample_fn,
    create_forward_posterior_fn,
    get_lr_scheduler_fn,
    save_checkpoint,
    load_checkpoint,
    get_experiment_config,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_NAME = 'ddpm_mean_fixed_variance'

# Reproducibility
SEED = 0

# Diffusion hyperparameters
N_STEPS = 50
BETA_START = 1e-4
BETA_END = 0.02

# Model hyperparameters
EMA_MU = 0.99

# Training hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
NUM_ITERS = 100000 + 1
WARMUP_ITERS = 100

# Reporting hyperparameters
REPORT_EVERY = 100
EVAL_EVERY = 1000
SAMPLES_LOGVAR_STATS_REPORT = 2000
SAMPLES_EVAL = 4000

# Data hyperparameters
DATA_SIZE = 10000

# ==============================================================================
# SETUP
# ==============================================================================
setup_seed(SEED)
device = get_device()

# Setup paths
results_dir = Path("../../results") / EXPERIMENT_NAME
fig_dir = results_dir / "figures"
checkpoint_dir = results_dir / "checkpoints"

fig_dir.mkdir(parents=True, exist_ok=True)
clean_figure_directories(fig_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*70}")
print(f"TRAINING: {EXPERIMENT_NAME}")
print(f"{'='*70}\n")

# ==============================================================================
# DATA
# ==============================================================================
print("[1/5] Loading data...")
data = sample_batch(DATA_SIZE)
dataset = torch.tensor(data).float().to(device)
plot_data_samples(data, fig_dir)

# ==============================================================================
# DIFFUSION SCHEDULE
# ==============================================================================
print("[2/5] Setting up diffusion schedule...")
schedule = setup_diffusion_schedule(N_STEPS, BETA_START, BETA_END, device)

# Fixed variance schedule (use posterior variance)
fixed_log_var = torch.log(torch.clamp(schedule['posterior_variance'], min=1e-20))

# Create diffusion functions
q_sample = create_q_sample_fn(schedule['alphas_bar_sqrt'], schedule['one_minus_alphas_bar_sqrt'])
forward_posterior = create_forward_posterior_fn(
    schedule['posterior_mean_coef_x0'],
    schedule['posterior_mean_coef_xt'],
    schedule['posterior_variance']
)

print("  Visualizing forward diffusion process...")
plot_forward_diffusion(q_sample, data, steps=N_STEPS, fig_dir=fig_dir)

# ==============================================================================
# MODEL
# ==============================================================================
print("[3/5] Initializing model...")
config = get_experiment_config(EXPERIMENT_NAME)

model = config['model_class'](N_STEPS, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
ema = EMA(EMA_MU)
ema.register(model)

print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Timesteps: {N_STEPS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")

# ==============================================================================
# TRAINING SETUP
# ==============================================================================
print("[4/5] Preparing training...")
monitor = TrainingMonitor()
get_lr_multiplier = get_lr_scheduler_fn(WARMUP_ITERS)

# Check for existing checkpoint
start_epoch = 0
latest_checkpoint = checkpoint_dir / "checkpoint_latest.pt"
if latest_checkpoint.exists():
    print(f"\n  Found existing checkpoint: {latest_checkpoint}")
    response = input("  Load checkpoint? [y/N]: ").strip().lower()
    if response == 'y':
        start_epoch = load_checkpoint(latest_checkpoint, model, optimizer, ema, monitor, device)
        print(f"  Resuming from epoch {start_epoch}")

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
print(f"\n[5/5] Training...\n")
print("-" * 70)

epoch_iterator = tqdm(range(start_epoch, NUM_ITERS), desc="Training", ncols=120,
                      initial=start_epoch, total=NUM_ITERS)

for epoch in epoch_iterator:
    # Learning rate warmup
    lr_mult = get_lr_multiplier(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE * lr_mult

    # Batch training
    permutation = torch.randperm(dataset.size()[0])
    for batch_start in range(0, dataset.size()[0], BATCH_SIZE):
        batch_idxs = permutation[batch_start:batch_start + BATCH_SIZE]
        batch_x = dataset[batch_idxs]

        # Compute loss (MSE for fixed variance)
        loss = config['loss_fn'](
            model=model,
            x0=batch_x,
            n_steps=N_STEPS,
            device=device,
            q_sample_fn=q_sample,
            forward_posterior_fn=forward_posterior,
            fixed_log_var=fixed_log_var,
        )

        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss at epoch {epoch}, skipping batch")
            continue

        optimizer.zero_grad()
        loss.backward()

        total_norm = sum(p.grad.data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None) ** 0.5
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        if total_norm > 100:
            print(f"WARNING: Extreme gradients ({total_norm:.1f}) at epoch {epoch}, skipping")
            continue

        optimizer.step()
        ema.update(model)

    epoch_iterator.set_postfix(
        loss=f"{loss.item():.4f}",
        grad=f"{grad_norm:.2f}",
        lr=f"{optimizer.param_groups[0]['lr']:.6f}"
    )

    # Tracking
    if epoch % REPORT_EVERY == 0:
        with torch.no_grad():
            sample_idx = torch.randperm(dataset.size()[0])[:SAMPLES_LOGVAR_STATS_REPORT]
            sample_x = dataset[sample_idx]
            sample_t = torch.randint(1, N_STEPS, size=(SAMPLES_LOGVAR_STATS_REPORT,), device=device)
            sample_xt = q_sample(sample_x, sample_t)
            _, pred_logvar = denoising_model_mean_fixed_variance(model, sample_xt, sample_t, fixed_log_var)
            mean_logvar = pred_logvar.mean().item()
            std_logvar = pred_logvar.std().item()

        monitor.update(
            epoch=epoch,
            loss=loss.item(),
            lr=optimizer.param_groups[0]['lr'],
            grad_norm=grad_norm.item(),
            mean_logvar=mean_logvar,
            std_logvar=std_logvar,
        )

        print(f"Epoch {epoch:05d}: loss = {loss:.6f}, lr = {optimizer.param_groups[0]['lr']:.6f}, "
              f"grad_norm = {grad_norm:.3f}, logvar = {mean_logvar:.2f}±{std_logvar:.2f}")

    # Evaluation
    if epoch % EVAL_EVERY == 0:
        print(f"\n{'='*70}")
        print(f"EVALUATION AT EPOCH {epoch}")
        print(f"{'='*70}")

        param_backup = {name: param.data.clone()
                       for name, param in model.named_parameters() if param.requires_grad}
        ema.ema(model)

        print("[1/5] Generating samples...")
        samples_traj, samples_final = config['sampling_fn'](
            model, shape=(SAMPLES_EVAL, 2), n_steps=N_STEPS, device=device, fixed_log_var=fixed_log_var
        )

        print("[2/5] Plotting sampling trajectory...")
        plot_sampling_trajectory(samples_traj, N_STEPS, epoch, fig_dir)

        print("[3/5] Analyzing sample quality...")
        plot_sample_quality_diagnostics(data, samples_final, epoch, fig_dir)

        print("[4/5] Analyzing model predictions...")
        eval_batch_size = min(2000, len(dataset))
        eval_indices = torch.randperm(len(dataset))[:eval_batch_size]
        eval_x0 = dataset[eval_indices]
        eval_t = torch.randint(1, N_STEPS, size=(eval_batch_size,), device=device)
        eval_xt = q_sample(eval_x0, eval_t)
        plot_model_predictions_analysis(model, eval_xt, eval_t, epoch, fig_dir, n_samples=5000)

        print("[5/5] Computing loss per timestep...")
        analyze_loss_per_timestep(model, dataset, N_STEPS, device, forward_posterior, q_sample, fig_dir, epoch)

        print("[Bonus] Plotting training curves...")
        monitor.plot_training_curves(fig_dir)

        print(f"{'='*70}\n")

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(param_backup[name])

        save_checkpoint(epoch, model, optimizer, ema, monitor, checkpoint_dir, EXPERIMENT_NAME)

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
