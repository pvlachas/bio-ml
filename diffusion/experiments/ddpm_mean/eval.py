"""
Evaluation script for DDPM with learned mean and variance.

Loads the best checkpoint, generates samples, and computes evaluation metrics.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json
from tqdm import tqdm

from ema import EMA
from utils import sample_batch, get_device
from metrics import compute_all_metrics
from loss import compute_vlb, compute_bpd
from sampling import denoising_model_mean_variance
from common import (
    setup_seed,
    setup_diffusion_schedule,
    create_q_sample_fn,
    create_forward_posterior_fn,
    get_experiment_config,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_NAME = 'ddpm_mean'

# Reproducibility
SEED = 42

# Hyperparameters (must match training)
N_STEPS = 50
BETA_START = 1e-4
BETA_END = 0.02
EMA_MU = 0.99

# Evaluation parameters
N_EVAL_SAMPLES = 10000
N_REAL_SAMPLES = 10000
VLB_BATCH_SIZE = 100

# ==============================================================================
# SETUP
# ==============================================================================
setup_seed(SEED)
device = get_device()

# Setup paths
results_dir = Path("../../results") / EXPERIMENT_NAME
checkpoint_dir = results_dir / "checkpoints"
results_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print(f"EVALUATION: {EXPERIMENT_NAME}")
print("="*70)

# ==============================================================================
# LOAD CHECKPOINT
# ==============================================================================
print("\n[1/6] Finding best checkpoint...")
checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))

if len(checkpoint_files) == 0:
    print(f"ERROR: No checkpoints found in {checkpoint_dir}")
    print("Please train the model first")
    exit(1)

best_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
print(f"  Found: {best_checkpoint}")
epoch_num = int(best_checkpoint.stem.split('_')[-1])
print(f"  Epoch: {epoch_num}")

# ==============================================================================
# LOAD MODEL
# ==============================================================================
print("\n[2/6] Loading model...")

config = get_experiment_config(EXPERIMENT_NAME)
model = config['model_class'](N_STEPS, device).to(device)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load checkpoint and apply EMA
checkpoint = torch.load(best_checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

ema = EMA(EMA_MU)
ema.shadow = checkpoint['ema_shadow']
ema.ema(model)

model.eval()
print("  Model loaded with EMA weights applied")

# ==============================================================================
# SETUP DIFFUSION
# ==============================================================================
print("\n[3/6] Setting up diffusion...")
schedule = setup_diffusion_schedule(N_STEPS, BETA_START, BETA_END, device)
q_sample = create_q_sample_fn(schedule['alphas_bar_sqrt'], schedule['one_minus_alphas_bar_sqrt'])
forward_posterior = create_forward_posterior_fn(
    schedule['posterior_mean_coef_x0'],
    schedule['posterior_mean_coef_xt'],
    schedule['posterior_variance']
)

# ==============================================================================
# GENERATE SAMPLES
# ==============================================================================
print(f"\n[4/6] Generating {N_EVAL_SAMPLES} samples...")

with torch.no_grad():
    samples_traj, samples_final = config['sampling_fn'](
        model,
        shape=(N_EVAL_SAMPLES, 2),
        n_steps=N_STEPS,
        device=device,
    )

print(f"  Shape: {samples_final.shape}")
print(f"  Mean: {samples_final.mean(dim=0).cpu().numpy()}")
print(f"  Std: {samples_final.std(dim=0).cpu().numpy()}")

# ==============================================================================
# LOAD REAL DATA
# ==============================================================================
print(f"\n[5/6] Loading real data...")
real_data = sample_batch(N_REAL_SAMPLES)
print(f"  Shape: {real_data.shape}")
print(f"  Mean: {real_data.mean(dim=0)}")
print(f"  Std: {real_data.std(dim=0)}")

# ==============================================================================
# COMPUTE VLB
# ==============================================================================
print(f"\n[6/6] Computing Variational Lower Bound...")

n_batches = (N_REAL_SAMPLES + VLB_BATCH_SIZE - 1) // VLB_BATCH_SIZE
vlb_values = []

for i in tqdm(range(n_batches), desc="  VLB batches"):
    start_idx = i * VLB_BATCH_SIZE
    end_idx = min((i + 1) * VLB_BATCH_SIZE, N_REAL_SAMPLES)
    batch_real = real_data[start_idx:end_idx].to(device)

    vlb_batch = compute_vlb(
        model=model,
        x0=batch_real,
        n_steps=N_STEPS,
        device=device,
        q_sample_fn=q_sample,
        forward_posterior_fn=forward_posterior,
        denoising_fn=denoising_model_mean_variance,
        alphas_prod=schedule['alphas_prod'],
    )
    vlb_values.append(vlb_batch.cpu())

# Statistics
vlb_all = torch.cat(vlb_values, dim=0)
vlb_mean = vlb_all.mean().item()
vlb_std = vlb_all.std().item()

dim = real_data.shape[1]
bpd_values = compute_bpd(vlb_all, dim)
bpd_mean = bpd_values.mean().item()
bpd_std = bpd_values.std().item()

print(f"  VLB (nats): {vlb_mean:.4f} ± {vlb_std:.4f}")
print(f"  Bits/dim:   {bpd_mean:.4f} ± {bpd_std:.4f}")

# ==============================================================================
# COMPUTE METRICS
# ==============================================================================
print("\nComputing evaluation metrics...")
metrics = compute_all_metrics(real_data, samples_final.cpu().numpy())

# Add log likelihood metrics
metrics['vlb_mean'] = vlb_mean
metrics['vlb_std'] = vlb_std
metrics['bpd_mean'] = bpd_mean
metrics['bpd_std'] = bpd_std

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
output_file = results_dir / f"metrics_{EXPERIMENT_NAME}.json"
with open(output_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'='*70}")
print("EVALUATION RESULTS")
print(f"{'='*70}\n")

print("Distribution Distance:")
print(f"  MMD:                  {metrics['mmd']:.6f}")
print(f"  Sliced Wasserstein:   {metrics['sliced_wasserstein']:.6f}")
print(f"  Wasserstein (avg):    {metrics['wasserstein_avg_per_dim']:.6f}")

print("\nHistogram Metrics:")
print(f"  Chi-Square:           {metrics['chi_square']:.6f}")
print(f"  KL Divergence:        {metrics['kl_divergence']:.6f}")
print(f"  JS Divergence:        {metrics['js_divergence']:.6f}")
print(f"  Histogram Inter.:     {metrics['histogram_intersection']:.6f}")

print("\nQuality Metrics:")
print(f"  Precision:            {metrics['precision']:.6f}")
print(f"  Recall:               {metrics['recall']:.6f}")
print(f"  F1 Score:             {metrics['f1_score']:.6f}")
print(f"  Coverage:             {metrics['coverage']:.6f}")

print("\nLog Likelihood:")
print(f"  VLB (nats):           {vlb_mean:.4f} ± {vlb_std:.4f}")
print(f"  Bits/dim:             {bpd_mean:.4f} ± {bpd_std:.4f}")

print(f"\n{'='*70}")
print(f"Results saved to: {output_file}")
print(f"{'='*70}")
