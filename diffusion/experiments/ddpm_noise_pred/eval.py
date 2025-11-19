"""
Evaluation script for DDPM with noise prediction.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import json

from ema import EMA
from utils import sample_batch, get_device
from metrics import compute_all_metrics
from common import (
    setup_seed,
    setup_diffusion_schedule,
    get_experiment_config,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EXPERIMENT_NAME = 'ddpm_noise_pred'
SEED = 42
N_STEPS = 50
BETA_START = 1e-4
BETA_END = 0.02
EMA_MU = 0.99
N_EVAL_SAMPLES = 10000
N_REAL_SAMPLES = 10000

# ==============================================================================
# SETUP
# ==============================================================================
setup_seed(SEED)
device = get_device()

results_dir = Path("../../results") / EXPERIMENT_NAME
checkpoint_dir = results_dir / "checkpoints"
results_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print(f"EVALUATION: {EXPERIMENT_NAME}")
print("="*70)

# ==============================================================================
# LOAD CHECKPOINT
# ==============================================================================
print("\n[1/5] Finding best checkpoint...")
checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))

if len(checkpoint_files) == 0:
    print(f"ERROR: No checkpoints found in {checkpoint_dir}")
    exit(1)

best_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
print(f"  Found: {best_checkpoint}")
epoch_num = int(best_checkpoint.stem.split('_')[-1])
print(f"  Epoch: {epoch_num}")

# ==============================================================================
# LOAD MODEL
# ==============================================================================
print("\n[2/5] Loading model...")

config = get_experiment_config(EXPERIMENT_NAME)
model = config['model_class'](N_STEPS, device).to(device)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
print("\n[3/5] Setting up diffusion...")
schedule = setup_diffusion_schedule(N_STEPS, BETA_START, BETA_END, device)
fixed_log_var = torch.log(torch.clamp(schedule['posterior_variance'], min=1e-20))

# ==============================================================================
# GENERATE SAMPLES
# ==============================================================================
print(f"\n[4/5] Generating {N_EVAL_SAMPLES} samples...")

with torch.no_grad():
    samples_traj, samples_final = config['sampling_fn'](
        model,
        shape=(N_EVAL_SAMPLES, 2),
        n_steps=N_STEPS,
        device=device,
        alphas=schedule['alphas'],
        alphas_bar_sqrt=schedule['alphas_bar_sqrt'],
        one_minus_alphas_bar_sqrt=schedule['one_minus_alphas_bar_sqrt'],
        betas=schedule['betas'],
        fixed_log_var=fixed_log_var,
    )

print(f"  Shape: {samples_final.shape}")
print(f"  Mean: {samples_final.mean(dim=0).cpu().numpy()}")
print(f"  Std: {samples_final.std(dim=0).cpu().numpy()}")

# ==============================================================================
# LOAD REAL DATA
# ==============================================================================
print(f"\n[5/5] Loading real data...")
real_data = sample_batch(N_REAL_SAMPLES)
print(f"  Shape: {real_data.shape}")

# ==============================================================================
# COMPUTE METRICS
# ==============================================================================
print("\nComputing evaluation metrics...")
metrics = compute_all_metrics(real_data, samples_final.cpu().numpy())

# Note: VLB computation is more complex for noise prediction models
# and would require converting noise predictions to denoising distributions
# For simplicity, we skip VLB for this experiment
metrics['vlb_mean'] = None
metrics['vlb_std'] = None
metrics['bpd_mean'] = None
metrics['bpd_std'] = None

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

print("\nQuality Metrics:")
print(f"  Precision:            {metrics['precision']:.6f}")
print(f"  Recall:               {metrics['recall']:.6f}")
print(f"  F1 Score:             {metrics['f1_score']:.6f}")

print(f"\n{'='*70}")
print(f"Results saved to: {output_file}")
print(f"{'='*70}")
