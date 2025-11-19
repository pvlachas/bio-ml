# DDPM Experiments Structure

This directory contains code for DDPM implementation & experiments.
Currently, the following implementations are included:
1. DDPM Mean+Variance prediction
2. DDPM Fixed Variance (Mean prediction only)
3. DDPM Noise Prediction

## Directory Structure

```
diffusion/
├── experiments/                    # Experiment-specific scripts
│   ├── ddpm_mean/                 # Mean+Variance prediction
│   │   ├── train.py
│   │   └── eval.py
│   ├── ddpm_mean_fixed_variance/  # Mean prediction with fixed variance
│   │   ├── train.py
│   │   └── eval.py
│   └── ddpm_noise_pred/           # Noise prediction
│       ├── train.py
│       └── eval.py
├── results/                        # Results organized by experiment
│   ├── ddpm_mean/
│   │   ├── figures/               # Training visualizations
│   │   ├── checkpoints/           # Model checkpoints
│   │   └── metrics_ddpm_mean.json
│   ├── ddpm_mean_fixed_variance/
│   │   ├── figures/
│   │   ├── checkpoints/
│   │   └── metrics_ddpm_mean_fixed_variance.json
│   ├── ddpm_noise_pred/
│   │   ├── figures/
│   │   ├── checkpoints/
│   │   └── metrics_ddpm_noise_pred.json
│   └── comparisons/                # Cross-experiment comparisons
├── common.py                       # Shared experiment utilities
├── compare_models.py               # Model comparison script
├── models.py                       # Model architectures
├── loss.py                         # Loss functions
├── sampling.py                     # Sampling functions
├── utils.py                        # General utilities
├── diagnostics.py                  # Training diagnostics
├── plotting.py                     # Plotting utilities
├── metrics.py                      # Evaluation metrics
└── ema.py                          # Exponential moving average
```

## Running Experiments

### Training

To train a specific experiment:

```bash
# Mean+Variance prediction
cd experiments/ddpm_mean
python train.py

# Fixed variance
cd experiments/ddpm_mean_fixed_variance
python train.py

# Noise prediction
cd experiments/ddpm_noise_pred
python train.py
```

### Evaluation

After training, evaluate the model:

```bash
# Mean+Variance prediction
cd experiments/ddpm_mean
python eval.py

# Fixed variance
cd experiments/ddpm_mean_fixed_variance
python eval.py

# Noise prediction
cd experiments/ddpm_noise_pred
python eval.py
```

### Comparing Models

After running evaluation for all experiments, compare the results:

```bash
# Run from the root diffusion directory
python compare_models.py
```

This will generate:
- Comparison plots in `results/comparisons/`
- A summary table in `results/comparisons/metrics_summary.txt`

## Experiment Types

### 1. DDPM Mean+Variance (`ddpm_mean`)
- **Model**: `ConditionalModel` (outputs both mean and log-variance)
- **Loss**: Variational lower bound (KL divergence)
- **Description**: Learns both the mean and variance of the denoising distribution

### 2. DDPM Fixed Variance (`ddpm_mean_fixed_variance`)
- **Model**: `ConditionalModelMeanOnly` (outputs only mean)
- **Loss**: MSE between predicted and true mean
- **Variance**: Fixed according to posterior variance schedule
- **Description**: Simpler model that only learns the mean

### 3. DDPM Noise Prediction (`ddpm_noise_pred`)
- **Model**: `ConditionalModelNoisePredictor` (outputs predicted noise)
- **Loss**: MSE between predicted and true noise
- **Description**: Predicts the noise added during diffusion (most common formulation)

