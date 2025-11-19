# Diffusion Model Training Diagnostics Guide

This short guide explains how to interpret the diagnostic plots and metrics generated 
during training.

## 📊 Diagnostic Plots Explained

### 1. **Training Curves**

4 subplots tracking metrics over time

**Subplots:**
- **Top-left (Loss)**: Training loss evolution
  - ✅ **Healthy**: Smooth decrease, converges to low value
  - ⚠️ **Warning**: Sudden spikes after initial decrease
  - 🔴 **Problem**: Loss increases after convergence

- **Top-right (Learning Rate)**: LR schedule

- **Bottom-left (Gradient Norm)**: Magnitude of gradients
  - ✅ **Healthy**: Stable, moderate values (0.1-10)
  - ⚠️ **Warning**: Increasing trend → model struggling
  - 🔴 **Problem**: Very high (>100) → **gradient explosion**
  - 🔴 **Problem**: Very low (<0.001) → **vanishing gradients**

- **Bottom-right (Log-Variance Stats)**: Model's predicted log σ²
  - ✅ **Healthy**: Stays within [-10, 0], stable std
  - ⚠️ **Warning**: Mean approaching clamp limits (-20 or 2)
  - 🔴 **Problem**: Std → 0 → **variance collapse** (all predictions same)
  - 🔴 **Problem**: Hitting clamps → model trying to predict extreme values

---

### 2. **Sample Quality Diagnostics** (`sample_quality_epoch_X.png`)

Side-by-side comparison of real vs generated data

**Components:**
- **Left (Real Data)**: Ground truth Swiss roll
- **Middle (Generated)**: Model's samples
- **Right (Marginal Distributions)**: 1D histograms for each dimension 

---

### 3. **Model Predictions Analysis** (`model_predictions_epoch_X.png`)

**What it shows:** 6 subplots analyzing model outputs

**Subplots:**

1. **Input x_t**: Noisy inputs colored by timestep
   - All timesteps should be represented
   - Initial timesteps (t~T) should be more spread out
   - Final timesteps (t~0) should cluster near data manifold

2. **Predicted Mean**: Model's denoised predictions
   - ✅ **Healthy**: Spreads similar to real data
   - Can be misleading, as the model might learn the identity mapping

3. **Predicted Log-Var (dim 0 & dim 1)**: Histogram of variance predictions
   - ✅ **Healthy**: Bell curve between [-10, 0]
   - ⚠️ **Warning**: Peaks at -20 or 2, hitting clamping limits frequently
   - 🔴 **Problem**: Narrow spike → all predictions have same variance

4. **Log-Variance vs Timestep**: Scatter plot
   - ✅ **Healthy**: Higher variance (closer to 2) at large t, lower at small t
   - 🔴 **Problem**: Flat line → not learning timestep-dependent variance

5. **Mean Magnitude vs Timestep**: ||μ||₂ vs t
   - ✅ **Healthy**: Decreases as t → 0 (denoising more aggressively)
   - 🔴 **Problem**: Flat or increasing → model not denoising properly

---

### 4. **Loss per Timestep** (`loss_per_timestep_epoch_X.png`)

**What it shows:** KL divergence for each timestep t=0...T

✅ **Healthy pattern:**
- Relatively uniform loss across timesteps
- Slightly higher loss at early timesteps (t near 0) is OK

🔴 **Problem patterns:**
- **Spike at specific timesteps** → model struggles with those timesteps
  - If spike at t~0: Can't predict final denoising step
  - If spike at t~T: Can't handle very noisy inputs
  - If spike in middle: Specific noise levels are hard

- **Exponentially increasing at low t** → Final denoising steps failing

---