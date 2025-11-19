"""
Comparison script for all three DDPM models.

Loads evaluation metrics from all models and creates comparison plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Directories
results_dir = Path("./results")
comparison_dir = results_dir / "comparisons"
comparison_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("MODEL COMPARISON")
print("="*70)

# --------------------------------------------------
# 1. Load metrics from all models
# --------------------------------------------------
print("\n[1/3] Loading metrics from all models...")

models_info = [
    {
        'name': 'Mean+Variance',
        'experiment': 'ddpm_mean',
        'color': '#1f77b4',
        'marker': 'o'
    },
    {
        'name': 'Fixed Variance',
        'experiment': 'ddpm_mean_fixed_variance',
        'color': '#ff7f0e',
        'marker': 's'
    },
    {
        'name': 'Noise Prediction',
        'experiment': 'ddpm_noise_pred',
        'color': '#2ca02c',
        'marker': '^'
    },
]

# Load all metrics
metrics_data = {}
for model in models_info:
    # New structure: results/<experiment_name>/metrics_<experiment_name>.json
    file_path = results_dir / model['experiment'] / f"metrics_{model['experiment']}.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            metrics_data[model['name']] = json.load(f)
        print(f"  ✓ Loaded {model['name']}")
    else:
        print(f"  ✗ Missing {model['name']} - {file_path}")

if len(metrics_data) == 0:
    print("\nERROR: No metrics found. Please run evaluation scripts first:")
    print("  - python experiments/ddpm_mean/eval.py")
    print("  - python experiments/ddpm_mean_fixed_variance/eval.py")
    print("  - python experiments/ddpm_noise_pred/eval.py")
    exit(1)

print(f"\nLoaded metrics for {len(metrics_data)} models")

# --------------------------------------------------
# 2. Create comparison table
# --------------------------------------------------
print("\n[2/3] Creating comparison table...")

# Define metric groups
metric_groups = {
    'Distribution Distance': [
        ('mmd', 'MMD', 'lower'),
        ('sliced_wasserstein', 'Sliced Wasserstein', 'lower'),
        ('wasserstein_avg_per_dim', 'Wasserstein (per-dim)', 'lower'),
    ],
    'Histogram Metrics': [
        ('chi_square', 'Chi-Square', 'lower'),
        ('kl_divergence', 'KL Divergence', 'lower'),
        ('js_divergence', 'JS Divergence', 'lower'),
        ('histogram_intersection', 'Histogram Intersection', 'higher'),
    ],
    'Quality Metrics': [
        ('precision', 'Precision', 'higher'),
        ('recall', 'Recall', 'higher'),
        ('f1_score', 'F1 Score', 'higher'),
        ('coverage', 'Coverage', 'higher'),
    ],
    'Log Likelihood': [
        ('bpd_mean', 'Bits per Dimension (bpd)', 'lower'),
        ('vlb_mean', 'VLB (nats)', 'higher'),
    ],
}

# Print comparison table
print("\n" + "="*70)
print("METRICS COMPARISON")
print("="*70)

for group_name, metrics in metric_groups.items():
    print(f"\n{group_name}:")
    print("-" * 70)

    for metric_key, metric_name, better in metrics:
        # Check if metric exists in all models
        if not all(metric_key in metrics_data[model] for model in metrics_data):
            continue

        # Get values
        values = {model: metrics_data[model][metric_key] for model in metrics_data}

        # Find best
        if better == 'lower':
            best_model = min(values, key=values.get)
        else:
            best_model = max(values, key=values.get)

        # Print
        print(f"\n  {metric_name}:")
        for model in metrics_data:
            value = values[model]
            marker = " ★" if model == best_model else "  "
            print(f"    {model:20s}: {value:12.6f}{marker}")

# --------------------------------------------------
# 3. Create comparison plots
# --------------------------------------------------
print("\n[3/3] Creating comparison plots...")

# Figure 1: Distribution Distance Metrics
fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
fig1.suptitle('Distribution Distance Metrics (Lower is Better)', fontsize=14, fontweight='bold')

distance_metrics = [
    ('mmd', 'Maximum Mean\nDiscrepancy (MMD)'),
    ('sliced_wasserstein', 'Sliced Wasserstein\nDistance'),
    ('wasserstein_avg_per_dim', 'Wasserstein Distance\n(Per-Dimension Average)'),
]

for idx, (metric_key, metric_title) in enumerate(distance_metrics):
    ax = axes[idx]

    model_names = []
    values = []
    colors = []

    for model_info in models_info:
        model_name = model_info['name']
        if model_name in metrics_data and metric_key in metrics_data[model_name]:
            model_names.append(model_name)
            values.append(metrics_data[model_name][metric_key])
            colors.append(model_info['color'])

    bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(metric_title)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(comparison_dir / "comparison_distance_metrics.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {comparison_dir / 'comparison_distance_metrics.png'}")
plt.close()

# Figure 2: Quality Metrics (Precision, Recall, F1)
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle('Quality Metrics (Higher is Better)', fontsize=14, fontweight='bold')

quality_metrics = [
    ('precision', 'Precision\n(Realism)'),
    ('recall', 'Recall\n(Coverage)'),
    ('f1_score', 'F1 Score\n(Harmonic Mean)'),
]

for idx, (metric_key, metric_title) in enumerate(quality_metrics):
    ax = axes[idx]

    model_names = []
    values = []
    colors = []

    for model_info in models_info:
        model_name = model_info['name']
        if model_name in metrics_data and metric_key in metrics_data[model_name]:
            model_names.append(model_name)
            values.append(metrics_data[model_name][metric_key])
            colors.append(model_info['color'])

    bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(metric_title)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(comparison_dir / "comparison_quality_metrics.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {comparison_dir / 'comparison_quality_metrics.png'}")
plt.close()

# Figure 3: Histogram-based Metrics
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
fig3.suptitle('Histogram-Based Metrics', fontsize=14, fontweight='bold')

histogram_metrics = [
    ('chi_square', 'Chi-Square Distance\n(Lower is Better)', 0),
    ('kl_divergence', 'KL Divergence\n(Lower is Better)', 1),
    ('js_divergence', 'JS Divergence\n(Lower is Better)', 2),
    ('histogram_intersection', 'Histogram Intersection\n(Higher is Better)', 3),
]

for metric_key, metric_title, idx in histogram_metrics:
    ax = axes[idx // 2, idx % 2]

    model_names = []
    values = []
    colors = []

    for model_info in models_info:
        model_name = model_info['name']
        if model_name in metrics_data and metric_key in metrics_data[model_name]:
            model_names.append(model_name)
            values.append(metrics_data[model_name][metric_key])
            colors.append(model_info['color'])

    bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(metric_title)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(comparison_dir / "comparison_histogram_metrics.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {comparison_dir / 'comparison_histogram_metrics.png'}")
plt.close()

# Figure 4: Log Likelihood Metrics
fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
fig4.suptitle('Log Likelihood Metrics', fontsize=14, fontweight='bold')

likelihood_metrics = [
    ('bpd_mean', 'Bits per Dimension\n(Lower is Better)'),
    ('vlb_mean', 'VLB (nats)\n(Higher is Better)'),
]

for idx, (metric_key, metric_title) in enumerate(likelihood_metrics):
    ax = axes[idx]

    model_names = []
    values = []
    colors = []

    for model_info in models_info:
        model_name = model_info['name']
        if model_name in metrics_data and metric_key in metrics_data[model_name]:
            model_names.append(model_name)
            values.append(metrics_data[model_name][metric_key])
            colors.append(model_info['color'])

    if len(values) > 0:
        bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title(metric_title)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(comparison_dir / "comparison_log_likelihood.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {comparison_dir / 'comparison_log_likelihood.png'}")
plt.close()

# Figure 5: Radar chart for overall comparison
fig5, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
fig5.suptitle('Overall Model Comparison (Radar Chart)\nNormalized to Best Model Performance',
              fontsize=14, fontweight='bold', y=0.98)

# Define metrics with their ideal values for scoring
# For each metric: (key, display_name, direction)
# direction: 'lower' or 'higher' is better
radar_metrics = [
    ('mmd', 'MMD', 'lower'),
    ('sliced_wasserstein', 'Sliced\nWasserstein', 'lower'),
    ('js_divergence', 'JS\nDivergence', 'lower'),
    ('precision', 'Precision', 'higher'),
    ('recall', 'Recall', 'higher'),
    ('f1_score', 'F1 Score', 'higher'),
    ('bpd_mean', 'Bits/dim', 'lower'),
]

# Compute scores for each model (0-100 scale)
# Score = absolute value normalized to best model (100 = best model)
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for model_info in models_info:
    model_name = model_info['name']
    if model_name not in metrics_data:
        continue

    scores = []
    for metric_key, _, direction in radar_metrics:
        if metric_key not in metrics_data[model_name]:
            scores.append(0)  # Zero score if missing
            continue

        # Collect all model values for this metric
        all_model_values = []
        for m in metrics_data:
            if metric_key in metrics_data[m]:
                all_model_values.append(metrics_data[m][metric_key])

        if len(all_model_values) == 0:
            scores.append(0)
            continue

        value = metrics_data[model_name][metric_key]
        best_value = min(all_model_values) if direction == 'lower' else max(all_model_values)

        # Compute score normalized to best model
        # Best model gets 100, others get proportionally less
        if abs(best_value) > 1e-10:
            if direction == 'lower':
                # For lower is better: score = 100 * (best / value)
                # Best model: score = 100 * (best / best) = 100
                # Worse model: score = 100 * (best / worse) < 100
                score = 100 * (best_value / value)
            else:
                # For higher is better: score = 100 * (value / best)
                # Best model: score = 100 * (best / best) = 100
                # Worse model: score = 100 * (worse / best) < 100
                score = 100 * (value / best_value)
        else:
            score = 0  # Best value is zero, can't normalize

        # Clamp to reasonable range [0, 100]
        score = max(0, min(100, score))
        scores.append(score)

    scores += scores[:1]  # Complete the circle

    ax.plot(angles, scores, 'o-', linewidth=2.5, label=model_name,
            color=model_info['color'], markersize=8)
    ax.fill(angles, scores, alpha=0.2, color=model_info['color'])

# Set labels and styling
ax.set_xticks(angles[:-1])
ax.set_xticklabels([label for _, label, _ in radar_metrics], fontsize=11)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.grid(True, linestyle='--', alpha=0.7)

# Add reference circles for relative performance
ax.plot(np.linspace(0, 2*np.pi, 100), [90]*100, 'k--', linewidth=0.8, alpha=0.2)
ax.plot(np.linspace(0, 2*np.pi, 100), [80]*100, 'k--', linewidth=0.8, alpha=0.2)

ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11)

# Add explanation text
fig5.text(0.5, 0.02, 'Score: 100 = Best model (reference), others normalized to best model absolute performance',
          ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(comparison_dir / "comparison_radar_chart.png", dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {comparison_dir / 'comparison_radar_chart.png'}")
plt.close()

# --------------------------------------------------
# 4. Create summary table file
# --------------------------------------------------
print("\n[4/4] Creating summary table...")

summary_file = comparison_dir / "metrics_summary.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DDPM MODELS COMPARISON SUMMARY\n")
    f.write("="*80 + "\n\n")

    for group_name, metrics in metric_groups.items():
        f.write(f"\n{group_name}:\n")
        f.write("-" * 80 + "\n")

        for metric_key, metric_name, better in metrics:
            # Check if metric exists
            if not all(metric_key in metrics_data[model] for model in metrics_data):
                continue

            # Get values
            values = {model: metrics_data[model][metric_key] for model in metrics_data}

            # Find best
            if better == 'lower':
                best_model = min(values, key=values.get)
                improvement_ref = max(values, key=values.get)
                improvement = ((values[improvement_ref] - values[best_model]) / values[improvement_ref]) * 100
            else:
                best_model = max(values, key=values.get)
                improvement_ref = min(values, key=values.get)
                if values[improvement_ref] > 0:
                    improvement = ((values[best_model] - values[improvement_ref]) / values[improvement_ref]) * 100
                else:
                    improvement = 0

            f.write(f"\n{metric_name}:\n")
            for model in metrics_data:
                value = values[model]
                marker = " ★ BEST" if model == best_model else ""
                f.write(f"  {model:25s}: {value:12.6f}{marker}\n")
            f.write(f"  Best model improvement: {improvement:.1f}%\n")

    f.write("\n" + "="*80 + "\n")
    f.write("Legend:\n")
    f.write("  ★ - Best performing model for this metric\n")
    f.write("  Improvement - Percentage improvement of best model over worst\n")
    f.write("="*80 + "\n")

print(f"  ✓ Saved: {summary_file}")

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)
print(f"\nGenerated files:")
print(f"  • {comparison_dir / 'comparison_distance_metrics.png'}")
print(f"  • {comparison_dir / 'comparison_quality_metrics.png'}")
print(f"  • {comparison_dir / 'comparison_histogram_metrics.png'}")
print(f"  • {comparison_dir / 'comparison_log_likelihood.png'}")
print(f"  • {comparison_dir / 'comparison_radar_chart.png'}")
print(f"  • {comparison_dir / 'metrics_summary.txt'}")
print("="*70)