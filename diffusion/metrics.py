"""
Evaluation metrics for 2D generative models.

Provides quantitative metrics to evaluate quality of generated samples:
- 2D Histogram Comparison (Chi-square, KL, JS divergence)
- Maximum Mean Discrepancy (MMD)
- 2D Wasserstein Distance
- Precision and Recall for Distributions
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
from sklearn.neighbors import NearestNeighbors


def compute_2d_histogram_metrics(real_samples, generated_samples, bins=50, range_percentile=99):
    """
    Compute 2D histogram comparison metrics between real and generated samples.

    Parameters
    ----------
    real_samples : torch.Tensor or np.ndarray
        Real data samples of shape (n_samples, 2)
    generated_samples : torch.Tensor or np.ndarray
        Generated samples of shape (n_samples, 2)
    bins : int
        Number of bins for histogram in each dimension
    range_percentile : float
        Percentile to use for determining histogram range (handles outliers)

    Returns
    -------
    dict with:
        - chi_square: Chi-square distance
        - kl_divergence: KL divergence (real || generated)
        - js_divergence: Jensen-Shannon divergence (symmetric)
        - histogram_intersection: Histogram intersection (similarity measure)
    """
    # Convert to numpy if needed
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()

    # Determine histogram range from real data (robust to outliers)
    all_data = np.concatenate([real_samples, generated_samples], axis=0)
    x_min, x_max = np.percentile(all_data[:, 0], [100-range_percentile, range_percentile])
    y_min, y_max = np.percentile(all_data[:, 1], [100-range_percentile, range_percentile])
    hist_range = [[x_min, x_max], [y_min, y_max]]

    # Compute 2D histograms
    hist_real, xedges, yedges = np.histogram2d(
        real_samples[:, 0], real_samples[:, 1],
        bins=bins, range=hist_range, density=True
    )
    hist_gen, _, _ = np.histogram2d(
        generated_samples[:, 0], generated_samples[:, 1],
        bins=bins, range=hist_range, density=True
    )

    # Normalize to probability distributions
    hist_real = hist_real / (hist_real.sum() + 1e-10)
    hist_gen = hist_gen / (hist_gen.sum() + 1e-10)

    # Add small constant to avoid log(0) and division by zero
    eps = 1e-10
    hist_real_smooth = hist_real + eps
    hist_gen_smooth = hist_gen + eps

    # 1. Chi-square distance
    chi_square = np.sum((hist_real - hist_gen) ** 2 / (hist_real_smooth + hist_gen_smooth))

    # 2. KL divergence: D_KL(real || generated)
    kl_divergence = np.sum(hist_real_smooth * np.log(hist_real_smooth / hist_gen_smooth))

    # 3. Jensen-Shannon divergence (symmetric version of KL)
    hist_mean = 0.5 * (hist_real_smooth + hist_gen_smooth)
    js_divergence = 0.5 * (
        np.sum(hist_real_smooth * np.log(hist_real_smooth / hist_mean)) +
        np.sum(hist_gen_smooth * np.log(hist_gen_smooth / hist_mean))
    )

    # 4. Histogram Intersection (similarity measure, higher is better)
    histogram_intersection = np.sum(np.minimum(hist_real, hist_gen))

    return {
        'chi_square': float(chi_square),
        'kl_divergence': float(kl_divergence),
        'js_divergence': float(js_divergence),
        'histogram_intersection': float(histogram_intersection),
    }


def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute Gaussian (RBF) kernel matrix between samples x and y.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    # Compute pairwise squared distances
    dists_sq = cdist(x, y, 'sqeuclidean')
    return np.exp(-dists_sq / (2 * sigma ** 2))


def compute_mmd(real_samples, generated_samples, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between real and generated samples.

    MMD measures the distance between two distributions using kernel embeddings.
    Lower MMD indicates more similar distributions.

    Parameters
    ----------
    real_samples : torch.Tensor or np.ndarray
        Real data samples of shape (n_samples, 2)
    generated_samples : torch.Tensor or np.ndarray
        Generated samples of shape (n_samples, 2)
    sigma : float
        Bandwidth parameter for Gaussian kernel

    Returns
    -------
    dict with:
        - mmd: Maximum Mean Discrepancy value
        - mmd_std: Standard deviation (from unbiased estimator)
    """
    # Convert to numpy if needed
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()

    # Sample subset for efficiency (if datasets are large)
    n_max = 5000
    if len(real_samples) > n_max:
        indices_real = np.random.choice(len(real_samples), n_max, replace=False)
        real_samples = real_samples[indices_real]
    if len(generated_samples) > n_max:
        indices_gen = np.random.choice(len(generated_samples), n_max, replace=False)
        generated_samples = generated_samples[indices_gen]

    n_real = len(real_samples)
    n_gen = len(generated_samples)

    # Compute kernel matrices
    K_real_real = gaussian_kernel(real_samples, real_samples, sigma)
    K_gen_gen = gaussian_kernel(generated_samples, generated_samples, sigma)
    K_real_gen = gaussian_kernel(real_samples, generated_samples, sigma)

    # Unbiased MMD^2 estimator
    # Remove diagonal elements for unbiased estimate
    K_real_real_offdiag = K_real_real - np.diag(np.diag(K_real_real))
    K_gen_gen_offdiag = K_gen_gen - np.diag(np.diag(K_gen_gen))

    mmd_sq = (
        K_real_real_offdiag.sum() / (n_real * (n_real - 1)) +
        K_gen_gen_offdiag.sum() / (n_gen * (n_gen - 1)) -
        2 * K_real_gen.sum() / (n_real * n_gen)
    )

    # MMD can be negative due to sampling variance, take max with 0
    mmd = np.sqrt(max(mmd_sq, 0))

    return {
        'mmd': float(mmd),
        'mmd_squared': float(mmd_sq),
    }


def compute_wasserstein_2d(real_samples, generated_samples, num_projections=1000):
    """
    Compute 2D Wasserstein distance using sliced Wasserstein approximation.

    The sliced Wasserstein distance projects samples onto random 1D directions
    and computes the average 1D Wasserstein distance across projections.

    Parameters
    ----------
    real_samples : torch.Tensor or np.ndarray
        Real data samples of shape (n_samples, 2)
    generated_samples : torch.Tensor or np.ndarray
        Generated samples of shape (n_samples, 2)
    num_projections : int
        Number of random projections to use

    Returns
    -------
    dict with:
        - sliced_wasserstein: Sliced Wasserstein distance
        - wasserstein_per_dim: List of Wasserstein distances for each original dimension
    """
    # Convert to numpy if needed
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()

    # 1. Sliced Wasserstein Distance
    # Generate random projection directions (unit vectors)
    np.random.seed(42)  # For reproducibility
    thetas = np.random.randn(num_projections, 2)
    thetas = thetas / np.linalg.norm(thetas, axis=1, keepdims=True)

    wasserstein_distances = []

    for theta in thetas:
        # Project samples onto direction theta
        proj_real = real_samples @ theta
        proj_gen = generated_samples @ theta

        # Sort projections
        proj_real_sorted = np.sort(proj_real)
        proj_gen_sorted = np.sort(proj_gen)

        # Make same length by interpolation if needed
        if len(proj_real_sorted) != len(proj_gen_sorted):
            n = min(len(proj_real_sorted), len(proj_gen_sorted))
            proj_real_sorted = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(proj_real_sorted)),
                proj_real_sorted
            )
            proj_gen_sorted = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(proj_gen_sorted)),
                proj_gen_sorted
            )

        # Compute 1D Wasserstein distance (L1 distance between sorted samples)
        w_dist = np.mean(np.abs(proj_real_sorted - proj_gen_sorted))
        wasserstein_distances.append(w_dist)

    sliced_wasserstein = np.mean(wasserstein_distances)

    # 2. Per-dimension Wasserstein distances (for reference)
    wasserstein_per_dim = []
    for dim in range(2):
        real_dim = np.sort(real_samples[:, dim])
        gen_dim = np.sort(generated_samples[:, dim])

        # Make same length
        if len(real_dim) != len(gen_dim):
            n = min(len(real_dim), len(gen_dim))
            real_dim = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(real_dim)), real_dim)
            gen_dim = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(gen_dim)), gen_dim)

        w_dist_dim = np.mean(np.abs(real_dim - gen_dim))
        wasserstein_per_dim.append(float(w_dist_dim))

    return {
        'sliced_wasserstein': float(sliced_wasserstein),
        'wasserstein_dim_0': wasserstein_per_dim[0],
        'wasserstein_dim_1': wasserstein_per_dim[1],
        'wasserstein_avg_per_dim': float(np.mean(wasserstein_per_dim)),
    }


def compute_precision_recall(real_samples, generated_samples, k=5):
    """
    Compute precision and recall for distributions.

    - Precision: Fraction of generated samples that are close to real samples
                 (measures if generated samples are realistic)
    - Recall: Fraction of real samples that are close to generated samples
              (measures if generator covers all modes)

    Parameters
    ----------
    real_samples : torch.Tensor or np.ndarray
        Real data samples of shape (n_samples, 2)
    generated_samples : torch.Tensor or np.ndarray
        Generated samples of shape (n_samples, 2)
    k : int
        Number of nearest neighbors to consider

    Returns
    -------
    dict with:
        - precision: Precision score [0, 1]
        - recall: Recall score [0, 1]
        - f1_score: Harmonic mean of precision and recall
        - coverage: Fraction of real data modes covered
    """
    # Convert to numpy if needed
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()

    # Sample subset for efficiency
    n_max = 10000
    if len(real_samples) > n_max:
        indices_real = np.random.choice(len(real_samples), n_max, replace=False)
        real_samples_subset = real_samples[indices_real]
    else:
        real_samples_subset = real_samples

    if len(generated_samples) > n_max:
        indices_gen = np.random.choice(len(generated_samples), n_max, replace=False)
        generated_samples_subset = generated_samples[indices_gen]
    else:
        generated_samples_subset = generated_samples

    # Build KNN models
    real_nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    real_nn.fit(real_samples_subset)

    gen_nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    gen_nn.fit(generated_samples_subset)

    # Compute precision: how many generated samples have a real sample nearby
    # Find k-th nearest neighbor distance in real data
    real_distances, _ = real_nn.kneighbors(real_samples_subset)
    real_radius = real_distances[:, -1]  # k-th nearest neighbor distance

    # For each generated sample, check if it's within the k-NN radius of some real sample
    gen_distances, _ = real_nn.kneighbors(generated_samples_subset)
    gen_nearest_dist = gen_distances[:, 0]  # Distance to nearest real sample

    # Compare with the radius at the nearest real sample
    nearest_indices = real_nn.kneighbors(generated_samples_subset, n_neighbors=1, return_distance=False)
    nearest_radii = real_radius[nearest_indices.flatten()]

    precision = np.mean(gen_nearest_dist <= nearest_radii)

    # Compute recall: how many real samples have a generated sample nearby
    # Find k-th nearest neighbor distance in generated data
    gen_distances_internal, _ = gen_nn.kneighbors(generated_samples_subset)
    gen_radius = gen_distances_internal[:, -1]

    # For each real sample, check if it's within the k-NN radius of some generated sample
    real_to_gen_distances, _ = gen_nn.kneighbors(real_samples_subset)
    real_nearest_dist = real_to_gen_distances[:, 0]

    nearest_gen_indices = gen_nn.kneighbors(real_samples_subset, n_neighbors=1, return_distance=False)
    nearest_gen_radii = gen_radius[nearest_gen_indices.flatten()]

    recall = np.mean(real_nearest_dist <= nearest_gen_radii)

    # F1 score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # Coverage: approximate number of modes covered
    # A mode is "covered" if at least one generated sample is nearby
    coverage = recall  # Simple approximation

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'coverage': float(coverage),
    }


def compute_all_metrics(real_samples, generated_samples, verbose=True):
    """
    Compute all evaluation metrics at once.

    Parameters
    ----------
    real_samples : torch.Tensor or np.ndarray
        Real data samples of shape (n_samples, 2)
    generated_samples : torch.Tensor or np.ndarray
        Generated samples of shape (n_samples, 2)
    verbose : bool
        If True, print metrics as they are computed

    Returns
    -------
    dict
        Dictionary containing all computed metrics
    """
    metrics = {}

    if verbose:
        print("\nComputing evaluation metrics...")
        print("-" * 60)

    # 2D Histogram metrics
    if verbose:
        print("  [1/4] Computing 2D histogram metrics...")
    hist_metrics = compute_2d_histogram_metrics(real_samples, generated_samples)
    metrics.update(hist_metrics)

    # MMD
    if verbose:
        print("  [2/4] Computing MMD...")
    mmd_metrics = compute_mmd(real_samples, generated_samples)
    metrics.update(mmd_metrics)

    # 2D Wasserstein
    if verbose:
        print("  [3/4] Computing Wasserstein distances...")
    wasserstein_metrics = compute_wasserstein_2d(real_samples, generated_samples)
    metrics.update(wasserstein_metrics)

    # Precision/Recall
    if verbose:
        print("  [4/4] Computing Precision/Recall...")
    pr_metrics = compute_precision_recall(real_samples, generated_samples)
    metrics.update(pr_metrics)

    if verbose:
        print("-" * 60)
        print("\nMetrics Summary:")
        print(f"  Histogram Metrics:")
        print(f"    Chi-square distance:      {metrics['chi_square']:.6f}")
        print(f"    KL divergence:            {metrics['kl_divergence']:.6f}")
        print(f"    JS divergence:            {metrics['js_divergence']:.6f}")
        print(f"    Histogram intersection:   {metrics['histogram_intersection']:.6f}")
        print(f"  MMD:                        {metrics['mmd']:.6f}")
        print(f"  Wasserstein (sliced):       {metrics['sliced_wasserstein']:.6f}")
        print(f"  Wasserstein (per-dim avg):  {metrics['wasserstein_avg_per_dim']:.6f}")
        print(f"  Precision:                  {metrics['precision']:.6f}")
        print(f"  Recall:                     {metrics['recall']:.6f}")
        print(f"  F1 Score:                   {metrics['f1_score']:.6f}")

    return metrics