from __future__ import annotations

"""
Batch position refinement comparison across entire dataset.

Scans all samples in a dataset folder and compares three position-finding methods:
1. Discrete heatmap peak (argmax)
2. Softmax expectation refinement (probabilistic weighting)
3. Gaussian peak refinement (multi-stage sub-pixel estimator)

Computes comprehensive statistics for each method:
- RMS errors (X, Y, Distance)
- P50 (median) values
- P99 (99th percentile) values

Plots sparse scatter showing prediction errors (distance from ground truth center).
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure package imports work when executed directly
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.infer_sample_vis import (  # noqa: E402
    compute_embeddings_and_correlation,
    load_checkpoint,
    load_sample,
    _softmax_refinement,
)

from Train.gaussian_peak_refine import gaussian_peak_refine  # noqa: E402
from Train.theta_peak_refine import theta_peak_refine  # noqa: E402


def find_all_samples(dataset_folder: Path) -> List[Path]:
    """
    Find all sample directories in the dataset folder.

    Args:
        dataset_folder: Root folder containing sample subdirectories.

    Returns:
        List of sample directory paths.
    """
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

    # Each sample is a directory containing .npy files
    sample_dirs = [d for d in dataset_folder.iterdir() if d.is_dir()]

    if not sample_dirs:
        raise ValueError(f"No sample directories found in {dataset_folder}")

    return sorted(sample_dirs)


def process_single_sample(
    model,
    sample_path: Path,
    lidar_variant: str,
    device: str,
    resolution: float,
) -> dict:
    """
    Process a single sample and extract position and rotation estimates from three methods.

    Args:
        model: Trained ReLL model.
        sample_path: Path to sample directory.
        lidar_variant: LiDAR variant to use ('gicp' or 'non_aligned').
        device: Computation device.
        resolution: Map resolution in meters per pixel.

    Returns:
        Dictionary containing estimates:
        - 'peak': (x, y) from discrete heatmap peak
        - 'softmax': (x, y) from softmax refinement
        - 'gaussian': (x, y) from Gaussian fit
        - 'theta': predicted rotation angle in degrees
        - 'success': True if processing succeeded
    """
    try:
        # Load sample
        sample = load_sample(sample_path, lidar_variant=lidar_variant)
        lidar_tensor = sample["lidar"]

        # Compute LiDAR fill rate (percentage of valid pixels)
        if lidar_tensor.dim() >= 3 and lidar_tensor.shape[0] >= 3:
            mask_channel = lidar_tensor[2]
            fill_rate = float((mask_channel >= 0.5).float().mean().item())
        else:
            reference_channel = lidar_tensor[0]
            finite_mask = torch.isfinite(reference_channel)
            fill_rate = float(finite_mask.float().mean().item())

        lidar = lidar_tensor.to(device)
        geospatial = sample["map"].to(device)

        # Compute correlation heatmap
        results = compute_embeddings_and_correlation(model, lidar, geospatial)
        heatmap = results["correlation_heatmap"]

        # Extract dimensions
        H, W = heatmap.shape
        center_row = H // 2
        center_col = W // 2

        # Method 1: Discrete peak
        peak_idx = int(torch.argmax(heatmap))
        peak_row = peak_idx // W
        peak_col = peak_idx % W
        peak_y_offset_px = peak_row - center_row
        peak_x_offset_px = peak_col - center_col
        peak_x_m = peak_x_offset_px * resolution
        peak_y_m = peak_y_offset_px * resolution

        # Method 2: Softmax refinement
        (
            softmax_probs,
            softmax_mu_x_px,
            softmax_mu_y_px,
            softmax_sigma_x_px,
            softmax_sigma_y_px,
        ) = _softmax_refinement(heatmap)
        softmax_x_m = softmax_mu_x_px * resolution
        softmax_y_m = softmax_mu_y_px * resolution

        # Method 3: Gaussian peak refinement (multi-stage)
        gaussian_result = gaussian_peak_refine(heatmap)
        gaussian_x_m = gaussian_result.x * resolution
        gaussian_y_m = gaussian_result.y * resolution

        # Theta (rotation) prediction - run full model forward pass
        lidar_batch = lidar.unsqueeze(0)  # Add batch dimension
        geospatial_batch = geospatial.unsqueeze(0)
        with torch.no_grad():
            predictions = model(lidar_batch, geospatial_batch)

        theta_logits = predictions["theta_logits"].squeeze(0).cpu()
        theta_candidates_deg = predictions["theta_candidates_deg"].cpu()

        # Apply theta peak refinement
        theta_result = theta_peak_refine(theta_logits, theta_candidates_deg)
        theta_pred_deg = theta_result.theta_deg

        # Ground truth is 0° (samples are centered without rotation)
        theta_error_deg = theta_pred_deg - 0.0

        return {
            'peak': (float(peak_x_m), float(peak_y_m)),
            'softmax': (float(softmax_x_m), float(softmax_y_m)),
            'gaussian': (float(gaussian_x_m), float(gaussian_y_m)),
            'theta': float(theta_error_deg),
            'fill_rate': float(fill_rate),
            'success': True,
        }

    except Exception as e:
        print(f"[Warning] Failed to process {sample_path.name}: {e}")
        return {
            'peak': (0.0, 0.0),
            'softmax': (0.0, 0.0),
            'gaussian': (0.0, 0.0),
            'theta': 0.0,
            'fill_rate': 0.0,
            'success': False,
        }


def plot_refinement_comparison(
    peak_positions: np.ndarray,
    softmax_positions: np.ndarray,
    gaussian_positions: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Plot sparse scatter comparison of three position-finding methods.

    Args:
        peak_positions: Nx2 array of (x, y) errors from discrete peak method.
        softmax_positions: Nx2 array of (x, y) errors from softmax method.
        gaussian_positions: Nx2 array of (x, y) errors from Gaussian method.
        save_path: Optional path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot three methods with different colors and markers
    ax.scatter(
        peak_positions[:, 0],
        peak_positions[:, 1],
        c='blue',
        marker='o',
        s=50,
        alpha=0.6,
        label='Discrete Peak',
        edgecolors='black',
        linewidths=0.5,
    )

    ax.scatter(
        softmax_positions[:, 0],
        softmax_positions[:, 1],
        c='red',
        marker='^',
        s=50,
        alpha=0.6,
        label='Softmax Refinement',
        edgecolors='black',
        linewidths=0.5,
    )

    ax.scatter(
        gaussian_positions[:, 0],
        gaussian_positions[:, 1],
        c='green',
        marker='s',
        s=50,
        alpha=0.6,
        label='Gaussian Surface Fit (LS)',
        edgecolors='black',
        linewidths=0.5,
    )

    # Mark ground truth center
    ax.scatter(
        [0],
        [0],
        c='gold',
        marker='*',
        s=400,
        edgecolors='black',
        linewidths=2,
        label='Ground Truth (0, 0)',
        zorder=10,
    )

    # Add grid and labels
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    ax.set_xlabel('X Error (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Error (m)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Position Refinement Comparison (N={len(peak_positions)} samples)',
        fontsize=14,
        fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[Save] Refinement comparison plot saved to: {save_path}")

    plt.show()


def compute_statistics(positions: np.ndarray, method_name: str) -> None:
    """
    Compute and print comprehensive statistics for a position-finding method.

    Args:
        positions: Nx2 array of (x, y) errors from ground truth.
        method_name: Name of the method for display.
    """
    # Separate x and y errors
    x_errors = positions[:, 0]
    y_errors = positions[:, 1]

    # Compute 2D Euclidean distances from origin
    distances = np.sqrt(x_errors**2 + y_errors**2)

    # RMS (Root Mean Square) errors
    rms_x = np.sqrt(np.mean(x_errors**2))
    rms_y = np.sqrt(np.mean(y_errors**2))
    rms_dist = np.sqrt(np.mean(distances**2))

    # Percentile statistics
    # X direction
    p50_x = np.percentile(np.abs(x_errors), 50)  # Median absolute error in x
    p99_x = np.percentile(np.abs(x_errors), 99)  # 99th percentile in x

    # Y direction
    p50_y = np.percentile(np.abs(y_errors), 50)  # Median absolute error in y
    p99_y = np.percentile(np.abs(y_errors), 99)  # 99th percentile in y

    # Distance
    p50_dist = np.percentile(distances, 50)  # Median distance
    p99_dist = np.percentile(distances, 99)  # 99th percentile distance

    # Mean and max for reference
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)

    print(f"\n[{method_name}]")
    print(f"  RMS Errors:")
    print(f"    X:        {rms_x:.4f} m")
    print(f"    Y:        {rms_y:.4f} m")
    print(f"    Distance: {rms_dist:.4f} m")
    print(f"  P50 (Median):")
    print(f"    X:        {p50_x:.4f} m")
    print(f"    Y:        {p50_y:.4f} m")
    print(f"    Distance: {p50_dist:.4f} m")
    print(f"  P99 (99th percentile):")
    print(f"    X:        {p99_x:.4f} m")
    print(f"    Y:        {p99_y:.4f} m")
    print(f"    Distance: {p99_dist:.4f} m")
    print(f"  Other:")
    print(f"    Mean dist: {mean_dist:.4f} m")
    print(f"    Max dist:  {max_dist:.4f} m")


def plot_error_distributions(
    theta_errors: np.ndarray,
    x_errors: np.ndarray,
    y_errors: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Plot combined 2D distribution showing relationship between position (x,y) and rotation (theta) errors.

    Creates a scatter plot where:
    - X-axis: x position error (Gaussian method)
    - Y-axis: y position error (Gaussian method)
    - Color: theta rotation error (colormap)

    This reveals correlations between position and rotation accuracy.

    Args:
        theta_errors: N-length array of theta errors in degrees.
        x_errors: N-length array of x errors in meters (Gaussian method).
        y_errors: N-length array of y errors in meters (Gaussian method).
        save_path: Optional path to save the plot.
    """
    # Compute statistics for title
    rms_x = np.sqrt(np.mean(x_errors**2))
    rms_y = np.sqrt(np.mean(y_errors**2))
    rms_theta = np.sqrt(np.mean(theta_errors**2))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create scatter plot with theta as color dimension
    scatter = ax.scatter(
        x_errors,
        y_errors,
        c=theta_errors,
        cmap='coolwarm',  # Blue (negative) to Red (positive) theta errors
        s=60,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5,
    )

    # Add colorbar for theta errors
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Theta Error (°)', fontsize=12, fontweight='bold', rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    # Mark ground truth center
    ax.scatter(
        [0],
        [0],
        c='gold',
        marker='*',
        s=600,
        edgecolors='black',
        linewidths=3,
        label='Ground Truth (0, 0)',
        zorder=10,
    )

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Labels and title
    ax.set_xlabel('X Error (m) - Gaussian Method', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Error (m) - Gaussian Method', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Combined Position & Rotation Error Distribution (N={len(theta_errors)} samples)\n'
        f'RMS: X={rms_x:.4f}m, Y={rms_y:.4f}m, Theta={rms_theta:.3f}°',
        fontsize=14,
        fontweight='bold',
        pad=15,
    )

    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Equal aspect ratio for position errors
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Generate second filename by appending '_distributions' before extension
        base_path = save_path.parent / (save_path.stem + '_distributions' + save_path.suffix)
        plt.savefig(base_path, dpi=200, bbox_inches='tight')
        print(f"[Save] Error distributions plot saved to: {base_path}")

    plt.show()


def plot_fill_rate_vs_distance(
    gaussian_positions: np.ndarray,
    fill_rates: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Analyze fill rate vs radial distance from ground truth.
    This directly answers: does position error correlate with fill rate?

    Args:
        gaussian_positions: Nx2 array of (x, y) errors from Gaussian method (in meters).
        fill_rates: N-length array of LiDAR fill rates (0.0–1.0).
        save_path: Optional path to save the plot (suffix `_fill_vs_dist` will be appended).
    """
    if gaussian_positions.size == 0 or fill_rates.size == 0:
        print("[Plot] Skipping fill rate vs distance analysis (no data).")
        return

    # Compute radial distance from ground truth
    distances = np.sqrt(gaussian_positions[:, 0]**2 + gaussian_positions[:, 1]**2)

    # Compute Pearson correlation
    correlation = np.corrcoef(distances, fill_rates)[0, 1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top subplot: Scatter plot with trend line
    ax1 = axes[0]
    scatter = ax1.scatter(
        distances,
        fill_rates * 100,  # Convert to percentage
        c=fill_rates,
        cmap='viridis',
        s=50,
        alpha=0.5,
        edgecolors='black',
        linewidths=0.3,
        vmin=0.0,
        vmax=1.0,
    )

    # Add trend line
    z = np.polyfit(distances, fill_rates * 100, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(distances.min(), distances.max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", linewidth=2,
            label=f'Trend (Pearson r={correlation:.3f})')

    ax1.set_xlabel('Distance from Ground Truth (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fill Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'Fill Rate vs Position Error Distance (N={len(fill_rates)} samples)',
        fontsize=13, fontweight='bold'
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    ax1.set_ylim([0, 105])

    # Bottom subplot: Box plot by distance bins
    ax2 = axes[1]
    max_dist = distances.max()

    # Define distance bins dynamically
    if max_dist <= 2.0:
        distance_bins = [0, 0.5, 1.0, 2.0, max_dist + 0.01]
        bin_labels = ['0-0.5m', '0.5-1m', '1-2m', f'2m+']
    elif max_dist <= 5.0:
        distance_bins = [0, 0.5, 1.0, 2.0, 5.0, max_dist + 0.01]
        bin_labels = ['0-0.5m', '0.5-1m', '1-2m', '2-5m', f'5m+']
    else:
        distance_bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0, max_dist + 0.01]
        bin_labels = ['0-0.5m', '0.5-1m', '1-2m', '2-5m', '5-10m', '10m+']

    binned_data = []
    actual_labels = []
    bin_stats = []

    for i in range(len(distance_bins) - 1):
        mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
        if mask.sum() > 0:
            binned_fill_rates = fill_rates[mask] * 100
            binned_data.append(binned_fill_rates)
            actual_labels.append(f'{bin_labels[i]}\n(n={mask.sum()})')
            bin_stats.append({
                'count': mask.sum(),
                'mean': np.mean(binned_fill_rates),
                'median': np.median(binned_fill_rates),
            })

    if binned_data:
        bp = ax2.boxplot(binned_data, labels=actual_labels, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2),
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='green', markersize=6))

        ax2.set_xlabel('Distance Bin', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fill Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Fill Rate Distribution by Error Distance', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 105])

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = save_path.parent / (save_path.stem + '_fill_vs_dist' + save_path.suffix)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[Save] Radial fill rate analysis saved to: {output_path}")

    plt.show()

    # Print correlation statistics
    print(f"\n[Analysis] Fill Rate vs Position Error Distance:")
    print(f"  Pearson Correlation: {correlation:.4f}")
    if abs(correlation) < 0.1:
        print(f"  → Very weak/no correlation (fill rate doesn't explain position error)")
    elif abs(correlation) < 0.3:
        print(f"  → Weak correlation")
    elif abs(correlation) < 0.7:
        print(f"  → Moderate correlation")
    else:
        print(f"  → Strong correlation")

    # Print bin statistics
    if bin_stats:
        print(f"\n  Distance Bin Statistics:")
        for i, (label, stats) in enumerate(zip(bin_labels[:len(bin_stats)], bin_stats)):
            print(f"    {label:12s}: n={stats['count']:4d}, mean={stats['mean']:5.1f}%, median={stats['median']:5.1f}%")


def plot_fill_rate_directional(
    gaussian_positions: np.ndarray,
    fill_rates: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Analyze fill rate by direction/angle from ground truth.
    Shows if certain directions have systematically different fill rates.

    Args:
        gaussian_positions: Nx2 array of (x, y) errors from Gaussian method (in meters).
        fill_rates: N-length array of LiDAR fill rates (0.0–1.0).
        save_path: Optional path to save the plot (suffix `_geometry_directional` will be appended).
    """
    if gaussian_positions.size == 0 or fill_rates.size == 0:
        print("[Plot] Skipping directional analysis (no data).")
        return

    x_errors = gaussian_positions[:, 0]
    y_errors = gaussian_positions[:, 1]

    # Compute polar coordinates
    distances = np.sqrt(x_errors**2 + y_errors**2)
    angles_rad = np.arctan2(y_errors, x_errors)  # -π to π
    angles_deg = np.degrees(angles_rad)  # -180 to 180

    # Normalize to 0-360
    angles_deg = (angles_deg + 360) % 360

    fig = plt.figure(figsize=(14, 11))

    # Top: Polar scatter plot
    ax1 = plt.subplot(2, 1, 1, projection='polar')
    scatter = ax1.scatter(
        angles_rad,
        distances,
        c=fill_rates * 100,
        cmap='RdYlGn',
        s=60,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.3,
        vmin=0,
        vmax=100,
    )

    ax1.set_theta_zero_location('E')  # 0° = East (positive X)
    ax1.set_theta_direction(-1)  # Clockwise
    ax1.set_title('Polar View: Fill Rate by Direction & Distance',
                 fontsize=13, fontweight='bold', pad=20)
    ax1.set_ylabel('Distance (m)', fontsize=11, fontweight='bold')

    cbar1 = plt.colorbar(scatter, ax=ax1, pad=0.1, fraction=0.046)
    cbar1.set_label('Fill Rate (%)', fontsize=10, fontweight='bold')

    # Bottom: Angular binning analysis
    ax2 = plt.subplot(2, 1, 2)

    # Define angular sectors (8 directions)
    n_sectors = 8
    sector_size = 360 / n_sectors
    sector_labels = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']

    sector_data = []
    sector_names = []
    sector_stats = []

    for i in range(n_sectors):
        start_angle = i * sector_size
        end_angle = (i + 1) * sector_size

        mask = (angles_deg >= start_angle) & (angles_deg < end_angle)
        count = mask.sum()

        if count > 0:
            sector_fill = fill_rates[mask] * 100
            sector_data.append(sector_fill)
            sector_names.append(f'{sector_labels[i]}\n(n={count})')
            sector_stats.append({
                'direction': sector_labels[i],
                'count': count,
                'mean': np.mean(sector_fill),
                'median': np.median(sector_fill),
            })

    if sector_data:
        bp = ax2.boxplot(sector_data, labels=sector_names, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2),
                        boxprops=dict(facecolor='lightgreen', alpha=0.6),
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='blue', markersize=6))

        ax2.set_xlabel('Direction Sector', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fill Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Fill Rate Distribution by Direction (8 Sectors)',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 105])

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = save_path.parent / (save_path.stem + '_geometry_directional' + save_path.suffix)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"[Save] Directional analysis plot saved to: {output_path}")

    plt.show()

    # Print directional statistics
    if sector_stats:
        print(f"\n[Analysis] Fill Rate by Direction:")
        for stats in sector_stats:
            print(f"  {stats['direction']:3s}: n={stats['count']:4d}, "
                  f"mean={stats['mean']:5.1f}%, median={stats['median']:5.1f}%")


def compute_theta_statistics(theta_errors: np.ndarray) -> None:
    """
    Compute and print comprehensive statistics for theta (rotation) prediction.

    Args:
        theta_errors: N-length array of theta errors in degrees from ground truth (0°).
    """
    # Absolute errors
    abs_errors = np.abs(theta_errors)

    # RMS (Root Mean Square) error
    rms_theta = np.sqrt(np.mean(theta_errors**2))

    # Percentile statistics
    p50_theta = np.percentile(abs_errors, 50)  # Median absolute error
    p99_theta = np.percentile(abs_errors, 99)  # 99th percentile absolute error

    # Mean and max for reference
    mean_abs = np.mean(abs_errors)
    max_abs = np.max(abs_errors)

    print(f"\n[Theta (Rotation) Prediction]")
    print(f"  RMS Error:         {rms_theta:.4f}°")
    print(f"  P50 (Median):      {p50_theta:.4f}°")
    print(f"  P99 (99th pctl):   {p99_theta:.4f}°")
    print(f"  Mean Abs Error:    {mean_abs:.4f}°")
    print(f"  Max Abs Error:     {max_abs:.4f}°")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch position refinement comparison for ReLL dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"G:\GithubProject\ReLL\model-save\best_1000.ckpt"),
        help="Path to model checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"Rell-sample-raster-0p2"),
        help="Path to dataset folder containing sample subdirectories.",
    )
    parser.add_argument(
        "--lidar-variant",
        type=str,
        default="gicp",
        choices=["gicp", "non_aligned"],
        help="LiDAR variant to use for correlation.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.2,
        help="Map resolution in meters per pixel (default: 0.2m).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the comparison plot (e.g., refinement_comparison.png).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing).",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ReLL Position Refinement Batch Analysis")
    print("=" * 70)

    # Load trained model
    print(f"\n[Load] Loading checkpoint from: {args.checkpoint}")
    model, model_cfg = load_checkpoint(args.checkpoint, device=args.device)
    print(f"[Load] Model loaded successfully on {args.device}")

    # Find all samples
    print(f"\n[Scan] Scanning dataset folder: {args.dataset}")
    sample_dirs = find_all_samples(args.dataset)

    if args.max_samples is not None:
        sample_dirs = sample_dirs[:args.max_samples]

    print(f"[Scan] Found {len(sample_dirs)} samples")

    # Process all samples
    print(f"\n[Process] Processing samples...")
    peak_results = []
    softmax_results = []
    gaussian_results = []
    theta_results = []
    fill_rates = []

    for sample_path in tqdm(sample_dirs, desc="Processing samples"):
        result = process_single_sample(
            model,
            sample_path,
            args.lidar_variant,
            args.device,
            args.resolution,
        )

        if result['success']:
            peak_results.append(result['peak'])
            softmax_results.append(result['softmax'])
            gaussian_results.append(result['gaussian'])
            theta_results.append(result['theta'])
            fill_rates.append(result['fill_rate'])

    # Convert to numpy arrays
    peak_positions = np.array(peak_results)
    softmax_positions = np.array(softmax_results)
    gaussian_positions = np.array(gaussian_results)
    theta_errors = np.array(theta_results)
    fill_rates = np.array(fill_rates)

    print(f"\n[Process] Successfully processed {len(peak_positions)} samples")

    # Compute statistics
    print("\n" + "=" * 70)
    print("Position Error Statistics (from ground truth center)")
    print("=" * 70)

    compute_statistics(peak_positions, "Method 1: Discrete Peak")
    compute_statistics(softmax_positions, "Method 2: Softmax Refinement")
    compute_statistics(gaussian_positions, "Method 3: Gaussian Peak Refinement")
    compute_theta_statistics(theta_errors)

    # Print comparison table
    print("\n" + "=" * 70)
    print("Comparison Summary Table")
    print("=" * 70)

    def get_stats(positions):
        x_errors = positions[:, 0]
        y_errors = positions[:, 1]
        distances = np.sqrt(x_errors**2 + y_errors**2)
        return {
            'rms_x': np.sqrt(np.mean(x_errors**2)),
            'rms_y': np.sqrt(np.mean(y_errors**2)),
            'rms_dist': np.sqrt(np.mean(distances**2)),
            'p50_dist': np.percentile(distances, 50),
            'p99_dist': np.percentile(distances, 99),
        }

    peak_stats = get_stats(peak_positions)
    softmax_stats = get_stats(softmax_positions)
    gaussian_stats = get_stats(gaussian_positions)

    print(f"\n{'Metric':<20} {'Peak':<12} {'Softmax':<12} {'Gaussian':<12} {'Best':<10}")
    print("-" * 70)

    # RMS X
    best_rms_x = min(peak_stats['rms_x'], softmax_stats['rms_x'], gaussian_stats['rms_x'])
    print(f"{'RMS X (m)':<20} {peak_stats['rms_x']:>11.4f} {softmax_stats['rms_x']:>11.4f} {gaussian_stats['rms_x']:>11.4f} ", end="")
    if gaussian_stats['rms_x'] == best_rms_x:
        print("Gaussian")
    elif softmax_stats['rms_x'] == best_rms_x:
        print("Softmax")
    else:
        print("Peak")

    # RMS Y
    best_rms_y = min(peak_stats['rms_y'], softmax_stats['rms_y'], gaussian_stats['rms_y'])
    print(f"{'RMS Y (m)':<20} {peak_stats['rms_y']:>11.4f} {softmax_stats['rms_y']:>11.4f} {gaussian_stats['rms_y']:>11.4f} ", end="")
    if gaussian_stats['rms_y'] == best_rms_y:
        print("Gaussian")
    elif softmax_stats['rms_y'] == best_rms_y:
        print("Softmax")
    else:
        print("Peak")

    # RMS Distance
    best_rms_dist = min(peak_stats['rms_dist'], softmax_stats['rms_dist'], gaussian_stats['rms_dist'])
    print(f"{'RMS Distance (m)':<20} {peak_stats['rms_dist']:>11.4f} {softmax_stats['rms_dist']:>11.4f} {gaussian_stats['rms_dist']:>11.4f} ", end="")
    if gaussian_stats['rms_dist'] == best_rms_dist:
        print("Gaussian")
    elif softmax_stats['rms_dist'] == best_rms_dist:
        print("Softmax")
    else:
        print("Peak")

    # P50 Distance
    best_p50 = min(peak_stats['p50_dist'], softmax_stats['p50_dist'], gaussian_stats['p50_dist'])
    print(f"{'P50 Distance (m)':<20} {peak_stats['p50_dist']:>11.4f} {softmax_stats['p50_dist']:>11.4f} {gaussian_stats['p50_dist']:>11.4f} ", end="")
    if gaussian_stats['p50_dist'] == best_p50:
        print("Gaussian")
    elif softmax_stats['p50_dist'] == best_p50:
        print("Softmax")
    else:
        print("Peak")

    # P99 Distance
    best_p99 = min(peak_stats['p99_dist'], softmax_stats['p99_dist'], gaussian_stats['p99_dist'])
    print(f"{'P99 Distance (m)':<20} {peak_stats['p99_dist']:>11.4f} {softmax_stats['p99_dist']:>11.4f} {gaussian_stats['p99_dist']:>11.4f} ", end="")
    if gaussian_stats['p99_dist'] == best_p99:
        print("Gaussian")
    elif softmax_stats['p99_dist'] == best_p99:
        print("Softmax")
    else:
        print("Peak")

    print("-" * 70)

    # Theta (rotation) statistics
    abs_theta_errors = np.abs(theta_errors)
    rms_theta = np.sqrt(np.mean(theta_errors**2))
    p50_theta = np.percentile(abs_theta_errors, 50)
    p99_theta = np.percentile(abs_theta_errors, 99)

    print(f"{'RMS Theta (°)':<20} {rms_theta:>11.4f} {'':>12} {'':>12} {'Theta':<10}")
    print(f"{'P50 Theta (°)':<20} {p50_theta:>11.4f} {'':>12} {'':>12} {'Theta':<10}")
    print(f"{'P99 Theta (°)':<20} {p99_theta:>11.4f} {'':>12} {'':>12} {'Theta':<10}")

    print("=" * 70)

    # Plot comparison
    print("\n[Plot] Generating comparison plot...")
    plot_refinement_comparison(
        peak_positions,
        softmax_positions,
        gaussian_positions,
        save_path=args.save,
    )

    # Plot error distributions (using Gaussian method for x/y)
    print("\n[Plot] Generating error distribution plot...")
    plot_error_distributions(
        theta_errors,
        gaussian_positions[:, 0],  # X errors from Gaussian method
        gaussian_positions[:, 1],  # Y errors from Gaussian method
        save_path=args.save,
    )

    # Plot LiDAR fill rate vs distance analysis
    if len(fill_rates) > 0:
        print("\n[Plot] Analyzing LiDAR fill rate vs. position error distance...")
        plot_fill_rate_vs_distance(
            gaussian_positions,
            fill_rates,
            save_path=args.save,
        )

    # Plot directional/geometric analysis of fill rate
    if len(fill_rates) > 0:
        print("\n[Plot] Analyzing fill rate by direction (geometric analysis)...")
        plot_fill_rate_directional(
            gaussian_positions,
            fill_rates,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
