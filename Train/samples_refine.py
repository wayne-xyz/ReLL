from __future__ import annotations

"""
Batch position refinement comparison across entire dataset.

Scans all samples in a dataset folder and compares three position-finding methods:
1. Discrete heatmap peak (argmax)
2. Softmax expectation refinement (probabilistic weighting)
3. Gaussian surface fit refinement (least-squares fitting over patch)

The Gaussian method uses fit_gaussian_peak() from correlation_3d_visualizer.py,
which fits log(Z) = a + bx*X + by*Y - cx*X^2 - cy*Y^2 over a patch around the
discrete peak to simultaneously find both the refined peak position (μx, μy) and
Gaussian shape parameters (σx, σy).

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

from Train.infer_visualize import (  # noqa: E402
    compute_embeddings_and_correlation,
    load_checkpoint,
    load_sample,
    _softmax_refinement,
)

from Train.correlation_3d_visualizer import fit_gaussian_peak  # noqa: E402


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
    Process a single sample and extract position estimates from three methods.

    Args:
        model: Trained ReLL model.
        sample_path: Path to sample directory.
        lidar_variant: LiDAR variant to use ('gicp' or 'non_aligned').
        device: Computation device.
        resolution: Map resolution in meters per pixel.

    Returns:
        Dictionary containing position estimates in meters:
        - 'peak': (x, y) from discrete heatmap peak
        - 'softmax': (x, y) from softmax refinement
        - 'gaussian': (x, y) from Gaussian fit
        - 'success': True if processing succeeded
    """
    try:
        # Load sample
        sample = load_sample(sample_path, lidar_variant=lidar_variant)
        lidar = sample["lidar"].to(device)
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

        # Method 3: Gaussian surface fit (using least-squares over larger patch)
        radius_px = results.get("correlation_radius_px", heatmap.shape[-1] // 2)
        patch_radius = max(3, min(radius_px, 6))
        gaussian_mu_x_px, gaussian_mu_y_px, gaussian_sigma_x_px, gaussian_sigma_y_px = fit_gaussian_peak(
            heatmap, patch_radius=patch_radius
        )
        gaussian_x_m = gaussian_mu_x_px * resolution
        gaussian_y_m = gaussian_mu_y_px * resolution

        return {
            'peak': (float(peak_x_m), float(peak_y_m)),
            'softmax': (float(softmax_x_m), float(softmax_y_m)),
            'gaussian': (float(gaussian_x_m), float(gaussian_y_m)),
            'success': True,
        }

    except Exception as e:
        print(f"[Warning] Failed to process {sample_path.name}: {e}")
        return {
            'peak': (0.0, 0.0),
            'softmax': (0.0, 0.0),
            'gaussian': (0.0, 0.0),
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
    Compute and print statistics for a position-finding method.

    Args:
        positions: Nx2 array of (x, y) errors.
        method_name: Name of the method for display.
    """
    distances = np.sqrt(np.sum(positions**2, axis=1))

    mean_error = np.mean(distances)
    median_error = np.median(distances)
    std_error = np.std(distances)
    max_error = np.max(distances)

    print(f"\n[{method_name}]")
    print(f"  Mean error:   {mean_error:.4f} m")
    print(f"  Median error: {median_error:.4f} m")
    print(f"  Std error:    {std_error:.4f} m")
    print(f"  Max error:    {max_error:.4f} m")


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

    # Convert to numpy arrays
    peak_positions = np.array(peak_results)
    softmax_positions = np.array(softmax_results)
    gaussian_positions = np.array(gaussian_results)

    print(f"\n[Process] Successfully processed {len(peak_positions)} samples")

    # Compute statistics
    print("\n" + "=" * 70)
    print("Position Error Statistics (from ground truth center)")
    print("=" * 70)

    compute_statistics(peak_positions, "Discrete Peak")
    compute_statistics(softmax_positions, "Softmax Refinement")
    compute_statistics(gaussian_positions, "Gaussian Surface Fit (Least-Squares)")

    # Plot comparison
    print("\n[Plot] Generating comparison plot...")
    plot_refinement_comparison(
        peak_positions,
        softmax_positions,
        gaussian_positions,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
