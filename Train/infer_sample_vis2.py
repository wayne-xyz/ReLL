from __future__ import annotations

"""
Correlation visualization tools for ReLL translation search.

This script implements three position-finding methods from infer_visualize.py:

Method 1: Heatmap Cross-Correlation (Discrete Peak)
    - Finds the maximum value in the correlation heatmap
    - Returns integer pixel coordinates
    - Simple and fast, but limited to discrete grid positions

Method 2: Softmax Refinement (Expectation-Based)
    - Applies softmax to convert heatmap to probability distribution
    - Computes expectation (weighted average) for sub-pixel accuracy
    - Also provides uncertainty estimates (σx, σy)
    - Imported from: _softmax_refinement() in infer_visualize.py

Method 3: Gaussian Peak Fitting (Sub-pixel, 3×3 Parabolic)
    - Fits a parabolic surface to a 3×3 neighborhood around the peak
    - Uses simple, robust log-space quadratic fitting
    - Most accurate for sharp, well-defined peaks
    - Provided by: gaussian_peak_refine() in Train.gaussian_peak_refine

The script renders 2D contour panels comparing the three methods side-by-side.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure package imports work when executed directly
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.gaussian_peak_refine import GaussianPeakResult, bilinear_sample, gaussian_peak_refine
from Train.theta_peak_refine import ThetaPeakResult, theta_peak_refine
from Train.infer_sample_vis import (  # noqa: E402
    compute_embeddings_and_correlation,
    load_checkpoint,
    load_sample,
    _softmax_refinement,
)



def plot_correlation_2d(
    heatmap: torch.Tensor,
    softmax_probs: torch.Tensor,
    softmax_mu: tuple[float, float],
    softmax_sigma: tuple[float, float],
    gaussian_mu: tuple[float, float],
    resolution: float,
    use_meters: bool,
    peak_offset: tuple[int, int],
    peak_score: float,
    softmax_score: float,
    gaussian_score: float,
    theta_result: ThetaPeakResult | None = None,
    theta_candidates_deg: torch.Tensor | None = None,
    theta_logits: torch.Tensor | None = None,
    theta_gt_deg: float | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Render 2D contour comparison for the three position-finding methods:
    1. Heatmap cross-correlation (discrete peak)
    2. Softmax refinement (expectation-based)
    3. Gaussian peak fitting (sub-pixel fitting)

    Plus optional theta (rotation) response curve visualization.
    """
    heat_np = heatmap.cpu().numpy()
    softmax_np = softmax_probs.cpu().numpy()

    H, W = heat_np.shape
    center_row = H // 2
    center_col = W // 2
    x_offsets_px = np.arange(W) - center_col
    y_offsets_px = np.arange(H) - center_row
    X_px, Y_px = np.meshgrid(x_offsets_px, y_offsets_px)

    if use_meters:
        X_plot = X_px * resolution
        Y_plot = Y_px * resolution
        unit = "m"
    else:
        X_plot = X_px
        Y_plot = Y_px
        unit = "px"

    softmax_sigma_x, softmax_sigma_y = softmax_sigma

    if use_meters:
        softmax_sigma_x_disp = softmax_sigma_x * resolution
        softmax_sigma_y_disp = softmax_sigma_y * resolution
    else:
        softmax_sigma_x_disp = softmax_sigma_x
        softmax_sigma_y_disp = softmax_sigma_y

    # Determine layout: add theta row if theta data is provided
    has_theta = theta_result is not None and theta_candidates_deg is not None and theta_logits is not None
    if has_theta:
        fig = plt.figure(figsize=(20, 11), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_theta = fig.add_subplot(gs[1, :])  # Theta spans full width
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    # === Method 1: Raw Heatmap Cross-Correlation (Discrete Peak) ===
    im0 = axes[0].imshow(
        heat_np,
        cmap="coolwarm",
        origin="lower",
        extent=[X_plot.min(), X_plot.max(), Y_plot.min(), Y_plot.max()],
    )
    axes[0].set_title(
        f"Method 1: Heatmap Cross-Correlation\nPeak Score = {peak_score:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel(f"ΔX ({unit})")
    axes[0].set_ylabel(f"ΔY ({unit})")
    # Plot discrete peak
    axes[0].scatter(
        [peak_offset[0] * resolution if use_meters else peak_offset[0]],
        [peak_offset[1] * resolution if use_meters else peak_offset[1]],
        c="gold",
        s=140,
        marker="*",
        edgecolors="black",
        linewidths=1.5,
        label=f"Discrete Peak ({peak_offset[0]:+.1f}{unit}, {peak_offset[1]:+.1f}{unit})",
    )
    # Also show refined positions for comparison
    sx, sy = softmax_mu
    gx, gy = gaussian_mu
    axes[0].scatter(
        [sx * resolution if use_meters else sx],
        [sy * resolution if use_meters else sy],
        c="red",
        marker="^",
        s=90,
        edgecolors="white",
        linewidths=1.0,
        alpha=0.7,
        label=f"Softmax μ ({sx:+.2f}{unit}, {sy:+.2f}{unit})",
    )
    axes[0].scatter(
        [gx * resolution if use_meters else gx],
        [gy * resolution if use_meters else gy],
        facecolors="none",
        edgecolors="yellow",
        marker="s",
        s=100,
        linewidths=1.5,
        alpha=0.7,
        label=f"Gaussian μ ({gx:+.2f}{unit}, {gy:+.2f}{unit})",
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Correlation")

    # === Method 2: Softmax Refinement (Expectation-Based) ===
    softmax_display = np.log(softmax_np + 1e-8)
    cf1 = axes[1].contourf(
        X_plot,
        Y_plot,
        softmax_display,
        levels=40,
        cmap="magma",
    )
    axes[1].scatter(
        [sx * resolution if use_meters else sx],
        [sy * resolution if use_meters else sy],
        c="red",
        marker="^",
        s=120,
        edgecolors="white",
        linewidths=1.5,
        label=f"Softmax μ ({sx:+.2f}{unit}, {sy:+.2f}{unit})",
    )
    # Show discrete peak for reference
    axes[1].scatter(
        [peak_offset[0] * resolution if use_meters else peak_offset[0]],
        [peak_offset[1] * resolution if use_meters else peak_offset[1]],
        c="gold",
        s=100,
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        alpha=0.5,
        label=f"Discrete Peak",
    )
    axes[1].set_title(
        f"Method 2: Softmax Refinement\nScore ≈ {softmax_score:.4f}, σ≈({softmax_sigma_x_disp:.2f},{softmax_sigma_y_disp:.2f}) {unit}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel(f"ΔX ({unit})")
    axes[1].set_ylabel(f"ΔY ({unit})")
    axes[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.colorbar(cf1, ax=axes[1], fraction=0.046, pad=0.04, label="log Probability")

    # === Method 3: Gaussian Peak Fitting (Enhanced Sub-pixel) ===
    axes[2].imshow(
        heat_np,
        cmap="coolwarm",
        origin="lower",
        extent=[X_plot.min(), X_plot.max(), Y_plot.min(), Y_plot.max()],
    )

    gx_plot = gx * resolution if use_meters else gx
    gy_plot = gy * resolution if use_meters else gy
    peak_x_disp = peak_offset[0] * resolution if use_meters else peak_offset[0]
    peak_y_disp = peak_offset[1] * resolution if use_meters else peak_offset[1]

    # Highlight refined locations
    axes[2].scatter(
        [gx_plot],
        [gy_plot],
        facecolors="yellow",
        edgecolors="white",
        marker="s",
        s=150,
        linewidths=2.5,
        label=f"Gaussian μ ({gx:+.2f}{unit}, {gy:+.2f}{unit})",
        zorder=10,
    )
    axes[2].scatter(
        [peak_x_disp],
        [peak_y_disp],
        c="gold",
        s=110,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        alpha=0.5,
        label="Discrete Peak",
        zorder=9,
    )
    axes[2].scatter(
        [sx * resolution if use_meters else sx],
        [sy * resolution if use_meters else sy],
        c="red",
        marker="^",
        s=110,
        edgecolors="white",
        linewidths=1.2,
        alpha=0.7,
        label="Softmax μ",
        zorder=9,
    )

    # Add cross-hairs to emphasise refined alignment
    axes[2].axvline(gx_plot, color="yellow", linestyle="--", linewidth=1.1, alpha=0.65)
    axes[2].axhline(gy_plot, color="yellow", linestyle="--", linewidth=1.1, alpha=0.65)

    # Zoom into a local neighbourhood around the refined peak for clarity
    zoom_radius = 6  # pixels
    zoom_span = zoom_radius * (resolution if use_meters else 1.0)
    axes[2].set_xlim(gx_plot - zoom_span, gx_plot + zoom_span)
    axes[2].set_ylim(gy_plot - zoom_span, gy_plot + zoom_span)

    axes[2].set_title(
        f"Method 3: Gaussian Peak Fitting (Refined)\nScore ≈ {gaussian_score:.4f}",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_xlabel(f"ΔX ({unit})")
    axes[2].set_ylabel(f"ΔY ({unit})")
    axes[2].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_aspect('equal', adjustable='box')

    # === Theta (Rotation) Response Curve (if provided) ===
    if has_theta:
        theta_angles = theta_candidates_deg.cpu().numpy()
        theta_scores = theta_logits.cpu().numpy()
        theta_probs = theta_result.prob_distribution.numpy()

        # Plot raw logits
        ax_theta.plot(
            theta_angles,
            theta_scores,
            'b-',
            linewidth=2.0,
            label='Raw Logits',
            alpha=0.7,
        )

        # Plot softmax probabilities on secondary y-axis
        ax_theta_prob = ax_theta.twinx()
        ax_theta_prob.fill_between(
            theta_angles,
            theta_probs,
            alpha=0.3,
            color='green',
            label='Softmax Probability',
        )
        ax_theta_prob.plot(
            theta_angles,
            theta_probs,
            'g-',
            linewidth=1.5,
            alpha=0.8,
        )

        # Mark predicted theta
        ax_theta.axvline(
            theta_result.theta_deg,
            color='blue',
            linestyle='--',
            linewidth=2.5,
            label=f'Predicted θ = {theta_result.theta_deg:.2f}°',
        )

        # Mark ground truth theta if provided
        if theta_gt_deg is not None:
            ax_theta.axvline(
                theta_gt_deg,
                color='red',
                linestyle='--',
                linewidth=2.5,
                label=f'Ground Truth θ = {theta_gt_deg:.2f}°',
            )
            # Show error
            theta_error = abs(theta_result.theta_deg - theta_gt_deg)
            error_text = f'Error = {theta_error:.3f}°'
        else:
            error_text = ''

        # Mark uncertainty region
        sigma = theta_result.sigma_deg
        ax_theta.axvspan(
            theta_result.theta_deg - sigma,
            theta_result.theta_deg + sigma,
            alpha=0.2,
            color='blue',
            label=f'±1σ = ±{sigma:.2f}°',
        )

        ax_theta.set_xlabel('Rotation Angle (degrees)', fontsize=11, fontweight='bold')
        ax_theta.set_ylabel('Correlation Score (Logits)', fontsize=11, fontweight='bold', color='blue')
        ax_theta_prob.set_ylabel('Softmax Probability', fontsize=11, fontweight='bold', color='green')
        ax_theta.tick_params(axis='y', labelcolor='blue')
        ax_theta_prob.tick_params(axis='y', labelcolor='green')

        # Title with stats
        title_parts = [f'Theta (Rotation) Response Curve']
        if theta_gt_deg is not None:
            title_parts.append(error_text)
        title_parts.append(f'Confidence = {theta_result.confidence:.3f}')
        ax_theta.set_title(' | '.join(title_parts), fontsize=12, fontweight='bold')

        # Combine legends from both y-axes
        lines1, labels1 = ax_theta.get_legend_handles_labels()
        lines2, labels2 = ax_theta_prob.get_legend_handles_labels()
        ax_theta.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

        ax_theta.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax_theta.set_xlim(theta_angles.min() - 0.5, theta_angles.max() + 0.5)

    # Overall title
    title = "Three Position-Finding Methods Comparison (2D Mode)"
    if has_theta:
        title += " + Rotation Analysis"
    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        y=0.995 if has_theta else 1.02,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Save] 2D contour figure saved to: {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ReLL translation cross-correlation with detailed 2D contour panels."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"G:\GithubProject\ReLL\model-save\best_1000.ckpt"),
        help="Path to model checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(r"Rell-sample-raster2\0As5GhcTExRFWNboqeHwpegdeGf2j0YW_036"),
        help="Path to sample directory (containing .npy files).",
    )
    parser.add_argument(
        "--lidar-variant",
        type=str,
        default="gicp",
        choices=["gicp", "non_aligned"],
        help="LiDAR variant to use for correlation.",
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
        help="Optional path to save the visualization (e.g., output.png).",
    )
    parser.add_argument(
        "--meters",
        action="store_true",
        help="Plot X/Y axes in meters instead of pixels.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Show visualization GUI. Without this flag, only prints results.",
    )
    parser.add_argument(
        "--gaussian-method",
        type=str,
        default="improved",
        choices=["improved"],
        help="Gaussian fitting method: improved (multi-stage refinement over the correlation peak).",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ReLL Cross-Correlation Visualizer")
    print("=" * 70)

    # Load trained model
    model, model_cfg = load_checkpoint(args.checkpoint, device=args.device)

    # Load single sample (no augmentation)
    sample = load_sample(args.sample, lidar_variant=args.lidar_variant)
    lidar = sample["lidar"].to(args.device)
    geospatial = sample["map"].to(args.device)
    resolution = sample["resolution"]

    print("\n[Inference] Computing correlation heatmap...")
    results = compute_embeddings_and_correlation(model, lidar, geospatial)
    heatmap = results["correlation_heatmap"]
    # Extract discrete peak info (already centered grid)
    peak_idx = int(torch.argmax(heatmap))
    H, W = heatmap.shape
    peak_row = peak_idx // W
    peak_col = peak_idx % W
    center_row = H // 2
    center_col = W // 2
    peak_y_offset = peak_row - center_row
    peak_x_offset = peak_col - center_col
    peak_score = float(heatmap[peak_row, peak_col].item())

    # Get theta (rotation) predictions by running full model forward pass
    print("\n[Inference] Computing rotation predictions...")
    lidar_batch = lidar.unsqueeze(0)  # Add batch dimension
    geospatial_batch = geospatial.unsqueeze(0)
    with torch.no_grad():
        predictions = model(lidar_batch, geospatial_batch)

    theta_logits = predictions["theta_logits"].squeeze(0).cpu()  # Remove batch dim
    theta_candidates_deg = predictions["theta_candidates_deg"].cpu()  # Get from predictions, not model buffer

    # Apply theta peak refinement
    theta_result = theta_peak_refine(theta_logits, theta_candidates_deg)

    print(f"  - Theta candidates: {theta_candidates_deg.numpy()}")
    print(f"  - Predicted theta: {theta_result.theta_deg:.3f}°")
    print(f"  - Theta uncertainty (σ): {theta_result.sigma_deg:.3f}°")
    print(f"  - Theta confidence: {theta_result.confidence:.3f}")

    (
        softmax_probs,
        softmax_mu_x_px,
        softmax_mu_y_px,
        softmax_sigma_x_px,
        softmax_sigma_y_px,
    ) = _softmax_refinement(heatmap)

    # Select Gaussian fitting method based on args
    gaussian_result: GaussianPeakResult = gaussian_peak_refine(heatmap)
    gaussian_mu_x_px = gaussian_result.x
    gaussian_mu_y_px = gaussian_result.y
    gaussian_score = gaussian_result.score

    softmax_score = bilinear_sample(heatmap, softmax_mu_x_px, softmax_mu_y_px)

    # Calculate distances from ground truth (0, 0) for final summary
    import math
    peak_dist_px = math.sqrt(peak_x_offset**2 + peak_y_offset**2)
    peak_dist_m = peak_dist_px * resolution
    softmax_dist_px = math.sqrt(softmax_mu_x_px**2 + softmax_mu_y_px**2)
    softmax_dist_m = softmax_dist_px * resolution
    gaussian_dist_px = math.sqrt(gaussian_mu_x_px**2 + gaussian_mu_y_px**2)
    gaussian_dist_m = gaussian_dist_px * resolution

    print("\n" + "=" * 70)
    print("Three Position-Finding Methods Summary")
    print("=" * 70)
    print(
        f"[Method 1: Heatmap Cross-Correlation]\n"
        f"  Discrete Peak: ({peak_x_offset:+.1f}px, {peak_y_offset:+.1f}px) = "
        f"({peak_x_offset * resolution:+.3f}m, {peak_y_offset * resolution:+.3f}m)\n"
        f"  Distance from GT (0,0): {peak_dist_px:.3f}px = {peak_dist_m:.4f}m\n"
        f"  Score: {peak_score:.4f}"
    )
    print(
        f"\n[Method 2: Softmax Refinement]\n"
        f"  Refined μ: ({softmax_mu_x_px:+.3f}px, {softmax_mu_y_px:+.3f}px) = "
        f"({softmax_mu_x_px * resolution:+.3f}m, {softmax_mu_y_px * resolution:+.3f}m)\n"
        f"  Distance from GT (0,0): {softmax_dist_px:.3f}px = {softmax_dist_m:.4f}m\n"
        f"  σ: ({softmax_sigma_x_px:.3f}px, {softmax_sigma_y_px:.3f}px) = "
        f"({softmax_sigma_x_px * resolution:.3f}m, {softmax_sigma_y_px * resolution:.3f}m)\n"
        f"  Score: {softmax_score:.4f}"
    )

    # Determine if Gaussian beats baselines
    gaussian_beats_peak = "✓" if gaussian_dist_px < peak_dist_px else "✗"
    gaussian_beats_softmax = "✓" if gaussian_dist_px < softmax_dist_px else "✗"
    gaussian_best = " 🏆 BEST!" if gaussian_dist_px < min(peak_dist_px, softmax_dist_px) else ""

    print(
        f"\n[Method 3: Gaussian Peak Fitting ({args.gaussian_method})]\n"
        f"  Refined μ: ({gaussian_mu_x_px:+.3f}px, {gaussian_mu_y_px:+.3f}px) = "
        f"({gaussian_mu_x_px * resolution:+.3f}m, {gaussian_mu_y_px * resolution:+.3f}m)\n"
        f"  Distance from GT (0,0): {gaussian_dist_px:.3f}px = {gaussian_dist_m:.4f}m "
        f"[vs Peak:{gaussian_beats_peak} vs Softmax:{gaussian_beats_softmax}]{gaussian_best}\n"
        f"  Score: {gaussian_score:.4f}"
    )
    print(
        f"\n[Theta (Rotation) Prediction]\n"
        f"  Predicted θ: {theta_result.theta_deg:+.3f}°\n"
        f"  Uncertainty (σ): {theta_result.sigma_deg:.3f}°\n"
        f"  Confidence: {theta_result.confidence:.3f} (max softmax probability)\n"
        f"  Search range: [{theta_candidates_deg.min():.0f}°, {theta_candidates_deg.max():.0f}°] "
        f"({len(theta_candidates_deg)} candidates)"
    )
    print("=" * 70)

    if args.vis:
        plot_correlation_2d(
            heatmap,
            softmax_probs,
            (softmax_mu_x_px, softmax_mu_y_px),
            (softmax_sigma_x_px, softmax_sigma_y_px),
            (gaussian_mu_x_px, gaussian_mu_y_px),
            resolution,
            use_meters=args.meters,
            peak_offset=(peak_x_offset, peak_y_offset),
            peak_score=peak_score,
            softmax_score=softmax_score,
            gaussian_score=gaussian_score,
            theta_result=theta_result,
            theta_candidates_deg=theta_candidates_deg,
            theta_logits=theta_logits,
            theta_gt_deg=0.0,  # Ground truth is 0 for centered samples
            save_path=args.save,
        )
    else:
        print("\n[Info] Skipping visualization (use --vis to show plots)")
        print("[Done] Results printed above.")


if __name__ == "__main__":
    main()
