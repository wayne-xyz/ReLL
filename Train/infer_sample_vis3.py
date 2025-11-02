from __future__ import annotations

"""
Correlation visualization tools for ReLL translation search - Focus on Position Finding.

This script implements three position-finding methods:

Method 1: Heatmap Cross-Correlation (Discrete Peak)
    - Finds the maximum value in the correlation heatmap
    - Returns integer pixel coordinates
    - Simple and fast, but limited to discrete grid positions

Method 2: Softmax Refinement (Expectation-Based)
    - Applies softmax to convert heatmap to probability distribution
    - Computes expectation (weighted average) for sub-pixel accuracy
    - Also provides uncertainty estimates (σx, σy)

Method 3: Gaussian Peak Fitting (Sub-pixel, Multi-Stage)
    - Uses advanced multi-strategy ensemble refinement
    - Most accurate for sharp, well-defined peaks

Method 4: Imagery Overlay Visualization
    - Overlays heatmap and contours on satellite imagery
    - Shows spatial context of correlation peaks
    - Displays all three methods' predictions on real imagery
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

# Ensure package imports work when executed directly
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.gaussian_peak_refine import GaussianPeakResult, bilinear_sample, gaussian_peak_refine
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
    imagery: np.ndarray | None = None,
    lidar_image: np.ndarray | None = None,
    save_path: Path | None = None,
    show_all: bool = False,
) -> None:
    """
    Render 2D comparison for position-finding methods with imagery overlay.

    Args:
        heatmap: Correlation heatmap [H, W]
        softmax_probs: Softmax probability distribution [H, W]
        softmax_mu: Softmax mean position (x, y) in pixels
        softmax_sigma: Softmax uncertainty (σx, σy) in pixels
        gaussian_mu: Gaussian peak position (x, y) in pixels
        resolution: Pixel resolution in meters/pixel
        use_meters: If True, plot in meters; else pixels
        peak_offset: Discrete peak offset (x, y) in pixels
        peak_score: Correlation score at discrete peak
        softmax_score: Correlation score at softmax position
        gaussian_score: Correlation score at Gaussian position
        imagery: Optional RGB imagery [3, H, W] for overlay
        lidar_image: Optional LiDAR raster to display alongside imagery overlays
        save_path: Optional path to save figure
        show_all: If True, show top row (method plots); else show only imagery overlays
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

    # Extract common values needed for all plots
    sx, sy = softmax_mu
    gx, gy = gaussian_mu

    lidar_height_display: np.ndarray | np.ma.MaskedArray | None = None
    lidar_intensity_display: np.ndarray | np.ma.MaskedArray | None = None
    lidar_hw: tuple[int, int] | None = None
    total_valid_pixels = 0
    mask_valid_count: int | None = None
    valid_counts: dict[str, int] = {}

    if lidar_image is not None:
        lidar_np = np.array(lidar_image)
        if lidar_np.ndim > 0:
            lidar_np = np.squeeze(lidar_np)
        if lidar_np.ndim == 3 and lidar_np.shape[0] <= 4 and lidar_np.shape[1] > 4:
            lidar_np = np.transpose(lidar_np, (1, 2, 0))
        if lidar_np.ndim == 2:
            lidar_np = lidar_np[:, :, None]

        lidar_np = lidar_np.astype(np.float32, copy=False)

        total_valid_pixels = lidar_np.shape[0] * lidar_np.shape[1]

        def _normalize_channel(channel: np.ndarray) -> np.ndarray:
            channel = np.array(channel, dtype=np.float32, copy=False)
            finite_mask_local = np.isfinite(channel)
            if np.any(finite_mask_local):
                min_val_local = float(channel[finite_mask_local].min())
                max_val_local = float(channel[finite_mask_local].max())
                if max_val_local > min_val_local:
                    channel = (channel - min_val_local) / (max_val_local - min_val_local)
                else:
                    channel = np.zeros_like(channel)
            else:
                channel = np.zeros_like(channel)
            channel[~finite_mask_local] = 0.0
            return channel

        channels = lidar_np.shape[2] if lidar_np.ndim == 3 else 1
        height_ch = lidar_np[..., 0] if channels >= 1 else None
        intensity_ch = lidar_np[..., 1] if channels >= 2 else None
        mask_ch = lidar_np[..., 2] if channels >= 3 else None

        mask_array: np.ndarray | None = None
        if mask_ch is not None:
            mask_array = np.array(mask_ch >= 0.5, dtype=bool)
            mask_valid_count = int(mask_array.sum())

        if height_ch is not None:
            valid_counts["height"] = int(np.isfinite(height_ch).sum())
            height_norm = _normalize_channel(height_ch)
            if mask_array is not None:
                lidar_height_display = np.ma.array(height_norm, mask=~mask_array)
            else:
                lidar_height_display = height_norm
        if intensity_ch is not None:
            valid_counts["intensity"] = int(np.isfinite(intensity_ch).sum())
            intensity_norm = _normalize_channel(intensity_ch)
            if mask_array is not None:
                lidar_intensity_display = np.ma.array(intensity_norm, mask=~mask_array)
            else:
                lidar_intensity_display = intensity_norm

        if lidar_height_display is not None:
            lidar_hw = lidar_height_display.shape
        elif lidar_intensity_display is not None:
            lidar_hw = lidar_intensity_display.shape
        elif mask_array is not None:
            lidar_hw = mask_array.shape

    num_method_cols = 3
    num_lidar_cols = int(lidar_height_display is not None) + int(lidar_intensity_display is not None)
    lidar_enabled = num_lidar_cols > 0
    max_cols = max(num_method_cols, num_lidar_cols if lidar_enabled else 1)

    if show_all:
        rows = 3 if lidar_enabled else 2
    else:
        rows = 2 if lidar_enabled else 1

    figsize_width = max(24, 8 * max_cols)
    figsize_height = max(7 * rows, 14 if show_all else 7 * rows)
    fig, axes_grid = plt.subplots(rows, max_cols, figsize=(figsize_width, figsize_height), constrained_layout=True)
    axes_grid = np.atleast_2d(axes_grid)

    axes_top: np.ndarray | None
    lidar_row_index: int | None = None
    if show_all:
        top_row = np.atleast_1d(axes_grid[0])
        axes_top = top_row[:num_method_cols]
        for extra_ax in top_row[num_method_cols:]:
            extra_ax.axis("off")
        if lidar_enabled:
            lidar_row_index = 1
            imagery_row_index = 2
        else:
            imagery_row_index = 1
    else:
        axes_top = None
        imagery_row_index = 0
        if lidar_enabled:
            lidar_row_index = 1

    imagery_row = np.atleast_1d(axes_grid[imagery_row_index])
    axes_bottom = imagery_row[:num_method_cols]
    for extra_ax in imagery_row[num_method_cols:]:
        extra_ax.axis("off")

    lidar_axes: list[plt.Axes] = []
    if lidar_enabled and lidar_row_index is not None:
        lidar_row = np.atleast_1d(axes_grid[lidar_row_index])
        lidar_axes = list(lidar_row[:num_lidar_cols])
        for extra_ax in lidar_row[num_lidar_cols:]:
            extra_ax.axis("off")
    lidar_extent: list[float] | None = None

    # === Top Row: Method Visualizations (only if show_all=True) ===
    if axes_top is not None:
        # === Method 1: Raw Heatmap Cross-Correlation (Discrete Peak) ===
        im0 = axes_top[0].imshow(
            heat_np,
            cmap="coolwarm",
            origin="lower",
            extent=[X_plot.min(), X_plot.max(), Y_plot.min(), Y_plot.max()],
        )
        axes_top[0].set_title(
            f"Method 1: Heatmap Cross-Correlation\nPeak Score = {peak_score:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        axes_top[0].set_xlabel(f"ΔX ({unit})")
        axes_top[0].set_ylabel(f"ΔY ({unit})")
        # Plot discrete peak
        axes_top[0].scatter(
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
        axes_top[0].scatter(
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
        axes_top[0].scatter(
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
        axes_top[0].legend(loc="upper right", fontsize=8)
        axes_top[0].grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
        fig.colorbar(im0, ax=axes_top[0], fraction=0.046, pad=0.04, label="Correlation")

        # === Method 2: Softmax Refinement (Expectation-Based) ===
        softmax_display = np.log(softmax_np + 1e-8)
        cf1 = axes_top[1].contourf(
            X_plot,
            Y_plot,
            softmax_display,
            levels=40,
            cmap="magma",
        )
        axes_top[1].scatter(
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
        axes_top[1].scatter(
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
        axes_top[1].set_title(
            f"Method 2: Softmax Refinement\nScore ≈ {softmax_score:.4f}, σ≈({softmax_sigma_x_disp:.2f},{softmax_sigma_y_disp:.2f}) {unit}",
            fontsize=12,
            fontweight="bold",
        )
        axes_top[1].set_xlabel(f"ΔX ({unit})")
        axes_top[1].set_ylabel(f"ΔY ({unit})")
        axes_top[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        axes_top[1].legend(loc="upper right", fontsize=8)
        fig.colorbar(cf1, ax=axes_top[1], fraction=0.046, pad=0.04, label="log Probability")

        # === Method 3: Gaussian Peak Fitting (Enhanced Sub-pixel) ===
        axes_top[2].imshow(
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
        axes_top[2].scatter(
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
        axes_top[2].scatter(
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
        axes_top[2].scatter(
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
        axes_top[2].axvline(gx_plot, color="yellow", linestyle="--", linewidth=1.1, alpha=0.65)
        axes_top[2].axhline(gy_plot, color="yellow", linestyle="--", linewidth=1.1, alpha=0.65)

        axes_top[2].set_title(
            f"Method 3: Gaussian Peak Fitting\nScore ≈ {gaussian_score:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        axes_top[2].set_xlabel(f"ΔX ({unit})")
        axes_top[2].set_ylabel(f"ΔY ({unit})")
        axes_top[2].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        axes_top[2].legend(loc="upper right", fontsize=8)

    # === Bottom Row: 3 Imagery Overlays (one per method) ===
    if imagery is not None:
        # Prepare imagery (convert from CHW to HWC and normalize if needed)
        if imagery.shape[0] == 3:  # CHW format
            img_display = np.transpose(imagery, (1, 2, 0))
        else:
            img_display = imagery

        # Ensure imagery is in [0, 1] range
        if img_display.max() > 1.0:
            img_display = img_display / 255.0

        # Prepare different overlays for each method
        # 1. Raw heatmap overlay (for Method 1)
        # Mask out low correlation values (similar to softmax)
        threshold_heat = heat_np.max() * 0.001  # Show only values > 0.1% of max
        heat_masked = np.ma.masked_where(heat_np < threshold_heat, heat_np)
        heat_norm = (heat_masked - heat_masked.min()) / (heat_masked.max() - heat_masked.min() + 1e-8)
        heatmap_colored = cm.hot(heat_norm)
        # Set alpha channel: 0 for masked values, 1 for valid values
        heatmap_colored[:, :, 3] = np.where(heat_np < threshold_heat, 0, 1)
        contour_levels_heat = np.linspace(heat_masked.min(), heat_masked.max(), 10)

        # 2. Softmax probability overlay (for Method 2)
        # Mask out very low probability values (near zero)
        threshold = softmax_np.max() * 0.001  # Show only values > 0.1% of max
        softmax_masked = np.ma.masked_where(softmax_np < threshold, softmax_np)
        softmax_norm = (softmax_masked - softmax_masked.min()) / (softmax_masked.max() - softmax_masked.min() + 1e-8)
        softmax_colored = cm.viridis(softmax_norm)
        # Set alpha channel: 0 for masked values, 1 for valid values
        softmax_colored[:, :, 3] = np.where(softmax_np < threshold, 0, 1)
        contour_levels_softmax = np.linspace(softmax_masked.min(), softmax_masked.max(), 12)

        # 3. No density map for Method 3
        # Gaussian peak refinement is a discrete optimization (not probabilistic/differentiable)
        # So there's no meaningful convergence density to show - just the peak point

        # Define ORIGINAL imagery extent (not centered, full range from 0)
        img_H, img_W = img_display.shape[:2]
        if use_meters:
            imagery_extent = [0, img_W * resolution, 0, img_H * resolution]
        else:
            imagery_extent = [0, img_W, 0, img_H]

        if lidar_hw is not None:
            if lidar_hw == img_display.shape[:2]:
                lidar_extent = list(imagery_extent)
            else:
                if use_meters:
                    lidar_extent = [0, lidar_hw[1] * resolution, 0, lidar_hw[0] * resolution]
                else:
                    lidar_extent = [0, lidar_hw[1], 0, lidar_hw[0]]

        # Extract peak offset values
        peak_x_offset, peak_y_offset = peak_offset

        # Calculate where to position heatmap overlay on the full imagery
        # Heatmap center should align with imagery center
        heatmap_center_x = img_W / 2.0
        heatmap_center_y = img_H / 2.0
        heatmap_half_w = W / 2.0
        heatmap_half_h = H / 2.0

        if use_meters:
            heatmap_extent = [
                (heatmap_center_x - heatmap_half_w) * resolution,
                (heatmap_center_x + heatmap_half_w) * resolution,
                (heatmap_center_y - heatmap_half_h) * resolution,
                (heatmap_center_y + heatmap_half_h) * resolution,
            ]
        else:
            heatmap_extent = [
                heatmap_center_x - heatmap_half_w,
                heatmap_center_x + heatmap_half_w,
                heatmap_center_y - heatmap_half_h,
                heatmap_center_y + heatmap_half_h,
            ]

        # Adjust prediction coordinates to imagery coordinate system
        # All prediction offsets are in pixels, so add to center position
        peak_x_img = heatmap_center_x + peak_x_offset
        peak_y_img = heatmap_center_y + peak_y_offset
        sx_img = heatmap_center_x + sx
        sy_img = heatmap_center_y + sy
        gx_img = heatmap_center_x + gx
        gy_img = heatmap_center_y + gy
        gt_x_img = heatmap_center_x
        gt_y_img = heatmap_center_y

        # Convert to meters if needed
        if use_meters:
            peak_x_img *= resolution
            peak_y_img *= resolution
            sx_img *= resolution
            sy_img *= resolution
            gx_img *= resolution
            gy_img *= resolution
            gt_x_img *= resolution
            gt_y_img *= resolution

        # === Imagery Overlay 1: Discrete Peak ===
        # Show only raw heatmap overlay - no markers
        axes_bottom[0].imshow(
            img_display,
            origin="lower",
            extent=imagery_extent,
        )
        axes_bottom[0].imshow(
            heatmap_colored,
            origin="lower",
            extent=heatmap_extent,
        )
        # Add offset value text for Method 1
        peak_x_offset, peak_y_offset = peak_offset
        if use_meters:
            offset_text = f"Offset: ({peak_x_offset * resolution:+.3f}m, {peak_y_offset * resolution:+.3f}m)"
        else:
            offset_text = f"Offset: ({peak_x_offset:+.1f}px, {peak_y_offset:+.1f}px)"
        axes_bottom[0].text(
            0.02, 0.98, offset_text,
            transform=axes_bottom[0].transAxes,
            fontsize=10,
            verticalalignment='top',
            color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )

        axes_bottom[0].set_title(
            "Method 1 on Imagery\nRaw Correlation Heatmap Overlay",
            fontsize=12,
            fontweight="bold",
        )
        axes_bottom[0].set_xlabel(f"X ({unit})")
        axes_bottom[0].set_ylabel(f"Y ({unit})")
        # Set explicit axis limits to ensure original full scale
        axes_bottom[0].set_xlim(imagery_extent[0], imagery_extent[1])
        axes_bottom[0].set_ylim(imagery_extent[2], imagery_extent[3])
        axes_bottom[0].set_aspect('equal', adjustable='box')
        axes_bottom[0].grid(True, alpha=0.3, linestyle=":", linewidth=0.5, color='white')

        # Create coordinate grids for contours (both overlays 2 and 3 use same size as heatmap)
        X_heatmap = np.linspace(heatmap_extent[0], heatmap_extent[1], W)
        Y_heatmap = np.linspace(heatmap_extent[2], heatmap_extent[3], H)
        X_heat_grid, Y_heat_grid = np.meshgrid(X_heatmap, Y_heatmap)

        # === Imagery Overlay 2: Softmax ===
        axes_bottom[1].imshow(
            img_display,
            origin="lower",
            extent=imagery_extent,
        )
        axes_bottom[1].imshow(
            softmax_colored,
            origin="lower",
            extent=heatmap_extent,
        )
        axes_bottom[1].contour(
            X_heat_grid, Y_heat_grid, softmax_np,
            levels=contour_levels_softmax,
            colors='cyan',
            linewidths=0.8,
            alpha=0.6,
        )
        # Add offset value text for Method 2
        if use_meters:
            offset_text = f"Offset: ({sx * resolution:+.3f}m, {sy * resolution:+.3f}m)"
        else:
            offset_text = f"Offset: ({sx:+.2f}px, {sy:+.2f}px)"
        axes_bottom[1].text(
            0.02, 0.98, offset_text,
            transform=axes_bottom[1].transAxes,
            fontsize=10,
            verticalalignment='top',
            color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )

        axes_bottom[1].set_title(
            "Method 2 on Imagery\nSoftmax Probability Density Overlay",
            fontsize=12,
            fontweight="bold",
        )
        axes_bottom[1].set_xlabel(f"X ({unit})")
        axes_bottom[1].set_ylabel(f"Y ({unit})")
        # Set explicit axis limits to ensure original full scale
        axes_bottom[1].set_xlim(imagery_extent[0], imagery_extent[1])
        axes_bottom[1].set_ylim(imagery_extent[2], imagery_extent[3])
        axes_bottom[1].set_aspect('equal', adjustable='box')
        axes_bottom[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5, color='white')

        # === Imagery Overlay 3: Gaussian Peak Point (Discrete Optimization) ===
        # No density overlay - just show the refined peak point
        # Method 3 is discrete optimization, not a convergence-based method
        axes_bottom[2].imshow(
            img_display,
            origin="lower",
            extent=imagery_extent,
        )

        # Mark the Gaussian peak refinement result with simple point
        axes_bottom[2].plot(
            [gx_img], [gy_img],
            'o',
            color='lime',
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.5,
            zorder=10,
        )

        # Add offset value text for Method 3
        if use_meters:
            offset_text = f"Offset: ({gx * resolution:+.3f}m, {gy * resolution:+.3f}m)"
        else:
            offset_text = f"Offset: ({gx:+.2f}px, {gy:+.2f}px)"
        axes_bottom[2].text(
            0.02, 0.98, offset_text,
            transform=axes_bottom[2].transAxes,
            fontsize=10,
            verticalalignment='top',
            color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )

        axes_bottom[2].set_title(
            "Method 3 on Imagery\nGaussian Peak Refinement (Discrete Point)",
            fontsize=12,
            fontweight="bold",
        )
        axes_bottom[2].set_xlabel(f"X ({unit})")
        axes_bottom[2].set_ylabel(f"Y ({unit})")
        # Set explicit axis limits to ensure original full scale
        axes_bottom[2].set_xlim(imagery_extent[0], imagery_extent[1])
        axes_bottom[2].set_ylim(imagery_extent[2], imagery_extent[3])
        axes_bottom[2].set_aspect('equal', adjustable='box')
        axes_bottom[2].grid(True, alpha=0.3, linestyle=":", linewidth=0.5, color='white')
    else:
        # No imagery available, show message in all bottom panels
        for i in range(num_method_cols):
            axes_bottom[i].text(
                0.5, 0.5,
                "Imagery Not Available",
                ha='center', va='center',
                fontsize=14,
                transform=axes_bottom[i].transAxes,
            )
            axes_bottom[i].set_title(f"Method {i+1} Imagery Overlay", fontsize=12, fontweight="bold")
            axes_bottom[i].axis('off')

    if lidar_extent is None and lidar_hw is not None:
        if use_meters:
            lidar_extent = [0, lidar_hw[1] * resolution, 0, lidar_hw[0] * resolution]
        else:
            lidar_extent = [0, lidar_hw[1], 0, lidar_hw[0]]

    if lidar_axes:
        panel_specs: list[
            tuple[str, np.ndarray | np.ma.MaskedArray, str, str, float | None, str | None]
        ] = []
        if lidar_height_display is not None:
            fill_rate = None
            fill_source: str | None = None
            if mask_valid_count is not None and total_valid_pixels > 0:
                fill_rate = mask_valid_count / total_valid_pixels
                fill_source = "mask"
            elif total_valid_pixels > 0 and "height" in valid_counts:
                fill_rate = valid_counts["height"] / total_valid_pixels
                fill_source = "finite"
            panel_specs.append(
                ("LiDAR Height", lidar_height_display, "turbo", "Normalized Height", fill_rate, fill_source)
            )
        if lidar_intensity_display is not None:
            fill_rate = None
            fill_source = None
            if mask_valid_count is not None and total_valid_pixels > 0:
                fill_rate = mask_valid_count / total_valid_pixels
                fill_source = "mask"
            elif total_valid_pixels > 0 and "intensity" in valid_counts:
                fill_rate = valid_counts["intensity"] / total_valid_pixels
                fill_source = "finite"
            panel_specs.append(
                ("LiDAR Intensity", lidar_intensity_display, "gray", "Normalized Intensity", fill_rate, fill_source)
            )

        for ax, (title, data, cmap_name, colorbar_label, fill_rate, fill_source) in zip(lidar_axes, panel_specs):
            if data is not None:
                if lidar_extent is not None:
                    im_panel = ax.imshow(
                        data,
                        origin="lower",
                        extent=lidar_extent,
                        cmap=cmap_name,
                    )
                else:
                    im_panel = ax.imshow(
                        data,
                        origin="lower",
                        cmap=cmap_name,
                    )
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel(f"X ({unit})")
                ax.set_ylabel(f"Y ({unit})")
                if lidar_extent is not None:
                    ax.set_xlim(lidar_extent[0], lidar_extent[1])
                    ax.set_ylim(lidar_extent[2], lidar_extent[3])
                ax.set_aspect('equal', adjustable='box')
                grid_color = "white" if cmap_name != "gray" else "black"
                ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5, color=grid_color)
                fig.colorbar(im_panel, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
                if fill_rate is not None:
                    suffix = " (mask)" if fill_source == "mask" else " (finite)"
                    ax.text(
                        0.02,
                        0.94,
                        f"Fill: {fill_rate * 100:.1f}%{suffix}",
                        transform=ax.transAxes,
                        fontsize=10,
                        fontweight="bold",
                        color="white" if cmap_name != "gray" else "black",
                        bbox=dict(boxstyle="round", facecolor="black" if cmap_name != "gray" else "white", alpha=0.6),
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "LiDAR Channel Not Available",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=ax.transAxes,
                )
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.axis("off")

        # Hide any leftover axes if panel specs fewer than axes
        if len(panel_specs) < len(lidar_axes):
            for extra_ax in lidar_axes[len(panel_specs):]:
                extra_ax.text(
                    0.5,
                    0.5,
                    "LiDAR Channel Not Available",
                    ha="center",
                    va="center",
                    fontsize=14,
                    transform=extra_ax.transAxes,
                )
                extra_ax.set_title("LiDAR Panel", fontsize=12, fontweight="bold")
                extra_ax.axis("off")

    # Overall title
    fig.suptitle(
        "Position-Finding Methods Comparison with Imagery Overlay",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[Save] Visualization saved to: {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ReLL position finding with imagery overlay (no rotation)."
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
        "--all",
        action="store_true",
        help="Show all plots including top row (method visualizations). Default: show only imagery overlays.",
    )
    parser.add_argument(
        "--lidar",
        action="store_true",
        help="Add an extra panel visualizing the LiDAR raster for the selected sample.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ReLL Position-Finding Visualizer (Focus on X,Y)")
    print("=" * 70)

    # Load trained model
    model, model_cfg = load_checkpoint(args.checkpoint, device=args.device)

    # Load single sample (no augmentation)
    sample = load_sample(args.sample, lidar_variant=args.lidar_variant)
    lidar_tensor = sample["lidar"]
    lidar = lidar_tensor.to(args.device)
    geospatial = sample["map"].to(args.device)
    resolution = sample["resolution"]
    lidar_np = lidar_tensor.cpu().numpy() if args.lidar else None

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

    # Softmax refinement
    (
        softmax_probs,
        softmax_mu_x_px,
        softmax_mu_y_px,
        softmax_sigma_x_px,
        softmax_sigma_y_px,
    ) = _softmax_refinement(heatmap)

    # Gaussian peak refinement
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
        f"\n[Method 3: Gaussian Peak Fitting]\n"
        f"  Refined μ: ({gaussian_mu_x_px:+.3f}px, {gaussian_mu_y_px:+.3f}px) = "
        f"({gaussian_mu_x_px * resolution:+.3f}m, {gaussian_mu_y_px * resolution:+.3f}m)\n"
        f"  Distance from GT (0,0): {gaussian_dist_px:.3f}px = {gaussian_dist_m:.4f}m "
        f"[vs Peak:{gaussian_beats_peak} vs Softmax:{gaussian_beats_softmax}]{gaussian_best}\n"
        f"  Score: {gaussian_score:.4f}"
    )
    print("=" * 70)

    # Extract imagery for overlay
    imagery_tensor = geospatial[:3, :, :]  # RGB channels from map
    imagery_np = imagery_tensor.cpu().numpy()

    # Generate visualization (always shown)
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
        imagery=imagery_np,
        lidar_image=lidar_np,
        save_path=args.save,
        show_all=args.all,
    )


if __name__ == "__main__":
    main()
