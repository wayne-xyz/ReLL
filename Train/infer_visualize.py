from __future__ import annotations

"""
Inference visualization tool for ReLL localization model.

Loads a trained model and visualizes:
- Embedding features from both encoders (LiDAR and Map)
- Cross-correlation heatmap from sliding window search
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.serialization import add_safe_globals

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.config import DatasetConfig, ModelConfig, OptimConfig, SaveConfig, EarlyStopConfig
from Train.data import raster_builder_from_processed_dir, replace_nan_with_zero
from Train.model import LocalizationModel


def load_checkpoint(ckpt_path: Path, device: str = "cpu") -> Tuple[LocalizationModel, ModelConfig]:
    """Load trained model from checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Load] Loading checkpoint from: {ckpt_path}")

    # Register safe types for unpickling config objects
    safe_types = [DatasetConfig, ModelConfig, OptimConfig, SaveConfig, EarlyStopConfig]
    try:
        from pathlib import WindowsPath
        safe_types.append(WindowsPath)
    except ImportError:
        pass
    try:
        from pathlib import PosixPath
        safe_types.append(PosixPath)
    except ImportError:
        pass
    add_safe_globals(safe_types)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint (key is "model_cfg" not "model_config")
    if "model_cfg" not in checkpoint:
        raise KeyError(f"Checkpoint missing 'model_cfg'. Available keys: {list(checkpoint.keys())}")

    model_cfg = checkpoint["model_cfg"]
    print(f"[Config] Model config loaded:")
    print(f"  - embed_dim: {model_cfg.embed_dim}")
    print(f"  - proj_dim: {model_cfg.proj_dim}")
    print(f"  - search_radius: {model_cfg.search_radius}")
    print(f"  - theta_search_deg: {model_cfg.theta_search_deg}")

    # Reconstruct model (state dict key is "model_state" not "model_state_dict")
    model = LocalizationModel(model_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"[Load] Model loaded successfully!")
    return model, model_cfg


def load_sample(sample_dir: Path, lidar_variant: str = "gicp") -> Dict[str, Tensor]:
    """Load a single sample without augmentation (no perturbation)."""
    print(f"[Data] Loading sample from: {sample_dir}")

    # Load raw rasters
    rasters = raster_builder_from_processed_dir(sample_dir)

    # Select LiDAR variant
    if lidar_variant.lower() == "gicp":
        lidar_height = torch.tensor(rasters["gicp_height"], dtype=torch.float32)
        lidar_intensity = torch.tensor(rasters["gicp_intensity"], dtype=torch.float32)
    else:
        lidar_height = torch.tensor(rasters["non_aligned_height"], dtype=torch.float32)
        lidar_intensity = torch.tensor(rasters["non_aligned_intensity"], dtype=torch.float32)

    dsm_height = torch.tensor(rasters["dsm_height"], dtype=torch.float32)
    imagery = torch.tensor(rasters["imagery"], dtype=torch.float32)

    # Ensure imagery is RGB only
    if imagery.shape[0] > 3:
        imagery = imagery[:3, :, :]

    # Clean NaN values
    lidar_height, lidar_mask = replace_nan_with_zero(lidar_height)
    lidar_intensity, _ = replace_nan_with_zero(lidar_intensity)
    dsm_height, dsm_mask = replace_nan_with_zero(dsm_height)

    # Build tensors (same as dataset)
    lidar_tensor = torch.stack(
        [lidar_height, lidar_intensity / 255.0, lidar_mask],
        dim=0,
    )  # [3, H, W]

    if imagery.dtype == torch.uint8:
        imagery = imagery.float() / 255.0

    dsm_height = dsm_height.unsqueeze(0)
    dsm_mask = dsm_mask.unsqueeze(0)
    map_tensor = torch.cat([imagery, dsm_height, dsm_mask], dim=0)  # [5, H, W]

    resolution = float(rasters["resolution"])

    print(f"[Data] Sample loaded:")
    print(f"  - LiDAR shape: {lidar_tensor.shape}")
    print(f"  - Map shape: {map_tensor.shape}")
    print(f"  - Resolution: {resolution:.4f} m/px")

    return {
        "lidar": lidar_tensor,
        "map": map_tensor,
        "resolution": resolution,
    }


@torch.no_grad()
def compute_embeddings_and_correlation(
    model: LocalizationModel,
    lidar: Tensor,
    geospatial: Tensor,
) -> Dict[str, Tensor]:
    """
    Compute embeddings and cross-correlation heatmap.

    Returns:
        dict with keys:
            - lidar_embed: [C, H, W] - LiDAR embedding features
            - map_embed: [C, H, W] - Map embedding features
            - lidar_proj: [D, H, W] - Projected LiDAR (used in correlation)
            - map_proj: [D, H, W] - Projected map (used in correlation)
            - correlation_heatmap: [H_out, W_out] - Cross-correlation scores
    """
    # Add batch dimension
    lidar = lidar.unsqueeze(0)  # [1, 3, H, W]
    geospatial = geospatial.unsqueeze(0)  # [1, 5, H, W]

    # Encode through adapters + encoders
    lidar_feat = model._encode(lidar, model.lidar_adapter, model.lidar_encoder)
    map_feat = model._encode(geospatial, model.map_adapter, model.map_encoder)

    # Project to lower dimension
    lidar_proj = model._l2norm(model.lidar_projection(lidar_feat))
    map_proj = model._l2norm(model.map_projection(map_feat))

    # Compute cross-correlation via sliding window
    translation_logits = model.compute_translation_logits(lidar_proj, map_proj)

    # Remove batch dimension for visualization
    lidar_embed = lidar_feat.squeeze(0).cpu()  # [C, H, W]
    map_embed = map_feat.squeeze(0).cpu()  # [C, H, W]
    lidar_proj_vis = lidar_proj.squeeze(0).cpu()  # [D, H, W]
    map_proj_vis = map_proj.squeeze(0).cpu()  # [D, H, W]
    correlation_heatmap = translation_logits.squeeze(0).cpu()  # [H_out, W_out]

    print(f"[Inference] Embeddings computed:")
    print(f"  - LiDAR embedding: {lidar_embed.shape}")
    print(f"  - Map embedding: {map_embed.shape}")
    print(f"  - LiDAR projection: {lidar_proj_vis.shape}")
    print(f"  - Map projection: {map_proj_vis.shape}")
    print(f"  - Correlation heatmap: {correlation_heatmap.shape}")

    return {
        "lidar_embed": lidar_embed,
        "map_embed": map_embed,
        "lidar_proj": lidar_proj_vis,
        "map_proj": map_proj_vis,
        "correlation_heatmap": correlation_heatmap,
    }


def visualize_results(
    results: Dict[str, Tensor],
    resolution: float,
    search_radius: int,
    save_path: Path | None = None,
    lidar_height: Tensor | None = None,
    lidar_intensity: Tensor | None = None,
    dsm_height: Tensor | None = None,
    imagery: Tensor | None = None,
) -> None:
    """
    Visualize projection features and correlation heatmap with original image preview.

    Shows:
    - Top section: Original images (LiDAR height, intensity, DSM, imagery)
    - Bottom section: All 4 projection channels separately:
      - Row 1: LiDAR projection ch0, ch1, ch2
      - Row 2: LiDAR projection ch3, Map projection ch0, ch1
      - Row 3: Map projection ch2, ch3, Cross-correlation heatmap
    """
    # Create figure with two sections: preview (1x4) + features (3x3)
    fig = plt.figure(figsize=(28, 20))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    # Preview section (top row) - 4 subplots
    preview_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    # Features section (bottom 3 rows) - 3x3 grid
    feature_axes = [fig.add_subplot(gs[i+1, j]) for i in range(3) for j in range(3)]
    axes = feature_axes

    # === Preview Section: Original Images ===
    if lidar_height is not None:
        h = lidar_height.numpy() if isinstance(lidar_height, Tensor) else lidar_height
        h_norm = (h - h.min()) / (h.max() - h.min() + 1e-9)
        preview_axes[0].imshow(h_norm, cmap="viridis")
        preview_axes[0].set_title("LiDAR Height", fontsize=12, fontweight="bold")
        preview_axes[0].axis("off")
        plt.colorbar(preview_axes[0].images[0], ax=preview_axes[0], fraction=0.046, pad=0.04)

    if lidar_intensity is not None:
        intensity = lidar_intensity.numpy() if isinstance(lidar_intensity, Tensor) else lidar_intensity
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-9)
        preview_axes[1].imshow(intensity_norm, cmap="gray")
        preview_axes[1].set_title("LiDAR Intensity", fontsize=12, fontweight="bold")
        preview_axes[1].axis("off")
        plt.colorbar(preview_axes[1].images[0], ax=preview_axes[1], fraction=0.046, pad=0.04)

    if dsm_height is not None:
        dsm = dsm_height.numpy() if isinstance(dsm_height, Tensor) else dsm_height
        dsm_norm = (dsm - dsm.min()) / (dsm.max() - dsm.min() + 1e-9)
        preview_axes[2].imshow(dsm_norm, cmap="plasma")
        preview_axes[2].set_title("DSM Height", fontsize=12, fontweight="bold")
        preview_axes[2].axis("off")
        plt.colorbar(preview_axes[2].images[0], ax=preview_axes[2], fraction=0.046, pad=0.04)

    if imagery is not None:
        img = imagery.numpy() if isinstance(imagery, Tensor) else imagery
        # Handle different image formats
        if img.shape[0] == 3:  # CHW format
            img = img.transpose(1, 2, 0)
        elif img.shape[2] == 3:  # HWC format
            pass
        # Normalize if needed
        if img.max() > 1.0:
            img = img / 255.0
        preview_axes[3].imshow(img)
        preview_axes[3].set_title("Imagery", fontsize=12, fontweight="bold")
        preview_axes[3].axis("off")


    lidar_proj = results["lidar_proj"]  # [D, H, W] - D=proj_dim (usually 4)
    map_proj = results["map_proj"]  # [D, H, W] - D=proj_dim (usually 4)
    proj_dim = lidar_proj.shape[0]

    # === LiDAR Projection Channels (0-3) ===
    for i in range(min(proj_dim, 4)):
        channel_data = lidar_proj[i].numpy()  # [H, W]
        # Normalize to [0, 1] for display
        channel_norm = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-9)

        axes[i].imshow(channel_norm, cmap="gray")
        axes[i].set_title(f"LiDAR Proj Ch{i}", fontsize=11, fontweight="bold")
        axes[i].axis("off")
        # Add colorbar
        plt.colorbar(axes[i].images[0], ax=axes[i], fraction=0.046, pad=0.04)

    # === Map Projection Channels (0-3) ===
    for i in range(min(proj_dim, 4)):
        channel_data = map_proj[i].numpy()  # [H, W]
        # Normalize to [0, 1] for display
        channel_norm = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-9)

        plot_idx = 4 + i  # Start after LiDAR channels
        axes[plot_idx].imshow(channel_norm, cmap="gray")
        axes[plot_idx].set_title(f"Map Proj Ch{i}", fontsize=11, fontweight="bold")
        axes[plot_idx].axis("off")
        # Add colorbar
        plt.colorbar(axes[plot_idx].images[0], ax=axes[plot_idx], fraction=0.046, pad=0.04)

    # === Cross-Correlation Heatmap (last subplot) ===
    correlation_heatmap = results["correlation_heatmap"].numpy()  # [H_out, W_out]

    # Find peak
    peak_idx = np.argmax(correlation_heatmap)
    H_out, W_out = correlation_heatmap.shape
    peak_row = peak_idx // W_out
    peak_col = peak_idx % W_out
    peak_score = correlation_heatmap[peak_row, peak_col]

    # Compute offset in pixels and meters
    center_row = H_out // 2
    center_col = W_out // 2
    offset_y_px = peak_row - center_row
    offset_x_px = peak_col - center_col
    offset_y_m = offset_y_px * resolution
    offset_x_m = offset_x_px * resolution

    corr_ax = axes[8]  # Last subplot (3x3 grid, index 8)
    im = corr_ax.imshow(correlation_heatmap, cmap="gray", interpolation="nearest")
    corr_ax.plot(peak_col, peak_row, "bx", markersize=15, markeredgewidth=3, label="Peak")
    corr_ax.plot(center_col, center_row, "g+", markersize=15, markeredgewidth=2, label="Center")
    corr_ax.set_title(
        f"Cross-Correlation Heatmap\n"
        f"Window: {2*search_radius+1}×{2*search_radius+1} px\n"
        f"Peak: ({peak_row}, {peak_col}) = {peak_score:.4f}",
        fontsize=11,
        fontweight="bold",
    )
    corr_ax.set_xlabel(f"X offset (center at {center_col})", fontsize=9)
    corr_ax.set_ylabel(f"Y offset (center at {center_row})", fontsize=9)
    corr_ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=corr_ax, fraction=0.046, pad=0.04)

    # Add grid
    corr_ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.suptitle(
        f"ReLL Localization Inference - All Projection Channels\n"
        f"Detected Offset: ({offset_x_m:.3f}m, {offset_y_m:.3f}m) = ({offset_x_px}px, {offset_y_px}px)\n"
        f"Proj dim: {proj_dim} channels (L2-normalized), Resolution: {resolution:.4f} m/px",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Save] Visualization saved to: {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ReLL model inference.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"G:\GithubProject\ReLL\model-save\best_1000.ckpt"),
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(r"Rell-sample-raster2\0As5GhcTExRFWNboqeHwpegdeGf2j0YW_036"),
        help="Path to sample directory (containing .npy files)",
    )
    parser.add_argument(
        "--lidar-variant",
        type=str,
        default="gicp",
        choices=["gicp", "non_aligned"],
        help="LiDAR variant to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save visualization (e.g., output.png)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ReLL Localization Inference Visualization")
    print("=" * 80)

    # Load model
    model, model_cfg = load_checkpoint(args.checkpoint, device=args.device)

    # Load sample (no perturbation - aligned)
    sample = load_sample(args.sample, lidar_variant=args.lidar_variant)

    # Move to device
    lidar = sample["lidar"].to(args.device)
    geospatial = sample["map"].to(args.device)
    resolution = sample["resolution"]

    # Extract original image data for preview
    rasters = raster_builder_from_processed_dir(args.sample)
    if args.lidar_variant.lower() == "gicp":
        lidar_height = torch.tensor(rasters["gicp_height"], dtype=torch.float32)
        lidar_intensity = torch.tensor(rasters["gicp_intensity"], dtype=torch.float32)
    else:
        lidar_height = torch.tensor(rasters["non_aligned_height"], dtype=torch.float32)
        lidar_intensity = torch.tensor(rasters["non_aligned_intensity"], dtype=torch.float32)
    
    dsm_height = torch.tensor(rasters["dsm_height"], dtype=torch.float32)
    imagery = torch.tensor(rasters["imagery"], dtype=torch.float32)
    
    # Ensure imagery is RGB only
    if imagery.shape[0] > 3:
        imagery = imagery[:3, :, :]
    
    # Clean NaN values
    lidar_height, _ = replace_nan_with_zero(lidar_height)
    lidar_intensity, _ = replace_nan_with_zero(lidar_intensity)
    dsm_height, _ = replace_nan_with_zero(dsm_height)

    print(f"\n[Inference] Running model on {args.device}...")

    # Compute embeddings and correlation
    results = compute_embeddings_and_correlation(model, lidar, geospatial)

    # Visualize with original image preview
    print(f"\n[Visualize] Creating visualization...")
    visualize_results(
        results,
        resolution,
        model_cfg.search_radius,
        save_path=args.save,
        lidar_height=lidar_height,
        lidar_intensity=lidar_intensity,
        dsm_height=dsm_height,
        imagery=imagery,
    )

    print("\n[Done] Inference visualization complete!")


if __name__ == "__main__":
    main()
