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

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from Train.config import ModelConfig
from Train.data import raster_builder_from_processed_dir, replace_nan_with_zero
from Train.model import LocalizationModel


def load_checkpoint(ckpt_path: Path, device: str = "cpu") -> Tuple[LocalizationModel, ModelConfig]:
    """Load trained model from checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Load] Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint
    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint missing 'model_config'. Cannot reconstruct model.")

    model_cfg = checkpoint["model_config"]
    print(f"[Config] Model config loaded:")
    print(f"  - embed_dim: {model_cfg.embed_dim}")
    print(f"  - proj_dim: {model_cfg.proj_dim}")
    print(f"  - search_radius: {model_cfg.search_radius}")
    print(f"  - theta_search_deg: {model_cfg.theta_search_deg}")

    # Reconstruct model
    model = LocalizationModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
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
) -> None:
    """
    Visualize embeddings and correlation heatmap.

    Shows:
    1. LiDAR embedding (first 3 channels as RGB)
    2. Map embedding (first 3 channels as RGB)
    3. Cross-correlation heatmap with peak marker
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # === 1. LiDAR Embedding Visualization ===
    lidar_embed = results["lidar_embed"]  # [C, H, W]
    # Take first 3 channels and normalize to [0, 1] for RGB display
    lidar_rgb = lidar_embed[:3].permute(1, 2, 0).numpy()  # [H, W, 3]
    lidar_rgb = (lidar_rgb - lidar_rgb.min()) / (lidar_rgb.max() - lidar_rgb.min() + 1e-9)

    axes[0].imshow(lidar_rgb)
    axes[0].set_title(f"LiDAR Embedding\n(first 3/{lidar_embed.shape[0]} channels)", fontsize=12)
    axes[0].axis("off")

    # === 2. Map Embedding Visualization ===
    map_embed = results["map_embed"]  # [C, H, W]
    # Take first 3 channels and normalize to [0, 1] for RGB display
    map_rgb = map_embed[:3].permute(1, 2, 0).numpy()  # [H, W, 3]
    map_rgb = (map_rgb - map_rgb.min()) / (map_rgb.max() - map_rgb.min() + 1e-9)

    axes[1].imshow(map_rgb)
    axes[1].set_title(f"Map Embedding\n(first 3/{map_embed.shape[0]} channels)", fontsize=12)
    axes[1].axis("off")

    # === 3. Cross-Correlation Heatmap ===
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

    im = axes[2].imshow(correlation_heatmap, cmap="hot", interpolation="nearest")
    axes[2].plot(peak_col, peak_row, "bx", markersize=15, markeredgewidth=3, label="Peak")
    axes[2].plot(center_col, center_row, "g+", markersize=15, markeredgewidth=2, label="Center")
    axes[2].set_title(
        f"Cross-Correlation Heatmap\n"
        f"Window: {2*search_radius+1}×{2*search_radius+1} px\n"
        f"Peak: ({peak_row}, {peak_col}) = {peak_score:.4f}",
        fontsize=12,
    )
    axes[2].set_xlabel(f"X offset (center at {center_col})")
    axes[2].set_ylabel(f"Y offset (center at {center_row})")
    axes[2].legend(loc="upper right")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Add grid
    axes[2].grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    plt.suptitle(
        f"ReLL Localization Inference Visualization\n"
        f"Detected Offset: ({offset_x_m:.3f}m, {offset_y_m:.3f}m) = ({offset_x_px}px, {offset_y_px}px)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

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

    print(f"\n[Inference] Running model on {args.device}...")

    # Compute embeddings and correlation
    results = compute_embeddings_and_correlation(model, lidar, geospatial)

    # Visualize
    print(f"\n[Visualize] Creating visualization...")
    visualize_results(results, resolution, model_cfg.search_radius, save_path=args.save)

    print("\n[Done] Inference visualization complete!")


if __name__ == "__main__":
    main()
