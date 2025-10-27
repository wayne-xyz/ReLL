import os
import time
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import pickle  # Need pickle to save non-numpy objects like transform and profile
import torch


def _fmt_seconds(sec: float) -> str:
    """Return H:MM:SS for a duration in seconds."""
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def raster_builder_from_sample_dir(sample_dir: Path):
    """
    Builds raster data for a single sample directory using build_training_rasters.

    Looks for specific files within the sample_dir with known filenames:
      - LiDAR (GICP): segment_gicp_utm.parquet
      - LiDAR (non-aligned): segment_utm.parquet
      - DSM points: segment_dsm_utm.laz
      - Imagery (orthophoto): segment_imagery_utm.tif
    """
    print(f"Attempting to build rasters for sample: {sample_dir}")
    try:
        gicp_path = sample_dir / "segment_gicp_utm.parquet"
        non_path  = sample_dir / "segment_utm.parquet"
        dsm_path  = sample_dir / "segment_dsm_utm.laz"
        img_path  = sample_dir / "segment_imagery_utm.tif"

        # Check if files exist before proceeding
        if not gicp_path.exists():
            raise FileNotFoundError(f"Missing GICP parquet file: {gicp_path}")
        if not non_path.exists():
            raise FileNotFoundError(f"Missing non-aligned parquet file: {non_path}")
        if not dsm_path.exists():
            raise FileNotFoundError(f"Missing DSM laz file: {dsm_path}")
        if not img_path.exists():
            raise FileNotFoundError(f"Missing imagery tif file: {img_path}")

        # Call YOUR builder (already defined earlier in the notebook)
        ras = build_training_rasters(
            gicp_parquet=gicp_path,
            non_aligned_parquet=non_path,
            dsm_points_path=dsm_path,
            imagery_path=img_path,
            # ---- optional knobs you can tune:
            splat_mode="none",            # "none" | "bilinear" | "gaussian"
            target_m_per_px=None,         # set (e.g., 0.2) to force output resolution; else keep source
            sampling=1.0,                 # scale factor if you want quick low-res tests (e.g., 2.0 downsample 2x)
            coarsen_factor=None,          # extra integer coarsening
            preview=False,                # set True to visualize once
        )
        print(f"Successfully built rasters for sample: {sample_dir}")
        return ras
    except FileNotFoundError as e:
        print(f"Skipping sample {sample_dir} due to missing file: {e}")
        return None  # Return None or raise a specific error if you prefer to stop
    except Exception as e:  # Catch any other exceptions during raster building
        print(f"Skipping sample {sample_dir} due to error during raster building: {e}")
        return None


def save_all_rasters_to_disk(source_folder, target_folder):
    """
    Iterates through sample directories in source_folder, builds rasters,
    and saves the results to corresponding subfolders in target_folder.
    Includes logging of raster information, per-sample timing, and ETA.
    """
    source_path = Path(source_folder)
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting raster preparation and saving from {source_path} to {target_path}")

    sample_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    if not sample_dirs:
        print(f"No sample subfolders found in {source_folder}. Exiting.")
        return

    total = len(sample_dirs)
    print(f"Found {total} sample subfolders to process.")

    # Timing trackers
    t0 = time.monotonic()
    completed = 0
    cumulative_seconds = 0.0

    for i, sample_dir in enumerate(sample_dirs, start=1):
        sample_name = sample_dir.name
        sample_target_dir = target_path / sample_name
        sample_target_dir.mkdir(parents=True, exist_ok=True)

        start_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{i}/{total}] Processing sample: {sample_name} (started at {start_wall})")
        t_sample = time.monotonic()

        try:
            # Build rasters for this sample
            rasters = raster_builder_from_sample_dir(sample_dir)

            if rasters is None:
                print(f"Skipping sample {sample_name} as raster building failed.")
                continue

            # --- Raster information logging ---
            print(f"  Raster information for {sample_name}:")
            if "resolution" in rasters:
                print(f"    Resolution: {rasters['resolution']:.3f} m/px")
            if "transform" in rasters:
                transform = rasters['transform']
                print(
                    "    Transform (Affine): "
                    f"a={getattr(transform,'a',float('nan')):.3f}, "
                    f"b={getattr(transform,'b',float('nan')):.3f}, "
                    f"c={getattr(transform,'c',float('nan')):.3f}, "
                    f"d={getattr(transform,'d',float('nan')):.3f}, "
                    f"e={getattr(transform,'e',float('nan')):.3f}, "
                    f"f={getattr(transform,'f',float('nan')):.3f}"
                )
            if "imagery" in rasters:
                imagery_shape = rasters['imagery'].shape
                print(f"    Imagery shape (C, H, W): {imagery_shape}")
            if "gicp_height" in rasters:
                height_shape = rasters['gicp_height'].shape
                print(f"    LiDAR/DSM raster shape (H, W): {height_shape}")

            # Define keys for data to save
            tensor_keys = [
                "gicp_height", "gicp_intensity",
                "non_aligned_height", "non_aligned_intensity",
                "dsm_height", "imagery",
            ]
            metadata_keys = ["transform", "crs", "profile", "resolution"]

            # Save tensors
            for key in tensor_keys:
                if key in rasters and isinstance(rasters[key], (np.ndarray, torch.Tensor)):
                    save_path = sample_target_dir / f"{key}.npy"
                    data_to_save = rasters[key].numpy() if isinstance(rasters[key], torch.Tensor) else rasters[key]
                    np.save(save_path, data_to_save)
                else:
                    print(f"Warning: Tensor key '{key}' not found or not a supported type in rasters for {sample_name}")

            # Save metadata
            for key in metadata_keys:
                if key in rasters:
                    save_path = sample_target_dir / f"{key}.pkl"
                    with open(save_path, 'wb') as f:
                        pickle.dump(rasters[key], f)
                else:
                    print(f"Warning: Metadata key '{key}' not found in rasters for {sample_name}")

            print(f"Successfully processed and saved rasters for sample: {sample_name}")

        except Exception as e:
            print(f"An error occurred while processing and saving sample {sample_name}: {e}")

        finally:
            # Per-sample timing + ETA
            elapsed_sample = time.monotonic() - t_sample
            completed += 1
            cumulative_seconds += elapsed_sample
            avg_per_sample = cumulative_seconds / completed if completed else float('inf')
            remaining = total - completed
            eta_seconds = avg_per_sample * remaining
            finish_time = datetime.now() + timedelta(seconds=eta_seconds)

            print(
                f"⏱  Time for {sample_name}: {_fmt_seconds(elapsed_sample)} | "
                f"Avg/sample: {_fmt_seconds(avg_per_sample)} | "
                f"Remaining: {remaining} | "
                f"ETA: {_fmt_seconds(eta_seconds)} (finish by {finish_time.strftime('%Y-%m-%d %H:%M:%S')})"
            )

    total_elapsed = time.monotonic() - t0
    print(f"\nFinished preparing and saving all rasters.")
    print(f"Total time: {_fmt_seconds(total_elapsed)} for {total} samples "
          f"(avg {_fmt_seconds(total_elapsed / total)})")

# Example:
# save_all_rasters_to_disk(source_folder, target_folder)
