"""
Point Cloud to Raster Projection Preview Tool

Reads LiDAR point cloud (parquet or LAZ), projects to raster, and shows
two processing methods side-by-side:
1. Gap-filled with zeros (training method)
2. Shifted (0.5th percentile baseline) + gap-filled (data pipeline method)

Usage:
    python utilities/projection_compare.py PATH/TO/FILE.parquet
    python utilities/projection_compare.py PATH/TO/FILE.laz --resolution 0.2
    python utilities/projection_compare.py PATH/TO/FILE.parquet --crop --crop-size 30.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio import Affine

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from laspy import LazBackend
from laspy.errors import LaspyException


def read_parquet_points(path: Path) -> pd.DataFrame:
    """Read point cloud from parquet file."""
    df = pd.read_parquet(path)

    # Normalize column names
    col_map = {}
    if 'utm_e' in df.columns:
        col_map['utm_e'] = 'x'
    if 'utm_n' in df.columns:
        col_map['utm_n'] = 'y'
    if 'elevation' in df.columns:
        col_map['elevation'] = 'z'

    if col_map:
        df = df.rename(columns=col_map)

    required = {'x', 'y', 'z'}
    if not required.issubset(df.columns):
        raise ValueError(f"Parquet must contain columns: {required}. Found: {set(df.columns)}")

    return df[['x', 'y', 'z']].dropna()


def read_laz_points(path: Path) -> pd.DataFrame:
    """Read point cloud from LAZ/LAS file."""
    last_exc = None

    if path.suffix.lower() == '.las':
        import laspy
        with laspy.open(path) as las_file:
            points = las_file.read()
            header = las_file.header
    else:
        import laspy
        for backend in (LazBackend.Lazrs, LazBackend.Laszip):
            try:
                with laspy.open(path, laz_backend=backend) as las_file:
                    points = las_file.read()
                    header = las_file.header
                    break
            except LaspyException as exc:
                last_exc = exc
                continue
        else:
            raise RuntimeError(
                f"Unable to read LAZ file. Install backend with: pip install 'laspy[lazrs,laszip]'\n"
                f"Error: {last_exc}"
            )

    # Extract coordinates with scale/offset
    scale = np.array(header.scales, dtype=np.float64)
    offset = np.array(header.offsets, dtype=np.float64)

    x = (points.X * scale[0] + offset[0]).astype(np.float32)
    y = (points.Y * scale[1] + offset[1]).astype(np.float32)
    z = (points.Z * scale[2] + offset[2]).astype(np.float32)

    return pd.DataFrame({'x': x, 'y': y, 'z': z})


def compute_grid_from_points(
    df: pd.DataFrame,
    resolution: float,
    margin: float = 1.0,
) -> Tuple[Affine, int, int]:
    """Compute raster grid parameters from point cloud extent."""
    x_min = df['x'].min() - margin
    x_max = df['x'].max() + margin
    y_min = df['y'].min() - margin
    y_max = df['y'].max() + margin

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    transform = Affine.translation(x_min, y_max) * Affine.scale(resolution, -resolution)

    return transform, width, height


def rasterize_points(
    df: pd.DataFrame,
    transform: Affine,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize point cloud to 2D grid using nearest neighbor."""
    if df.empty:
        return np.full((height, width), np.nan, dtype=np.float32)

    xs = df['x'].to_numpy(np.float64)
    ys = df['y'].to_numpy(np.float64)
    vals = df['z'].to_numpy(np.float32)

    # Transform world coords to pixel coords
    inv = ~transform
    cols_f, rows_f = inv * (xs, ys)

    # Round to nearest pixel
    rows = np.round(rows_f).astype(np.int32)
    cols = np.round(cols_f).astype(np.int32)

    # Filter in-bounds points
    inb = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width) & np.isfinite(vals)
    rows, cols, vals = rows[inb], cols[inb], vals[inb]

    # Mean aggregation per pixel
    raster = np.full((height, width), np.nan, dtype=np.float32)
    sums = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint32)
    np.add.at(sums, (rows, cols), vals)
    np.add.at(counts, (rows, cols), 1)
    valid = counts > 0
    raster[valid] = sums[valid] / counts[valid]

    return raster


def fill_empty_pixels(raster: np.ndarray) -> np.ndarray:
    """Fill NaN gaps with zeros (same as training)."""
    filled = raster.copy()
    filled[np.isnan(filled)] = 0.0
    return filled


def shift_and_fill_raster(raster: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Shift height by 0.5th percentile, clamp to >= 0, then fill NaN with zeros.
    Same as Data-pipeline-fetch/raster.py _shift_for_display() + gap filling

    Returns:
        (shifted_filled_raster, shift_value)
    """
    # Compute 0.5th percentile of finite values
    finite = np.isfinite(raster)
    if not np.any(finite):
        return np.zeros_like(raster), 0.0

    shift = float(np.percentile(raster[finite], 0.5))

    # Subtract shift and clamp to >= 0 (only for finite values)
    shifted = raster.copy()
    shifted[finite] = shifted[finite] - shift
    shifted[finite] = np.maximum(shifted[finite], 0.0)

    # Fill NaN/empty pixels with zeros
    shifted[~finite] = 0.0

    return shifted, shift


def crop_raster_center(
    raster: np.ndarray,
    transform: Affine,
    crop_size_m: float,
    resolution: float,
) -> Tuple[np.ndarray, Affine]:
    """
    Crop a square region from the center of the raster.

    Args:
        raster: Input raster array (H, W)
        transform: Rasterio Affine transform
        crop_size_m: Size of crop in meters (square)
        resolution: Resolution in meters per pixel

    Returns:
        (cropped_raster, updated_transform)
    """
    height, width = raster.shape
    crop_size_px = int(np.round(crop_size_m / resolution))

    # Compute center pixel
    center_row = height // 2
    center_col = width // 2

    # Compute crop bounds
    half_crop = crop_size_px // 2
    row_start = max(0, center_row - half_crop)
    row_end = min(height, center_row + half_crop)
    col_start = max(0, center_col - half_crop)
    col_end = min(width, center_col + half_crop)

    # Crop raster
    cropped = raster[row_start:row_end, col_start:col_end]

    # Update transform to reflect new origin
    new_x_min, new_y_max = transform * (col_start, row_start)
    new_transform = Affine.translation(new_x_min, new_y_max) * Affine.scale(resolution, -resolution)

    return cropped, new_transform


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Preview point cloud rasterization with 2 processing methods: gap-filled (training) and shifted+filled (data pipeline).',
    )

    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to point cloud file (.parquet, .pq, .laz, .las)'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.2,
        help='Raster resolution in meters per pixel (default: 0.2)'
    )
    parser.add_argument(
        '--crop',
        action='store_true',
        help='Crop a centered region from the raster'
    )
    parser.add_argument(
        '--crop-size',
        type=float,
        default=30.0,
        help='Size of crop in meters (square region, default: 30.0)'
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}")
        return 1

    suffix = args.input_file.suffix.lower()
    if suffix not in {'.parquet', '.pq', '.laz', '.las'}:
        print(f"Error: Unsupported file format: {suffix}")
        print("Supported formats: .parquet, .pq, .laz, .las")
        return 1

    # Read point cloud
    print(f"Reading: {args.input_file}")
    try:
        if suffix in {'.parquet', '.pq'}:
            df = read_parquet_points(args.input_file)
        else:
            df = read_laz_points(args.input_file)
    except Exception as exc:
        print(f"Error reading file: {exc}")
        return 1

    print(f"Loaded {len(df):,} points")

    if len(df) == 0:
        print("Error: No valid points found")
        return 1

    # Compute grid
    transform, width, height = compute_grid_from_points(df, args.resolution)

    # Rasterize
    print(f"Rasterizing to {width} x {height} grid...")
    height_raster = rasterize_points(df, transform, width, height)

    # Crop if requested
    if args.crop:
        height_raster, transform = crop_raster_center(
            height_raster, transform, args.crop_size, args.resolution
        )
        crop_h, crop_w = height_raster.shape
        print(f"Cropped to {args.crop_size:.1f}m x {args.crop_size:.1f}m ({crop_w} x {crop_h} pixels)")
        width, height = crop_w, crop_h

    # Statistics
    finite_before = np.isfinite(height_raster)
    filled_before = finite_before.sum()
    total_pixels = finite_before.size

    print(f"\nOriginal raster:")
    print(f"  Filled pixels: {filled_before:,} / {total_pixels:,} ({filled_before / total_pixels * 100:.1f}%)")
    print(f"  Empty pixels:  {total_pixels - filled_before:,} ({(total_pixels - filled_before) / total_pixels * 100:.1f}%)")

    if np.any(finite_before):
        h_vals = height_raster[finite_before]
        print(f"  Height range:  {h_vals.min():.3f} to {h_vals.max():.3f} m")

    # Method 1: Fill gaps with zeros (training method)
    height_filled = fill_empty_pixels(height_raster)
    print(f"\nMethod 1 - Gap-filled (zeros):")
    print(f"  All pixels filled: {total_pixels:,} (100.0%)")
    print(f"  Height range: 0.000 to {h_vals.max():.3f} m")

    # Method 2: Shift + fill (data pipeline method)
    height_shifted, shift_value = shift_and_fill_raster(height_raster)
    print(f"\nMethod 2 - Shifted + filled:")
    print(f"  Shift baseline (p0.5): {shift_value:.3f} m")
    print(f"  All pixels filled: {total_pixels:,} (100.0%)")
    h_vals_shifted = height_shifted[height_shifted > 0]
    if len(h_vals_shifted) > 0:
        print(f"  Height range: 0.000 to {h_vals_shifted.max():.3f} m")

    # Compute extent for plotting
    x_min = transform.c
    x_max = x_min + transform.a * width
    y_max = transform.f
    y_min = y_max + transform.e * height
    extent = (x_min, x_max, y_min, y_max)

    # Compute vmax from original data for consistent color scale
    if np.any(finite_before):
        vmax_original = np.nanpercentile(height_raster[finite_before], 99)
    else:
        vmax_original = 1.0

    # Plot 2x2 grid: rasters on top, histograms on bottom
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    # Top row: Raster images
    # Panel 1: Gap-filled with zeros
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('Method 1: Gap-filled (zeros)', fontweight='bold', fontsize=13)
    vmin_filled = 0.0
    vmax_filled = vmax_original
    im = ax.imshow(height_filled, cmap='terrain', extent=extent, origin='upper', vmin=vmin_filled, vmax=vmax_filled)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='Height (m)')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.text(0.02, 0.98, 'Training method\nFill: 100.0%',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Panel 2: Shifted + filled
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(f'Method 2: Shifted + filled (p0.5={shift_value:.2f}m)', fontweight='bold', fontsize=13)
    vmin_shifted = 0.0
    non_zero = height_shifted > 0
    if np.any(non_zero):
        vmax_shifted = np.percentile(height_shifted[non_zero], 99)
    else:
        vmax_shifted = 1.0
    im = ax.imshow(height_shifted, cmap='terrain', extent=extent, origin='upper', vmin=vmin_shifted, vmax=vmax_shifted)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='Height (m)')
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_aspect('equal')
    ax.text(0.02, 0.98, 'Data pipeline method\nFill: 100.0%',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Bottom row: Histograms
    # Histogram 1: Gap-filled values
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title('Distribution: Gap-filled (zeros)', fontweight='bold', fontsize=12)
    values_filled = height_filled.flatten()
    # Separate zeros and non-zeros for better visualization
    zeros_count = np.sum(values_filled == 0)
    non_zeros = values_filled[values_filled > 0]

    ax.hist(non_zeros, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5, label='Non-zero values')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Zeros: {zeros_count:,} px')
    ax.set_xlabel('Height (m)', fontsize=11)
    ax.set_ylabel('Pixel Count', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text
    if len(non_zeros) > 0:
        stats_text = f'Mean: {non_zeros.mean():.2f}m\nMedian: {np.median(non_zeros):.2f}m\nStd: {non_zeros.std():.2f}m'
    else:
        stats_text = 'All zeros'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Histogram 2: Shifted + filled values
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title('Distribution: Shifted + filled', fontweight='bold', fontsize=12)
    values_shifted = height_shifted.flatten()
    zeros_count_shifted = np.sum(values_shifted == 0)
    non_zeros_shifted = values_shifted[values_shifted > 0]

    ax.hist(non_zeros_shifted, bins=50, color='seagreen', alpha=0.7, edgecolor='black', linewidth=0.5, label='Non-zero values')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Zeros: {zeros_count_shifted:,} px')
    ax.set_xlabel('Height (m)', fontsize=11)
    ax.set_ylabel('Pixel Count', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text
    if len(non_zeros_shifted) > 0:
        stats_text = f'Mean: {non_zeros_shifted.mean():.2f}m\nMedian: {np.median(non_zeros_shifted):.2f}m\nStd: {non_zeros_shifted.std():.2f}m'
    else:
        stats_text = 'All zeros'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    title_parts = [
        f'Point Cloud Rasterization Comparison',
        f'Resolution: {args.resolution:.3f} m/px',
    ]
    if args.crop:
        title_parts.append(f'Crop: {args.crop_size:.1f}m x {args.crop_size:.1f}m')
    title_parts.append(f'Points: {len(df):,}')

    fig.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold', y=0.98)
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
