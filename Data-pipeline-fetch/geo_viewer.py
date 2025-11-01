#!/usr/bin/env python3
"""
Geographic Viewer for ReLL Dataset Samples

Visualizes the spatial distribution of all samples in a dataset folder
by plotting their center positions in UTM coordinates.

Usage:
    python geo_viewer.py <dataset_folder>

Example:
    python geo_viewer.py G:\GithubProject\ReLL\Rell-sample-raster-test-0p2
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Suppress PROJ library version warnings (harmless conflicts with PostgreSQL's PROJ)
os.environ['PROJ_DEBUG'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='pyproj')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class SampleLocation:
    """Container for sample geographic information."""

    def __init__(self, name: str, center_utm: Tuple[float, float],
                 bounds: Tuple[float, float, float, float], crs: str = None):
        self.name = name
        self.center_utm = center_utm  # (easting, northing)
        self.bounds = bounds  # (min_x, min_y, max_x, max_y)
        self.crs = crs

    def __repr__(self):
        return f"Sample({self.name}, UTM: {self.center_utm})"


def load_sample_metadata(sample_dir: Path) -> Dict:
    """
    Load metadata from a sample directory.

    Args:
        sample_dir: Path to sample directory

    Returns:
        Dictionary with transform, crs, and metadata
    """
    metadata = {}

    # Load transform (affine transformation)
    transform_file = sample_dir / "transform.pkl"
    if transform_file.exists():
        with open(transform_file, 'rb') as f:
            metadata['transform'] = pickle.load(f)

    # Load CRS (coordinate reference system)
    crs_file = sample_dir / "crs.pkl"
    if crs_file.exists():
        with open(crs_file, 'rb') as f:
            metadata['crs'] = pickle.load(f)

    # Load general metadata
    metadata_file = sample_dir / "metadata.pkl"
    if metadata_file.exists():
        with open(metadata_file, 'rb') as f:
            metadata['metadata'] = pickle.load(f)

    # Load resolution
    resolution_file = sample_dir / "resolution.pkl"
    if resolution_file.exists():
        with open(resolution_file, 'rb') as f:
            metadata['resolution'] = pickle.load(f)

    return metadata


def extract_sample_location(sample_dir: Path) -> SampleLocation | None:
    """
    Extract geographic location from sample directory.

    Args:
        sample_dir: Path to sample directory

    Returns:
        SampleLocation object or None if extraction fails
    """
    try:
        metadata = load_sample_metadata(sample_dir)

        if 'transform' not in metadata:
            print(f"[Warning] No transform found in {sample_dir.name}")
            return None

        transform = metadata['transform']

        # Get image dimensions from any .npy file
        npy_file = next(sample_dir.glob("*.npy"), None)
        if npy_file is None:
            print(f"[Warning] No .npy files found in {sample_dir.name}")
            return None

        # Load to get shape
        data = np.load(npy_file, mmap_mode='r')
        if data.ndim == 3:
            height, width = data.shape[1], data.shape[2]
        elif data.ndim == 2:
            height, width = data.shape
        else:
            print(f"[Warning] Unexpected data shape in {sample_dir.name}: {data.shape}")
            return None

        # Extract affine transform parameters
        # Transform format: (a, b, c, d, e, f)
        # or Affine object with attributes a, b, c, d, e, f
        if hasattr(transform, 'a'):
            # Affine object (from rasterio)
            a, b, c = transform.a, transform.b, transform.c
            d, e, f = transform.d, transform.e, transform.f
        else:
            # Tuple format
            a, b, c, d, e, f = transform

        # Affine transformation:
        # X_geo = a * col + b * row + c
        # Y_geo = d * col + e * row + f

        # Calculate image center in pixel coordinates
        center_col = width / 2.0
        center_row = height / 2.0

        # Transform center to UTM coordinates
        center_x = a * center_col + b * center_row + c
        center_y = d * center_col + e * center_row + f

        # Calculate bounds (corners of the image)
        min_x = a * 0 + b * 0 + c
        max_x = a * width + b * height + c
        min_y = d * 0 + e * 0 + f
        max_y = d * width + e * height + f

        # Ensure min/max are correct (in case of negative pixel sizes)
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y

        crs_str = str(metadata.get('crs', 'Unknown'))

        return SampleLocation(
            name=sample_dir.name,
            center_utm=(center_x, center_y),
            bounds=(min_x, min_y, max_x, max_y),
            crs=crs_str
        )

    except Exception as e:
        print(f"[Error] Failed to extract location from {sample_dir.name}: {e}")
        return None


def scan_dataset(dataset_folder: Path) -> List[SampleLocation]:
    """
    Scan all samples in a dataset folder.

    Args:
        dataset_folder: Path to dataset root folder

    Returns:
        List of SampleLocation objects
    """
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

    sample_dirs = sorted([d for d in dataset_folder.iterdir() if d.is_dir()])

    if not sample_dirs:
        raise ValueError(f"No sample directories found in {dataset_folder}")

    print(f"[Scan] Found {len(sample_dirs)} sample directories")

    locations = []
    for sample_dir in sample_dirs:
        location = extract_sample_location(sample_dir)
        if location is not None:
            locations.append(location)

    print(f"[Scan] Successfully extracted {len(locations)} sample locations")

    return locations


def plot_sample_locations(locations: List[SampleLocation], save_path: Path | None = None):
    """
    Plot all sample locations on a map.

    Args:
        locations: List of SampleLocation objects
        save_path: Optional path to save the figure
    """
    if not locations:
        print("[Error] No locations to plot")
        return

    # Extract coordinates
    eastings = [loc.center_utm[0] for loc in locations]
    northings = [loc.center_utm[1] for loc in locations]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot sample centers
    scatter = ax.scatter(
        eastings,
        northings,
        c='red',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidths=1.5,
        marker='o',
        label='Sample Centers',
        zorder=10
    )

    # Optionally plot bounding boxes
    for loc in locations:
        min_x, min_y, max_x, max_y = loc.bounds
        width = max_x - min_x
        height = max_y - min_y
        rect = Rectangle(
            (min_x, min_y),
            width,
            height,
            linewidth=0.5,
            edgecolor='blue',
            facecolor='none',
            alpha=0.3,
            zorder=5
        )
        ax.add_patch(rect)

    # Add sample name labels (for small datasets)
    if len(locations) <= 50:
        for loc in locations:
            ax.annotate(
                loc.name,
                xy=loc.center_utm,
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=6,
                alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
            )

    # Formatting
    ax.set_xlabel('Easting (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Northing (m)', fontsize=12, fontweight='bold')

    # Get CRS from first sample
    crs_info = locations[0].crs if locations[0].crs else "Unknown CRS"

    ax.set_title(
        f'Sample Distribution in Geographic Coordinates\n'
        f'Dataset: {len(locations)} samples | CRS: {crs_info}',
        fontsize=14,
        fontweight='bold',
        pad=15
    )

    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')

    # Add statistics
    stats_text = (
        f'Easting range: {min(eastings):.1f} - {max(eastings):.1f} m\n'
        f'Northing range: {min(northings):.1f} - {max(northings):.1f} m\n'
        f'Coverage: {max(eastings)-min(eastings):.1f} × {max(northings)-min(northings):.1f} m'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[Save] Figure saved to: {save_path}")

    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize geographic distribution of ReLL dataset samples.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python geo_viewer.py G:\\GithubProject\\ReLL\\Rell-sample-raster-test-0p2
  python geo_viewer.py /path/to/dataset --save output.png
        """
    )

    parser.add_argument(
        'dataset_folder',
        type=Path,
        help='Path to dataset folder containing sample subdirectories'
    )

    parser.add_argument(
        '--save',
        type=Path,
        default=None,
        help='Optional path to save the figure (e.g., sample_map.png)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ReLL Dataset Geographic Viewer")
    print("=" * 70)

    try:
        # Scan dataset
        locations = scan_dataset(args.dataset_folder)

        if not locations:
            print("[Error] No valid sample locations found")
            sys.exit(1)

        # Print summary
        print(f"\n[Summary]")
        print(f"  Total samples: {len(locations)}")
        eastings = [loc.center_utm[0] for loc in locations]
        northings = [loc.center_utm[1] for loc in locations]
        print(f"  Easting range: {min(eastings):.2f} - {max(eastings):.2f} m")
        print(f"  Northing range: {min(northings):.2f} - {max(northings):.2f} m")
        print(f"  Spatial extent: {max(eastings)-min(eastings):.2f} × {max(northings)-min(northings):.2f} m")

        # Plot
        print(f"\n[Plot] Generating visualization...")
        plot_sample_locations(locations, save_path=args.save)

    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
