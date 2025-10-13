"""Extract DSM points within proximity of LiDAR points.

This module provides functionality to extract DSM points that are within
a specified distance threshold from any LiDAR point (XY plane only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, NamedTuple, Optional

import numpy as np
import laspy
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


class DSMExtractionResult(NamedTuple):
    """Result of DSM extraction operation."""
    extracted_dsm_path: Path
    original_dsm_count: int
    extracted_dsm_count: int
    reduction_ratio: float
    distance_stats: Dict[str, float]


def extract_dsm_near_lidar(
    lidar_parquet_path: Path,
    dsm_laz_path: Path,
    output_path: Path,
    max_distance: float = 0.5,
    compression: str = "zstd",
) -> DSMExtractionResult:
    """Extract DSM points within max_distance of any LiDAR point (XY only).

    Args:
        lidar_parquet_path: Path to LiDAR UTM parquet file
        dsm_laz_path: Path to DSM LAZ file
        output_path: Path to save extracted DSM parquet
        max_distance: Maximum distance threshold in meters (default: 0.5m)
        compression: Parquet compression codec

    Returns:
        DSMExtractionResult with extraction statistics

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If no points remain after extraction
    """
    # Load LiDAR points (XY only)
    lidar_table = pq.read_table(lidar_parquet_path, columns=["utm_e", "utm_n"])
    lidar_xy = np.column_stack([
        lidar_table["utm_e"].to_numpy(),
        lidar_table["utm_n"].to_numpy()
    ])

    # Load DSM points
    las = laspy.read(dsm_laz_path)
    dsm_points = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
    original_count = len(dsm_points)

    # Build KDTree of LiDAR points (XY only)
    lidar_tree = cKDTree(lidar_xy)

    # For each DSM point, find distance to nearest LiDAR point
    distances, _ = lidar_tree.query(dsm_points[:, :2], k=1)

    # Filter DSM points
    mask = distances <= max_distance
    extracted_dsm = dsm_points[mask]

    if len(extracted_dsm) == 0:
        raise ValueError(
            f"No DSM points within {max_distance}m of LiDAR points. "
            f"Consider increasing max_distance."
        )

    # Compute statistics
    distance_stats = {
        "min": float(distances[mask].min()),
        "max": float(distances[mask].max()),
        "mean": float(distances[mask].mean()),
        "p50": float(np.percentile(distances[mask], 50)),
        "p95": float(np.percentile(distances[mask], 95)),
    }

    # Save extracted DSM as parquet
    import pyarrow as pa
    extracted_table = pa.Table.from_pydict({
        "utm_e": pa.array(extracted_dsm[:, 0], type=pa.float64()),
        "utm_n": pa.array(extracted_dsm[:, 1], type=pa.float64()),
        "elevation": pa.array(extracted_dsm[:, 2], type=pa.float32()),
    })
    pq.write_table(extracted_table, output_path, compression=compression, use_dictionary=False)

    return DSMExtractionResult(
        extracted_dsm_path=output_path,
        original_dsm_count=original_count,
        extracted_dsm_count=len(extracted_dsm),
        reduction_ratio=float(1 - len(extracted_dsm) / original_count),
        distance_stats=distance_stats,
    )
