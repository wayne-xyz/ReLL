from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

try:
    from .gicp_core import (
        GICPParams,
        build_cloud,
        prepare_downsampled_clouds as base_prepare_downsampled_clouds,
        run_gicp as base_run_gicp,
        compose_transform as base_compose_transform,
        apply_transform as base_apply_transform,
    )
except ImportError:
    from gicp_core import (
        GICPParams,
        build_cloud,
        prepare_downsampled_clouds as base_prepare_downsampled_clouds,
        run_gicp as base_run_gicp,
        compose_transform as base_compose_transform,
        apply_transform as base_apply_transform,
    )


@dataclass
class Op2Config:
    """Op2 configuration: DSM-style filtering + no downsampling before GICP."""
    voxel_size: float = 0.5
    normal_k: int = 20
    max_corr_dist: float = 0.8
    max_iter: int = 60

    # Vertical grid cell size for extracting highest LiDAR points (DSM-style)
    vertical_cell_size_m: float = 0.05

    # Disable downsampling before GICP
    disable_downsampling: bool = True


def _summarize(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }



def _extract_highest_per_vertical_cell(
    points: np.ndarray,
    cell_size: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract the highest point in each vertical cell (DSM-style filtering).

    Args:
        points: Nx3 array of points (E, N, Z)
        cell_size: Grid cell size in meters (horizontal)

    Returns:
        Tuple of (filtered_points, kept_indices, diagnostics)
    """
    if points.size == 0:
        return points, np.array([], dtype=np.int32), {"original_count": 0, "filtered_count": 0, "cells_created": 0}

    # Create 2D grid based on XY coordinates. Cast to int64 to avoid overflow when the crop spans many cells.
    xy = points[:, :2]
    min_xy = np.min(xy, axis=0)
    grid_idx = np.floor((xy - min_xy) / cell_size).astype(np.int64)

    # Sort by grid coordinates (primary) and negative Z (secondary) so the first row per cell has the highest Z.
    sort_order = np.lexsort((-points[:, 2], grid_idx[:, 1], grid_idx[:, 0]))
    sorted_cells = grid_idx[sort_order]

    keep_mask = np.zeros(sorted_cells.shape[0], dtype=bool)
    if keep_mask.size > 0:
        keep_mask[0] = True
        if keep_mask.size > 1:
            keep_mask[1:] = np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)

    highest_indices = sort_order[keep_mask].astype(np.int32, copy=False)
    filtered_points = points[highest_indices]

    diagnostics = {
        "original_count": int(points.shape[0]),
        "filtered_count": int(filtered_points.shape[0]),
        "cells_created": int(highest_indices.shape[0]),
        "cell_size_m": float(cell_size),
        "reduction_ratio": float(filtered_points.shape[0] / points.shape[0]) if points.shape[0] > 0 else 0.0,
    }

    return filtered_points, highest_indices, diagnostics

def _nearest_dsm_z(dsm_points: np.ndarray, query_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(dsm_points[:, :2])
    nn_dist, idx = tree.query(query_xy, k=1)
    nn_z = dsm_points[idx, 2]
    return nn_z, nn_dist




def _extract_yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _build_rz(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz


def _compose_from_xy_yaw_z(tx: float, ty: float, yaw: float, tz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _build_rz(yaw)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T



def register_with_dsm_style_filtering(
    shifted_points: np.ndarray,
    dsm_points: np.ndarray,
    shared_origin: Optional[np.ndarray] = None,
    config: Optional[Op2Config] = None,
) -> Dict[str, Any]:
    """Op2 registration: extract highest LiDAR points per cell (DSM-style), then GICP with full DSM.

    Args:
        shifted_points: Vertically shifted LiDAR points (Nx3)
        dsm_points: DSM points (Mx3)
        shared_origin: Optional shared origin for centering
        config: Op2 configuration

    Returns:
        Dictionary with transform, diagnostics, and metadata
    """
    cfg = config or Op2Config()

    # Step 1: Extract highest LiDAR point per vertical cell (DSM-style filtering)
    lidar_filtered, kept_indices, filter_diag = _extract_highest_per_vertical_cell(
        shifted_points,
        cfg.vertical_cell_size_m,
    )

    source_centroid = np.mean(lidar_filtered, axis=0) if lidar_filtered.size else np.zeros(3, dtype=np.float64)
    target_centroid = np.mean(dsm_points, axis=0) if dsm_points.size else np.zeros(3, dtype=np.float64)
    centroid_delta = target_centroid - source_centroid

    source_min = np.min(lidar_filtered, axis=0) if lidar_filtered.size else np.zeros(3, dtype=np.float64)
    source_max = np.max(lidar_filtered, axis=0) if lidar_filtered.size else np.zeros(3, dtype=np.float64)
    target_min = np.min(dsm_points, axis=0) if dsm_points.size else np.zeros(3, dtype=np.float64)
    target_max = np.max(dsm_points, axis=0) if dsm_points.size else np.zeros(3, dtype=np.float64)

    # Step 2: Use full DSM without gating
    target_for_gicp = dsm_points

    # Step 3: Prepare clouds (with or without downsampling)
    params = GICPParams(
        voxel_size=cfg.voxel_size if not cfg.disable_downsampling else 0.0,
        normal_k=cfg.normal_k,
        max_corr_dist=cfg.max_corr_dist,
        max_iter=cfg.max_iter,
        enforce_z_up=True,
    )

    clouds = base_prepare_downsampled_clouds(
        lidar_filtered,
        target_for_gicp,
        shared_origin=shared_origin,
        params=params,
    )

    # Step 4: Run GICP
    gicp_result = base_run_gicp(clouds["source_centered"], clouds["target_centered"], params=params)
    gicp_T = base_compose_transform(gicp_result, clouds["src_centroid"], clouds["tgt_centroid"])

    horiz_delta = float(np.hypot(centroid_delta[0], centroid_delta[1]))
    centroid_diag = {
        "source": {
            "utm_e": float(source_centroid[0]),
            "utm_n": float(source_centroid[1]),
            "z": float(source_centroid[2]),
        },
        "target": {
            "utm_e": float(target_centroid[0]),
            "utm_n": float(target_centroid[1]),
            "z": float(target_centroid[2]),
        },
        "delta_m": {
            "utm_e": float(centroid_delta[0]),
            "utm_n": float(centroid_delta[1]),
            "z": float(centroid_delta[2]),
            "horizontal_norm": horiz_delta,
        },
    }

    source_extent_diag = {
        "min": {
            "utm_e": float(source_min[0]),
            "utm_n": float(source_min[1]),
            "z": float(source_min[2]),
        },
        "max": {
            "utm_e": float(source_max[0]),
            "utm_n": float(source_max[1]),
            "z": float(source_max[2]),
        },
    }

    target_extent_diag = {
        "min": {
            "utm_e": float(target_min[0]),
            "utm_n": float(target_min[1]),
            "z": float(target_min[2]),
        },
        "max": {
            "utm_e": float(target_max[0]),
            "utm_n": float(target_max[1]),
            "z": float(target_max[2]),
        },
    }

    # Step 5: Post-correction: extract yaw + XY from GICP, estimate Z from median
    R = gicp_T[:3, :3]
    t = gicp_T[:3, 3]
    yaw = _extract_yaw(R)
    tx, ty = float(t[0]), float(t[1])

    # Estimate Z using median dz after applying XY+yaw only
    T_xy_yaw = _compose_from_xy_yaw_z(tx, ty, yaw, 0.0)
    src_xyyaw = base_apply_transform(shifted_points, T_xy_yaw)
    nn_z_after_xy, _ = _nearest_dsm_z(dsm_points, src_xyyaw[:, :2])
    dz_all = nn_z_after_xy - src_xyyaw[:, 2]
    z_median = float(np.median(dz_all)) if dz_all.size > 0 else 0.0

    T_final = _compose_from_xy_yaw_z(tx, ty, yaw, z_median)

    # Step 6: Compute final diagnostics
    transformed_points = base_apply_transform(shifted_points, T_final)
    nn_z_after, nn_xy = _nearest_dsm_z(dsm_points, transformed_points[:, :2])
    dz_after = nn_z_after - transformed_points[:, 2]

    diagnostics = {
        "lidar_dsm_style_filter": filter_diag,
        "pre_icp_centroids": centroid_diag,
        "source_extent_m": source_extent_diag,
        "target_extent_m": target_extent_diag,
        "gicp": {
            "fitness": float(getattr(gicp_result, "fitness", 0.0)),
            "rmse": float(getattr(gicp_result, "inlier_rmse", 0.0)),
            "downsampling_disabled": cfg.disable_downsampling,
        },
        "post_correction": {
            "yaw_rad": yaw,
            "xy_translation_m": float(np.hypot(tx, ty)),
            "z_translation_m": z_median,
        },
        "dz_summary_after_xyyaw": _summarize(dz_all),
        "dz_summary_after_final": _summarize(dz_after),
        "nn_xy_dist_summary_after_final": _summarize(nn_xy),
        "selected_source_points": int(lidar_filtered.shape[0]),
    }

    # Readable counts
    try:
        diagnostics["readable_counts"] = {
            "source_original_count": int(shifted_points.shape[0]),
            "source_filtered_count": int(lidar_filtered.shape[0]),
            "source_reduction_ratio": filter_diag.get("reduction_ratio", 0.0),
            "target_original_count": int(dsm_points.shape[0]),
            "target_used_count": int(target_for_gicp.shape[0]),
        }
    except Exception:
        pass

    return {
        "transform": T_final,
        "diagnostics": diagnostics,
        "selected_source_points": int(lidar_filtered.shape[0]),
    }



# Convenience exports
def compose_transform(
    result: o3d.pipelines.registration.RegistrationResult,
    src_centroid: np.ndarray,
    tgt_centroid: np.ndarray,
) -> np.ndarray:
    return base_compose_transform(result, src_centroid, tgt_centroid)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return base_apply_transform(points, transform)
