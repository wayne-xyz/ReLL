"""GICP alignment of LiDAR points to DSM reference.

This module provides functionality to perform Generalized ICP alignment
between LiDAR points (source) and DSM points (reference/target).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "open3d is required for GICP alignment. Install with: pip install open3d"
    ) from exc


class GICPResult(NamedTuple):
    """Result of GICP alignment operation."""
    aligned_lidar_path: Path
    metrics_json_path: Path
    fitness: float
    inlier_rmse: float
    transform_matrix: np.ndarray
    translation_m: Tuple[float, float, float]
    yaw_deg: float
    nn_rmse: float
    nn_mean_abs_distance: float


class GICPParams:
    """Parameters for GICP alignment."""

    def __init__(
        self,
        voxel_size: float = 0.3,
        normal_k: int = 20,
        max_corr_dist: float = 0.8,
        max_iter: int = 60,
        enforce_z_up: bool = True,
    ):
        self.voxel_size = voxel_size
        self.normal_k = normal_k
        self.max_corr_dist = max_corr_dist
        self.max_iter = max_iter
        self.enforce_z_up = enforce_z_up


def _build_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Build Open3D point cloud from numpy array."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pc


def _orient_normals(
    pc: o3d.geometry.PointCloud,
    normal_k: int,
    enforce_z_up: bool
) -> None:
    """Estimate and orient normals for a point cloud."""
    search = o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
    pc.estimate_normals(search)
    try:
        pc.orient_normals_consistent_tangent_plane(normal_k)
    except Exception:
        pass
    pc.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

    if enforce_z_up and len(pc.normals) > 0:
        normals = np.asarray(pc.normals)
        if float(np.nanmean(normals[:, 2])) < 0.0:
            pc.normals = o3d.utility.Vector3dVector(-normals)


def prepare_downsampled_clouds(
    source_points: np.ndarray,
    target_points: np.ndarray,
    params: GICPParams,
    anchor: Optional[np.ndarray] = None,
) -> Dict:
    """Prepare downsampled and centered clouds for GICP.

    Args:
        source_points: Source point cloud (Nx3)
        target_points: Target point cloud (Mx3)
        params: GICP parameters
        anchor: Optional anchor point for centering (if None, uses mean of downsampled points)

    Returns:
        Dictionary with downsampled clouds and centroids
    """
    # If anchor provided, shift points to anchor-centered frame first
    if anchor is not None:
        anchor = np.asarray(anchor, dtype=np.float64)
        source_points_local = source_points.astype(np.float64) - anchor
        target_points_local = target_points.astype(np.float64) - anchor
    else:
        source_points_local = source_points.astype(np.float64)
        target_points_local = target_points.astype(np.float64)

    source_pc = _build_cloud(source_points_local)
    target_pc = _build_cloud(target_points_local)

    # Downsample
    if params.voxel_size > 0:
        source_down = source_pc.voxel_down_sample(params.voxel_size)
        target_down = target_pc.voxel_down_sample(params.voxel_size)
    else:
        source_down = source_pc
        target_down = target_pc

    if len(source_down.points) == 0 or len(target_down.points) == 0:
        raise RuntimeError(
            "Downsampled cloud is empty. Adjust voxel_size or check point counts."
        )

    # Estimate normals
    _orient_normals(source_down, params.normal_k, params.enforce_z_up)
    _orient_normals(target_down, params.normal_k, params.enforce_z_up)

    # Compute centroids in local frame
    src_centroid_local = np.mean(np.asarray(source_down.points), axis=0)
    tgt_centroid_local = np.mean(np.asarray(target_down.points), axis=0)

    # Center the clouds (in local frame)
    source_centered = source_down.translate(-src_centroid_local)
    target_centered = target_down.translate(-tgt_centroid_local)

    # Convert centroids back to global frame
    if anchor is not None:
        src_centroid = src_centroid_local + anchor
        tgt_centroid = tgt_centroid_local + anchor
    else:
        src_centroid = src_centroid_local
        tgt_centroid = tgt_centroid_local

    return {
        "source_pc": source_pc,
        "target_pc": target_pc,
        "source_centered": source_centered,
        "target_centered": target_centered,
        "src_centroid": src_centroid,
        "tgt_centroid": tgt_centroid,
        "anchor": anchor if anchor is not None else np.zeros(3, dtype=np.float64),
    }


def run_gicp(
    source_centered: o3d.geometry.PointCloud,
    target_centered: o3d.geometry.PointCloud,
    params: GICPParams,
) -> o3d.pipelines.registration.RegistrationResult:
    """Run GICP registration.

    Args:
        source_centered: Centered source cloud
        target_centered: Centered target cloud
        params: GICP parameters

    Returns:
        Open3D registration result
    """
    estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=params.max_iter
    )

    return o3d.pipelines.registration.registration_generalized_icp(
        source_centered,
        target_centered,
        params.max_corr_dist,
        np.eye(4),
        estimation,
        criteria,
    )


def compose_transform(
    result: o3d.pipelines.registration.RegistrationResult,
    src_centroid: np.ndarray,
    tgt_centroid: np.ndarray,
) -> np.ndarray:
    """Compose final transform from centered registration result.

    Args:
        result: GICP registration result
        src_centroid: Source centroid
        tgt_centroid: Target centroid

    Returns:
        4x4 transformation matrix in global frame
    """
    # Transform from global to source-centered frame
    T_src = np.eye(4)
    T_src[:3, 3] = -np.asarray(src_centroid)

    # Transform from target-centered frame to global
    T_tgt = np.eye(4)
    T_tgt[:3, 3] = np.asarray(tgt_centroid)

    # Compose: global -> src-centered -> tgt-centered -> global
    return T_tgt @ result.transformation @ T_src


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform to Nx3 points.

    Args:
        points: Nx3 point cloud
        transform: 4x4 transformation matrix

    Returns:
        Transformed Nx3 points
    """
    homo = np.hstack([points, np.ones((points.shape[0], 1))])
    return (transform @ homo.T).T[:, :3]


def extract_yaw(R: np.ndarray) -> float:
    """Extract yaw angle from rotation matrix."""
    return math.atan2(R[1, 0], R[0, 0])


def summarize_nn_error(
    aligned: np.ndarray,
    target: np.ndarray
) -> Tuple[float, float]:
    """Compute nearest-neighbor RMSE and mean absolute distance.

    Args:
        aligned: Aligned source points
        target: Target points

    Returns:
        Tuple of (RMSE, mean absolute distance)
    """
    if aligned.size == 0 or target.size == 0:
        return 0.0, 0.0

    pc = _build_cloud(target)
    tree = o3d.geometry.KDTreeFlann(pc)

    sq_errors = []
    abs_errors = []

    for pt in aligned:
        pt_vec = pt.astype(np.float64)
        result = tree.search_knn_vector_3d(pt_vec, 1)
        if result[0] == 0:
            continue
        nn = target[result[1][0]]
        diff = nn - pt_vec
        sq_errors.append(float(np.dot(diff, diff)))
        abs_errors.append(float(np.linalg.norm(diff)))

    if not sq_errors:
        return 0.0, 0.0

    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
    mean_abs = sum(abs_errors) / len(abs_errors)
    return rmse, mean_abs


def load_anchor_from_metadata(meta_path: Path) -> np.ndarray:
    """Load anchor point (center position) from metadata parquet.

    Args:
        meta_path: Path to segment metadata parquet file

    Returns:
        3D anchor point [utm_e, utm_n, z] from metadata center position

    Raises:
        ValueError: If metadata is missing required columns
    """
    import pandas as pd

    df = pd.read_parquet(meta_path)

    if len(df) == 0:
        raise ValueError("Metadata file is empty")

    row = df.iloc[0]

    # Extract center UTM coordinates
    if "center_utm_easting_m" not in df.columns or "center_utm_northing_m" not in df.columns:
        raise ValueError(
            "Metadata missing required columns: center_utm_easting_m, center_utm_northing_m"
        )

    utm_e = float(row["center_utm_easting_m"])
    utm_n = float(row["center_utm_northing_m"])

    # For Z, prefer city frame Z if available (matches sensor position Z)
    if "center_city_tz_m" in df.columns:
        z = float(row["center_city_tz_m"])
    else:
        # Fallback: will be updated from LiDAR mean later
        z = 0.0

    return np.array([utm_e, utm_n, z], dtype=np.float64)


def compute_distance_percentiles(
    aligned: np.ndarray,
    target: np.ndarray,
    percentiles: list = [50, 75, 90, 95, 99]
) -> Dict[str, float]:
    """Compute distance percentiles for alignment quality assessment.

    Args:
        aligned: Aligned source points
        target: Target points
        percentiles: List of percentiles to compute

    Returns:
        Dictionary of percentile values
    """
    if aligned.size == 0 or target.size == 0:
        return {f"p{p}": 0.0 for p in percentiles}

    pc = _build_cloud(target)
    tree = o3d.geometry.KDTreeFlann(pc)

    distances = []
    for pt in aligned:
        pt_vec = pt.astype(np.float64)
        result = tree.search_knn_vector_3d(pt_vec, 1)
        if result[0] == 0:
            continue
        nn = target[result[1][0]]
        dist = float(np.linalg.norm(nn - pt_vec))
        distances.append(dist)

    if not distances:
        return {f"p{p}": 0.0 for p in percentiles}

    distances = np.array(distances)
    return {f"p{p}": float(np.percentile(distances, p)) for p in percentiles}


def align_lidar_to_dsm(
    lidar_parquet_path: Path,
    dsm_parquet_path: Path,
    meta_path: Path,
    output_lidar_path: Path,
    output_metrics_path: Path,
    params: Optional[GICPParams] = None,
    compression: str = "zstd",
) -> GICPResult:
    """Align LiDAR points to DSM reference using GICP.

    Args:
        lidar_parquet_path: Path to source LiDAR UTM parquet
        dsm_parquet_path: Path to target DSM parquet (extracted)
        meta_path: Path to metadata parquet (for anchor point)
        output_lidar_path: Path to save aligned LiDAR parquet
        output_metrics_path: Path to save metrics JSON
        params: GICP parameters (default: GICPParams())
        compression: Parquet compression codec

    Returns:
        GICPResult with alignment statistics and paths

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If GICP alignment fails
    """
    if params is None:
        params = GICPParams()

    # Load LiDAR points
    lidar_table = pq.read_table(lidar_parquet_path)
    lidar_points = np.column_stack([
        lidar_table["utm_e"].to_numpy(),
        lidar_table["utm_n"].to_numpy(),
        lidar_table["elevation"].to_numpy()
    ]).astype(np.float64)

    # Load DSM points
    dsm_table = pq.read_table(dsm_parquet_path)
    dsm_points = np.column_stack([
        dsm_table["utm_e"].to_numpy(),
        dsm_table["utm_n"].to_numpy(),
        dsm_table["elevation"].to_numpy()
    ]).astype(np.float64)

    # Load anchor point from metadata
    anchor = load_anchor_from_metadata(meta_path)

    # If Z was not in metadata, use mean of LiDAR Z
    if anchor[2] == 0.0:
        anchor[2] = lidar_points[:, 2].mean()

    # Prepare clouds for GICP with metadata anchor
    clouds = prepare_downsampled_clouds(lidar_points, dsm_points, params, anchor=anchor)

    # Run GICP (source=LiDAR, target=DSM)
    result = run_gicp(
        clouds["source_centered"],
        clouds["target_centered"],
        params
    )

    # Compose global transform
    transform_global = compose_transform(
        result,
        clouds["src_centroid"],
        clouds["tgt_centroid"]
    )

    # Apply transform to LiDAR
    aligned_lidar = apply_transform(lidar_points, transform_global)

    # Extract transform components
    R = transform_global[:3, :3]
    t = transform_global[:3, 3]
    yaw = extract_yaw(R)

    # Compute alignment quality metrics
    nn_rmse, nn_mean_abs = summarize_nn_error(aligned_lidar, dsm_points)
    distance_percentiles = compute_distance_percentiles(aligned_lidar, dsm_points)

    # Save aligned LiDAR
    aligned_table = pa.Table.from_pydict({
        "utm_e": pa.array(aligned_lidar[:, 0], type=pa.float64()),
        "utm_n": pa.array(aligned_lidar[:, 1], type=pa.float64()),
        "elevation": pa.array(aligned_lidar[:, 2], type=pa.float32()),
        "intensity": lidar_table["intensity"],
        "laser_number": lidar_table["laser_number"],
        "offset_ns": lidar_table["offset_ns"],
        "source_index": lidar_table["source_index"],
        "source_timestamp_ns": lidar_table["source_timestamp_ns"],
    })
    pq.write_table(aligned_table, output_lidar_path, compression=compression, use_dictionary=False)

    # Save metrics
    metrics = {
        "inputs": {
            "lidar_file": str(lidar_parquet_path),
            "dsm_file": str(dsm_parquet_path),
            "meta_file": str(meta_path),
            "lidar_points": int(len(lidar_points)),
            "dsm_points": int(len(dsm_points)),
        },
        "local_frame_anchor_utm": {
            "utm_e": float(anchor[0]),
            "utm_n": float(anchor[1]),
            "z": float(anchor[2]),
            "note": "Anchor point from metadata (center sweep's vehicle position) - used for local frame centering during GICP",
        },
        "gicp_parameters": {
            "voxel_size": params.voxel_size,
            "normal_k": params.normal_k,
            "max_corr_dist": params.max_corr_dist,
            "max_iter": params.max_iter,
            "enforce_z_up": params.enforce_z_up,
        },
        "gicp_result": {
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
        },
        "transform_global_frame": {
            "matrix_4x4": transform_global.tolist(),
            "translation_m": [float(t[0]), float(t[1]), float(t[2])],
            "yaw_deg": float(math.degrees(yaw)),
            "note": "Transform in global UTM frame (already applied to aligned LiDAR)",
        },
        "alignment_quality": {
            "nn_rmse_m": nn_rmse,
            "nn_mean_abs_distance_m": nn_mean_abs,
            "distance_percentiles_m": distance_percentiles,
        },
        "outputs": {
            "aligned_lidar": str(output_lidar_path),
            "metrics_json": str(output_metrics_path),
        },
    }

    with output_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return GICPResult(
        aligned_lidar_path=output_lidar_path,
        metrics_json_path=output_metrics_path,
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        transform_matrix=transform_global,
        translation_m=(float(t[0]), float(t[1]), float(t[2])),
        yaw_deg=float(math.degrees(yaw)),
        nn_rmse=nn_rmse,
        nn_mean_abs_distance=nn_mean_abs,
    )
