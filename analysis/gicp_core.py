from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass
class GICPParams:
    voxel_size: float = 0.5
    normal_k: int = 20
    max_corr_dist: float = 1.0
    max_iter: int = 30
    enforce_z_up: bool = True


def build_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pc


def _orient_normals_consistently(pc: o3d.geometry.PointCloud, normal_k: int, enforce_z_up: bool = True) -> None:
    knn_param = o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
    pc.estimate_normals(knn_param)
    # In some Open3D builds this may not be available; guard it.
    try:
        pc.orient_normals_consistent_tangent_plane(normal_k)
    except Exception:
        pass
    # Orient towards a common camera location (origin of local coords)
    pc.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    if enforce_z_up and len(pc.normals) > 0:
        n = np.asarray(pc.normals)
        if float(np.nanmean(n[:, 2])) < 0.0:
            pc.normals = o3d.utility.Vector3dVector(-n)


def prepare_downsampled_clouds(
    source_points: np.ndarray,
    target_points: np.ndarray,
    shared_origin: Optional[np.ndarray] = None,
    params: Optional[GICPParams] = None,
):
    params = params or GICPParams()
    origin = np.zeros(3, dtype=np.float64) if shared_origin is None else np.asarray(shared_origin, dtype=np.float64)
    if origin.shape[0] == 2:
        origin = np.array([origin[0], origin[1], 0.0], dtype=np.float64)

    src_local = source_points.astype(np.float64) - origin
    tgt_local = target_points.astype(np.float64) - origin

    source_pc = build_cloud(src_local)
    target_pc = build_cloud(tgt_local)

    # Skip downsampling if voxel_size is 0 or negative (disabled)
    if params.voxel_size > 0:
        source_down = source_pc.voxel_down_sample(params.voxel_size)
        target_down = target_pc.voxel_down_sample(params.voxel_size)
    else:
        source_down = source_pc
        target_down = target_pc

    if len(source_down.points) == 0 or len(target_down.points) == 0:
        raise RuntimeError("Downsampled cloud empty. Adjust voxel size or ground mask.")

    _orient_normals_consistently(source_down, params.normal_k, params.enforce_z_up)
    _orient_normals_consistently(target_down, params.normal_k, params.enforce_z_up)

    src_centroid_local = np.mean(np.asarray(source_down.points), axis=0)
    tgt_centroid_local = np.mean(np.asarray(target_down.points), axis=0)

    # Center clouds around their own centroids
    source_centered = source_down.translate(-src_centroid_local)
    target_centered = target_down.translate(-tgt_centroid_local)

    src_centroid = src_centroid_local + origin
    tgt_centroid = tgt_centroid_local + origin

    return {
        "source_pc": source_pc,
        "target_pc": target_pc,
        "source_centered": source_centered,
        "target_centered": target_centered,
        "src_centroid": src_centroid,
        "tgt_centroid": tgt_centroid,
    }


def run_gicp(
    source_centered: o3d.geometry.PointCloud,
    target_centered: o3d.geometry.PointCloud,
    params: Optional[GICPParams] = None,
) -> o3d.pipelines.registration.RegistrationResult:
    params = params or GICPParams()
    estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=params.max_iter)
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
    T_src = np.eye(4)
    T_src[:3, 3] = -src_centroid
    T_tgt = np.eye(4)
    T_tgt[:3, 3] = tgt_centroid
    return T_tgt @ result.transformation @ T_src


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return (transform @ homogeneous.T).T[:, :3]
