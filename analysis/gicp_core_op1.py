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
class Op1Config:
    voxel_size: float = 0.5
    normal_k: int = 20
    max_corr_dist: float = 0.8
    max_iter: int = 60

    # ROI / correspondence gating
    vertical_gate_m: float = 0.5
    min_points_after_gate: int = 1000
    fallback_fraction: float = 0.25  # if gating too strict, keep this frac by |dz| ranking
    target_xy_margin_m: float = 10.0  # expand source ROI bbox to crop target
    target_min_points_after_gate: int = 5000

    # Optional Z downweighting during ICP (reduce vertical attraction)
    z_downweight_factor: float = 2.0  # use >1.0 to downweight Z (e.g., 5.0). 1.0 = disabled


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


def _nearest_dsm_z(dsm_points: np.ndarray, query_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(dsm_points[:, :2])
    nn_dist, idx = tree.query(query_xy, k=1)
    nn_z = dsm_points[idx, 2]
    return nn_z, nn_dist


def _gate_ground_like(
    shifted_points: np.ndarray,
    dsm_points: np.ndarray,
    vertical_gate_m: float,
    min_points_after_gate: int,
    fallback_fraction: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    nn_z, nn_xy_dist = _nearest_dsm_z(dsm_points, shifted_points[:, :2])
    dz = nn_z - shifted_points[:, 2]
    mask = np.abs(dz) <= vertical_gate_m
    num_mask = int(np.count_nonzero(mask))
    diagnostics = {
        "gate_vertical_threshold_m": float(vertical_gate_m),
        "preselection_total": int(shifted_points.shape[0]),
        "preselection_kept_by_gate": num_mask,
        "preselection_dz_summary": _summarize(dz),
        "preselection_xy_dist_summary": _summarize(nn_xy_dist),
    }

    if num_mask < min_points_after_gate:
        order = np.argsort(np.abs(dz))
        take = max(min_points_after_gate, int(fallback_fraction * shifted_points.shape[0]))
        take = min(take, shifted_points.shape[0])
        sel_idx = order[:take]
    else:
        sel_idx = np.nonzero(mask)[0]

    diagnostics["preselection_final_kept"] = int(sel_idx.size)
    return sel_idx, diagnostics


def _crop_and_gate_target_by_source(
    source_points: np.ndarray,
    dsm_points: np.ndarray,
    vertical_gate_m: float,
    xy_margin_m: float,
    min_points_after_gate: int,
    fallback_fraction: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # 1) Crop DSM to source XY bbox with margin
    src_xy = source_points[:, :2]
    min_xy = np.min(src_xy, axis=0) - xy_margin_m
    max_xy = np.max(src_xy, axis=0) + xy_margin_m
    in_roi = (
        (dsm_points[:, 0] >= min_xy[0])
        & (dsm_points[:, 0] <= max_xy[0])
        & (dsm_points[:, 1] >= min_xy[1])
        & (dsm_points[:, 1] <= max_xy[1])
    )
    dsm_roi_idx = np.nonzero(in_roi)[0]
    dsm_roi = dsm_points[dsm_roi_idx]

    diagnostics: Dict[str, Any] = {
        "target_crop": {
            "xy_margin_m": float(xy_margin_m),
            "roi_count": int(dsm_roi.shape[0]),
        }
    }

    if dsm_roi.shape[0] == 0:
        # Fallback: no crop effective, operate on full DSM
        dsm_roi = dsm_points
        dsm_roi_idx = np.arange(dsm_points.shape[0])
        diagnostics["target_crop"]["roi_count"] = int(dsm_roi.shape[0])
        diagnostics["target_crop"]["fallback_full_target"] = True

    # 2) Gate DSM by vertical consistency relative to nearest source point
    src_tree = cKDTree(source_points[:, :2])
    _, nn_src_idx = src_tree.query(dsm_roi[:, :2], k=1)
    nn_src_z = source_points[nn_src_idx, 2]
    dz = dsm_roi[:, 2] - nn_src_z
    mask = np.abs(dz) <= vertical_gate_m
    num_mask = int(np.count_nonzero(mask))

    diagnostics["target_gate"] = {
        "gate_vertical_threshold_m": float(vertical_gate_m),
        "preselection_total": int(dsm_roi.shape[0]),
        "preselection_kept_by_gate": num_mask,
        "preselection_dz_summary": _summarize(dz),
    }

    if num_mask < min_points_after_gate:
        order = np.argsort(np.abs(dz))
        take = max(min_points_after_gate, int(fallback_fraction * dsm_roi.shape[0]))
        take = min(take, dsm_roi.shape[0])
        sel_local_idx = order[:take]
    else:
        sel_local_idx = np.nonzero(mask)[0]

    sel_global_idx = dsm_roi_idx[sel_local_idx]
    diagnostics["target_gate"]["final_kept"] = int(sel_global_idx.size)
    return sel_global_idx, diagnostics


def _scale_z(points: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return points
    scaled = points.copy()
    scaled[:, 2] = scaled[:, 2] / factor
    return scaled


def _unscale_transform_z(T: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return T
    # Adjust translation z back to original units
    Tout = T.copy()
    Tout[2, 3] = Tout[2, 3] * factor
    return Tout


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


def register_with_roi_and_post_correction(
    shifted_points: np.ndarray,
    dsm_points: np.ndarray,
    shared_origin: Optional[np.ndarray] = None,
    config: Optional[Op1Config] = None,
) -> Dict[str, Any]:
    cfg = config or Op1Config()

    sel_idx, gate_diag = _gate_ground_like(
        shifted_points,
        dsm_points,
        cfg.vertical_gate_m,
        cfg.min_points_after_gate,
        cfg.fallback_fraction,
    )
    source_for_gicp = shifted_points[sel_idx]

    # Target ROI and gating to symmetrize with source
    tgt_idx, tgt_diag = _crop_and_gate_target_by_source(
        source_for_gicp,
        dsm_points,
        cfg.vertical_gate_m,
        cfg.target_xy_margin_m,
        cfg.target_min_points_after_gate,
        cfg.fallback_fraction,
    )
    target_for_gicp = dsm_points[tgt_idx]

    # Optionally downweight z before ICP to reduce vertical attraction
    src_for_icp = _scale_z(source_for_gicp, cfg.z_downweight_factor)
    tgt_for_icp = _scale_z(target_for_gicp, cfg.z_downweight_factor)

    # Prepare and run GICP
    params = GICPParams(
        voxel_size=cfg.voxel_size,
        normal_k=cfg.normal_k,
        max_corr_dist=cfg.max_corr_dist,
        max_iter=cfg.max_iter,
        enforce_z_up=True,
    )

    clouds = base_prepare_downsampled_clouds(src_for_icp, tgt_for_icp, shared_origin=shared_origin, params=params)
    gicp_result = base_run_gicp(clouds["source_centered"], clouds["target_centered"], params=params)
    gicp_T_scaled = base_compose_transform(gicp_result, clouds["src_centroid"], clouds["tgt_centroid"])
    gicp_T = _unscale_transform_z(gicp_T_scaled, cfg.z_downweight_factor)

    # Post-correction: use yaw + XY from GICP, set Z from robust median using ground-like correspondences
    R = gicp_T[:3, :3]
    t = gicp_T[:3, 3]
    yaw = _extract_yaw(R)
    # Keep XY translation from GICP directly
    tx, ty = float(t[0]), float(t[1])

    # Estimate Z using median dz after applying XY+yaw only
    T_xy_yaw = _compose_from_xy_yaw_z(tx, ty, yaw, 0.0)
    src_xyyaw = base_apply_transform(shifted_points, T_xy_yaw)
    nn_z_after_xy, _ = _nearest_dsm_z(dsm_points, src_xyyaw[:, :2])
    dz_all = nn_z_after_xy - src_xyyaw[:, 2]
    # Focus on ground-like subset again for robust Z
    dz_ground = dz_all[sel_idx]
    z_median = float(np.median(dz_ground)) if dz_ground.size > 0 else float(np.median(dz_all))

    T_final = _compose_from_xy_yaw_z(tx, ty, yaw, z_median)

    # Diagnostics
    transformed_points = base_apply_transform(shifted_points, T_final)
    nn_z_after, nn_xy = _nearest_dsm_z(dsm_points, transformed_points[:, :2])
    dz_after = nn_z_after - transformed_points[:, 2]

    diagnostics = {
        "roi": gate_diag,
        "target": tgt_diag,
        "gicp": {
            "fitness": float(getattr(gicp_result, "fitness", 0.0)),
            "rmse": float(getattr(gicp_result, "inlier_rmse", 0.0)),
        },
        "post_correction": {
            "yaw_rad": yaw,
            "xy_translation_m": float(np.hypot(tx, ty)),
            "z_translation_m": z_median,
        },
        "dz_summary_after_xyyaw": _summarize(dz_all),
        "dz_summary_after_final": _summarize(dz_after),
        "nn_xy_dist_summary_after_final": _summarize(nn_xy),
        "selected_source_points": int(source_for_gicp.shape[0]),
    }

    return {
        "transform": T_final,
        "diagnostics": diagnostics,
        "selected_source_points": int(source_for_gicp.shape[0]),
    }


# Convenience exports aligned with base core naming
def compose_transform(
    result: o3d.pipelines.registration.RegistrationResult,
    src_centroid: np.ndarray,
    tgt_centroid: np.ndarray,
) -> np.ndarray:
    return base_compose_transform(result, src_centroid, tgt_centroid)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return base_apply_transform(points, transform)


