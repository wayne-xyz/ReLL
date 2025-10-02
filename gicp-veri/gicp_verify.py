import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np

# Ensure we can import the shared GICP helpers when present
ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
if ANALYSIS_DIR.exists():
    sys.path.append(str(ANALYSIS_DIR))

try:
    from gicp_core import (
        GICPParams,
        prepare_downsampled_clouds,
        run_gicp,
        compose_transform,
        apply_transform,
    )
except ImportError:
    import open3d as o3d

    class GICPParams:
        def __init__(self, voxel_size=0.5, normal_k=20, max_corr_dist=0.8, max_iter=60, enforce_z_up=True):
            self.voxel_size = voxel_size
            self.normal_k = normal_k
            self.max_corr_dist = max_corr_dist
            self.max_iter = max_iter
            self.enforce_z_up = enforce_z_up

    def _build_cloud(points: np.ndarray) -> "o3d.geometry.PointCloud":
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        return pc

    def _orient_normals(pc: "o3d.geometry.PointCloud", normal_k: int, enforce_z_up: bool) -> None:
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

    def prepare_downsampled_clouds(source_points: np.ndarray, target_points: np.ndarray, shared_origin=None, params=None):
        params = params or GICPParams()
        origin = np.zeros(3, dtype=np.float64) if shared_origin is None else np.asarray(shared_origin, dtype=np.float64)
        src_local = source_points.astype(np.float64) - origin
        tgt_local = target_points.astype(np.float64) - origin
        source_pc = _build_cloud(src_local)
        target_pc = _build_cloud(tgt_local)
        if params.voxel_size > 0:
            source_down = source_pc.voxel_down_sample(params.voxel_size)
            target_down = target_pc.voxel_down_sample(params.voxel_size)
        else:
            source_down = source_pc
            target_down = target_pc
        if len(source_down.points) == 0 or len(target_down.points) == 0:
            raise RuntimeError("Downsampled cloud empty. Adjust voxel size or point counts.")
        _orient_normals(source_down, params.normal_k, params.enforce_z_up)
        _orient_normals(target_down, params.normal_k, params.enforce_z_up)
        src_centroid = np.mean(np.asarray(source_down.points), axis=0)
        tgt_centroid = np.mean(np.asarray(target_down.points), axis=0)
        source_centered = source_down.translate(-src_centroid)
        target_centered = target_down.translate(-tgt_centroid)
        return {
            "source_pc": source_pc,
            "target_pc": target_pc,
            "source_centered": source_centered,
            "target_centered": target_centered,
            "src_centroid": src_centroid + origin,
            "tgt_centroid": tgt_centroid + origin,
        }

    def run_gicp(source_centered, target_centered, params=None):
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

    def compose_transform(result, src_centroid, tgt_centroid):
        T_src = np.eye(4)
        T_src[:3, 3] = -np.asarray(src_centroid)
        T_tgt = np.eye(4)
        T_tgt[:3, 3] = np.asarray(tgt_centroid)
        return T_tgt @ result.transformation @ T_src

    def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        homo = np.hstack([points, np.ones((points.shape[0], 1))])
        return (transform @ homo.T).T[:, :3]

import open3d as o3d

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit("pandas is required to read parquet files for this verification script.") from exc


@dataclass
class VerifyConfig:
    translation_m: Tuple[float, float, float] = (0.3, -0.25, 0.15)
    rotation_deg: Tuple[float, float, float] = (0.3, -0.2, 0.5)  # roll, pitch, yaw
    voxel_size: float = 0.3
    normal_k: int = 20
    max_corr_dist: float = 0.8
    max_iter: int = 60


def load_config(path: Optional[Path]) -> VerifyConfig:
    if path is None:
        return VerifyConfig()
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    cfg = VerifyConfig()
    if "translation_m" in data:
        cfg.translation_m = tuple(float(v) for v in data["translation_m"])
    if "rotation_deg" in data:
        cfg.rotation_deg = tuple(float(v) for v in data["rotation_deg"])
    if "voxel_size" in data:
        cfg.voxel_size = float(data["voxel_size"])
    if "normal_k" in data:
        cfg.normal_k = int(data["normal_k"])
    if "max_corr_dist" in data:
        cfg.max_corr_dist = float(data["max_corr_dist"])
    if "max_iter" in data:
        cfg.max_iter = int(data["max_iter"])
    return cfg


def infer_xyz_columns(df: "pd.DataFrame") -> np.ndarray:
    candidates = [
        ("utm_e", "utm_n", "elevation"),
        ("utm_e", "utm_n", "z"),
        ("x", "y", "z"),
    ]
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return df.loc[:, cols].to_numpy(dtype=float)
    raise ValueError("Could not find a trio of XYZ columns in the parquet file. Expected utm_e/utm_n/elevation or x/y/z.")


def build_transform(tx: float, ty: float, tz: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def extract_yaw(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def summarize_nn_error(aligned: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    if aligned.size == 0 or target.size == 0:
        return 0.0, 0.0

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(target.astype(np.float64))
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


def run_verification(target_path: Path, output_dir: Path, cfg: VerifyConfig, visualize: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(target_path)
    target_points = infer_xyz_columns(df)

    tx, ty, tz = cfg.translation_m
    roll_deg, pitch_deg, yaw_deg = cfg.rotation_deg
    offset_transform = build_transform(
        tx,
        ty,
        tz,
        math.radians(roll_deg),
        math.radians(pitch_deg),
        math.radians(yaw_deg),
    )

    anchor = np.mean(target_points, axis=0)
    target_local = target_points - anchor
    source_local = apply_transform(target_local, offset_transform)
    source_points = source_local + anchor

    params = GICPParams(
        voxel_size=cfg.voxel_size,
        normal_k=cfg.normal_k,
        max_corr_dist=cfg.max_corr_dist,
        max_iter=cfg.max_iter,
        enforce_z_up=True,
    )

    clouds = prepare_downsampled_clouds(source_points, target_points, shared_origin=None, params=params)
    result = run_gicp(clouds["source_centered"], clouds["target_centered"], params=params)
    estimated_transform = compose_transform(result, clouds["src_centroid"], clouds["tgt_centroid"])

    aligned_points = apply_transform(source_points, estimated_transform)

    gt_transform = np.linalg.inv(offset_transform)
    delta_transform = np.linalg.inv(estimated_transform) @ gt_transform

    translation_error = float(np.linalg.norm(delta_transform[:3, 3]))
    yaw_error = math.degrees(abs(extract_yaw(delta_transform[:3, :3])))

    rmse_to_target, mean_abs_to_target = summarize_nn_error(aligned_points, target_points)

    target_out = output_dir / "target_reference.parquet"
    source_out = output_dir / "source_offset.parquet"
    aligned_out = output_dir / "source_aligned.parquet"
    metrics_out = output_dir / "gicp_metrics.json"

    df_target = df.copy()
    df_target[df_target.columns[0]] = target_points[:, 0]
    df_target[df_target.columns[1]] = target_points[:, 1]
    df_target[df_target.columns[2]] = target_points[:, 2]
    df_target.to_parquet(target_out, index=False)

    df_source = df.copy()
    df_source[df_source.columns[0]] = source_points[:, 0]
    df_source[df_source.columns[1]] = source_points[:, 1]
    df_source[df_source.columns[2]] = source_points[:, 2]
    df_source.to_parquet(source_out, index=False)

    df_aligned = df.copy()
    df_aligned[df_aligned.columns[0]] = aligned_points[:, 0]
    df_aligned[df_aligned.columns[1]] = aligned_points[:, 1]
    df_aligned[df_aligned.columns[2]] = aligned_points[:, 2]
    df_aligned.to_parquet(aligned_out, index=False)

    metrics = {
        "config": asdict(cfg),
        "input_points": int(target_points.shape[0]),
        "local_anchor_utm": {
            "utm_e": float(anchor[0]),
            "utm_n": float(anchor[1]),
            "z": float(anchor[2]),
        },
        "offset_applied": {
            "translation_m": [float(tx), float(ty), float(tz)],
            "rotation_deg": [float(roll_deg), float(pitch_deg), float(yaw_deg)],
            "matrix_4x4": offset_transform.tolist(),
        },
        "gicp": {
            "fitness": float(getattr(result, "fitness", 0.0)),
            "rmse": float(getattr(result, "inlier_rmse", 0.0)),
            "transform_matrix": estimated_transform.tolist(),
        },
        "alignment_quality": {
            "rmse_to_target_m": rmse_to_target,
            "mean_abs_distance_m": mean_abs_to_target,
        },
        "transform_error": {
            "matrix_4x4": delta_transform.tolist(),
            "translation_error_m": translation_error,
            "yaw_error_deg": yaw_error,
        },
        "outputs": {
            "target_reference": str(target_out),
            "source_offset": str(source_out),
            "source_aligned": str(aligned_out),
            "metrics_json": str(metrics_out),
        },
    }

    with metrics_out.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    if visualize:
        visualize_clouds(target_points, source_points, aligned_points)

    return metrics_out


def visualize_clouds(target: np.ndarray, source: np.ndarray, aligned: np.ndarray) -> None:
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target)
    target_pc.paint_uniform_color([0.1, 0.4, 0.9])  # blue-ish

    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source)
    source_pc.paint_uniform_color([0.9, 0.2, 0.2])  # red

    aligned_pc = o3d.geometry.PointCloud()
    aligned_pc.points = o3d.utility.Vector3dVector(aligned)
    aligned_pc.paint_uniform_color([0.2, 0.8, 0.3])  # green

    print("Visualization legend: target/reference = blue, perturbed source = red, GICP-aligned source = green.")
    print("Close the viewer window to return to the terminal.")

    o3d.visualization.draw_geometries(
        [target_pc, source_pc, aligned_pc],
        window_name="GICP Verification",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify GICP by aligning a perturbed copy of a point cloud back to its reference.")
    parser.add_argument("--target", type=Path, default=Path("data3_utm.parquet"), help="Target/reference point cloud in parquet format.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to store generated clouds and metrics.")
    parser.add_argument("--config", type=Path, help="Optional JSON config file with transform and GICP parameters.")
    parser.add_argument("--tx", type=float, help="Override translation X (meters) applied to create the source cloud.")
    parser.add_argument("--ty", type=float, help="Override translation Y (meters) applied to create the source cloud.")
    parser.add_argument("--tz", type=float, help="Override translation Z (meters) applied to create the source cloud.")
    parser.add_argument("--roll-deg", type=float, help="Override roll offset in degrees applied to the source cloud.")
    parser.add_argument("--pitch-deg", type=float, help="Override pitch offset in degrees applied to the source cloud.")
    parser.add_argument("--yaw-deg", type=float, help="Override yaw offset in degrees applied to the source cloud.")
    parser.add_argument("--voxel-size", type=float, help="Override voxel size for downsampling before GICP (meters).")
    parser.add_argument("--normal-k", type=int, help="Override neighbours for normal estimation.")
    parser.add_argument("--max-corr-dist", type=float, help="Override maximum correspondence distance for GICP (meters).")
    parser.add_argument("--max-iter", type=int, help="Override maximum number of GICP iterations.")
    parser.add_argument("--visualize", action="store_true", help="Open an Open3D window to compare target/source/aligned clouds.")
    return parser.parse_args()


def merge_overrides(cfg: VerifyConfig, args: argparse.Namespace) -> VerifyConfig:
    tx, ty, tz = cfg.translation_m
    roll_deg, pitch_deg, yaw_deg = cfg.rotation_deg
    if args.tx is not None:
        tx = args.tx
    if args.ty is not None:
        ty = args.ty
    if args.tz is not None:
        tz = args.tz
    if args.roll_deg is not None:
        roll_deg = args.roll_deg
    if args.pitch_deg is not None:
        pitch_deg = args.pitch_deg
    if args.yaw_deg is not None:
        yaw_deg = args.yaw_deg
    voxel_size = args.voxel_size if args.voxel_size is not None else cfg.voxel_size
    normal_k = args.normal_k if args.normal_k is not None else cfg.normal_k
    max_corr_dist = args.max_corr_dist if args.max_corr_dist is not None else cfg.max_corr_dist
    max_iter = args.max_iter if args.max_iter is not None else cfg.max_iter
    return VerifyConfig(
        translation_m=(tx, ty, tz),
        rotation_deg=(roll_deg, pitch_deg, yaw_deg),
        voxel_size=voxel_size,
        normal_k=normal_k,
        max_corr_dist=max_corr_dist,
        max_iter=max_iter,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_overrides(cfg, args)

    target_path = args.target
    if not target_path.is_file():
        raise FileNotFoundError(f"Target cloud not found: {target_path}")

    metrics_path = run_verification(target_path, args.output_dir, cfg, visualize=args.visualize)
    print(f"Verification metrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
