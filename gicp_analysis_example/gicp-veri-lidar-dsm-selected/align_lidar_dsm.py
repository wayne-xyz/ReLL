"""
Align shifted LiDAR to extracted DSM using GICP.

This script:
1. Loads preprocessed shifted LiDAR and extracted DSM
2. Computes centroid of the 100x100 square
3. Runs GICP in local frame (centered at centroid)
4. Saves aligned LiDAR and metrics
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

# Import GICP helpers from parent analysis directory
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


def load_parquet(path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load point cloud from parquet file.

    Returns:
        Tuple of (points as Nx3 array, original dataframe)
    """
    print(f"Loading {path}")
    df = pd.read_parquet(path)

    # Try to find XYZ columns
    if all(col in df.columns for col in ["utm_e", "utm_n", "elevation"]):
        points = df[["utm_e", "utm_n", "elevation"]].to_numpy(dtype=float)
    elif all(col in df.columns for col in ["utm_e", "utm_n", "z"]):
        points = df[["utm_e", "utm_n", "z"]].to_numpy(dtype=float)
    elif all(col in df.columns for col in ["x", "y", "z"]):
        points = df[["x", "y", "z"]].to_numpy(dtype=float)
    else:
        raise ValueError(f"Could not find XYZ columns in {path}")

    print(f"Loaded {len(points):,} points")
    return points, df


def compute_centroid(points: np.ndarray) -> np.ndarray:
    """Compute centroid of point cloud.

    Args:
        points: Nx3 array

    Returns:
        3D centroid point
    """
    return points.mean(axis=0)


def extract_yaw(R: np.ndarray) -> float:
    """Extract yaw angle from rotation matrix."""
    return math.atan2(R[1, 0], R[0, 0])


def make_translation(tx: float, ty: float, tz: float) -> np.ndarray:
    """Create 4x4 translation matrix."""
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def summarize_nn_error(aligned: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Compute nearest-neighbor RMSE and mean absolute distance.

    Args:
        aligned: Aligned source points
        target: Target points

    Returns:
        Tuple of (RMSE, mean absolute distance)
    """
    import open3d as o3d

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


def load_center_from_meta(meta_path: Path) -> np.ndarray:
    """Load center point coordinates from metadata parquet.

    Args:
        meta_path: Path to metadata parquet file

    Returns:
        3D center point [utm_e, utm_n, z]
    """
    print(f"Loading center point from metadata: {meta_path}")
    df = pd.read_parquet(meta_path)

    if len(df) == 0:
        raise ValueError("Metadata file is empty")

    # Take first row
    row = df.iloc[0]

    # Extract center coordinates
    if "center_utm_easting_m" in df.columns and "center_utm_northing_m" in df.columns:
        utm_e = float(row["center_utm_easting_m"])
        utm_n = float(row["center_utm_northing_m"])

        # For Z, prefer city frame Z if available
        if "center_city_tz_m" in df.columns:
            z = float(row["center_city_tz_m"])
        else:
            # Fallback: use mean of LiDAR Z (will be computed later)
            z = 0.0

        center = np.array([utm_e, utm_n, z], dtype=float)
        print(f"Center from metadata: E={utm_e:.3f}, N={utm_n:.3f}, Z={z:.3f}")
        return center
    else:
        raise ValueError("Metadata missing required columns: center_utm_easting_m, center_utm_northing_m")


def run_alignment(
    shifted_lidar_path: Path,
    meta_path: Path,
    extracted_dsm_path: Path,
    output_dir: Path,
    voxel_size: float = 0.3,
    normal_k: int = 20,
    max_corr_dist: float = 0.8,
    max_iter: int = 60,
) -> Path:
    """Run GICP alignment between shifted LiDAR and extracted DSM.

    Args:
        shifted_lidar_path: Path to shifted LiDAR parquet
        meta_path: Path to metadata parquet with center coordinates
        extracted_dsm_path: Path to extracted DSM parquet
        output_dir: Output directory
        voxel_size: Voxel size for downsampling
        normal_k: Number of neighbors for normal estimation
        max_corr_dist: Maximum correspondence distance for GICP
        max_iter: Maximum GICP iterations

    Returns:
        Path to metrics JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    shifted_lidar, lidar_df = load_parquet(shifted_lidar_path)
    extracted_dsm, _ = load_parquet(extracted_dsm_path)

    print(f"\n=== Loading Center Point (Local Frame Anchor) ===")
    # Load center from metadata
    anchor = load_center_from_meta(meta_path)

    # If Z was not in metadata, use mean of shifted LiDAR Z
    if anchor[2] == 0.0:
        anchor[2] = shifted_lidar[:, 2].mean()
        print(f"Updated Z from shifted LiDAR mean: {anchor[2]:.3f}")

    print(f"Local frame anchor (from metadata): E={anchor[0]:.3f}, N={anchor[1]:.3f}, Z={anchor[2]:.3f}")

    print(f"\n=== Running GICP ===")
    print(f"Parameters: voxel_size={voxel_size}, normal_k={normal_k}, max_corr_dist={max_corr_dist}, max_iter={max_iter}")

    # Setup GICP parameters
    params = GICPParams(
        voxel_size=voxel_size,
        normal_k=normal_k,
        max_corr_dist=max_corr_dist,
        max_iter=max_iter,
        enforce_z_up=True,
    )

    # Prepare clouds (source=shifted_lidar, target=extracted_dsm)
    clouds = prepare_downsampled_clouds(
        shifted_lidar,
        extracted_dsm,
        shared_origin=None,
        params=params
    )

    # Run GICP
    result = run_gicp(clouds["source_centered"], clouds["target_centered"], params=params)
    estimated_transform_global = compose_transform(result, clouds["src_centroid"], clouds["tgt_centroid"])

    print(f"GICP fitness: {result.fitness:.6f}")
    print(f"GICP RMSE: {result.inlier_rmse:.6f} m")

    # Convert to local frame
    anchor_to_origin = make_translation(-anchor[0], -anchor[1], -anchor[2])
    origin_to_anchor = make_translation(anchor[0], anchor[1], anchor[2])
    estimated_transform_local = anchor_to_origin @ estimated_transform_global @ origin_to_anchor

    # Extract transform components from local frame
    R_local = estimated_transform_local[:3, :3]
    t_local = estimated_transform_local[:3, 3]
    yaw_local = extract_yaw(R_local)

    print(f"\nTransform (local frame):")
    print(f"  Translation: [{t_local[0]:.6f}, {t_local[1]:.6f}, {t_local[2]:.6f}] m")
    print(f"  Yaw: {math.degrees(yaw_local):.6f} deg")

    # Apply transform to shifted LiDAR
    aligned_lidar = apply_transform(shifted_lidar, estimated_transform_global)

    # Compute alignment quality
    rmse_to_target, mean_abs_to_target = summarize_nn_error(aligned_lidar, extracted_dsm)
    print(f"\nAlignment quality:")
    print(f"  NN RMSE: {rmse_to_target:.6f} m")
    print(f"  NN mean abs distance: {mean_abs_to_target:.6f} m")

    # Save aligned LiDAR
    aligned_path = output_dir / "lidar_aligned.parquet"
    df_aligned = lidar_df.copy()
    if "utm_e" in df_aligned.columns:
        df_aligned["utm_e"] = aligned_lidar[:, 0]
        df_aligned["utm_n"] = aligned_lidar[:, 1]
        if "elevation" in df_aligned.columns:
            df_aligned["elevation"] = aligned_lidar[:, 2]
        elif "z" in df_aligned.columns:
            df_aligned["z"] = aligned_lidar[:, 2]
    elif "x" in df_aligned.columns:
        df_aligned["x"] = aligned_lidar[:, 0]
        df_aligned["y"] = aligned_lidar[:, 1]
        df_aligned["z"] = aligned_lidar[:, 2]
    df_aligned.to_parquet(aligned_path, index=False)
    print(f"\nSaved aligned LiDAR: {aligned_path}")

    # Save metrics
    metrics_path = output_dir / "alignment_metrics.json"
    metrics = {
        "inputs": {
            "shifted_lidar": str(shifted_lidar_path),
            "meta_data": str(meta_path),
            "extracted_dsm": str(extracted_dsm_path),
            "shifted_lidar_points": int(len(shifted_lidar)),
            "extracted_dsm_points": int(len(extracted_dsm)),
        },
        "local_frame_anchor_utm": {
            "utm_e": float(anchor[0]),
            "utm_n": float(anchor[1]),
            "z": float(anchor[2]),
            "note": "Center point from metadata - all transforms below are in local frame centered at this anchor point",
        },
        "gicp_parameters": {
            "voxel_size": voxel_size,
            "normal_k": normal_k,
            "max_corr_dist": max_corr_dist,
            "max_iter": max_iter,
        },
        "gicp_result": {
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
        },
        "transform_local_frame": {
            "matrix_4x4": estimated_transform_local.tolist(),
            "translation_m": [float(t_local[0]), float(t_local[1]), float(t_local[2])],
            "yaw_deg": float(math.degrees(yaw_local)),
            "note": "Transform in local frame (centered at anchor)",
        },
        "transform_global_frame": {
            "matrix_4x4": estimated_transform_global.tolist(),
            "note": "Transform in global UTM frame (for applying to actual data)",
        },
        "alignment_quality": {
            "rmse_to_target_m": rmse_to_target,
            "mean_abs_distance_m": mean_abs_to_target,
        },
        "outputs": {
            "aligned_lidar": str(aligned_path),
            "metrics_json": str(metrics_path),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics: {metrics_path}")

    return metrics_path


def run_gui() -> None:
    """Launch GUI for file selection."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("LiDAR-DSM GICP Alignment")

    # Variables
    shifted_lidar_var = tk.StringVar(value="lidar_shifted.parquet")
    meta_var = tk.StringVar(value="data3_meta.parquet")
    extracted_dsm_var = tk.StringVar(value="dsm_extracted.parquet")
    output_dir_var = tk.StringVar(value=".")
    voxel_size_var = tk.DoubleVar(value=0.3)
    normal_k_var = tk.IntVar(value=20)
    max_corr_dist_var = tk.DoubleVar(value=0.8)
    max_iter_var = tk.IntVar(value=60)

    # Browse functions
    def browse_shifted_lidar():
        path = filedialog.askopenfilename(
            title="Select Shifted LiDAR Parquet File",
            filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            shifted_lidar_var.set(path)

    def browse_meta():
        path = filedialog.askopenfilename(
            title="Select Metadata Parquet File",
            filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            meta_var.set(path)

    def browse_extracted_dsm():
        path = filedialog.askopenfilename(
            title="Select Extracted DSM Parquet File",
            filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            extracted_dsm_var.set(path)

    def browse_output():
        path = filedialog.askdirectory(title="Select Output Directory", initialdir=Path.cwd())
        if path:
            output_dir_var.set(path)

    def run_alignment_gui():
        try:
            shifted_lidar_path = Path(shifted_lidar_var.get())
            meta_path = Path(meta_var.get())
            extracted_dsm_path = Path(extracted_dsm_var.get())
            output_dir = Path(output_dir_var.get())
            voxel_size = voxel_size_var.get()
            normal_k = normal_k_var.get()
            max_corr_dist = max_corr_dist_var.get()
            max_iter = max_iter_var.get()

            if not shifted_lidar_path.is_file():
                raise FileNotFoundError(f"Shifted LiDAR file not found: {shifted_lidar_path}")
            if not meta_path.is_file():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            if not extracted_dsm_path.is_file():
                raise FileNotFoundError(f"Extracted DSM file not found: {extracted_dsm_path}")

            # Close GUI and run alignment
            root.destroy()

            metrics_path = run_alignment(
                shifted_lidar_path,
                meta_path,
                extracted_dsm_path,
                output_dir,
                voxel_size=voxel_size,
                normal_k=normal_k,
                max_corr_dist=max_corr_dist,
                max_iter=max_iter,
            )

            print("\n=== GICP Alignment Complete ===")
            messagebox.showinfo("Success", f"GICP alignment complete!\n\nMetrics saved to:\n{metrics_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # GUI Layout
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(main_frame, text="LiDAR-DSM GICP Alignment", font=("", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=10)

    # Shifted LiDAR file
    ttk.Label(main_frame, text="Shifted LiDAR:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=shifted_lidar_var, width=50).grid(row=1, column=1, padx=5, pady=5)
    ttk.Button(main_frame, text="Browse", command=browse_shifted_lidar).grid(row=1, column=2, padx=5, pady=5)

    # Metadata file
    ttk.Label(main_frame, text="Metadata:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=meta_var, width=50).grid(row=2, column=1, padx=5, pady=5)
    ttk.Button(main_frame, text="Browse", command=browse_meta).grid(row=2, column=2, padx=5, pady=5)

    # Extracted DSM file
    ttk.Label(main_frame, text="Extracted DSM:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=extracted_dsm_var, width=50).grid(row=3, column=1, padx=5, pady=5)
    ttk.Button(main_frame, text="Browse", command=browse_extracted_dsm).grid(row=3, column=2, padx=5, pady=5)

    # Output directory
    ttk.Label(main_frame, text="Output Directory:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=output_dir_var, width=50).grid(row=4, column=1, padx=5, pady=5)
    ttk.Button(main_frame, text="Browse", command=browse_output).grid(row=4, column=2, padx=5, pady=5)

    # Parameters frame
    params_frame = ttk.LabelFrame(main_frame, text="GICP Parameters", padding="10")
    params_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=10, sticky=(tk.W, tk.E))

    ttk.Label(params_frame, text="Voxel Size (m):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
    ttk.Entry(params_frame, textvariable=voxel_size_var, width=15).grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)

    ttk.Label(params_frame, text="Normal K:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
    ttk.Entry(params_frame, textvariable=normal_k_var, width=15).grid(row=0, column=3, sticky=tk.W, padx=5, pady=3)

    ttk.Label(params_frame, text="Max Corr Dist (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
    ttk.Entry(params_frame, textvariable=max_corr_dist_var, width=15).grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)

    ttk.Label(params_frame, text="Max Iterations:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=3)
    ttk.Entry(params_frame, textvariable=max_iter_var, width=15).grid(row=1, column=3, sticky=tk.W, padx=5, pady=3)

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=6, column=0, columnspan=3, pady=20)
    ttk.Button(button_frame, text="Run GICP Alignment", command=run_alignment_gui).grid(row=0, column=0, padx=5)
    ttk.Button(button_frame, text="Exit", command=root.quit).grid(row=0, column=1, padx=5)

    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Align shifted LiDAR to extracted DSM using GICP")
    parser.add_argument("--shifted-lidar", type=Path, help="Shifted LiDAR parquet")
    parser.add_argument("--meta", type=Path, help="Metadata parquet with center coordinates")
    parser.add_argument("--extracted-dsm", type=Path, help="Extracted DSM parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument("--voxel-size", type=float, default=0.3, help="Voxel size for downsampling (meters)")
    parser.add_argument("--normal-k", type=int, default=20, help="Neighbors for normal estimation")
    parser.add_argument("--max-corr-dist", type=float, default=0.8, help="Maximum correspondence distance (meters)")
    parser.add_argument("--max-iter", type=int, default=60, help="Maximum GICP iterations")
    parser.add_argument("--gui", action="store_true", help="Launch GUI for file selection")
    args = parser.parse_args()

    # Launch GUI if requested or if no files specified
    if args.gui or not (args.shifted_lidar and args.meta and args.extracted_dsm):
        run_gui()
        return

    # Validate inputs
    if not args.shifted_lidar.is_file():
        raise FileNotFoundError(f"Shifted LiDAR file not found: {args.shifted_lidar}")
    if not args.meta.is_file():
        raise FileNotFoundError(f"Metadata file not found: {args.meta}")
    if not args.extracted_dsm.is_file():
        raise FileNotFoundError(f"Extracted DSM file not found: {args.extracted_dsm}")

    run_alignment(
        args.shifted_lidar,
        args.meta,
        args.extracted_dsm,
        args.output_dir,
        voxel_size=args.voxel_size,
        normal_k=args.normal_k,
        max_corr_dist=args.max_corr_dist,
        max_iter=args.max_iter,
    )

    print("\n=== GICP Alignment Complete ===")


if __name__ == "__main__":
    main()
