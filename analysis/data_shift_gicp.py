import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import laspy
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

VOXEL_SIZE = 0.5
NORMAL_RADIUS = 2.0
MAX_CORR_DIST = 1.0
MAX_ITER = 30
GROUND_THRESHOLD = -14.5


def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ])


def fit_affine(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.hstack([src_xy, np.ones((src_xy.shape[0], 1))])
    coeff_e, *_ = np.linalg.lstsq(X, dst_xy[:, 0], rcond=None)
    coeff_n, *_ = np.linalg.lstsq(X, dst_xy[:, 1], rcond=None)
    return coeff_e, coeff_n


def apply_affine(coeff: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    return pts_xy @ coeff[:2] + coeff[2]


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def evaluate(points: np.ndarray, dsm_points: np.ndarray) -> dict:
    tree = cKDTree(dsm_points[:, :2])
    nn_dist, idx = tree.query(points[:, :2], k=1)
    dsm_z = dsm_points[idx, 2]
    vertical_diff = dsm_z - points[:, 2]
    return {
        "vertical": summarize(vertical_diff),
        "horizontal": summarize(nn_dist),
    }


def make_output_paths(lidar_path: Path, output_dir: Path) -> dict:
    base = lidar_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "shifted": output_dir / f"{base}_shifted.parquet",
        "shifted_gicp": output_dir / f"{base}_shifted_gicp.parquet",
        "metrics_json": output_dir / f"{base}_alignment_metrics.json",
        "metrics_npz": output_dir / f"{base}_alignment_results.npz",
    }


def load_inputs(lidar_path: Path, meta_path: Path, dsm_path: Path):
    meta = pd.read_parquet(meta_path).iloc[0]
    lidar_df = pd.read_parquet(lidar_path)
    rotation = quat_to_rot(meta["center_city_qw"], meta["center_city_qx"], meta["center_city_qy"], meta["center_city_qz"])
    translation = np.array([
        meta["center_city_tx_m"],
        meta["center_city_ty_m"],
        meta["center_city_tz_m"],
    ])
    xyz_local = lidar_df[["x", "y", "z"]].to_numpy()
    xyz_city = (rotation @ xyz_local.T).T + translation

    city_positions = np.array(json.loads(meta["sensor_positions_city_m"]))
    utm_positions = np.array([[e, n] for _, e, n in json.loads(meta["sensor_positions_utm"])])
    coeff_e, coeff_n = fit_affine(city_positions[:, :2], utm_positions)
    utm_e = apply_affine(coeff_e, xyz_city[:, :2])
    utm_n = apply_affine(coeff_n, xyz_city[:, :2])

    las = laspy.read(dsm_path)
    dsm_points = np.column_stack([las.x, las.y, las.z])
    e_min, e_max = np.min(utm_e), np.max(utm_e)
    n_min, n_max = np.min(utm_n), np.max(utm_n)
    margin = 40.0
    mask = (
        (dsm_points[:, 0] >= e_min - margin)
        & (dsm_points[:, 0] <= e_max + margin)
        & (dsm_points[:, 1] >= n_min - margin)
        & (dsm_points[:, 1] <= n_max + margin)
    )
    dsm_subset = dsm_points[mask]
    return meta, lidar_df, xyz_city, utm_e, utm_n, dsm_subset


def query_dsm_height(dsm_points: np.ndarray, easting: float, northing: float) -> float:
    tree = cKDTree(dsm_points[:, :2])
    _, idx = tree.query([[easting, northing]], k=1)
    return float(dsm_points[idx[0], 2])


def build_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pc


def prepare_downsampled_clouds(source_points: np.ndarray, target_points: np.ndarray):
    source_pc = build_cloud(source_points)
    target_pc = build_cloud(target_points)

    source_down = source_pc.voxel_down_sample(VOXEL_SIZE)
    target_down = target_pc.voxel_down_sample(VOXEL_SIZE)

    if len(source_down.points) == 0 or len(target_down.points) == 0:
        raise RuntimeError("Downsampled cloud empty. Adjust voxel size or ground mask.")

    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=60)
    source_down.estimate_normals(search_param)
    target_down.estimate_normals(search_param)

    src_centroid = np.mean(np.asarray(source_down.points), axis=0)
    tgt_centroid = np.mean(np.asarray(target_down.points), axis=0)

    source_centered = source_down.translate(-src_centroid, relative=False)
    target_centered = target_down.translate(-tgt_centroid, relative=False)

    return {
        "source_pc": source_pc,
        "target_pc": target_pc,
        "source_centered": source_centered,
        "target_centered": target_centered,
        "src_centroid": src_centroid,
        "tgt_centroid": tgt_centroid,
    }


def run_gicp(source_centered: o3d.geometry.PointCloud, target_centered: o3d.geometry.PointCloud):
    estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=MAX_ITER)
    return o3d.pipelines.registration.registration_generalized_icp(
        source_centered,
        target_centered,
        MAX_CORR_DIST,
        np.eye(4),
        estimation,
        criteria,
    )


def compose_transform(result: o3d.pipelines.registration.RegistrationResult, src_centroid: np.ndarray, tgt_centroid: np.ndarray) -> np.ndarray:
    T_src = np.eye(4)
    T_src[:3, 3] = -src_centroid
    T_tgt = np.eye(4)
    T_tgt[:3, 3] = tgt_centroid
    return T_tgt @ result.transformation @ T_src


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return (transform @ homogeneous.T).T[:, :3]


def save_parquet(template_df: pd.DataFrame, points: np.ndarray, path: Path) -> None:
    out_df = template_df.copy()
    out_df["utm_e"] = points[:, 0]
    out_df["utm_n"] = points[:, 1]
    out_df["elevation"] = points[:, 2]
    out_df.to_parquet(path, index=False)


def process_alignment(lidar_path: Path, meta_path: Path, dsm_path: Path, output_dir: Path, run_gicp_flag: bool = True) -> dict:
    meta, lidar_df, xyz_city, utm_e, utm_n, dsm_subset = load_inputs(lidar_path, meta_path, dsm_path)

    baseline_points = np.column_stack([utm_e, utm_n, xyz_city[:, 2]])
    baseline_metrics = evaluate(baseline_points, dsm_subset)

    center_dsm_z = query_dsm_height(dsm_subset, meta["center_utm_easting_m"], meta["center_utm_northing_m"])
    center_city_z = float(meta["center_city_tz_m"])
    vertical_offset = center_dsm_z - center_city_z

    shifted_points = baseline_points.copy()
    shifted_points[:, 2] += vertical_offset
    shifted_metrics = evaluate(shifted_points, dsm_subset)

    outputs = make_output_paths(lidar_path, output_dir)
    save_parquet(lidar_df, shifted_points, outputs["shifted"])

    metrics = {
        "center_dsm_z": center_dsm_z,
        "center_city_z": center_city_z,
        "vertical_offset_applied": vertical_offset,
        "baseline": baseline_metrics,
        "after_shift": shifted_metrics,
    }

    gicp_info = None
    transformed_points = shifted_points

    if run_gicp_flag:
        ground_mask = xyz_city[:, 2] <= GROUND_THRESHOLD
        if not np.any(ground_mask):
            raise RuntimeError("Ground mask empty; adjust threshold before running GICP.")
        source_for_gicp = shifted_points[ground_mask]
        clouds = prepare_downsampled_clouds(source_for_gicp, dsm_subset)
        gicp_result = run_gicp(clouds["source_centered"], clouds["target_centered"])
        gicp_transform = compose_transform(gicp_result, clouds["src_centroid"], clouds["tgt_centroid"])
        transformed_points = apply_transform(shifted_points, gicp_transform)
        gicp_metrics = evaluate(transformed_points, dsm_subset)
        translation = gicp_transform[:3, 3]
        gicp_info = {
            "registration_fitness": float(gicp_result.fitness),
            "registration_rmse": float(gicp_result.inlier_rmse),
            "transform": gicp_transform.tolist(),
            "metrics": gicp_metrics,
            "horizontal_translation_m": float(np.linalg.norm(translation[:2])),
        }
        save_parquet(lidar_df, transformed_points, outputs["shifted_gicp"])
        metrics["gicp"] = gicp_info
    else:
        metrics["gicp"] = None
        outputs["shifted_gicp"] = None

    outputs["shifted_points"] = shifted_points
    outputs["transformed_points"] = transformed_points

    Path(outputs["metrics_json"]).write_text(json.dumps(metrics, indent=2))
    np.savez(outputs["metrics_npz"], metrics=metrics, shifted=shifted_points, transformed=transformed_points)

    return {
        "outputs": outputs,
        "metrics": metrics,
        "gicp_info": gicp_info,
    }


def launch_gui():
    root = tk.Tk()
    root.title("LiDAR Shift + GICP Alignment")

    path_vars = {
        "lidar": tk.StringVar(),
        "meta": tk.StringVar(),
        "dsm": tk.StringVar(),
        "output": tk.StringVar(value=str(Path.cwd() / "analysis")),
    }

    def browse_file(kind: str, filetypes):
        initialdir = Path(path_vars[kind].get() or Path.cwd())
        selected = filedialog.askopenfilename(parent=root, filetypes=filetypes, initialdir=initialdir)
        if selected:
            path_vars[kind].set(selected)

    def browse_dir():
        initialdir = Path(path_vars["output"].get() or Path.cwd())
        selected = filedialog.askdirectory(parent=root, initialdir=initialdir)
        if selected:
            path_vars["output"].set(selected)

    def run_processing():
        try:
            lidar_path = Path(path_vars["lidar"].get())
            meta_path = Path(path_vars["meta"].get())
            dsm_path = Path(path_vars["dsm"].get())
            output_dir = Path(path_vars["output"].get())
            if not lidar_path.is_file() or not meta_path.is_file() or not dsm_path.is_file():
                raise FileNotFoundError("Please select valid LiDAR, metadata, and DSM files.")
            result = process_alignment(lidar_path, meta_path, dsm_path, output_dir, run_gicp_flag=True)
            metrics_path = result["outputs"]["metrics_json"]
            shift_path = result["outputs"]["shifted"]
            gicp_path = result["outputs"].get("shifted_gicp")
            msg = [
                f"Metrics saved to: {metrics_path}",
                f"Shifted cloud: {shift_path}",
            ]
            if gicp_path:
                msg.append(f"Shifted + GICP cloud: {gicp_path}")
            messagebox.showinfo("Completed", "\n".join(msg))
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Error", str(exc))

    labels = {
        "lidar": "LiDAR parquet",
        "meta": "Metadata parquet",
        "dsm": "DSM LAZ",
        "output": "Output directory",
    }

    for idx, key in enumerate(["lidar", "meta", "dsm", "output"]):
        tk.Label(root, text=labels[key]).grid(row=idx, column=0, sticky="w", padx=6, pady=4)
        entry = tk.Entry(root, textvariable=path_vars[key], width=60)
        entry.grid(row=idx, column=1, padx=6, pady=4)
        if key == "output":
            button = tk.Button(root, text="Browse", command=browse_dir)
        else:
            ft = [("Parquet", "*.parquet")] if key != "dsm" else [("LAS/LAZ", "*.las *.laz"), ("All", "*.*")]
            button = tk.Button(root, text="Browse", command=lambda k=key, f=ft: browse_file(k, f))
        button.grid(row=idx, column=2, padx=6, pady=4)

    tk.Button(root, text="Run", command=run_processing).grid(row=4, column=0, columnspan=3, pady=12)
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Align LiDAR sweep to DSM by vertical shift and optional GICP refinement.")
    parser.add_argument("--lidar", type=Path, help="Path to LiDAR parquet file")
    parser.add_argument("--meta", type=Path, help="Path to metadata parquet file")
    parser.add_argument("--dsm", type=Path, help="Path to DSM LAS/LAZ file")
    parser.add_argument("--output-dir", type=Path, help="Directory for outputs")
    parser.add_argument("--skip-gicp", action="store_true", help="Skip the GICP refinement step")
    parser.add_argument("--gui", action="store_true", help="Launch GUI even if paths are provided")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gui or not (args.lidar and args.meta and args.dsm):
        launch_gui()
        return

    output_dir = args.output_dir or args.lidar.parent
    result = process_alignment(args.lidar, args.meta, args.dsm, output_dir, run_gicp_flag=not args.skip_gicp)
    metrics = result["metrics"]
    metrics_path = result["outputs"]["metrics_json"]
    print("Metrics saved to:", metrics_path)
    print("Vertical offset applied (m):", metrics["vertical_offset_applied"])
    print("Shifted cloud saved to:", result["outputs"]["shifted"])
    gicp_info = result.get("gicp_info")
    gicp_path = result["outputs"].get("shifted_gicp")
    if gicp_info and gicp_path:
        print("Shifted + GICP cloud saved to:", gicp_path)
        print("GICP fitness:", gicp_info["registration_fitness"], "RMSE:", gicp_info["registration_rmse"])
        print("GICP horizontal translation (m):", gicp_info["horizontal_translation_m"])
    elif not args.skip_gicp:
        print("GICP output unavailable; see metrics file for details.")


if __name__ == "__main__":
    main()
