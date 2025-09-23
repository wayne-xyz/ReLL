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

try:
    from .gicp_core import (
        GICPParams,
        prepare_downsampled_clouds as core_prepare_downsampled_clouds,
        run_gicp as core_run_gicp,
        compose_transform as core_compose_transform,
        apply_transform as core_apply_transform,
    )
    from .gicp_core_op1 import register_with_roi_and_post_correction as op1_register
except ImportError:
    from gicp_core import (
        GICPParams,
        prepare_downsampled_clouds as core_prepare_downsampled_clouds,
        run_gicp as core_run_gicp,
        compose_transform as core_compose_transform,
        apply_transform as core_apply_transform,
    )
    from gicp_core_op1 import register_with_roi_and_post_correction as op1_register

VOXEL_SIZE = 0.5
NORMAL_RADIUS = 2.0  # legacy: no longer used for normal estimation
NORMAL_K = 20        # use fixed K for PCA normals (density-invariant)
MAX_CORR_DIST = 1.0
MAX_ITER = 80
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


def evaluate(points: np.ndarray, dsm_points: np.ndarray) -> tuple[dict, np.ndarray]:
    tree = cKDTree(dsm_points[:, :2])
    nn_dist, idx = tree.query(points[:, :2], k=1)
    dsm_z = dsm_points[idx, 2]
    vertical_diff = dsm_z - points[:, 2]
    metrics = {
        "vertical": summarize(vertical_diff),
        "horizontal": summarize(nn_dist),
    }
    return metrics, vertical_diff


def make_output_paths(lidar_path: Path, output_dir: Path) -> dict:
    base = lidar_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "shifted": output_dir / f"{base}_shifted.parquet",
        "shifted_gicp": output_dir / f"{base}_shifted_gicp.parquet",
        "metrics_json": output_dir / f"{base}_alignment_metrics.json",
        "metrics_npz": output_dir / f"{base}_alignment_results.npz",
    }


def load_inputs(lidar_path: Path, meta_path: Path | None, dsm_path: Path):
    lidar_df = pd.read_parquet(lidar_path)
    meta = pd.read_parquet(meta_path).iloc[0] if meta_path is not None else None

    has_utm_columns = {"utm_e", "utm_n"}.issubset(lidar_df.columns)

    if has_utm_columns:
        utm_e = lidar_df["utm_e"].to_numpy(dtype=float)
        utm_n = lidar_df["utm_n"].to_numpy(dtype=float)
        if "elevation" in lidar_df.columns:
            z_values = lidar_df["elevation"].to_numpy(dtype=float)
        elif "z" in lidar_df.columns:
            z_values = lidar_df["z"].to_numpy(dtype=float)
        else:
            raise ValueError("LiDAR parquet must include an 'elevation' or 'z' column for height values.")
        xyz_city = np.column_stack([utm_e, utm_n, z_values])
    else:
        if meta is None:
            raise ValueError(
                "Metadata parquet is required when LiDAR parquet lacks 'utm_e'/'utm_n' columns."
            )
        rotation = quat_to_rot(
            meta["center_city_qw"],
            meta["center_city_qx"],
            meta["center_city_qy"],
            meta["center_city_qz"],
        )
        translation = np.array(
            [
                meta["center_city_tx_m"],
                meta["center_city_ty_m"],
                meta["center_city_tz_m"],
            ]
        )
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
    # Kept for backward compatibility; not used after refactor
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pc


def prepare_downsampled_clouds(
    source_points: np.ndarray,
    target_points: np.ndarray,
    shared_origin: np.ndarray | None = None,
):
    params = GICPParams(voxel_size=VOXEL_SIZE, normal_k=20, max_corr_dist=MAX_CORR_DIST, max_iter=MAX_ITER)
    return core_prepare_downsampled_clouds(source_points, target_points, shared_origin=shared_origin, params=params)


def run_gicp(source_centered: o3d.geometry.PointCloud, target_centered: o3d.geometry.PointCloud):
    params = GICPParams(voxel_size=VOXEL_SIZE, normal_k=20, max_corr_dist=MAX_CORR_DIST, max_iter=MAX_ITER)
    return core_run_gicp(source_centered, target_centered, params=params)


def compose_transform(result: o3d.pipelines.registration.RegistrationResult, src_centroid: np.ndarray, tgt_centroid: np.ndarray) -> np.ndarray:
    return core_compose_transform(result, src_centroid, tgt_centroid)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return core_apply_transform(points, transform)


def save_parquet(template_df: pd.DataFrame, points: np.ndarray, path: Path) -> None:
    out_df = template_df.copy()
    out_df["utm_e"] = points[:, 0]
    out_df["utm_n"] = points[:, 1]
    out_df["elevation"] = points[:, 2]
    out_df.to_parquet(path, index=False)


def process_alignment(
    lidar_path: Path,
    meta_path: Path | None,
    dsm_path: Path,
    output_dir: Path,
    run_gicp_flag: bool = True,
    gicp_strategy: str = "core",
) -> dict:
    meta, lidar_df, xyz_city, utm_e, utm_n, dsm_subset = load_inputs(lidar_path, meta_path, dsm_path)

    baseline_points = np.column_stack([utm_e, utm_n, xyz_city[:, 2]])
    baseline_metrics, baseline_vertical_diff = evaluate(baseline_points, dsm_subset)

    if meta is not None:
        center_dsm_z = query_dsm_height(
            dsm_subset,
            float(meta["center_utm_easting_m"]),
            float(meta["center_utm_northing_m"]),
        )
        center_city_z = float(meta["center_city_tz_m"])
        vertical_offset = center_dsm_z - center_city_z
        offset_details = {
            "method": "metadata_center",
            "center_dsm_z": center_dsm_z,
            "center_city_z": center_city_z,
        }
    else:
        center_dsm_z = None
        center_city_z = None
        vertical_offset = float(np.median(baseline_vertical_diff))
        offset_details = {
            "method": "median_vertical_difference",
            "baseline_vertical_median": float(np.median(baseline_vertical_diff)),
            "baseline_vertical_mean": float(np.mean(baseline_vertical_diff)),
        }

    shifted_points = baseline_points.copy()
    shifted_points[:, 2] += vertical_offset
    shifted_metrics, _ = evaluate(shifted_points, dsm_subset)

    outputs = make_output_paths(lidar_path, output_dir)
    save_parquet(lidar_df, shifted_points, outputs["shifted"])

    if meta is not None:
        anchor_xy = np.array([float(meta["center_utm_easting_m"]), float(meta["center_utm_northing_m"])] , dtype=np.float64)
    else:
        lidar_xy_mean = np.mean(shifted_points[:, :2], axis=0)
        dsm_xy_mean = np.mean(dsm_subset[:, :2], axis=0)
        anchor_xy = 0.5 * (lidar_xy_mean + dsm_xy_mean)
    lidar_z_median = float(np.median(shifted_points[:, 2]))
    dsm_z_median = float(np.median(dsm_subset[:, 2]))
    anchor_z = 0.5 * (lidar_z_median + dsm_z_median)
    shared_origin = np.array([anchor_xy[0], anchor_xy[1], anchor_z], dtype=np.float64)


    metrics = {
        "vertical_offset_applied": vertical_offset,
        "baseline": baseline_metrics,
        "after_shift": shifted_metrics,
        "offset_estimation": offset_details,
    }
    if center_dsm_z is not None:
        metrics["center_dsm_z"] = center_dsm_z
    if center_city_z is not None:
        metrics["center_city_z"] = center_city_z
    metrics["gicp_anchor_origin"] = shared_origin.tolist()

    gicp_info = None
    transformed_points = shifted_points

    if run_gicp_flag:
        if gicp_strategy == "op1":
            op1_result = op1_register(shifted_points, dsm_subset, shared_origin=shared_origin, config=None)
            gicp_transform = np.asarray(op1_result["transform"], dtype=float)
            transformed_points = apply_transform(shifted_points, gicp_transform)
            gicp_metrics, _ = evaluate(transformed_points, dsm_subset)
            translation = gicp_transform[:3, 3]
            gicp_info = {
                "strategy": "op1",
                "transform": gicp_transform.tolist(),
                "metrics": gicp_metrics,
                "horizontal_translation_m": float(np.linalg.norm(translation[:2])),
                "selected_source_points": int(op1_result.get("selected_source_points", 0)),
                "diagnostics": op1_result.get("diagnostics", {}),
            }
            save_parquet(lidar_df, transformed_points, outputs["shifted_gicp"])
            metrics["gicp"] = gicp_info
        else:
            # Default core strategy
            # Select ground-like LiDAR points by comparing shifted LiDAR Z to nearest DSM Z
            dsm_tree = cKDTree(dsm_subset[:, :2])
            nn_dist, idx = dsm_tree.query(shifted_points[:, :2], k=1)
            nn_dsm_z = dsm_subset[idx, 2]
            vdiff_after_shift = nn_dsm_z - shifted_points[:, 2]

            mask = np.abs(vdiff_after_shift) <= 1.0
            if np.count_nonzero(mask) < 500:
                mask = np.abs(vdiff_after_shift) <= 1.5
            if np.count_nonzero(mask) < 500:
                mask = np.abs(vdiff_after_shift) <= 2.5

            if np.count_nonzero(mask) < 100:
                order = np.argsort(np.abs(vdiff_after_shift))
                take = min(5000, max(1000, int(0.2 * len(order))))
                sel_idx = order[:take]
                source_for_gicp = shifted_points[sel_idx]
            else:
                source_for_gicp = shifted_points[mask]

            if len(source_for_gicp) == 0:
                raise RuntimeError("No suitable source points for GICP; check inputs or thresholds.")

            clouds = prepare_downsampled_clouds(source_for_gicp, dsm_subset, shared_origin=shared_origin)
            gicp_result = run_gicp(clouds["source_centered"], clouds["target_centered"])
            gicp_transform = compose_transform(gicp_result, clouds["src_centroid"], clouds["tgt_centroid"])
            transformed_points = apply_transform(shifted_points, gicp_transform)
            gicp_metrics, _ = evaluate(transformed_points, dsm_subset)
            translation = gicp_transform[:3, 3]
            gicp_info = {
                "strategy": "core",
                "registration_fitness": float(gicp_result.fitness),
                "registration_rmse": float(gicp_result.inlier_rmse),
                "transform": gicp_transform.tolist(),
                "metrics": gicp_metrics,
                "horizontal_translation_m": float(np.linalg.norm(translation[:2])),
                "gicp_source_points": int(len(source_for_gicp)),
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
        "strategy": tk.StringVar(value="op1"),
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
            lidar_value = path_vars["lidar"].get().strip()
            meta_value = path_vars["meta"].get().strip()
            dsm_value = path_vars["dsm"].get().strip()
            output_value = path_vars["output"].get().strip()
            strategy_value = path_vars["strategy"].get().strip() or "core"

            if not lidar_value:
                raise FileNotFoundError("Please select a LiDAR parquet file.")
            if not dsm_value:
                raise FileNotFoundError("Please select a DSM LAS/LAZ file.")

            lidar_path = Path(lidar_value)
            dsm_path = Path(dsm_value)
            meta_path = Path(meta_value) if meta_value else None
            output_dir = Path(output_value) if output_value else lidar_path.parent

            if not lidar_path.is_file():
                raise FileNotFoundError(f"LiDAR parquet not found: {lidar_path}")
            if not dsm_path.is_file():
                raise FileNotFoundError(f"DSM file not found: {dsm_path}")
            if meta_path is not None and not meta_path.is_file():
                raise FileNotFoundError(f"Metadata parquet not found: {meta_path}")

            result = process_alignment(
                lidar_path,
                meta_path,
                dsm_path,
                output_dir,
                run_gicp_flag=True,
                gicp_strategy=strategy_value,
            )
            metrics_path = result["outputs"]["metrics_json"]
            shift_path = result["outputs"]["shifted"]
            gicp_path = result["outputs"].get("shifted_gicp")
            gicp_info = result.get("gicp_info")
            msg = [
                f"Metrics saved to: {metrics_path}",
                f"Shifted cloud: {shift_path}",
            ]
            if gicp_path:
                msg.append(f"Shifted + GICP cloud: {gicp_path}")
            # Print op1 gating logs if available
            try:
                diag = (gicp_info or {}).get("diagnostics") or {}
                roi = diag.get("roi") or {}
                tgt = diag.get("target") or {}
                counts = diag.get("readable_counts") or {}
                if roi:
                    msg.append(
                        "Source gate: kept={} rejected={} z(p50)={:.3f}".format(
                            roi.get("preselection_final_kept"),
                            roi.get("preselection_final_rejected"),
                            (roi.get("z_summary_kept") or {}).get("p50", float("nan")),
                        )
                    )
                if tgt:
                    tgt_gate = tgt.get("target_gate") or {}
                    msg.append(
                        "Target gate: kept={} rejected={} z(p50)={:.3f}".format(
                            tgt_gate.get("final_kept"),
                            tgt_gate.get("final_rejected"),
                            (tgt_gate.get("z_summary_kept") or {}).get("p50", float("nan")),
                        )
                    )
                if counts:
                    msg.append(
                        "Counts: src_orig={} src_kept={} tgt_orig={} tgt_crop={} tgt_kept={}".format(
                            counts.get("source_original_count"),
                            counts.get("source_final_kept_count"),
                            counts.get("target_original_count"),
                            counts.get("target_after_crop_count"),
                            counts.get("target_final_kept_count"),
                        )
                    )
            except Exception:
                pass
            messagebox.showinfo("Completed", "\n".join(msg))
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Error", str(exc))
    labels = {
        "lidar": "LiDAR parquet",
        "meta": "Metadata parquet (optional)",
        "dsm": "DSM LAZ",
        "output": "Output directory",
        "strategy": "GICP strategy",
    }

    for idx, key in enumerate(["lidar", "meta", "dsm", "output", "strategy"]):
        tk.Label(root, text=labels[key]).grid(row=idx, column=0, sticky="w", padx=6, pady=4)
        if key == "strategy":
            options = ["core", "op1"]
            om = tk.OptionMenu(root, path_vars[key], *options)
            om.config(width=20)
            om.grid(row=idx, column=1, padx=6, pady=4, sticky="w")
            button = tk.Label(root, text=" ")
        else:
            entry = tk.Entry(root, textvariable=path_vars[key], width=60)
            entry.grid(row=idx, column=1, padx=6, pady=4)
            if key == "output":
                button = tk.Button(root, text="Browse", command=browse_dir)
            else:
                ft = [("Parquet", "*.parquet")] if key != "dsm" else [("LAS/LAZ", "*.las *.laz"), ("All", "*.*")]
                button = tk.Button(root, text="Browse", command=lambda k=key, f=ft: browse_file(k, f))
        button.grid(row=idx, column=2, padx=6, pady=4)

    tk.Button(root, text="Run", command=run_processing).grid(row=5, column=0, columnspan=3, pady=12)
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Align LiDAR sweep to DSM by vertical shift and optional GICP refinement.")
    parser.add_argument("--lidar", type=Path, help="Path to LiDAR parquet file")
    parser.add_argument("--meta", type=Path, help="Path to metadata parquet file (optional; required for sensor-frame inputs)")
    parser.add_argument("--dsm", type=Path, help="Path to DSM LAS/LAZ file")
    parser.add_argument("--output-dir", type=Path, help="Directory for outputs")
    parser.add_argument("--skip-gicp", action="store_true", help="Skip the GICP refinement step")
    parser.add_argument("--gicp-strategy", choices=["core", "op1"], default="op1", help="Choose GICP strategy: 'core' or 'op1'")
    parser.add_argument("--gui", action="store_true", help="Launch GUI even if paths are provided")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gui or not (args.lidar and args.dsm):
        launch_gui()
        return

    lidar_path = args.lidar
    dsm_path = args.dsm
    meta_path = args.meta

    if not lidar_path.is_file():
        raise FileNotFoundError(f"LiDAR parquet not found: {lidar_path}")
    if not dsm_path.is_file():
        raise FileNotFoundError(f"DSM file not found: {dsm_path}")
    if meta_path is not None and not meta_path.is_file():
        raise FileNotFoundError(f"Metadata parquet not found: {meta_path}")

    output_dir = args.output_dir or lidar_path.parent
    result = process_alignment(
        lidar_path,
        meta_path,
        dsm_path,
        output_dir,
        run_gicp_flag=not args.skip_gicp,
        gicp_strategy=args.gicp_strategy,
    )
    metrics = result["metrics"]
    metrics_path = result["outputs"]["metrics_json"]
    print("Metrics saved to:", metrics_path)
    print("Vertical offset applied (m):", metrics["vertical_offset_applied"])
    print("Shifted cloud saved to:", result["outputs"]["shifted"])
    gicp_info = result.get("gicp_info")
    gicp_path = result["outputs"].get("shifted_gicp")
    if gicp_info and gicp_path:
        print("Shifted + GICP cloud saved to:", gicp_path)
        if "registration_fitness" in gicp_info:
            print("GICP fitness:", gicp_info["registration_fitness"], "RMSE:", gicp_info["registration_rmse"])
        else:
            diag = (gicp_info.get("diagnostics") or {}).get("gicp") or {}
            if diag:
                print("GICP fitness:", diag.get("fitness"), "RMSE:", diag.get("rmse"))
        print("GICP horizontal translation (m):", gicp_info["horizontal_translation_m"])
    elif not args.skip_gicp:
        print("GICP output unavailable; see metrics file for details.")

if __name__ == "__main__":
    main()



