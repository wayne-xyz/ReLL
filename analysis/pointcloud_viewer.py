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

DEFAULT_LIDAR_SAMPLE = 200000
DEFAULT_DSM_SAMPLE = 400000
DEFAULT_MARGIN = 20.0


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


def downsample_points(points: np.ndarray, sample_size: int | None) -> np.ndarray:
    if sample_size is None or sample_size <= 0 or len(points) <= sample_size:
        return points.astype(np.float64)
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(points), size=sample_size, replace=False)
    return points[idx].astype(np.float64)


def infer_lidar_points(lidar_path: Path, meta_path: Path | None, sample_size: int | None) -> np.ndarray:
    print(f"Loading LiDAR parquet: {lidar_path}")
    df = pd.read_parquet(lidar_path)

    if {"utm_e", "utm_n", "elevation"}.issubset(df.columns):
        print("Detected utm_e/utm_n/elevation columns; treating LiDAR coordinates as UTM.")
        pts = df[["utm_e", "utm_n", "elevation"]].to_numpy()
    elif meta_path is not None:
        print(f"Using metadata {meta_path} to convert LiDAR to UTM.")
        meta = pd.read_parquet(meta_path).iloc[0]
        rotation = quat_to_rot(meta["center_city_qw"], meta["center_city_qx"], meta["center_city_qy"], meta["center_city_qz"])
        translation = np.array([
            meta["center_city_tx_m"],
            meta["center_city_ty_m"],
            meta["center_city_tz_m"],
        ])
        if not {"x", "y", "z"}.issubset(df.columns):
            raise ValueError("LiDAR parquet lacks x,y,z columns required for meta-based transform.")
        xyz_local = df[["x", "y", "z"]].to_numpy()
        xyz_city = (rotation @ xyz_local.T).T + translation

        city_positions = np.array(json.loads(meta["sensor_positions_city_m"]))
        utm_positions = np.array([[e, n] for _, e, n in json.loads(meta["sensor_positions_utm"])])
        coeff_e, coeff_n = fit_affine(city_positions[:, :2], utm_positions)
        utm_e = apply_affine(coeff_e, xyz_city[:, :2])
        utm_n = apply_affine(coeff_n, xyz_city[:, :2])
        pts = np.column_stack([utm_e, utm_n, xyz_city[:, 2]])
    elif {"x", "y", "z"}.issubset(df.columns):
        print("LiDAR lacks UTM columns and metadata; visualising raw x/y/z coordinates (likely sensor frame).")
        pts = df[["x", "y", "z"]].to_numpy()
    else:
        raise ValueError("Unable to infer LiDAR coordinate columns; provide metadata or standard fields.")

    pts = downsample_points(pts, sample_size)
    if pts.size:
        print(f"LiDAR sample size: {len(pts)} | min={pts.min(axis=0)}, max={pts.max(axis=0)}")
    else:
        print("LiDAR point set is empty after sampling.")
    return pts


def load_dsm_points(dsm_path: Path, bbox: tuple[np.ndarray, np.ndarray] | None, sample_size: int | None, margin: float) -> np.ndarray:
    print(f"Loading DSM file: {dsm_path}")
    las = laspy.read(dsm_path)
    pts = np.column_stack([las.x, las.y, las.z])
    if pts.size:
        print(f"DSM extent full set | min={pts.min(axis=0)}, max={pts.max(axis=0)}")
    else:
        print("DSM file contained no points.")

    if bbox is not None and pts.size:
        print(f"Clipping DSM to LiDAR XY bounds with {margin} m margin.")
        mins, maxs = bbox
        mins = mins.copy()
        maxs = maxs.copy()
        mins[:2] -= margin
        maxs[:2] += margin
        mask = (
            (pts[:, 0] >= mins[0]) & (pts[:, 0] <= maxs[0]) &
            (pts[:, 1] >= mins[1]) & (pts[:, 1] <= maxs[1])
        )
        pts = pts[mask]

    pts = downsample_points(pts, sample_size)
    if pts.size:
        print(f"DSM sample size: {len(pts)} | min={pts.min(axis=0)}, max={pts.max(axis=0)}")
    else:
        print("DSM point set is empty after sampling.")
    return pts


def create_o3d_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array(color, dtype=np.float64), (len(points), 1))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def visualize_points(lidar_points: np.ndarray, dsm_points: np.ndarray) -> None:
    geometries = []
    if len(lidar_points):
        geometries.append(create_o3d_cloud(lidar_points, (1.0, 0.2, 0.2)))
    if len(dsm_points):
        geometries.append(create_o3d_cloud(dsm_points, (0.2, 0.8, 0.2)))
    if not geometries:
        raise ValueError("No points available to visualize.")
    o3d.visualization.draw_geometries(geometries)


def run_viewer(lidar_path: Path, meta_path: Path | None, dsm_path: Path, output_sample: tuple[int | None, int | None], margin: float) -> None:
    lidar_pts = infer_lidar_points(lidar_path, meta_path, sample_size=output_sample[0])
    bbox = (lidar_pts.min(axis=0), lidar_pts.max(axis=0)) if len(lidar_pts) else None
    dsm_pts = load_dsm_points(dsm_path, bbox=bbox, sample_size=output_sample[1], margin=margin)
    visualize_points(lidar_pts, dsm_pts)


def launch_gui():
    root = tk.Tk()
    root.title("LiDAR + DSM Viewer")

    path_vars = {
        "lidar": tk.StringVar(),
        "meta": tk.StringVar(),
        "dsm": tk.StringVar(),
        "lidar_sample": tk.StringVar(value=str(DEFAULT_LIDAR_SAMPLE)),
        "dsm_sample": tk.StringVar(value=str(DEFAULT_DSM_SAMPLE)),
        "margin": tk.StringVar(value=str(DEFAULT_MARGIN)),
    }

    def browse_file(var: tk.StringVar, filetypes):
        initialdir = Path(var.get() or Path.cwd())
        selection = filedialog.askopenfilename(parent=root, initialdir=initialdir, filetypes=filetypes)
        if selection:
            var.set(selection)

    def run_view():
        try:
            lidar_path = Path(path_vars["lidar"].get())
            dsm_path = Path(path_vars["dsm"].get())
            meta_value = path_vars["meta"].get().strip()
            meta_path = Path(meta_value) if meta_value else None
            if not lidar_path.is_file() or not dsm_path.is_file():
                raise FileNotFoundError("Select existing LiDAR parquet and DSM LAS/LAZ files.")
            lidar_sample = parse_optional_int(path_vars["lidar_sample"].get())
            dsm_sample = parse_optional_int(path_vars["dsm_sample"].get())
            margin_val = float(path_vars["margin"].get())
            run_viewer(lidar_path, meta_path, dsm_path, (lidar_sample, dsm_sample), margin_val)
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Error", str(exc))

    rows = [
        ("LiDAR parquet", "lidar", [("Parquet", "*.parquet"), ("All", "*.*")]),
        ("Metadata parquet (optional)", "meta", [("Parquet", "*.parquet"), ("All", "*.*")]),
        ("DSM LAZ/LAS", "dsm", [("LAS/LAZ", "*.laz *.las"), ("All", "*.*")]),
    ]

    for idx, (label, key, filetypes) in enumerate(rows):
        tk.Label(root, text=label).grid(row=idx, column=0, sticky="w", padx=6, pady=4)
        tk.Entry(root, textvariable=path_vars[key], width=60).grid(row=idx, column=1, padx=6, pady=4)
        tk.Button(root, text="Browse", command=lambda v=path_vars[key], ft=filetypes: browse_file(v, ft)).grid(row=idx, column=2, padx=6, pady=4)

    tk.Label(root, text="LiDAR sample (blank = all)").grid(row=3, column=0, sticky="w", padx=6, pady=4)
    tk.Entry(root, textvariable=path_vars["lidar_sample"], width=15).grid(row=3, column=1, sticky="w", padx=6, pady=4)

    tk.Label(root, text="DSM sample (blank = all)").grid(row=4, column=0, sticky="w", padx=6, pady=4)
    tk.Entry(root, textvariable=path_vars["dsm_sample"], width=15).grid(row=4, column=1, sticky="w", padx=6, pady=4)

    tk.Label(root, text="Bounding margin (m)").grid(row=5, column=0, sticky="w", padx=6, pady=4)
    tk.Entry(root, textvariable=path_vars["margin"], width=15).grid(row=5, column=1, sticky="w", padx=6, pady=4)

    tk.Button(root, text="View", command=run_view).grid(row=6, column=0, columnspan=3, pady=12)
    root.mainloop()


def parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LiDAR parquet and DSM LAZ together.")
    parser.add_argument("--lidar", type=Path, help="Path to LiDAR parquet file")
    parser.add_argument("--dsm", type=Path, help="Path to DSM LAS/LAZ file")
    parser.add_argument("--meta", type=Path, default=None, help="Optional metadata parquet for LiDAR frame conversion")
    parser.add_argument("--lidar-sample", type=int, default=DEFAULT_LIDAR_SAMPLE, help="LiDAR random sample size (<=0 for all)")
    parser.add_argument("--dsm-sample", type=int, default=DEFAULT_DSM_SAMPLE, help="DSM random sample size (<=0 for all)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="Extra metres to include around LiDAR XY bounds")
    parser.add_argument("--gui", action="store_true", help="Launch GUI file picker")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gui or not (args.lidar and args.dsm):
        launch_gui()
        return

    lidar_sample = args.lidar_sample if args.lidar_sample and args.lidar_sample > 0 else None
    dsm_sample = args.dsm_sample if args.dsm_sample and args.dsm_sample > 0 else None
    run_viewer(args.lidar, args.meta, args.dsm, (lidar_sample, dsm_sample), args.margin)


if __name__ == "__main__":
    main()
