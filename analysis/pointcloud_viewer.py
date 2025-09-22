
import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import laspy
import numpy as np
import open3d as o3d
import pandas as pd


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


def log_bounds(name: str, mins: np.ndarray, maxs: np.ndarray) -> None:
    span = maxs - mins
    print(
        f"{name} bounds (width ? length ? height): "
        f"{span[0]:.2f} m ? {span[1]:.2f} m ? {span[2]:.2f} m"
    )
    print(
        f"{name} min [E={mins[0]:.3f}, N={mins[1]:.3f}, Z={mins[2]:.2f}] | "
        f"max [E={maxs[0]:.3f}, N={maxs[1]:.3f}, Z={maxs[2]:.2f}]"
    )


def infer_lidar_points(lidar_path: Path, meta_path: Path | None) -> tuple[np.ndarray, dict]:
    print(f"Loading LiDAR parquet: {lidar_path}")
    df = pd.read_parquet(lidar_path)

    info: dict[str, object] = {
        "frame": "unknown",
        "utm_zone": None,
        "utm_hemisphere": None,
        "quaternion": None,
        "translation": None,
    }

    if {"utm_e", "utm_n", "elevation"}.issubset(df.columns):
        print("Detected utm_e/utm_n/elevation; treating LiDAR coordinates as UTM.")
        pts = df[["utm_e", "utm_n", "elevation"]].to_numpy()
        info["frame"] = "UTM columns"
    elif meta_path is not None:
        print(f"Using metadata {meta_path} to convert LiDAR to UTM.")
        meta = pd.read_parquet(meta_path).iloc[0]
        quaternion = [meta["center_city_qw"], meta["center_city_qx"], meta["center_city_qy"], meta["center_city_qz"]]
        translation = [meta["center_city_tx_m"], meta["center_city_ty_m"], meta["center_city_tz_m"]]
        info["quaternion"] = quaternion
        info["translation"] = translation
        zone = meta.get("center_utm_zone")
        if isinstance(zone, str) and zone:
            info["utm_zone"] = zone
            info["utm_hemisphere"] = zone[-1].upper()

        rotation = quat_to_rot(*quaternion)
        translation_vec = np.array(translation, dtype=float)
        if not {"x", "y", "z"}.issubset(df.columns):
            raise ValueError("LiDAR parquet lacks x,y,z columns required for metadata transform.")
        xyz_local = df[["x", "y", "z"]].to_numpy()
        xyz_city = (rotation @ xyz_local.T).T + translation_vec

        city_positions = np.array(json.loads(meta["sensor_positions_city_m"]))
        utm_positions = np.array([[e, n] for _, e, n in json.loads(meta["sensor_positions_utm"])])
        coeff_e, coeff_n = fit_affine(city_positions[:, :2], utm_positions)
        utm_e = apply_affine(coeff_e, xyz_city[:, :2])
        utm_n = apply_affine(coeff_n, xyz_city[:, :2])
        pts = np.column_stack([utm_e, utm_n, xyz_city[:, 2]])
        info["frame"] = "UTM via metadata"
    elif {"x", "y", "z"}.issubset(df.columns):
        print("LiDAR missing UTM columns and metadata; visualising raw x/y/z coordinates (sensor frame).")
        pts = df[["x", "y", "z"]].to_numpy()
        info["frame"] = "sensor frame"
    else:
        raise ValueError("Unable to infer LiDAR coordinate columns; provide metadata or standard fields.")

    pts = np.asarray(pts, dtype=float)
    if pts.size:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        log_bounds("LiDAR", mins, maxs)
        info["bounds_min"] = mins.tolist()
        info["bounds_max"] = maxs.tolist()
        info["bounds_span"] = (maxs - mins).tolist()
    else:
        print("LiDAR dataset is empty.")
    return pts, info


def load_dsm_points(dsm_path: Path) -> tuple[np.ndarray, dict]:
    print(f"Loading DSM file: {dsm_path}")
    las = laspy.read(dsm_path)
    pts = np.column_stack([las.x, las.y, las.z])
    info: dict[str, object] = {
        "epsg": las.header.epsg,
        "utm_zone": None,
        "utm_hemisphere": None,
        "crs": None,
    }

    crs = las.header.parse_crs()
    if crs is not None:
        info["crs"] = crs.to_string()
        print(f"DSM CRS: {crs.to_string()} (EPSG: {crs.to_epsg()})")
    else:
        print("DSM CRS metadata unavailable; relying on file naming conventions.")

    epsg = las.header.epsg
    if epsg is not None:
        if 32601 <= epsg <= 32660:
            info["utm_zone"] = f"{epsg - 32600}N"
            info["utm_hemisphere"] = "N"
        elif 32701 <= epsg <= 32760:
            info["utm_zone"] = f"{epsg - 32700}S"
            info["utm_hemisphere"] = "S"
    if info["utm_zone"]:
        print(f"DSM UTM zone inferred as {info['utm_zone']}")

    pts = np.asarray(pts, dtype=float)
    if pts.size:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        log_bounds("DSM", mins, maxs)
        info["bounds_min"] = mins.tolist()
        info["bounds_max"] = maxs.tolist()
        info["bounds_span"] = (maxs - mins).tolist()
    else:
        print("DSM dataset is empty.")
    return pts, info


def create_o3d_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array(color, dtype=np.float64), (len(points), 1))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def visualize_points(lidar_points: np.ndarray, dsm_points: np.ndarray) -> None:
    geometries: list[o3d.geometry.Geometry] = []
    if len(lidar_points):
        geometries.append(create_o3d_cloud(lidar_points, (1.0, 0.2, 0.2)))
    if len(dsm_points):
        geometries.append(create_o3d_cloud(dsm_points, (0.2, 0.8, 0.2)))
    if not geometries:
        raise ValueError("No points available to visualize.")
    o3d.visualization.draw_geometries(geometries)


def run_viewer(lidar_path: Path, meta_path: Path | None, dsm_path: Path) -> None:
    lidar_pts, lidar_info = infer_lidar_points(lidar_path, meta_path)
    dsm_pts, dsm_info = load_dsm_points(dsm_path)

    lidar_zone = lidar_info.get("utm_zone")
    dsm_zone = dsm_info.get("utm_zone")
    print(f"LiDAR frame: {lidar_info.get('frame')}")
    if lidar_zone:
        hemi = lidar_info.get("utm_hemisphere")
        hemi_text = f" ({hemi} hemisphere)" if hemi else ""
        print(f"LiDAR UTM zone: {lidar_zone}{hemi_text}")
    if lidar_info.get("quaternion"):
        print(f"LiDAR center quaternion (w,x,y,z): {tuple(lidar_info['quaternion'])}")
    if lidar_info.get("translation"):
        print(f"LiDAR center translation (city frame, m): {tuple(lidar_info['translation'])}")
    if dsm_zone:
        hemi = dsm_info.get("utm_hemisphere")
        hemi_text = f" ({hemi} hemisphere)" if hemi else ""
        print(f"DSM UTM zone: {dsm_zone}{hemi_text}")
    if lidar_zone and dsm_zone and lidar_zone != dsm_zone:
        print("Warning: LiDAR and DSM report different UTM zones; verify inputs.")

    lidar_min = lidar_info.get("bounds_min")
    lidar_max = lidar_info.get("bounds_max")
    dsm_min = dsm_info.get("bounds_min")
    dsm_max = dsm_info.get("bounds_max")
    if lidar_min and lidar_max and dsm_min and dsm_max:
        lidar_center = (np.array(lidar_min[:2]) + np.array(lidar_max[:2])) * 0.5
        dsm_center = (np.array(dsm_min[:2]) + np.array(dsm_max[:2])) * 0.5
        center_delta = np.linalg.norm(lidar_center - dsm_center)
        print(f"LiDAR vs DSM XY center offset ? {center_delta:.2f} m")
        east_min_delta = abs(lidar_min[0] - dsm_min[0])
        east_max_delta = abs(lidar_max[0] - dsm_max[0])
        north_min_delta = abs(lidar_min[1] - dsm_min[1])
        north_max_delta = abs(lidar_max[1] - dsm_max[1])
        print(
            "Boundary deltas | east min/max: {:.1f} m / {:.1f} m | north min/max: {:.1f} m / {:.1f} m".format(
                east_min_delta, east_max_delta, north_min_delta, north_max_delta
            )
        )
    else:
        print("Insufficient information to compare spatial bounds.")

    visualize_points(lidar_pts, dsm_pts)


def launch_gui() -> None:
    root = tk.Tk()
    root.title("LiDAR + DSM Viewer")

    path_vars = {
        "lidar": tk.StringVar(),
        "meta": tk.StringVar(),
        "dsm": tk.StringVar(),
    }

    def browse_file(var: tk.StringVar, filetypes) -> None:
        initialdir = Path(var.get() or Path.cwd())
        selection = filedialog.askopenfilename(parent=root, initialdir=initialdir, filetypes=filetypes)
        if selection:
            var.set(selection)

    def run_view() -> None:
        try:
            lidar_path = Path(path_vars["lidar"].get())
            dsm_path = Path(path_vars["dsm"].get())
            meta_value = path_vars["meta"].get().strip()
            meta_path = Path(meta_value) if meta_value else None
            if not lidar_path.is_file() or not dsm_path.is_file():
                raise FileNotFoundError("Select existing LiDAR parquet and DSM LAS/LAZ files.")
            run_viewer(lidar_path, meta_path, dsm_path)
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

    tk.Button(root, text="View", command=run_view).grid(row=3, column=0, columnspan=3, pady=12)
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LiDAR parquet and DSM LAZ together.")
    parser.add_argument("--lidar", type=Path, help="Path to LiDAR parquet file")
    parser.add_argument("--dsm", type=Path, help="Path to DSM LAS/LAZ file")
    parser.add_argument("--meta", type=Path, default=None, help="Optional metadata parquet for LiDAR frame conversion")
    parser.add_argument("--gui", action="store_true", help="Launch GUI file picker")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gui or not (args.lidar and args.dsm):
        launch_gui()
        return

    if not args.lidar.is_file():
        raise FileNotFoundError(f"LiDAR parquet not found: {args.lidar}")
    if not args.dsm.is_file():
        raise FileNotFoundError(f"DSM file not found: {args.dsm}")
    if args.meta is not None and not args.meta.is_file():
        raise FileNotFoundError(f"Metadata parquet not found: {args.meta}")

    run_viewer(args.lidar, args.meta, args.dsm)


if __name__ == "__main__":
    main()
