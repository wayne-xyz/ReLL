"""GUI for previewing sample imagery GeoTIFFs, DSM LAZ point clouds, and LiDAR macro-sweep points."""
from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import rowcol

# ensure PROJ uses rasterio packaged data
os.environ.setdefault("PROJ_LIB", str(Path(rasterio.__file__).parent / "proj_data"))


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def fit_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve for uniform scale + rotation + translation mapping src -> dst."""
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = float(np.sum(S) / np.sum(src_centered ** 2))
    translation = centroid_dst - scale * centroid_src @ R
    return scale, R, translation


def fit_city_to_utm(meta: dict) -> tuple[float, np.ndarray, np.ndarray]:
    city_positions = np.array(json.loads(meta["sensor_positions_city_m"][0]))[:, :2]
    utm_entries = json.loads(meta["sensor_positions_utm"][0])
    utm_positions = []
    for item in utm_entries:
        if item is None or len(item) < 3:
            continue
        _, east, north = item
        utm_positions.append([float(east), float(north)])
    utm_positions = np.array(utm_positions)
    if utm_positions.shape[0] < 2:
        raise ValueError("Metadata does not contain enough UTM references")
    return fit_rigid_transform(city_positions, utm_positions)


def transform_points(meta: dict, point_table: pq.Table) -> dict[str, np.ndarray]:
    qw = float(meta["center_city_qw"][0])
    qx = float(meta["center_city_qx"][0])
    qy = float(meta["center_city_qy"][0])
    qz = float(meta["center_city_qz"][0])
    rot = quaternion_to_matrix(qw, qx, qy, qz)
    trans = np.array([
        float(meta["center_city_tx_m"][0]),
        float(meta["center_city_ty_m"][0]),
        float(meta["center_city_tz_m"][0]),
    ])

    points = point_table.to_pandas().values.astype(np.float64)
    city_points = (rot @ points.T).T + trans

    scale, rotation, translation = fit_city_to_utm(meta)
    city_xy = city_points[:, :2]
    utm_xy = scale * (city_xy @ rotation) + translation
    utm_points = np.column_stack([utm_xy, city_points[:, 2]])

    return {
        "city": city_points,
        "utm": utm_points,
    }


class ImageryDsmViewer:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title("Imagery / DSM / LiDAR Viewer")
        self.window.geometry("760x300")

        self.imagery_var = tk.StringVar()
        self.dsm_var = tk.StringVar()
        self.points_var = tk.StringVar()
        self.meta_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select files to preview.")

        self._cached_points: dict | None = None
        self._cached_key: tuple[str, str] | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        frame = tk.Frame(self.window)
        frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(frame, text="Imagery GeoTIFF:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(frame, textvariable=self.imagery_var, width=60).grid(row=0, column=1, padx=5, sticky=tk.W)
        tk.Button(frame, text="Browse", command=self.browse_imagery).grid(row=0, column=2)

        tk.Label(frame, text="DSM LAZ/LAS:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        tk.Entry(frame, textvariable=self.dsm_var, width=60).grid(row=1, column=1, padx=5, sticky=tk.W, pady=(6, 0))
        tk.Button(frame, text="Browse", command=self.browse_dsm).grid(row=1, column=2, pady=(6, 0))

        tk.Label(frame, text="LiDAR points (parquet):").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        tk.Entry(frame, textvariable=self.points_var, width=60).grid(row=2, column=1, padx=5, sticky=tk.W, pady=(6, 0))
        tk.Button(frame, text="Browse", command=self.browse_points).grid(row=2, column=2, pady=(6, 0))

        tk.Label(frame, text="Metadata parquet:").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        tk.Entry(frame, textvariable=self.meta_var, width=60).grid(row=3, column=1, padx=5, sticky=tk.W, pady=(6, 0))
        tk.Button(frame, text="Browse", command=self.browse_meta).grid(row=3, column=2, pady=(6, 0))

        btn_frame = tk.Frame(self.window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(btn_frame, text="Open Imagery", command=self.open_imagery).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Open DSM", command=self.open_dsm).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Quit", command=self.window.destroy).pack(side=tk.RIGHT)

        tk.Label(self.window, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, padx=10, pady=(0, 10))

    def browse_imagery(self) -> None:
        path = filedialog.askopenfilename(title="Select GeoTIFF", filetypes=[("GeoTIFF", "*.tif"), ("All files", "*.*")])
        if path:
            self.imagery_var.set(path)

    def browse_dsm(self) -> None:
        path = filedialog.askopenfilename(title="Select DSM", filetypes=[("LiDAR", "*.laz;*.las"), ("All files", "*.*")])
        if path:
            self.dsm_var.set(path)

    def browse_points(self) -> None:
        path = filedialog.askopenfilename(title="Select point cloud parquet", filetypes=[("Parquet", "*.parquet"), ("All files", "*.*")])
        if path:
            self.points_var.set(path)
            meta_candidate = Path(path).with_name(Path(path).stem + "_meta").with_suffix(".parquet")
            if meta_candidate.exists():
                self.meta_var.set(str(meta_candidate))

    def browse_meta(self) -> None:
        path = filedialog.askopenfilename(title="Select metadata parquet", filetypes=[("Parquet", "*.parquet"), ("All files", "*.*")])
        if path:
            self.meta_var.set(path)

    def _ensure_point_data(self) -> dict | None:
        point_path = Path(self.points_var.get())
        meta_path = Path(self.meta_var.get())
        if not point_path.exists() or not meta_path.exists():
            messagebox.showerror("Point data missing", "Select both the point parquet and metadata parquet.")
            return None
        key = (str(point_path), str(meta_path))
        if self._cached_key == key and self._cached_points is not None:
            return self._cached_points
        try:
            table = pq.read_table(point_path, columns=["x", "y", "z"])
            meta_table = pq.read_table(meta_path)
        except Exception as exc:
            messagebox.showerror("Failed to load parquet", str(exc))
            return None
        meta = meta_table.to_pydict()
        try:
            data = transform_points(meta, table)
        except Exception as exc:
            messagebox.showerror("Transform error", str(exc))
            return None
        data["meta"] = meta
        self._cached_points = data
        self._cached_key = key
        return data

    def open_imagery(self) -> None:
        path = Path(self.imagery_var.get())
        if not path.exists():
            messagebox.showerror("Missing imagery", "Please select a valid GeoTIFF file.")
            return
        point_data = self._ensure_point_data()
        if point_data is None:
            return
        try:
            with rasterio.open(path) as src:
                data = src.read()
                if data.shape[0] >= 3:
                    img = np.stack([data[i] for i in range(3)], axis=-1)
                else:
                    img = data[0]
                plt.figure(figsize=(6, 6))
                if img.ndim == 3:
                    norm = img.astype(np.float32)
                    if norm.max() > 0:
                        norm /= norm.max()
                    plt.imshow(np.clip(norm, 0, 1))
                else:
                    plt.imshow(img, cmap='gray')
                utm_points = point_data["utm"]
                if utm_points.size:
                    sample = utm_points
                    if sample.shape[0] > 200000:
                        idx = np.random.choice(sample.shape[0], 200000, replace=False)
                        sample = sample[idx]
                    cols, rows = rowcol(src.transform, sample[:, 0], sample[:, 1], op=float)
                    mask = (
                        (cols >= 0) & (cols < src.width) &
                        (rows >= 0) & (rows < src.height)
                    )
                    if np.any(mask):
                        plt.scatter(cols[mask], rows[mask], s=1, c='red', alpha=0.5)
                plt.title(path.name)
                plt.gca().invert_yaxis()
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                self.status_var.set(f"Displayed imagery {path.name} with LiDAR overlay")
        except Exception as exc:
            messagebox.showerror("Failed to load imagery", str(exc))

    def open_dsm(self) -> None:
        path = Path(self.dsm_var.get())
        if not path.exists():
            messagebox.showerror("Missing DSM", "Please select a valid LAZ/LAS file.")
            return
        point_data = self._ensure_point_data()
        if point_data is None:
            return
        try:
            las = laspy.read(path)
        except Exception as exc:
            messagebox.showerror("Failed to read DSM", str(exc))
            return
        dsm_coords = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
        if dsm_coords.size == 0:
            messagebox.showinfo("Empty DSM", "No points found in file.")
            return
        if dsm_coords.shape[0] > 500000:
            idx = np.random.choice(dsm_coords.shape[0], 500000, replace=False)
            dsm_coords = dsm_coords[idx]
        lidar_coords = point_data["utm"]
        if lidar_coords.shape[0] > 500000:
            idx = np.random.choice(lidar_coords.shape[0], 500000, replace=False)
            lidar_coords = lidar_coords[idx]
        dsm_cloud = o3d.geometry.PointCloud()
        dsm_cloud.points = o3d.utility.Vector3dVector(dsm_coords)
        dsm_cloud.paint_uniform_color([0.6, 0.6, 0.6])
        lidar_cloud = o3d.geometry.PointCloud()
        lidar_cloud.points = o3d.utility.Vector3dVector(lidar_coords)
        lidar_cloud.paint_uniform_color([1.0, 0.0, 0.0])
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(dsm_cloud.points)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=bbox.get_center())
        o3d.visualization.draw_geometries([dsm_cloud, lidar_cloud, frame], window_name=path.name, width=1280, height=800)
        self.status_var.set(f"Displayed DSM {path.name} with LiDAR overlay")

    def run(self) -> None:
        self.window.mainloop()


if __name__ == '__main__':
    ImageryDsmViewer().run()
