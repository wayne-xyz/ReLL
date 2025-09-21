"""GUI for previewing imagery, DSM, and LiDAR macro-sweep outputs."""
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

# ensure PROJ uses rasterio packaged data
os.environ.setdefault("PROJ_LIB", str(Path(rasterio.__file__).parent / "proj_data"))


def read_lidar_points(point_path: Path) -> np.ndarray:
    table = pq.read_table(point_path, columns=["x", "y", "z", "intensity"])
    points = np.column_stack([
        table["x"].to_numpy(zero_copy_only=False),
        table["y"].to_numpy(zero_copy_only=False),
        table["z"].to_numpy(zero_copy_only=False),
    ]).astype(np.float64)
    return points


def load_height_color(points: np.ndarray, intensities: np.ndarray | None = None) -> np.ndarray:
    if intensities is not None:
        norm = (intensities - intensities.min()) / (intensities.ptp() + 1e-6)
        return np.stack([norm, norm, 1.0 - norm], axis=1).astype(np.float64)
    z = points[:, 2]
    norm = (z - z.min()) / (z.ptp() + 1e-6)
    return np.stack([norm, 1.0 - norm, 0.5 * np.ones_like(norm)], axis=1).astype(np.float64)


def open_imagery(path: Path) -> None:
    try:
        with rasterio.open(path) as src:
            data = src.read()
    except Exception as exc:
        messagebox.showerror("Failed to open imagery", str(exc))
        return
    plt.figure(figsize=(6, 6))
    if data.shape[0] >= 3:
        img = np.stack([data[i] for i in range(3)], axis=-1).astype(np.float32)
        if img.max() > 0:
            img /= img.max()
        plt.imshow(np.clip(img, 0, 1))
    else:
        plt.imshow(data[0], cmap="gray")
    plt.title(path.name)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def open_lidar(point_path: Path) -> None:
    try:
        table = pq.read_table(point_path, columns=["x", "y", "z", "intensity"])
    except Exception as exc:
        messagebox.showerror("Failed to read LiDAR parquet", str(exc))
        return
    points = np.column_stack([
        table["x"].to_numpy(zero_copy_only=False),
        table["y"].to_numpy(zero_copy_only=False),
        table["z"].to_numpy(zero_copy_only=False),
    ]).astype(np.float64)
    intensity = table["intensity"].to_numpy(zero_copy_only=False).astype(np.float64)
    if points.shape[0] > 500000:
        idx = np.random.choice(points.shape[0], 500000, replace=False)
        points = points[idx]
        intensity = intensity[idx]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(load_height_color(points, intensity).astype(np.float64))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    o3d.visualization.draw_geometries([cloud, frame], window_name=point_path.name, width=1280, height=800)


def open_dsm(path: Path) -> None:
    if not path.exists():
        messagebox.showerror("Missing DSM", "Please select a valid LAZ/LAS file.")
        return
    try:
        las = laspy.read(path)
    except Exception as exc:
        messagebox.showerror("Failed to read DSM", str(exc))
        return
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    if points.shape[0] == 0:
        messagebox.showinfo("Empty DSM", "No points found in file.")
        return
    if points.shape[0] > 500000:
        idx = np.random.choice(points.shape[0], 500000, replace=False)
        points = points[idx]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color([0.6, 0.6, 0.6])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    o3d.visualization.draw_geometries([cloud, frame], window_name=path.name, width=1280, height=800)


class SimpleViewer:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title("Sample Previewer")
        self.window.geometry("640x260")

        self.imagery_var = tk.StringVar()
        self.dsm_var = tk.StringVar()
        self.points_var = tk.StringVar()

        self._build_layout()

    def _build_layout(self) -> None:
        frame = tk.Frame(self.window)
        frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(frame, text="Imagery GeoTIFF:").grid(row=0, column=0, sticky=tk.W, pady=4)
        tk.Entry(frame, textvariable=self.imagery_var, width=50).grid(row=0, column=1, padx=5, sticky=tk.W)
        tk.Button(frame, text="Browse", command=self.choose_imagery).grid(row=0, column=2)
        tk.Button(frame, text="View", command=self.view_imagery).grid(row=0, column=3, padx=5)

        tk.Label(frame, text="DSM LAZ/LAS:").grid(row=1, column=0, sticky=tk.W, pady=4)
        tk.Entry(frame, textvariable=self.dsm_var, width=50).grid(row=1, column=1, padx=5, sticky=tk.W)
        tk.Button(frame, text="Browse", command=self.choose_dsm).grid(row=1, column=2)
        tk.Button(frame, text="View", command=self.view_dsm).grid(row=1, column=3, padx=5)

        tk.Label(frame, text="LiDAR parquet:").grid(row=2, column=0, sticky=tk.W, pady=4)
        tk.Entry(frame, textvariable=self.points_var, width=50).grid(row=2, column=1, padx=5, sticky=tk.W)
        tk.Button(frame, text="Browse", command=self.choose_points).grid(row=2, column=2)
        tk.Button(frame, text="View", command=self.view_points).grid(row=2, column=3, padx=5)

        tk.Button(self.window, text="Quit", command=self.window.destroy).pack(pady=10)

    def choose_imagery(self) -> None:
        path = filedialog.askopenfilename(title="Select imagery GeoTIFF", filetypes=[("GeoTIFF", "*.tif"), ("All files", "*.*")])
        if path:
            self.imagery_var.set(path)

    def choose_dsm(self) -> None:
        path = filedialog.askopenfilename(title="Select DSM LAZ/LAS", filetypes=[("LAZ/LAS", "*.laz;*.las"), ("All files", "*.*")])
        if path:
            self.dsm_var.set(path)

    def choose_points(self) -> None:
        path = filedialog.askopenfilename(title="Select LiDAR parquet", filetypes=[("Parquet", "*.parquet"), ("All files", "*.*")])
        if path:
            self.points_var.set(path)

    def view_imagery(self) -> None:
        path = Path(self.imagery_var.get())
        if not path.exists():
            messagebox.showerror("Missing file", "Please select a valid GeoTIFF file")
            return
        open_imagery(path)

    def view_dsm(self) -> None:
        path = Path(self.dsm_var.get())
        if not path.exists():
            messagebox.showerror("Missing file", "Please select a valid DSM file")
            return
        open_dsm(path)

    def view_points(self) -> None:
        path = Path(self.points_var.get())
        if not path.exists():
            messagebox.showerror("Missing file", "Please select a valid LiDAR parquet")
            return
        open_lidar(path)

    def run(self) -> None:
        self.window.mainloop()


if __name__ == "__main__":
    SimpleViewer().run()
