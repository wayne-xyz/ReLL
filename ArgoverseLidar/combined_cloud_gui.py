"""Simple viewer for combined LiDAR macro-sweep parquet datasets."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pyarrow.parquet as pq
import open3d as o3d


PALETTE = np.array([
    [13, 110, 253],  # blue
    [25, 135, 84],   # green
    [220, 53, 69],   # red
    [255, 193, 7],   # yellow
    [111, 66, 193],  # purple
    [32, 201, 151],  # teal
    [255, 133, 27],  # orange
], dtype=np.float32) / 255.0


class MacroSweepViewer:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title("Combined LiDAR Viewer")
        self.window.geometry("760x520")

        self.point_path_var = tk.StringVar()
        self.meta_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a point parquet file to begin.")
        self.metadata: dict[str, str] | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        top = tk.Frame(self.window)
        top.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(top, text="Point parquet:").grid(row=0, column=0, sticky=tk.W)
        entry_points = tk.Entry(top, textvariable=self.point_path_var, width=70)
        entry_points.grid(row=0, column=1, padx=5, sticky=tk.W)
        tk.Button(top, text="Browse", command=self.browse_points).grid(row=0, column=2, sticky=tk.W)

        tk.Label(top, text="Metadata parquet:").grid(row=1, column=0, sticky=tk.W)
        entry_meta = tk.Entry(top, textvariable=self.meta_path_var, width=70)
        entry_meta.grid(row=1, column=1, padx=5, sticky=tk.W)
        tk.Button(top, text="Browse", command=self.browse_metadata).grid(row=1, column=2, sticky=tk.W)

        btn_frame = tk.Frame(self.window)
        btn_frame.pack(fill=tk.X, padx=10)
        tk.Button(btn_frame, text="Load Metadata", command=self.load_metadata).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Open Point Cloud", command=self.open_point_cloud).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Quit", command=self.window.destroy).pack(side=tk.RIGHT)

        meta_frame = tk.LabelFrame(self.window, text="Metadata")
        meta_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("Key", "Value")
        self.tree = ttk.Treeview(meta_frame, columns=columns, show="headings")
        self.tree.heading("Key", text="Key")
        self.tree.heading("Value", text="Value")
        self.tree.column("Key", width=220, anchor=tk.W)
        self.tree.column("Value", width=480, anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True)

        status_bar = tk.Label(self.window, textvariable=self.status_var, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

    def browse_points(self) -> None:
        path = filedialog.askopenfilename(
            title="Select point parquet",
            filetypes=[("Parquet", "*.parquet"), ("All files", "*.*")],
        )
        if path:
            self.point_path_var.set(path)
            self._guess_metadata_path()

    def browse_metadata(self) -> None:
        path = filedialog.askopenfilename(
            title="Select metadata parquet",
            filetypes=[("Parquet", "*.parquet"), ("All files", "*.*")],
        )
        if path:
            self.meta_path_var.set(path)

    def _guess_metadata_path(self) -> None:
        point_path = Path(self.point_path_var.get())
        if point_path.suffix != ".parquet":
            return
        candidate = point_path.with_name(point_path.stem + "_meta").with_suffix(".parquet")
        if candidate.exists():
            self.meta_path_var.set(str(candidate))

    def load_metadata(self) -> None:
        meta_path = Path(self.meta_path_var.get())
        if not meta_path.exists():
            messagebox.showerror("Metadata missing", "Select a valid metadata parquet file")
            return
        table = pq.read_table(meta_path)
        data = table.to_pydict()
        if not data:
            messagebox.showwarning("Empty metadata", "Metadata parquet contained no rows")
            return
        # Assume single row
        cleaned: dict[str, str] = {}
        for key, values in data.items():
            value = values[0] if isinstance(values, list) else values
            if isinstance(value, (list, dict)):
                cleaned[key] = json.dumps(value)
            else:
                cleaned[key] = str(value)
        self.metadata = cleaned
        for row in self.tree.get_children():
            self.tree.delete(row)
        for key, value in cleaned.items():
            self.tree.insert("", tk.END, values=(key, value))
        self.status_var.set(f"Loaded metadata from {meta_path.name}")

    def open_point_cloud(self) -> None:
        point_path = Path(self.point_path_var.get())
        if not point_path.exists():
            messagebox.showerror("Point file missing", "Select a valid point parquet file")
            return
        try:
            table = pq.read_table(point_path)
        except Exception as exc:
            messagebox.showerror("Failed to read parquet", str(exc))
            return
        required_columns = {"x", "y", "z"}
        if not required_columns.issubset(set(table.column_names)):
            messagebox.showerror("Invalid data", "Point parquet missing x/y/z columns")
            return

        points = np.column_stack([
            table["x"].to_numpy(zero_copy_only=False),
            table["y"].to_numpy(zero_copy_only=False),
            table["z"].to_numpy(zero_copy_only=False),
        ]).astype(np.float32)
        if points.size == 0:
            messagebox.showinfo("Empty point cloud", "No points to visualise")
            return

        if "source_index" in table.column_names:
            src = table["source_index"].to_numpy(zero_copy_only=False).astype(np.int32)
            palette = PALETTE
            colors = palette[src % len(palette)]
        else:
            intensity = table["intensity"].to_numpy(zero_copy_only=False).astype(np.float32)
            norm = (intensity - intensity.min()) / (intensity.ptp() + 1e-6)
            colors = np.stack([norm, norm, 1.0 - norm], axis=1)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        origin.paint_uniform_color([1.0, 0.0, 0.0])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

        geometries = [cloud, origin, frame]

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=point_path.name, width=1280, height=800)
        for geom in geometries:
            vis.add_geometry(geom)

        render_opt = vis.get_render_option()
        render_opt.point_size = 1.5

        view_ctl = vis.get_view_control()
        bbox = cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = max(bbox.get_extent().max(), 1.0)
        view_ctl.set_lookat(center)
        view_ctl.set_up([0.0, 0.0, 1.0])
        view_ctl.set_front([0.0, -1.0, 0.0])
        zoom = min(1.0, max(0.02, 40.0 / extent))
        view_ctl.set_zoom(zoom)

        vis.run()
        vis.destroy_window()

    def run(self) -> None:
        self.window.mainloop()


if __name__ == "__main__":
    MacroSweepViewer().run()
