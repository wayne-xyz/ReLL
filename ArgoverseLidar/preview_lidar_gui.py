"""GUI for browsing Argoverse 2 LiDAR feather files and previewing them with Open3D."""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import open3d as o3d
import pyarrow.feather as feather
import tkinter as tk
from tkinter import filedialog, messagebox

DEFAULT_ROOT = Path("..") / "argverse_data_preview"


@dataclass
class PointCloud:
    points: np.ndarray
    intensity: np.ndarray


def load_point_cloud(feather_path: Path, *, stride: int) -> PointCloud:
    table = feather.read_table(feather_path, columns=["x", "y", "z", "intensity"])
    frame = table.to_pandas()
    points = frame[["x", "y", "z"]].to_numpy(dtype=np.float32)
    intensity = frame["intensity"].to_numpy(dtype=np.float32)
    if stride > 1:
        points = points[::stride]
        intensity = intensity[::stride]
    return PointCloud(points=points, intensity=intensity)


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return intensity
    finite_mask = np.isfinite(intensity)
    if not np.any(finite_mask):
        return np.zeros_like(intensity)
    finite_values = intensity[finite_mask]
    low = float(finite_values.min())
    high = float(finite_values.max())
    if high - low <= 1e-6:
        scaled = np.zeros_like(intensity)
        scaled[finite_mask] = 0.5
        return scaled
    scaled = np.zeros_like(intensity)
    scaled[finite_mask] = (finite_values - low) / (high - low)
    return scaled


def intensity_to_color(intensity: np.ndarray) -> np.ndarray:
    scaled = normalize_intensity(intensity)
    colors = np.zeros((scaled.size, 3), dtype=np.float32)
    colors[:, 0] = scaled
    colors[:, 1] = scaled
    colors[:, 2] = 1.0 - 0.5 * scaled
    return colors


def visualize(cloud: PointCloud, title: str) -> None:
    if cloud.points.size == 0:
        raise ValueError("No points found in frame.")
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(cloud.points)
    geometry.colors = o3d.utility.Vector3dVector(intensity_to_color(cloud.intensity))
    o3d.visualization.draw_geometries([geometry], window_name=title, width=1280, height=800)


def find_feather_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.feather") if p.is_file())


class PreviewApp:
    def __init__(self, *, root: Path, stride: int) -> None:
        self.root_dir = root
        self.stride = stride
        self.files: List[Path] = []

        self.window = tk.Tk()
        self.window.title("Argoverse LiDAR Preview")
        self.window.geometry("720x480")

        self.path_var = tk.StringVar(value=str(root))
        self.stride_var = tk.IntVar(value=max(1, stride))

        self._build_layout()
        self.refresh_file_list()

    def _build_layout(self) -> None:
        top_frame = tk.Frame(self.window)
        top_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(top_frame, text="Preview folder:").pack(side=tk.LEFT)
        entry = tk.Entry(top_frame, textvariable=self.path_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(top_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)
        tk.Button(top_frame, text="Reload", command=self.refresh_file_list).pack(side=tk.LEFT, padx=(5, 0))

        middle_frame = tk.Frame(self.window)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        scrollbar = tk.Scrollbar(middle_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(middle_frame, activestyle="dotbox")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        control_frame = tk.Frame(self.window)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Label(control_frame, text="Stride:").pack(side=tk.LEFT)
        stride_spin = tk.Spinbox(control_frame, from_=1, to=50, textvariable=self.stride_var, width=5)
        stride_spin.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(control_frame, text="Open Selected", command=self.open_selected).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Open Random", command=self.open_random).pack(side=tk.LEFT, padx=(10, 0))
        tk.Button(control_frame, text="Quit", command=self.window.destroy).pack(side=tk.RIGHT)

    def browse_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.path_var.get() or str(DEFAULT_ROOT))
        if chosen:
            self.path_var.set(chosen)
            self.refresh_file_list()

    def refresh_file_list(self) -> None:
        try:
            root_path = Path(self.path_var.get()).expanduser().resolve()
        except Exception as exc:
            messagebox.showerror("Invalid path", f"Could not resolve folder: {exc}")
            return

        self.root_dir = root_path
        files = find_feather_files(root_path)
        self.files = files
        self.listbox.delete(0, tk.END)
        for file in files:
            try:
                relative = file.relative_to(root_path)
            except ValueError:
                relative = file
            self.listbox.insert(tk.END, str(relative))
        if not files:
            self.listbox.insert(tk.END, "(No .feather files found)")

    def open_selected(self) -> None:
        if not self.files:
            messagebox.showinfo("No files", "No feather files to open.")
            return
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Select a feather file first.")
            return
        index = selection[0]
        file_path = self.files[index]
        self._open_file(file_path)

    def open_random(self) -> None:
        if not self.files:
            messagebox.showinfo("No files", "No feather files to choose from.")
            return
        file_path = random.choice(self.files)
        self.listbox.selection_clear(0, tk.END)
        idx = self.files.index(file_path)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self._open_file(file_path)

    def _open_file(self, path: Path) -> None:
        stride = max(1, self.stride_var.get())
        try:
            cloud = load_point_cloud(path, stride=stride)
            print(f"Loaded {cloud.points.shape[0]} points from {path}")
            visualize(cloud, path.name)
        except Exception as exc:
            messagebox.showerror("Failed to open feather", f"{exc}")

    def run(self) -> None:
        self.window.mainloop()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse and preview Argoverse 2 LiDAR sweeps.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"Folder to search for .feather files (default: {DEFAULT_ROOT})")
    parser.add_argument("--stride", type=int, default=1,
                        help="Decimate the point cloud by keeping one of every N points (default: %(default)s).")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    app = PreviewApp(root=args.root.resolve(), stride=args.stride)
    app.run()


if __name__ == "__main__":
    main()
