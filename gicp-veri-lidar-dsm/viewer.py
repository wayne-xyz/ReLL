"""
Interactive viewer for comparing point cloud files.

Allows selecting any two parquet files for comparison.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import open3d as o3d


# Color scheme for two files
COLOR_FILE_1 = [1.0, 0.3, 0.3]      # Red
COLOR_FILE_2 = [0.2, 0.5, 1.0]      # Blue


def load_parquet_points(path: Path) -> Optional[np.ndarray]:
    """Load point cloud from parquet file.

    Returns:
        Nx3 array or None if failed
    """
    try:
        df = pd.read_parquet(path)

        # Try to find XYZ columns
        if all(col in df.columns for col in ["utm_e", "utm_n", "elevation"]):
            points = df[["utm_e", "utm_n", "elevation"]].to_numpy(dtype=float)
        elif all(col in df.columns for col in ["utm_e", "utm_n", "z"]):
            points = df[["utm_e", "utm_n", "z"]].to_numpy(dtype=float)
        elif all(col in df.columns for col in ["x", "y", "z"]):
            points = df[["x", "y", "z"]].to_numpy(dtype=float)
        else:
            print(f"Warning: Could not find XYZ columns in {path}")
            return None

        print(f"Loaded {len(points):,} points from {path}")
        return points

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def create_point_cloud(points: np.ndarray, color: list[float]) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud with uniform color.

    Args:
        points: Nx3 array
        color: RGB color [r, g, b] in range [0, 1]

    Returns:
        Open3D PointCloud
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.paint_uniform_color(color)
    return pc


class ViewerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Point Cloud Viewer")

        # File paths
        self.file1_path = tk.StringVar(value="")
        self.file2_path = tk.StringVar(value="")

        # State
        self.show_file1 = tk.BooleanVar(value=True)
        self.show_file2 = tk.BooleanVar(value=True)

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="Point Cloud Viewer", font=("", 14, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=10)

        # File 1 section
        file1_frame = ttk.LabelFrame(main_frame, text="File 1 (Red)", padding="10")
        file1_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Checkbutton(file1_frame, text="Show File 1", variable=self.show_file1).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(file1_frame, textvariable=self.file1_path, width=60).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(file1_frame, text="Browse", command=self.browse_file1).grid(row=1, column=1, padx=5, pady=5)

        # File 2 section
        file2_frame = ttk.LabelFrame(main_frame, text="File 2 (Blue)", padding="10")
        file2_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Checkbutton(file2_frame, text="Show File 2", variable=self.show_file2).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(file2_frame, textvariable=self.file2_path, width=60).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(file2_frame, text="Browse", command=self.browse_file2).grid(row=1, column=1, padx=5, pady=5)

        # Legend
        legend_frame = ttk.LabelFrame(main_frame, text="Color Legend", padding="10")
        legend_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        legend_text = "Red: File 1  |  Blue: File 2"
        ttk.Label(legend_frame, text=legend_text, font=("", 10)).grid(row=0, column=0)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="View", command=self.view_clouds, width=15).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit, width=15).grid(row=0, column=1, padx=5)

    def browse_file1(self):
        """Browse for File 1."""
        path = filedialog.askopenfilename(
            title="Select Point Cloud File 1",
            filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            self.file1_path.set(path)

    def browse_file2(self):
        """Browse for File 2."""
        path = filedialog.askopenfilename(
            title="Select Point Cloud File 2",
            filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            self.file2_path.set(path)

    def view_clouds(self):
        """Load and visualize selected point clouds."""
        geometries = []

        # Load File 1
        if self.show_file1.get():
            file1_str = self.file1_path.get().strip()
            if file1_str:
                file1_path = Path(file1_str)
                if file1_path.is_file():
                    points = load_parquet_points(file1_path)
                    if points is not None and points.size > 0:
                        pc = create_point_cloud(points, COLOR_FILE_1)
                        geometries.append(pc)
                        print(f"Loaded File 1: {file1_path.name} ({len(points):,} points)")
                    else:
                        print(f"Warning: Could not load File 1: {file1_path}")
                else:
                    messagebox.showwarning("File Not Found", f"File 1 does not exist:\n{file1_path}")

        # Load File 2
        if self.show_file2.get():
            file2_str = self.file2_path.get().strip()
            if file2_str:
                file2_path = Path(file2_str)
                if file2_path.is_file():
                    points = load_parquet_points(file2_path)
                    if points is not None and points.size > 0:
                        pc = create_point_cloud(points, COLOR_FILE_2)
                        geometries.append(pc)
                        print(f"Loaded File 2: {file2_path.name} ({len(points):,} points)")
                    else:
                        print(f"Warning: Could not load File 2: {file2_path}")
                else:
                    messagebox.showwarning("File Not Found", f"File 2 does not exist:\n{file2_path}")

        # Visualize
        if not geometries:
            messagebox.showwarning("No Data", "No point clouds loaded. Please select files and ensure they exist.")
            return

        print(f"\nVisualizing {len(geometries)} point cloud(s)...")
        print("Close the viewer window to return to the GUI.\n")

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Point Cloud Viewer",
            width=1200,
            height=800,
        )


def main():
    root = tk.Tk()
    app = ViewerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
