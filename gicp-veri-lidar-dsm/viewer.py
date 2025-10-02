"""
Interactive viewer for comparing LiDAR-DSM alignment results.

Allows selecting:
- LiDAR: Original / Shifted / Aligned
- DSM: Original / Extracted
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import open3d as o3d

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False


# Color scheme
COLOR_LIDAR_ORIGINAL = [1.0, 0.3, 0.3]      # Red
COLOR_LIDAR_SHIFTED = [1.0, 0.6, 0.2]       # Orange
COLOR_LIDAR_ALIGNED = [0.3, 0.8, 0.3]       # Green
COLOR_DSM_ORIGINAL = [0.2, 0.5, 1.0]        # Blue
COLOR_DSM_EXTRACTED = [0.5, 0.3, 1.0]       # Purple


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


def load_laz_points(path: Path) -> Optional[np.ndarray]:
    """Load point cloud from LAZ/LAS file.

    Returns:
        Nx3 array or None if failed
    """
    if not HAS_LASPY:
        print("Error: laspy not installed, cannot load LAZ/LAS files")
        return None

    try:
        las = laspy.read(path)
        points = np.column_stack([las.x, las.y, las.z]).astype(float)
        print(f"Loaded {len(points):,} points from {path}")
        return points

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def create_point_cloud(points: np.ndarray, color: List[float]) -> o3d.geometry.PointCloud:
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
        self.root.title("LiDAR-DSM Alignment Viewer")

        # File paths
        self.lidar_original_path = Path("data3_utm.parquet")
        self.lidar_shifted_path = Path("lidar_shifted.parquet")
        self.lidar_aligned_path = Path("lidar_aligned.parquet")
        self.dsm_original_path = Path("data3_dsm_utm.laz")
        self.dsm_extracted_path = Path("dsm_extracted.parquet")

        # State
        self.lidar_selection = tk.StringVar(value="shifted")
        self.dsm_selection = tk.StringVar(value="extracted")
        self.show_lidar = tk.BooleanVar(value=True)
        self.show_dsm = tk.BooleanVar(value=True)

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title = ttk.Label(main_frame, text="LiDAR-DSM Alignment Viewer", font=("", 14, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=10)

        # LiDAR section
        lidar_frame = ttk.LabelFrame(main_frame, text="LiDAR", padding="10")
        lidar_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Checkbutton(lidar_frame, text="Show LiDAR", variable=self.show_lidar).grid(row=0, column=0, sticky=tk.W)

        ttk.Radiobutton(lidar_frame, text="Original (red)", variable=self.lidar_selection, value="original").grid(row=1, column=0, sticky=tk.W, padx=20)
        ttk.Radiobutton(lidar_frame, text="Shifted (orange)", variable=self.lidar_selection, value="shifted").grid(row=2, column=0, sticky=tk.W, padx=20)
        ttk.Radiobutton(lidar_frame, text="Aligned (green)", variable=self.lidar_selection, value="aligned").grid(row=3, column=0, sticky=tk.W, padx=20)

        # DSM section
        dsm_frame = ttk.LabelFrame(main_frame, text="DSM", padding="10")
        dsm_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Checkbutton(dsm_frame, text="Show DSM", variable=self.show_dsm).grid(row=0, column=0, sticky=tk.W)

        ttk.Radiobutton(dsm_frame, text="Original (blue)", variable=self.dsm_selection, value="original").grid(row=1, column=0, sticky=tk.W, padx=20)
        ttk.Radiobutton(dsm_frame, text="Extracted (purple)", variable=self.dsm_selection, value="extracted").grid(row=2, column=0, sticky=tk.W, padx=20)

        # File paths display
        paths_frame = ttk.LabelFrame(main_frame, text="File Paths", padding="10")
        paths_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        paths_text = (
            f"LiDAR Original: {self.lidar_original_path}\n"
            f"LiDAR Shifted: {self.lidar_shifted_path}\n"
            f"LiDAR Aligned: {self.lidar_aligned_path}\n"
            f"DSM Original: {self.dsm_original_path}\n"
            f"DSM Extracted: {self.dsm_extracted_path}"
        )
        ttk.Label(paths_frame, text=paths_text, font=("Courier", 9)).grid(row=0, column=0, sticky=tk.W)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="View", command=self.view_clouds).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).grid(row=0, column=1, padx=5)

        # Legend
        legend_frame = ttk.LabelFrame(main_frame, text="Color Legend", padding="10")
        legend_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        legend_text = (
            "Red: LiDAR Original | Orange: LiDAR Shifted | Green: LiDAR Aligned\n"
            "Blue: DSM Original | Purple: DSM Extracted"
        )
        ttk.Label(legend_frame, text=legend_text, font=("", 9)).grid(row=0, column=0)

    def view_clouds(self):
        """Load and visualize selected point clouds."""
        geometries = []

        # Load LiDAR
        if self.show_lidar.get():
            lidar_choice = self.lidar_selection.get()

            if lidar_choice == "original":
                points = load_parquet_points(self.lidar_original_path)
                color = COLOR_LIDAR_ORIGINAL
                name = "LiDAR Original"
            elif lidar_choice == "shifted":
                points = load_parquet_points(self.lidar_shifted_path)
                color = COLOR_LIDAR_SHIFTED
                name = "LiDAR Shifted"
            elif lidar_choice == "aligned":
                points = load_parquet_points(self.lidar_aligned_path)
                color = COLOR_LIDAR_ALIGNED
                name = "LiDAR Aligned"
            else:
                points = None
                name = "Unknown"

            if points is not None and points.size > 0:
                pc = create_point_cloud(points, color)
                geometries.append(pc)
                print(f"Added {name}: {len(points):,} points")
            else:
                print(f"Warning: Could not load {name}")

        # Load DSM
        if self.show_dsm.get():
            dsm_choice = self.dsm_selection.get()

            if dsm_choice == "original":
                points = load_laz_points(self.dsm_original_path)
                color = COLOR_DSM_ORIGINAL
                name = "DSM Original"
            elif dsm_choice == "extracted":
                points = load_parquet_points(self.dsm_extracted_path)
                color = COLOR_DSM_EXTRACTED
                name = "DSM Extracted"
            else:
                points = None
                name = "Unknown"

            if points is not None and points.size > 0:
                pc = create_point_cloud(points, color)
                geometries.append(pc)
                print(f"Added {name}: {len(points):,} points")
            else:
                print(f"Warning: Could not load {name}")

        # Visualize
        if not geometries:
            messagebox.showwarning("No Data", "No point clouds loaded. Check file paths and selections.")
            return

        print(f"\nVisualizing {len(geometries)} point clouds...")
        print("Close the viewer window to return to the GUI.\n")

        o3d.visualization.draw_geometries(
            geometries,
            window_name="LiDAR-DSM Alignment Viewer",
            width=1200,
            height=800,
        )


def main():
    root = tk.Tk()
    app = ViewerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
