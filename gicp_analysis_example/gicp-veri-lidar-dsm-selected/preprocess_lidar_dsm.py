"""
Preprocess LiDAR and DSM data for GICP alignment.

This script performs two main tasks:
1. Vertical shift: Align LiDAR Z values to DSM using center point as anchor
2. DSM crop: Extract DSM points within 0.5m of any LiDAR point
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import laspy
from scipy.spatial import cKDTree


def load_lidar_parquet(path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load LiDAR data from parquet file.

    Returns:
        Tuple of (points as Nx3 array, original dataframe)
    """
    print(f"Loading LiDAR data from {path}")
    df = pd.read_parquet(path)

    # Try to find XYZ columns
    if all(col in df.columns for col in ["utm_e", "utm_n", "elevation"]):
        points = df[["utm_e", "utm_n", "elevation"]].to_numpy(dtype=float)
    elif all(col in df.columns for col in ["utm_e", "utm_n", "z"]):
        points = df[["utm_e", "utm_n", "z"]].to_numpy(dtype=float)
    elif all(col in df.columns for col in ["x", "y", "z"]):
        points = df[["x", "y", "z"]].to_numpy(dtype=float)
    else:
        raise ValueError("Could not find XYZ columns in LiDAR parquet. Expected utm_e/utm_n/elevation or x/y/z")

    print(f"Loaded {len(points):,} LiDAR points")
    print(f"LiDAR bounds: X=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    return points, df


def load_dsm_laz(path: Path) -> np.ndarray:
    """Load DSM data from LAZ/LAS file.

    Returns:
        Points as Nx3 array
    """
    print(f"Loading DSM data from {path}")
    las = laspy.read(path)
    points = np.column_stack([las.x, las.y, las.z]).astype(float)

    print(f"Loaded {len(points):,} DSM points")
    print(f"DSM bounds: X=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    return points


def compute_center_point(points: np.ndarray) -> np.ndarray:
    """Compute center point of the point cloud (mean of XY bounds).

    Args:
        points: Nx3 array

    Returns:
        3D center point [x, y, z_mean]
    """
    xy_min = points[:, :2].min(axis=0)
    xy_max = points[:, :2].max(axis=0)
    xy_center = (xy_min + xy_max) / 2.0
    z_mean = points[:, 2].mean()

    return np.array([xy_center[0], xy_center[1], z_mean], dtype=float)


def query_dsm_height(dsm_points: np.ndarray, x: float, y: float) -> float:
    """Query DSM height at given XY location using nearest neighbor.

    Args:
        dsm_points: Nx3 array of DSM points
        x: X coordinate
        y: Y coordinate

    Returns:
        Z value at nearest DSM point
    """
    tree = cKDTree(dsm_points[:, :2])
    dist, idx = tree.query([x, y], k=1)
    return float(dsm_points[idx, 2])


def vertical_shift_lidar(
    lidar_points: np.ndarray,
    dsm_points: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Shift LiDAR Z values to align with DSM at center point.

    Args:
        lidar_points: Nx3 LiDAR points
        dsm_points: Mx3 DSM points

    Returns:
        Tuple of (shifted LiDAR points, diagnostics dict)
    """
    print("\n=== Step 1: Vertical Shift ===")

    # Compute center point of LiDAR
    lidar_center = compute_center_point(lidar_points)
    print(f"LiDAR center point: X={lidar_center[0]:.3f}, Y={lidar_center[1]:.3f}, Z={lidar_center[2]:.3f}")

    # Query DSM height at center
    dsm_height_at_center = query_dsm_height(dsm_points, lidar_center[0], lidar_center[1])
    print(f"DSM height at center: Z={dsm_height_at_center:.3f}")

    # Compute vertical shift
    vertical_shift = dsm_height_at_center - lidar_center[2]
    print(f"Vertical shift to apply: {vertical_shift:.3f} m")

    # Apply shift
    shifted_lidar = lidar_points.copy()
    shifted_lidar[:, 2] += vertical_shift

    print(f"Shifted LiDAR Z range: [{shifted_lidar[:, 2].min():.2f}, {shifted_lidar[:, 2].max():.2f}]")

    diagnostics = {
        "lidar_center_utm": {
            "utm_e": float(lidar_center[0]),
            "utm_n": float(lidar_center[1]),
            "z_before_shift": float(lidar_center[2]),
        },
        "dsm_height_at_center": float(dsm_height_at_center),
        "vertical_shift_m": float(vertical_shift),
        "shifted_z_range": {
            "min": float(shifted_lidar[:, 2].min()),
            "max": float(shifted_lidar[:, 2].max()),
        },
    }

    return shifted_lidar, diagnostics


def extract_dsm_near_lidar(
    dsm_points: np.ndarray,
    lidar_points: np.ndarray,
    max_distance: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract DSM points within max_distance of any LiDAR point.

    Args:
        dsm_points: Nx3 DSM points
        lidar_points: Mx3 LiDAR points (should be shifted)
        max_distance: Maximum distance threshold in meters

    Returns:
        Tuple of (extracted DSM points, diagnostics dict)
    """
    print(f"\n=== Step 2: DSM Extraction (max distance: {max_distance} m) ===")

    # Build KDTree of LiDAR points (XY only)
    lidar_tree = cKDTree(lidar_points[:, :2])

    # For each DSM point, find nearest LiDAR point
    print("Computing distances from DSM to LiDAR...")
    distances, _ = lidar_tree.query(dsm_points[:, :2], k=1)

    # Filter DSM points
    mask = distances <= max_distance
    extracted_dsm = dsm_points[mask]

    print(f"Original DSM points: {len(dsm_points):,}")
    print(f"Extracted DSM points: {len(extracted_dsm):,}")
    print(f"Reduction: {100 * (1 - len(extracted_dsm) / len(dsm_points)):.1f}%")

    diagnostics = {
        "max_distance_m": float(max_distance),
        "original_dsm_count": int(len(dsm_points)),
        "extracted_dsm_count": int(len(extracted_dsm)),
        "reduction_ratio": float(1 - len(extracted_dsm) / len(dsm_points)),
        "distance_stats": {
            "min": float(distances[mask].min()) if mask.any() else 0.0,
            "max": float(distances[mask].max()) if mask.any() else 0.0,
            "mean": float(distances[mask].mean()) if mask.any() else 0.0,
            "p50": float(np.percentile(distances[mask], 50)) if mask.any() else 0.0,
            "p95": float(np.percentile(distances[mask], 95)) if mask.any() else 0.0,
        },
    }

    return extracted_dsm, diagnostics


def save_parquet(points: np.ndarray, template_df: pd.DataFrame, output_path: Path) -> None:
    """Save points to parquet using template dataframe structure.

    Args:
        points: Nx3 array of points
        template_df: Original dataframe to use as template
        output_path: Path to save parquet file
    """
    df = template_df.copy()

    # Update coordinate columns
    if "utm_e" in df.columns:
        df["utm_e"] = points[:, 0]
        df["utm_n"] = points[:, 1]
        if "elevation" in df.columns:
            df["elevation"] = points[:, 2]
        elif "z" in df.columns:
            df["z"] = points[:, 2]
    elif "x" in df.columns:
        df["x"] = points[:, 0]
        df["y"] = points[:, 1]
        df["z"] = points[:, 2]

    df.to_parquet(output_path, index=False)
    print(f"Saved {len(points):,} points to {output_path}")


def save_dsm_parquet(points: np.ndarray, output_path: Path) -> None:
    """Save DSM points to parquet.

    Args:
        points: Nx3 array of points
        output_path: Path to save parquet file
    """
    df = pd.DataFrame({
        "utm_e": points[:, 0],
        "utm_n": points[:, 1],
        "elevation": points[:, 2],
    })
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(points):,} DSM points to {output_path}")


def run_gui() -> None:
    """Launch GUI for file selection."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("LiDAR-DSM Preprocessing")

    # Variables
    lidar_path_var = tk.StringVar(value="data3_utm.parquet")
    dsm_path_var = tk.StringVar(value="data3_dsm_utm.laz")
    output_dir_var = tk.StringVar(value=".")
    max_distance_var = tk.DoubleVar(value=0.5)

    # Browse functions
    def browse_lidar():
        path = filedialog.askopenfilename(
            title="Select LiDAR Parquet File",
            filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            lidar_path_var.set(path)

    def browse_dsm():
        path = filedialog.askopenfilename(
            title="Select DSM LAZ/LAS File",
            filetypes=[("LAZ/LAS Files", "*.laz *.las"), ("All Files", "*.*")],
            initialdir=Path.cwd(),
        )
        if path:
            dsm_path_var.set(path)

    def browse_output():
        path = filedialog.askdirectory(title="Select Output Directory", initialdir=Path.cwd())
        if path:
            output_dir_var.set(path)

    def run_preprocessing():
        try:
            lidar_path = Path(lidar_path_var.get())
            dsm_path = Path(dsm_path_var.get())
            output_dir = Path(output_dir_var.get())
            max_distance = max_distance_var.get()

            if not lidar_path.is_file():
                raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
            if not dsm_path.is_file():
                raise FileNotFoundError(f"DSM file not found: {dsm_path}")

            output_dir.mkdir(parents=True, exist_ok=True)

            # Close GUI and run processing
            root.destroy()

            # Run the preprocessing
            lidar_points, lidar_df = load_lidar_parquet(lidar_path)
            dsm_points = load_dsm_laz(dsm_path)
            shifted_lidar, shift_diagnostics = vertical_shift_lidar(lidar_points, dsm_points)
            extracted_dsm, extraction_diagnostics = extract_dsm_near_lidar(
                dsm_points, shifted_lidar, max_distance=max_distance
            )

            shifted_lidar_path = output_dir / "lidar_shifted.parquet"
            extracted_dsm_path = output_dir / "dsm_extracted.parquet"
            metrics_path = output_dir / "preprocessing_metrics.json"

            save_parquet(shifted_lidar, lidar_df, shifted_lidar_path)
            save_dsm_parquet(extracted_dsm, extracted_dsm_path)

            metrics = {
                "inputs": {
                    "lidar_file": str(lidar_path),
                    "dsm_file": str(dsm_path),
                    "lidar_points": int(len(lidar_points)),
                    "dsm_points": int(len(dsm_points)),
                },
                "vertical_shift": shift_diagnostics,
                "dsm_extraction": extraction_diagnostics,
                "outputs": {
                    "shifted_lidar": str(shifted_lidar_path),
                    "extracted_dsm": str(extracted_dsm_path),
                },
            }

            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            print(f"\n=== Preprocessing Complete ===")
            print(f"Shifted LiDAR: {shifted_lidar_path}")
            print(f"Extracted DSM: {extracted_dsm_path}")
            print(f"Metrics: {metrics_path}")

            messagebox.showinfo("Success", f"Preprocessing complete!\n\nOutputs:\n{shifted_lidar_path}\n{extracted_dsm_path}\n{metrics_path}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # GUI Layout
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    ttk.Label(main_frame, text="LiDAR-DSM Preprocessing", font=("", 14, "bold")).grid(row=0, column=0, columnspan=3, pady=10)

    # LiDAR file
    ttk.Label(main_frame, text="LiDAR Parquet:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=lidar_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
    ttk.Button(main_frame, text="Browse", command=browse_lidar).grid(row=1, column=2, padx=5, pady=5)

    # DSM file
    ttk.Label(main_frame, text="DSM LAZ/LAS:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=dsm_path_var, width=50).grid(row=2, column=1, padx=5, pady=5)
    ttk.Button(main_frame, text="Browse", command=browse_dsm).grid(row=2, column=2, padx=5, pady=5)

    # Output directory
    ttk.Label(main_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=output_dir_var, width=50).grid(row=3, column=1, padx=5, pady=5)
    ttk.Button(main_frame, text="Browse", command=browse_output).grid(row=3, column=2, padx=5, pady=5)

    # Max distance
    ttk.Label(main_frame, text="Max Distance (m):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    ttk.Entry(main_frame, textvariable=max_distance_var, width=20).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)

    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=5, column=0, columnspan=3, pady=20)
    ttk.Button(button_frame, text="Run Preprocessing", command=run_preprocessing).grid(row=0, column=0, padx=5)
    ttk.Button(button_frame, text="Exit", command=root.quit).grid(row=0, column=1, padx=5)

    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess LiDAR and DSM data for GICP alignment")
    parser.add_argument("--lidar", type=Path, help="LiDAR parquet file")
    parser.add_argument("--dsm", type=Path, help="DSM LAZ/LAS file")
    parser.add_argument("--max-distance", type=float, default=0.5, help="Max distance for DSM extraction (meters)")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument("--gui", action="store_true", help="Launch GUI for file selection")
    args = parser.parse_args()

    # Launch GUI if requested or if no files specified
    if args.gui or not (args.lidar and args.dsm):
        run_gui()
        return

    # Validate inputs
    if not args.lidar.is_file():
        raise FileNotFoundError(f"LiDAR file not found: {args.lidar}")
    if not args.dsm.is_file():
        raise FileNotFoundError(f"DSM file not found: {args.dsm}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    lidar_points, lidar_df = load_lidar_parquet(args.lidar)
    dsm_points = load_dsm_laz(args.dsm)

    # Step 1: Vertical shift
    shifted_lidar, shift_diagnostics = vertical_shift_lidar(lidar_points, dsm_points)

    # Step 2: Extract DSM
    extracted_dsm, extraction_diagnostics = extract_dsm_near_lidar(
        dsm_points,
        shifted_lidar,
        max_distance=args.max_distance
    )

    # Save outputs
    shifted_lidar_path = args.output_dir / "lidar_shifted.parquet"
    extracted_dsm_path = args.output_dir / "dsm_extracted.parquet"
    metrics_path = args.output_dir / "preprocessing_metrics.json"

    save_parquet(shifted_lidar, lidar_df, shifted_lidar_path)
    save_dsm_parquet(extracted_dsm, extracted_dsm_path)

    # Save metrics
    metrics = {
        "inputs": {
            "lidar_file": str(args.lidar),
            "dsm_file": str(args.dsm),
            "lidar_points": int(len(lidar_points)),
            "dsm_points": int(len(dsm_points)),
        },
        "vertical_shift": shift_diagnostics,
        "dsm_extraction": extraction_diagnostics,
        "outputs": {
            "shifted_lidar": str(shifted_lidar_path),
            "extracted_dsm": str(extracted_dsm_path),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Preprocessing Complete ===")
    print(f"Shifted LiDAR: {shifted_lidar_path}")
    print(f"Extracted DSM: {extracted_dsm_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
