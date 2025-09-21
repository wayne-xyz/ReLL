"""Rasterize a LiDAR macro-sweep parquet into UTM-aligned GeoTIFF bands."""
from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_SAMPLE_DIR = BASE_DIR / "Data-Sample"
POINTS_PATH = DATA_SAMPLE_DIR / "combined_0p5s.parquet"
META_PATH = DATA_SAMPLE_DIR / "combined_0p5s_meta.parquet"
GRID_RESOLUTION = 0.2  # metres per pixel
NODATA_VALUE = -9999.0
BUFFER_METERS = 0.3


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def utm_zone_to_crs(zone: str) -> CRS:
    zone = zone.strip()
    if not zone:
        raise ValueError("UTM zone string is empty; cannot derive CRS")
    number = int(zone[:-1])
    hemisphere = zone[-1].upper()
    epsg = 32600 + number if hemisphere == "N" else 32700 + number
    return CRS.from_epsg(epsg)


def load_metadata(points_path: Path, meta_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CRS]:
    table = pq.read_table(meta_path)
    meta = table.to_pydict()
    translation = np.array([
        meta["center_city_tx_m"][0],
        meta["center_city_ty_m"][0],
        meta["center_city_tz_m"][0],
    ], dtype=np.float64)
    quat = np.array([
        meta["center_city_qw"][0],
        meta["center_city_qx"][0],
        meta["center_city_qy"][0],
        meta["center_city_qz"][0],
    ], dtype=np.float64)
    city_positions = json.loads(meta["sensor_positions_city_m"][0])
    utm_positions_raw = json.loads(meta["sensor_positions_utm"][0])

    city_xy = []
    utm_xy = []
    utm_zone = meta["center_utm_zone"][0]
    for city, utm in zip(city_positions, utm_positions_raw):
        if utm is None:
            continue
        zone, easting, northing = utm
        if not utm_zone:
            utm_zone = zone
        city_xy.append(city[:2])
        utm_xy.append([easting, northing])
    if not city_xy or not utm_xy:
        raise RuntimeError("Metadata missing paired city/UTM positions; cannot estimate transform")

    city_xy = np.array(city_xy, dtype=np.float64)
    utm_xy = np.array(utm_xy, dtype=np.float64)
    X = np.column_stack([city_xy, np.ones((city_xy.shape[0], 1))])
    A, _, _, _ = np.linalg.lstsq(X, utm_xy, rcond=None)

    crs = utm_zone_to_crs(utm_zone if utm_zone else utm_positions_raw[0][0])
    return translation, quat, A, city_xy, crs


def load_points(points_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(points_path, columns=["x", "y", "z", "intensity"])
    cols = {name: table[name].to_numpy(zero_copy_only=False) for name in ["x", "y", "z", "intensity"]}
    points = np.column_stack([
        cols["x"].astype(np.float64),
        cols["y"].astype(np.float64),
        cols["z"].astype(np.float64),
    ])
    intensity = cols["intensity"].astype(np.float32)
    return points, intensity


def project_to_utm(points_ref: np.ndarray, translation: np.ndarray, quat: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = quaternion_to_matrix(*quat)
    points_city = points_ref @ rotation.T + translation
    xy = points_city[:, :2]
    xy_h = np.column_stack([xy, np.ones((xy.shape[0], 1))])
    utm_xy = xy_h @ A
    return utm_xy[:, 0], utm_xy[:, 1], points_city[:, 2]


def rasterize(easting: np.ndarray, northing: np.ndarray, height: np.ndarray, intensity: np.ndarray, crs: CRS, output_tif: Path) -> None:
    min_e = float(easting.min() - BUFFER_METERS)
    max_e = float(easting.max() + BUFFER_METERS)
    min_n = float(northing.min() - BUFFER_METERS)
    max_n = float(northing.max() + BUFFER_METERS)

    width = int(np.ceil((max_e - min_e) / GRID_RESOLUTION)) + 1
    height_px = int(np.ceil((max_n - min_n) / GRID_RESOLUTION)) + 1

    cols = np.floor((easting - min_e) / GRID_RESOLUTION).astype(np.int64)
    rows = np.floor((max_n - northing) / GRID_RESOLUTION).astype(np.int64)
    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height_px - 1)

    linear_idx = rows * width + cols
    order = np.argsort(linear_idx)
    lin_sorted = linear_idx[order]
    z_sorted = height[order]
    intensity_sorted = intensity[order]

    unique_idx, start_idx = np.unique(lin_sorted, return_index=True)
    counts = np.diff(np.append(start_idx, lin_sorted.size))

    max_z = np.maximum.reduceat(z_sorted, start_idx)
    sum_intensity = np.add.reduceat(intensity_sorted, start_idx)
    mean_intensity = sum_intensity / counts

    height_raster = np.full(width * height_px, NODATA_VALUE, dtype=np.float32)
    intensity_raster = np.full(width * height_px, NODATA_VALUE, dtype=np.float32)
    mask_raster = np.zeros(width * height_px, dtype=np.uint8)

    height_raster[unique_idx] = max_z.astype(np.float32)
    intensity_raster[unique_idx] = mean_intensity.astype(np.float32)
    mask_raster[unique_idx] = 255

    height_grid = height_raster.reshape((height_px, width))
    intensity_grid = intensity_raster.reshape((height_px, width))
    mask_grid = mask_raster.reshape((height_px, width))

    transform = from_origin(min_e, max_n, GRID_RESOLUTION, GRID_RESOLUTION)
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=height_px,
        width=width,
        count=2,
        dtype="float32",
        crs=crs.to_wkt(),
        transform=transform,
        nodata=NODATA_VALUE,
        compress="zstd",
    ) as dst:
        dst.write(height_grid, 1)
        dst.write(intensity_grid, 2)
        dst.write_mask(mask_grid)


def rasterize_macro_sweep(points_path: Path, meta_path: Path, output_tif: Path) -> Path:
    translation, quat, A, _city_xy, crs = load_metadata(points_path, meta_path)
    points_ref, intensity = load_points(points_path)
    easting, northing, height = project_to_utm(points_ref, translation, quat, A)
    rasterize(easting, northing, height, intensity, crs, output_tif)
    return output_tif


def suggested_output_path(points_path: Path) -> Path:
    return points_path.with_name(points_path.stem + "_height_intensity.tif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rasterize a macro-sweep parquet to GeoTIFF")
    parser.add_argument("--points", type=Path, help="Path to combined points parquet")
    parser.add_argument("--meta", type=Path, help="Path to metadata parquet")
    parser.add_argument("--output", type=Path, help="Output GeoTIFF path")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
    return parser.parse_args()


def cli_main(args: argparse.Namespace) -> None:
    if args.points is None or args.meta is None:
        raise ValueError("--points and --meta are required unless --gui is set")
    points_path = args.points.resolve()
    meta_path = args.meta.resolve()
    output_path = args.output.resolve() if args.output else suggested_output_path(points_path)
    rasterize_macro_sweep(points_path, meta_path, output_path)
    print(f"Saved raster to {output_path}")


class RasterizerApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Macro-Sweep Rasterizer")
        self.root.geometry("520x260")

        self.folder_var = tk.StringVar()
        self.point_var = tk.StringVar()
        self.meta_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a folder containing macro-sweep parquets.")

        self._build_layout()

    def _build_layout(self) -> None:
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Folder selection
        row0 = tk.Frame(frame)
        row0.pack(fill=tk.X, pady=4)
        tk.Label(row0, text="Folder:", width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row0, textvariable=self.folder_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        tk.Button(row0, text="Browse", command=self.choose_folder).pack(side=tk.LEFT)

        # Point parquet
        row1 = tk.Frame(frame)
        row1.pack(fill=tk.X, pady=4)
        tk.Label(row1, text="Points:", width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row1, textvariable=self.point_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        tk.Button(row1, text="Browse", command=self.choose_points).pack(side=tk.LEFT)

        # Meta parquet
        row2 = tk.Frame(frame)
        row2.pack(fill=tk.X, pady=4)
        tk.Label(row2, text="Metadata:", width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row2, textvariable=self.meta_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        tk.Button(row2, text="Browse", command=self.choose_meta).pack(side=tk.LEFT)

        # Output
        row3 = tk.Frame(frame)
        row3.pack(fill=tk.X, pady=4)
        tk.Label(row3, text="Output:", width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row3, textvariable=self.output_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        tk.Button(row3, text="Browse", command=self.choose_output).pack(side=tk.LEFT)

        # Buttons
        btn_row = tk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=10)
        tk.Button(btn_row, text="Rasterize", command=self.run).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Quit", command=self.root.destroy).pack(side=tk.RIGHT)

        status = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill=tk.X, padx=10, pady=(0, 10))

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder with macro-sweep parquets")
        if folder:
            self.folder_var.set(folder)
            self.auto_populate(Path(folder))

    def choose_points(self) -> None:
        path = filedialog.askopenfilename(
            title="Select points parquet",
            filetypes=[("Parquet", "*.parquet"), ("All files", "*.*")],
            initialdir=self.folder_var.get() or None,
        )
        if path:
            self.point_var.set(path)
            self.guess_metadata(Path(path))
            self.guess_output(Path(path))

    def choose_meta(self) -> None:
        path = filedialog.askopenfilename(
            title="Select metadata parquet",
            filetypes=[("Parquet", "*.parquet"), ("All files", "*.*")],
            initialdir=self.folder_var.get() or None,
        )
        if path:
            self.meta_var.set(path)

    def choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select output GeoTIFF",
            defaultextension=".tif",
            filetypes=[("GeoTIFF", "*.tif"), ("All files", "*.*")],
            initialdir=self.folder_var.get() or None,
        )
        if path:
            self.output_var.set(path)

    def auto_populate(self, folder: Path) -> None:
        parquet_files = sorted(folder.glob("*.parquet"))
        points = None
        meta = None
        for file in parquet_files:
            if file.name.endswith("_meta.parquet"):
                if meta is None:
                    meta = file
            else:
                if points is None:
                    points = file
        if points:
            self.point_var.set(str(points))
            self.guess_output(points)
            if meta is None:
                self.guess_metadata(points)
        if meta:
            self.meta_var.set(str(meta))

    def guess_metadata(self, point_path: Path) -> None:
        candidate = point_path.with_name(point_path.stem + "_meta").with_suffix(".parquet")
        if candidate.exists():
            self.meta_var.set(str(candidate))

    def guess_output(self, point_path: Path) -> None:
        self.output_var.set(str(suggested_output_path(point_path)))

    def run(self) -> None:
        try:
            point_path = Path(self.point_var.get())
            meta_path = Path(self.meta_var.get())
            if not point_path.exists() or not meta_path.exists():
                raise FileNotFoundError("Point or metadata parquet not found")
            output_path = Path(self.output_var.get()) if self.output_var.get() else suggested_output_path(point_path)
            rasterize_macro_sweep(point_path, meta_path, output_path)
            self.status_var.set(f"Saved raster to {output_path}")
            messagebox.showinfo("Success", f"GeoTIFF written to\n{output_path}")
        except Exception as exc:  # pragma: no cover - GUI path
            messagebox.showerror("Rasterization failed", str(exc))
            self.status_var.set("Rasterization failed")

    def start(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    if args.gui or (args.points is None or args.meta is None):
        app = RasterizerApp()
        app.start()
    else:
        cli_main(args)


if __name__ == "__main__":
    main()
