"""Tkinter app to generate LiDAR macro-sweeps and aligned imagery/DSM samples."""
from __future__ import annotations

import json
import os
import sys
import threading
from enum import Enum, unique
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.util


def _configure_proj() -> None:
    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    package_path = Path(spec.origin).parent
    proj_data = package_path / "proj_data"
    gdal_data = package_path / "gdal_data"
    os.environ.setdefault("PROJ_LIB", str(proj_data))
    os.environ.setdefault("PROJ_DATA", str(proj_data))
    os.environ.setdefault("GDAL_DATA", str(gdal_data))
    os.environ.setdefault("PROJ_NETWORK", "OFF")


_configure_proj()

import laspy
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import tkinter as tk
from pyproj import CRS, Transformer, Proj

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from combine_lidar_sweeps import create_macro_sweep  # noqa: E402

DEFAULT_SWEEP_COUNT = 5
DEFAULT_BUFFER_METERS = 2.0
DEFAULT_GRID_RESOLUTION = 0.3
DEFAULT_SAMPLE_PREFIX = "combined_0p5s"
DEFAULT_CROP_MIN = 32.0

@unique
class CityName(str, Enum):
    ATX = "ATX"
    DTW = "DTW"
    MIA = "MIA"
    PAO = "PAO"
    PIT = "PIT"
    WDC = "WDC"


UTM_ZONE_MAP: Dict[CityName, int] = {
    CityName.ATX: 14,
    CityName.DTW: 17,
    CityName.MIA: 17,
    CityName.PAO: 10,
    CityName.PIT: 17,
    CityName.WDC: 18,
}

CITY_ORIGIN_LATLONG_DICT: Dict[CityName, Tuple[float, float]] = {
    CityName.ATX: (30.27464237939507, -97.7404457407424),
    CityName.DTW: (42.29993066912924, -83.17555750783717),
    CityName.MIA: (25.77452579915163, -80.19656914449405),
    CityName.PAO: (37.416065, -122.13571963362166),
    CityName.PIT: (40.44177902989321, -80.01294377242584),
    CityName.WDC: (38.889377, -77.0355047439081),
}


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
    if not zone:
        raise ValueError("UTM zone string is empty; cannot determine CRS")
    zone = zone.strip()
    number = int(zone[:-1])
    hemisphere = zone[-1].upper()
    epsg = 32600 + number if hemisphere == "N" else 32700 + number
    return CRS.from_epsg(epsg)


def infer_city_enum(city_name: str) -> CityName:
    if not city_name:
        raise ValueError("City name missing from metadata")
    token = city_name.split("_")[0].upper()
    return CityName[token]


def convert_city_coords_to_utm(points_city: np.ndarray, city: CityName) -> Tuple[np.ndarray, str]:
    latitude, longitude = CITY_ORIGIN_LATLONG_DICT[city]
    zone = UTM_ZONE_MAP[city]
    projector = Proj(proj="utm", zone=zone, ellps="WGS84", datum="WGS84", units="m")
    origin_easting, origin_northing = projector(longitude, latitude)
    offsets = np.array([origin_easting, origin_northing], dtype=np.float64)
    utm_points = points_city.astype(np.float64) + offsets
    zone_str = f"{zone}N"
    return utm_points, zone_str


def load_city_transform(meta_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, CRS]:
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
    zone = meta.get("center_utm_zone", [""])[0]
    if not zone:
        utm_list = json.loads(meta.get("sensor_positions_utm", ["[]"])[0])
        for item in utm_list:
            if item and item[0]:
                zone = item[0]
                break
    if not zone:
        raise RuntimeError("Metadata missing UTM zone; regenerate macro sweep with updated script")
    center_easting = float(meta["center_utm_easting_m"][0])
    center_northing = float(meta["center_utm_northing_m"][0])
    offset_xy = np.array([
        center_easting - translation[0],
        center_northing - translation[1],
    ], dtype=np.float64)
    target_crs = utm_zone_to_crs(zone)
    return translation, quat, offset_xy, zone, target_crs


def compute_utm_bounds(points_path: Path, translation: np.ndarray, quat: np.ndarray, offset_xy: np.ndarray, buffer_m: float, crop_square_m: float | None = None) -> Tuple[float, float, float, float]:
    points_table = pq.read_table(points_path, columns=["x", "y", "z"])
    points = np.column_stack([
        points_table["x"].to_numpy(zero_copy_only=False).astype(np.float64),
        points_table["y"].to_numpy(zero_copy_only=False).astype(np.float64),
        points_table["z"].to_numpy(zero_copy_only=False).astype(np.float64),
    ])
    rotation = quaternion_to_matrix(*quat)
    city_points = (rotation @ points.T).T + translation
    utm_xy = city_points[:, :2] + offset_xy

    if crop_square_m is not None:
        half = float(crop_square_m) / 2.0
        center_e = float(translation[0] + offset_xy[0])
        center_n = float(translation[1] + offset_xy[1])
        min_e = center_e - half
        max_e = center_e + half
        min_n = center_n - half
        max_n = center_n + half
    else:
        min_e = float(utm_xy[:, 0].min() - buffer_m)
        max_e = float(utm_xy[:, 0].max() + buffer_m)
        min_n = float(utm_xy[:, 1].min() - buffer_m)
        max_n = float(utm_xy[:, 1].max() + buffer_m)
    return min_e, max_e, min_n, max_n


def select_imagery_tiles(bounds: Tuple[float, float, float, float], bounds_csv: Path, target_crs: CRS) -> pd.DataFrame:
    min_e, max_e, min_n, max_n = bounds
    transformer = Transformer.from_crs(target_crs, CRS.from_epsg(4326), always_xy=True)
    west_lon, south_lat = transformer.transform(min_e, min_n)
    east_lon, north_lat = transformer.transform(max_e, max_n)
    df = pd.read_csv(bounds_csv)
    mask = (
        (df["east_lon"] >= west_lon)
        & (df["west_lon"] <= east_lon)
        & (df["north_lat"] >= south_lat)
        & (df["south_lat"] <= north_lat)
    )
    subset = df[mask]
    if subset.empty:
        raise RuntimeError("No imagery tiles overlap the requested bounding box")
    return subset


def build_imagery(bounds: Tuple[float, float, float, float], tiles_df: pd.DataFrame, imagery_dir: Path, target_crs: CRS, target_res: float, output_tif: Path) -> None:
    min_e, max_e, min_n, max_n = bounds
    if max_e <= min_e or max_n <= min_n:
        raise RuntimeError("Invalid bounding box for imagery generation")

    width = int(np.ceil((max_e - min_e) / target_res))
    height = int(np.ceil((max_n - min_n) / target_res))
    transform = from_origin(min_e, max_n, target_res, target_res)

    mosaic = None
    dest_nodata = 0

    for _, row in tiles_df.iterrows():
        jp2_dir = imagery_dir / row["source_dir"]
        jp2_name = Path(row["source_xml"]).with_suffix(".jp2").name
        src_path = jp2_dir / jp2_name
        if not src_path.exists():
            raise FileNotFoundError(f"Imagery tile missing: {src_path}")
        with rasterio.open(src_path) as src:
            if mosaic is None:
                mosaic = np.zeros((src.count, height, width), dtype=src.dtypes[0])
            temp = np.zeros_like(mosaic)
            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=temp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=dest_nodata,
                resampling=Resampling.bilinear,
            )
            mask = temp != dest_nodata
            mosaic = np.where(mask, temp, mosaic)

    if mosaic is None:
        raise RuntimeError("No imagery tiles were reprojected")

    meta = {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "count": mosaic.shape[0],
        "dtype": mosaic.dtype,
        "crs": target_crs.to_wkt(),
        "transform": transform,
        "compress": "zstd",
        "nodata": dest_nodata,
    }
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(mosaic)

def build_dsm(bounds: Tuple[float, float, float, float], tiles_df: pd.DataFrame, dsm_dir: Path, target_crs: CRS, output_laz: Path) -> None:
    min_e, max_e, min_n, max_n = bounds
    laz_paths: List[Path] = []
    for _, row in tiles_df.iterrows():
        tile_id = row["sub_tile_id"]
        big_tile = row["big_tile_id"]
        laz_folder = dsm_dir / f"stratmap21-28cm-50cm-bexar-travis_{big_tile}_lpc"
        laz_path = laz_folder / f"stratmap21-28cm_{tile_id}.laz"
        if laz_path.exists():
            laz_paths.append(laz_path)
    if not laz_paths:
        raise RuntimeError("No DSM LAZ tiles found for the selected bounding box")

    transformer = Transformer.from_crs(CRS.from_epsg(6578), target_crs, always_xy=True)
    z_scale = 0.3048006096012192
    collected_east: List[np.ndarray] = []
    collected_north: List[np.ndarray] = []
    collected_z: List[np.ndarray] = []

    for laz_path in laz_paths:
        las = laspy.read(laz_path)
        east, north = transformer.transform(las.x, las.y)
        z = las.z * z_scale
        mask = (east >= min_e) & (east <= max_e) & (north >= min_n) & (north <= max_n)
        if np.any(mask):
            collected_east.append(east[mask])
            collected_north.append(north[mask])
            collected_z.append(z[mask])

    if not collected_east:
        raise RuntimeError("DSM tiles found but no points intersect the LiDAR footprint")

    east = np.concatenate(collected_east)
    north = np.concatenate(collected_north)
    z = np.concatenate(collected_z)

    header = laspy.LasHeader(point_format=2, version="1.4")
    header.offsets = [float(east.min()), float(north.min()), float(z.min())]
    header.scales = [0.01, 0.01, 0.01]
    header.parse_crs = target_crs
    out_las = laspy.LasData(header)
    out_las.x = east
    out_las.y = north
    out_las.z = z

    try:
        out_las.write(output_laz)
    except Exception:
        out_las.write(output_laz.with_suffix(".las"))


def run_stage_two(points_path: Path, meta_path: Path, city_root: Path, output_dir: Path, buffer_m: float, target_res: float, base_name: str, crop_square_m: float | None = None) -> Tuple[Path, Path]:
    imagery_dir = city_root / "Imagery"
    dsm_dir = city_root / "DSM"
    bounds_csv = city_root / "imagery_tile_bounds.csv"

    if not imagery_dir.exists():
        raise FileNotFoundError(f"Imagery directory not found: {imagery_dir}")
    if not dsm_dir.exists():
        raise FileNotFoundError(f"DSM directory not found: {dsm_dir}")
    if not bounds_csv.exists():
        raise FileNotFoundError(f"Tile bounds CSV not found: {bounds_csv}")

    translation, quat, offset_xy, _zone, target_crs = load_city_transform(meta_path)
    effective_buffer = 0.0 if crop_square_m is not None else buffer_m
    bounds = compute_utm_bounds(points_path, translation, quat, offset_xy, effective_buffer, crop_square_m=crop_square_m)
    tiles_df = select_imagery_tiles(bounds, bounds_csv, target_crs)

    tif_path = output_dir / f"{base_name}_imagery_utm.tif"
    laz_path = output_dir / f"{base_name}_dsm_utm.laz"

    build_imagery(bounds, tiles_df, imagery_dir, target_crs, target_res, tif_path)
    build_dsm(bounds, tiles_df, dsm_dir, target_crs, laz_path)
    return tif_path, laz_path

    if not imagery_dir.exists():
        raise FileNotFoundError(f"Imagery directory not found: {imagery_dir}")
    if not dsm_dir.exists():
        raise FileNotFoundError(f"DSM directory not found: {dsm_dir}")
    if not bounds_csv.exists():
        raise FileNotFoundError(f"Tile bounds CSV not found: {bounds_csv}")

    translation, quat, offset_xy, _zone, target_crs = load_city_transform(meta_path)
    bounds = compute_utm_bounds(points_path, translation, quat, offset_xy, buffer_m)
    tiles_df = select_imagery_tiles(bounds, bounds_csv, target_crs)

    tif_path = output_dir / f"{base_name}_imagery_utm.tif"
    laz_path = output_dir / f"{base_name}_dsm_utm.laz"

    build_imagery(bounds, tiles_df, imagery_dir, target_crs, target_res, tif_path)
    build_dsm(bounds, tiles_df, dsm_dir, target_crs, laz_path)
    return tif_path, laz_path


class SampleBuilderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("AV2 Sample Builder")
        root.geometry("820x640")

        self.log_text: ScrolledText
        self.run_thread: threading.Thread | None = None

        self.log_dir_var = tk.StringVar()
        self.city_root_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(THIS_DIR.parent / "Data-Sample"))
        self.sample_prefix_var = tk.StringVar(value=DEFAULT_SAMPLE_PREFIX)
        self.sweep_count_var = tk.StringVar(value=str(DEFAULT_SWEEP_COUNT))
        self.crop_enable_var = tk.BooleanVar(value=False)
        self.crop_size_var = tk.StringVar(value=str(DEFAULT_CROP_MIN))
        self.crop_entry: tk.Entry | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        def add_path_row(label: str, var: tk.StringVar) -> None:
            row = tk.Frame(frame)
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text=label, width=22, anchor="w").pack(side=tk.LEFT)
            entry = tk.Entry(row, textvariable=var)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
            btn = tk.Button(row, text="Browse", command=lambda: self._browse_directory(var))
            btn.pack(side=tk.LEFT)

        add_path_row("AV2 log directory", self.log_dir_var)
        add_path_row("Imagery/DSM city root", self.city_root_var)
        add_path_row("Output directory", self.output_dir_var)

        row = tk.Frame(frame)
        row.pack(fill=tk.X, pady=4)
        tk.Label(row, text="Sample prefix", width=22, anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=self.sample_prefix_var).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(row, text="Sweep count", anchor="w").pack(side=tk.LEFT)
        tk.Entry(row, width=5, textvariable=self.sweep_count_var).pack(side=tk.LEFT, padx=(0, 6))

        crop_row = tk.Frame(frame)
        crop_row.pack(fill=tk.X, pady=(0, 6))
        chk = tk.Checkbutton(
            crop_row,
            text="Crop outputs to square (m)",
            variable=self.crop_enable_var,
            command=self._toggle_crop_entry,
        )
        chk.pack(side=tk.LEFT)
        entry = tk.Entry(crop_row, width=8, textvariable=self.crop_size_var, state=tk.DISABLED)
        entry.pack(side=tk.LEFT, padx=(6, 0))
        self.crop_entry = entry

        self.run_button = tk.Button(frame, text="Run pipeline", command=self.run_pipeline)
        self.run_button.pack(pady=(10, 8))

        self.log_text = ScrolledText(frame, height=20, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _toggle_crop_entry(self) -> None:
        state = tk.NORMAL if self.crop_enable_var.get() else tk.DISABLED
        if self.crop_entry is not None:
            self.crop_entry.configure(state=state)

    def _browse_directory(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="Select directory")
        if path:
            var.set(path)

    def log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def run_pipeline(self) -> None:
        if self.run_thread and self.run_thread.is_alive():
            messagebox.showinfo("Busy", "Pipeline is already running.")
            return
        try:
            sweep_count = int(self.sweep_count_var.get())
            if sweep_count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid input", "Sweep count must be a positive integer")
            return

        crop_size = None
        if self.crop_enable_var.get():
            try:
                crop_size = float(self.crop_size_var.get())
            except ValueError:
                messagebox.showerror("Invalid input", "Crop size must be a number")
                return
            if crop_size < DEFAULT_CROP_MIN:
                messagebox.showerror("Invalid input", f"Crop size must be at least {DEFAULT_CROP_MIN} m")
                return

        log_dir = Path(self.log_dir_var.get()).resolve()
        city_root = Path(self.city_root_var.get()).resolve()
        output_dir = Path(self.output_dir_var.get()).resolve()
        sample_prefix = self.sample_prefix_var.get().strip() or DEFAULT_SAMPLE_PREFIX

        missing = []
        if not log_dir.exists():
            missing.append(f"Log directory: {log_dir}")
        if not city_root.exists():
            missing.append(f"City root: {city_root}")
        if missing:
            messagebox.showerror("Path error", "\n".join(["Missing required paths:"] + missing))
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        params = (log_dir, city_root, output_dir, sweep_count, sample_prefix, crop_size)
        self.run_button.configure(state=tk.DISABLED)
        self.log("Starting sample pipeline...")
        if crop_size is not None:
            self.log(f"Cropping outputs to {crop_size:.2f} m square (min {DEFAULT_CROP_MIN} m).")
        self.run_thread = threading.Thread(target=self._pipeline_worker, args=params, daemon=True)
        self.run_thread.start()

    def _pipeline_worker(self, log_dir: Path, city_root: Path, output_dir: Path, sweep_count: int, sample_prefix: str, crop_size: float | None) -> None:
        try:
            self.log(f"Stage 1: combining {sweep_count} LiDAR sweeps from {log_dir}")
            lidar_dir = log_dir / "sensors" / "lidar"
            feather_files = sorted(lidar_dir.glob("*.feather"))
            if len(feather_files) < sweep_count:
                raise RuntimeError(f"Requested {sweep_count} sweeps but only found {len(feather_files)}")

            sweep_indices = list(range(sweep_count))
            center_index = len(sweep_indices) // 2
            output_prefix = output_dir / sample_prefix

            point_path, meta_path, applied_crop = create_macro_sweep(
                log_dir=log_dir,
                sweep_indices=sweep_indices,
                center_index=center_index,
                output_prefix=output_prefix,
                dataset_dir=THIS_DIR,
                crop_square_m=crop_size,
            )
            self.log(f"Generated point parquet: {point_path}")
            self.log(f"Generated metadata parquet: {meta_path}")
            if crop_size is not None:
                if applied_crop is not None and abs(applied_crop - crop_size) > 1e-6:
                    self.log(f"Requested crop {crop_size:.2f} m adjusted to {applied_crop:.2f} m based on available coverage.")
                elif applied_crop is None:
                    self.log("Crop request skipped; proceeding with full extent.")
                else:
                    self.log(f"Applied crop size: {applied_crop:.2f} m.")
            crop_for_stage2 = applied_crop if applied_crop is not None else None

            self.log("Stage 2: building aligned imagery and DSM")
            tif_path, laz_path = run_stage_two(
                points_path=point_path,
                meta_path=meta_path,
                city_root=city_root,
                output_dir=output_dir,
                buffer_m=DEFAULT_BUFFER_METERS,
                target_res=DEFAULT_GRID_RESOLUTION,
                base_name=sample_prefix,
                crop_square_m=crop_for_stage2,
            )
            self.log(f"Generated imagery GeoTIFF: {tif_path}")
            self.log(f"Generated DSM LAZ: {laz_path}")
            self.log("Pipeline completed successfully.")
            messagebox.showinfo("Done", f"Sample generated in {output_dir}")
        except Exception as exc:
            self.log(f"Error: {exc}")
            messagebox.showerror("Pipeline failed", str(exc))
        finally:
            self.run_button.configure(state=tk.NORMAL)


def main() -> None:
    root = tk.Tk()
    app = SampleBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()