"""Rasterize a LiDAR macro-sweep parquet into UTM-aligned GeoTIFF bands."""
import json
from pathlib import Path
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
OUTPUT_TIF = DATA_SAMPLE_DIR / "lidar_height_intensity.tif"
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


def load_metadata() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CRS]:
    table = pq.read_table(META_PATH)
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
    # affine matrix sending city XY (with bias term) to UTM easting/northing
    A, _, _, _ = np.linalg.lstsq(X, utm_xy, rcond=None)

    crs = utm_zone_to_crs(utm_zone if utm_zone else utm_positions_raw[0][0])
    return translation, quat, A, city_xy, crs


def load_points() -> Tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(POINTS_PATH, columns=["x", "y", "z", "intensity"])
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


def rasterize(easting: np.ndarray, northing: np.ndarray, height: np.ndarray, intensity: np.ndarray, crs: CRS) -> None:
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
    OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        OUTPUT_TIF,
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


def main() -> None:
    translation, quat, A, _city_xy, crs = load_metadata()
    points_ref, intensity = load_points()
    easting, northing, height = project_to_utm(points_ref, translation, quat, A)
    rasterize(easting, northing, height, intensity, crs)
    print(f"Saved raster to {OUTPUT_TIF}")


if __name__ == "__main__":
    main()
