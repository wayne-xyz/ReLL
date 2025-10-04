"""Imagery and DSM processing for LiDAR samples."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import importlib.util


def _configure_proj() -> None:
    """Configure PROJ and GDAL environment variables for rasterio."""
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
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from pyproj import CRS, Transformer, Proj
from scipy.spatial import cKDTree

# Import from lidar_processing to avoid duplication
from .lidar_processing import (
    CityName,
    UTM_ZONE_MAP,
    CITY_ORIGIN_LATLONG_DICT,
    quaternion_to_matrix,
)


DEFAULT_BUFFER_METERS = 2.0


def utm_zone_to_crs(zone: str) -> CRS:
    """Convert UTM zone string (e.g., '14N') to CRS object."""
    if not zone:
        raise ValueError("UTM zone string is empty; cannot determine CRS")
    zone = zone.strip()
    number = int(zone[:-1])
    hemisphere = zone[-1].upper()
    epsg = 32600 + number if hemisphere == "N" else 32700 + number
    return CRS.from_epsg(epsg)


def infer_city_enum(city_name: str) -> CityName:
    """Infer CityName enum from city string (e.g., 'ATX_city_123' -> CityName.ATX)."""
    if not city_name:
        raise ValueError("City name missing from metadata")
    token = city_name.split("_")[0].upper()
    return CityName[token]


def convert_city_coords_to_utm(points_city: np.ndarray, city: CityName) -> Tuple[np.ndarray, str]:
    """Convert city frame coordinates to UTM.

    Args:
        points_city: Nx2 array of city frame coordinates
        city: City name enum

    Returns:
        Tuple of (UTM coordinates, zone string like '14N')
    """
    latitude, longitude = CITY_ORIGIN_LATLONG_DICT[city]
    zone = UTM_ZONE_MAP[city]
    projector = Proj(proj="utm", zone=zone, ellps="WGS84", datum="WGS84", units="m")
    origin_easting, origin_northing = projector(longitude, latitude)
    offsets = np.array([origin_easting, origin_northing], dtype=np.float64)
    utm_points = points_city.astype(np.float64) + offsets
    zone_str = f"{zone}N"
    return utm_points, zone_str


def load_city_transform(meta_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, CRS]:
    """Load transformation parameters from metadata parquet.

    Returns:
        Tuple of (translation, quaternion, offset_xy, utm_zone, target_crs)
    """
    import json

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


def compute_utm_bounds(
    points_path: Path,
    translation: np.ndarray,
    quat: np.ndarray,
    offset_xy: np.ndarray,
    buffer_m: float,
    crop_square_m: float | None = None
) -> Tuple[float, float, float, float]:
    """Compute UTM bounding box for LiDAR points.

    Args:
        points_path: Path to sensor-frame point cloud parquet
        translation: City frame translation (3D)
        quat: Quaternion for rotation
        offset_xy: Offset from city to UTM (2D)
        buffer_m: Buffer to add around points (ignored if crop_square_m is set)
        crop_square_m: If set, use square crop centered at sensor position

    Returns:
        Tuple of (min_easting, max_easting, min_northing, max_northing)
    """
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


def select_imagery_tiles(
    bounds: Tuple[float, float, float, float],
    bounds_csv: Path,
    target_crs: CRS
) -> pd.DataFrame:
    """Select imagery tiles that overlap the given UTM bounding box.

    Args:
        bounds: (min_e, max_e, min_n, max_n) in UTM
        bounds_csv: Path to imagery_tile_bounds.csv
        target_crs: Target CRS for transformation

    Returns:
        DataFrame of overlapping tiles
    """
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


def detect_source_imagery_resolution(
    tiles_df: pd.DataFrame,
    imagery_dir: Path,
    target_crs: CRS
) -> float:
    """Detect the resolution of the source imagery by reading the first available tile.

    Args:
        tiles_df: DataFrame of tiles from select_imagery_tiles
        imagery_dir: Root directory containing imagery tiles
        target_crs: Target CRS for resolution calculation

    Returns:
        Resolution in meters per pixel in target CRS
    """
    for _, row in tiles_df.iterrows():
        jp2_dir = imagery_dir / row["source_dir"]
        jp2_name = Path(row["source_xml"]).with_suffix(".jp2").name
        src_path = jp2_dir / jp2_name
        if src_path.exists():
            with rasterio.open(src_path) as src:
                # Get resolution from the transform
                src_res_x = abs(src.transform.a)
                src_res_y = abs(src.transform.e)
                src_resolution = (src_res_x + src_res_y) / 2.0

                # If source CRS matches target CRS, use directly
                if src.crs == target_crs:
                    return src_resolution

                # Otherwise, approximate by reprojecting a small area
                center_x = (src.bounds.left + src.bounds.right) / 2
                center_y = (src.bounds.top + src.bounds.bottom) / 2

                # Create a small square in source CRS (1 pixel)
                transformer = Transformer.from_crs(src.crs, target_crs, always_xy=True)

                # Transform corners of one pixel
                x1, y1 = transformer.transform(center_x, center_y)
                x2, y2 = transformer.transform(center_x + src_res_x, center_y + src_res_y)

                # Calculate distance
                import math
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                return dist / math.sqrt(2)  # Diagonal distance for square pixel

    raise RuntimeError("Could not detect source imagery resolution - no tiles found")


def build_imagery(
    bounds: Tuple[float, float, float, float],
    tiles_df: pd.DataFrame,
    imagery_dir: Path,
    target_crs: CRS,
    target_res: float | None,
    output_tif: Path
) -> Tuple[None, float]:
    """Build imagery mosaic from tiles.

    Args:
        bounds: (min_e, max_e, min_n, max_n) in UTM
        tiles_df: DataFrame of tiles to mosaic
        imagery_dir: Root directory containing imagery tiles
        target_crs: Target CRS
        target_res: Target resolution in m/pixel. If None, auto-detects from source
        output_tif: Output GeoTIFF path

    Returns:
        Tuple of (None for compatibility, actual resolution used)
    """
    min_e, max_e, min_n, max_n = bounds
    if max_e <= min_e or max_n <= min_n:
        raise RuntimeError("Invalid bounding box for imagery generation")

    # Auto-detect resolution if not specified
    if target_res is None:
        target_res = detect_source_imagery_resolution(tiles_df, imagery_dir, target_crs)
        print(f"Auto-detected source imagery resolution: {target_res:.4f} m/pixel")

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

    return None, target_res


def build_dsm(
    bounds: Tuple[float, float, float, float],
    tiles_df: pd.DataFrame,
    dsm_dir: Path,
    target_crs: CRS,
    output_laz: Path
) -> None:
    """Build DSM point cloud from LAZ tiles.

    Args:
        bounds: (min_e, max_e, min_n, max_n) in UTM
        tiles_df: DataFrame of tiles
        dsm_dir: Root directory containing DSM tiles
        target_crs: Target CRS
        output_laz: Output LAZ path
    """
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


def compute_vertical_alignment(
    utm_point_path: Path,
    dsm_laz_path: Path,
    ground_percentile: float = 5.0,
    grid_size_m: float = 10.0
) -> float:
    """Compute vertical offset to align LiDAR ground points with DSM.

    Args:
        utm_point_path: Path to LiDAR UTM point cloud parquet
        dsm_laz_path: Path to DSM LAZ file
        ground_percentile: Percentile to use for ground point detection (default: 5.0)
        grid_size_m: Grid cell size for local ground detection in meters

    Returns:
        z_offset: Vertical offset to add to LiDAR elevations (median DSM_z - LiDAR_z)
    """
    # Read LiDAR UTM points
    lidar_table = pq.read_table(utm_point_path, columns=["utm_e", "utm_n", "elevation"])
    lidar_e = lidar_table["utm_e"].to_numpy(zero_copy_only=False)
    lidar_n = lidar_table["utm_n"].to_numpy(zero_copy_only=False)
    lidar_z = lidar_table["elevation"].to_numpy(zero_copy_only=False)

    # Compute grid bounds
    min_e, max_e = float(lidar_e.min()), float(lidar_e.max())
    min_n, max_n = float(lidar_n.min()), float(lidar_n.max())

    # Create grid cells
    n_cells_e = max(1, int(np.ceil((max_e - min_e) / grid_size_m)))
    n_cells_n = max(1, int(np.ceil((max_n - min_n) / grid_size_m)))

    ground_points_e = []
    ground_points_n = []
    ground_points_z = []

    # For each grid cell, find ground points (bottom percentile)
    for i in range(n_cells_e):
        for j in range(n_cells_n):
            cell_min_e = min_e + i * grid_size_m
            cell_max_e = min_e + (i + 1) * grid_size_m
            cell_min_n = min_n + j * grid_size_m
            cell_max_n = min_n + (j + 1) * grid_size_m

            # Find points in this cell
            mask = (
                (lidar_e >= cell_min_e) & (lidar_e < cell_max_e) &
                (lidar_n >= cell_min_n) & (lidar_n < cell_max_n)
            )

            if not np.any(mask):
                continue

            cell_z = lidar_z[mask]
            cell_e = lidar_e[mask]
            cell_n = lidar_n[mask]

            # Get ground threshold (bottom percentile)
            z_threshold = np.percentile(cell_z, ground_percentile)
            ground_mask = cell_z <= z_threshold

            if np.any(ground_mask):
                ground_points_e.extend(cell_e[ground_mask])
                ground_points_n.extend(cell_n[ground_mask])
                ground_points_z.extend(cell_z[ground_mask])

    if len(ground_points_z) == 0:
        raise RuntimeError("No ground points detected in LiDAR data")

    ground_points_e = np.array(ground_points_e)
    ground_points_n = np.array(ground_points_n)
    ground_points_z = np.array(ground_points_z)

    # Read DSM
    dsm_las = laspy.read(dsm_laz_path)
    dsm_e = np.array(dsm_las.x)
    dsm_n = np.array(dsm_las.y)
    dsm_z = np.array(dsm_las.z)

    # For each ground point, find nearest DSM point and compute difference
    dsm_tree = cKDTree(np.column_stack([dsm_e, dsm_n]))

    # Query nearest DSM points for each ground point
    distances, indices = dsm_tree.query(
        np.column_stack([ground_points_e, ground_points_n]),
        k=1,
        distance_upper_bound=2.0  # Max 2m search radius
    )

    # Filter out points without nearby DSM points
    valid_mask = np.isfinite(distances)

    if not np.any(valid_mask):
        raise RuntimeError("No LiDAR ground points have nearby DSM points")

    valid_lidar_z = ground_points_z[valid_mask]
    valid_dsm_z = dsm_z[indices[valid_mask]]

    # Compute offset: DSM - LiDAR (what to add to LiDAR to match DSM)
    differences = valid_dsm_z - valid_lidar_z

    # Use median for robustness against outliers
    z_offset = float(np.median(differences))

    return z_offset


def apply_vertical_alignment(utm_point_path: Path, z_offset_m: float) -> None:
    """Apply vertical offset to UTM point cloud parquet file (in-place update).

    Args:
        utm_point_path: Path to UTM point cloud parquet
        z_offset_m: Vertical offset to add to elevation values
    """
    # Read the full table
    table = pq.read_table(utm_point_path)
    data = table.to_pydict()

    # Apply offset to elevation
    elevation = np.array(data["elevation"])
    elevation_aligned = elevation + z_offset_m

    # Update the dictionary
    data["elevation"] = pa.array(elevation_aligned, type=pa.float32())

    # Write back to file
    aligned_table = pa.Table.from_pydict(data)

    # Preserve compression settings from original
    original_metadata = pq.read_metadata(utm_point_path)
    compression = "zstd"  # Default

    pq.write_table(aligned_table, utm_point_path, compression=compression, use_dictionary=False)


def update_metadata_with_alignment(meta_path: Path, z_offset_m: float) -> None:
    """Update metadata parquet with vertical alignment offset.

    Args:
        meta_path: Path to metadata parquet file
        z_offset_m: Vertical offset value to add
    """
    # Read existing metadata
    table = pq.read_table(meta_path)
    data = table.to_pydict()

    # Add z_offset_m field
    data["z_offset_m"] = pa.array([z_offset_m], type=pa.float32())

    # Write back
    updated_table = pa.Table.from_pydict(data)
    pq.write_table(updated_table, meta_path, compression="zstd", use_dictionary=True)


class StageTwoResult(NamedTuple):
    """Result of stage 2 processing (imagery + DSM)."""
    tif_path: Path
    laz_path: Path
    utm_extent_xy: Tuple[float, float]
    imagery_resolution_m: float
    z_offset_m: float


def run_stage_two(
    points_path: Path,
    meta_path: Path,
    city_root: Path,
    output_dir: Path,
    buffer_m: float,
    target_res: float | None,
    base_name: str,
    crop_square_m: float | None = None
) -> StageTwoResult:
    """Run stage 2 processing: generate imagery and DSM.

    Args:
        points_path: Path to sensor-frame point cloud parquet
        meta_path: Path to metadata parquet
        city_root: Root directory containing Imagery/ and DSM/ subdirectories
        output_dir: Output directory for generated files
        buffer_m: Buffer around LiDAR footprint (ignored if crop_square_m is set)
        target_res: Target imagery resolution in m/pixel. If None, auto-detects from source
        base_name: Base name for output files
        crop_square_m: If set, use square crop instead of footprint + buffer

    Returns:
        StageTwoResult with paths and metadata
    """
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
    bounds = compute_utm_bounds(
        points_path, translation, quat, offset_xy, effective_buffer, crop_square_m=crop_square_m
    )
    tiles_df = select_imagery_tiles(bounds, bounds_csv, target_crs)

    tif_path = output_dir / f"{base_name}_imagery_utm.tif"
    laz_path = output_dir / f"{base_name}_dsm_utm.laz"

    width_m = float(bounds[1] - bounds[0])
    height_m = float(bounds[3] - bounds[2])

    _, actual_resolution = build_imagery(bounds, tiles_df, imagery_dir, target_crs, target_res, tif_path)
    build_dsm(bounds, tiles_df, dsm_dir, target_crs, laz_path)

    # Get UTM point cloud path from metadata
    meta_table = pq.read_table(meta_path)
    meta_dict = meta_table.to_pydict()
    utm_point_file = meta_dict["utm_point_file"][0]
    utm_point_path = output_dir / utm_point_file

    # Compute and apply vertical alignment
    z_offset_m = 0.0
    try:
        z_offset_m = compute_vertical_alignment(utm_point_path, laz_path)
        apply_vertical_alignment(utm_point_path, z_offset_m)
        update_metadata_with_alignment(meta_path, z_offset_m)
        print(f"Vertical alignment applied: z_offset = {z_offset_m:.4f} m")
    except Exception as e:
        print(f"Warning: Vertical alignment failed: {e}")
        print("Continuing with unaligned elevations (z_offset = 0.0 m)")
        # Still update metadata with 0.0 offset to maintain consistency
        try:
            update_metadata_with_alignment(meta_path, 0.0)
        except:
            pass

    return StageTwoResult(
        tif_path=tif_path,
        laz_path=laz_path,
        utm_extent_xy=(width_m, height_m),
        imagery_resolution_m=actual_resolution,
        z_offset_m=z_offset_m,
    )
