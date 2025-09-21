import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from pyproj import Transformer, CRS
import laspy

os.environ.setdefault("PROJ_LIB", str(Path(rasterio.__file__).parent / "proj_data"))

BASE_DIR = Path(r"G:\GithubProject\ReLL")
DATA_SAMPLE_DIR = BASE_DIR / "Data-Sample"
IMAGERY_DIR = BASE_DIR / "Argoverse2-geoalign" / "ATX" / "Imagery"
DSM_DIR = BASE_DIR / "Argoverse2-geoalign" / "ATX" / "DSM"
IMAGERY_BOUNDS = BASE_DIR / "Argoverse2-geoalign" / "ATX" / "imagery_tile_bounds.csv"
POINTS_PATH = DATA_SAMPLE_DIR / "combined_0p5s.parquet"
META_PATH = DATA_SAMPLE_DIR / "combined_0p5s_meta.parquet"

OUTPUT_TIF = DATA_SAMPLE_DIR / "imagery_utm.tif"
OUTPUT_LAZ = DATA_SAMPLE_DIR / "dsm_utm.laz"

BUFFER_METERS = 2.0
TARGET_PROJ4 = "+proj=utm +zone=14 +datum=WGS84 +units=m +no_defs"
TARGET_CRS = CRS.from_proj4(TARGET_PROJ4)
TARGET_RES = 0.3


def load_city_to_utm_transform():
    meta = pq.read_table(META_PATH).to_pydict()
    city_positions = np.array(json.loads(meta['sensor_positions_city_m'][0]))[:, :2]
    utm_positions = []
    for item in json.loads(meta['sensor_positions_utm'][0]):
        if item is None:
            continue
        zone, easting, northing = item
        utm_positions.append([easting, northing])
    utm_positions = np.array(utm_positions)
    X = np.hstack([city_positions, np.ones((city_positions.shape[0], 1))])
    A, _, _, _ = np.linalg.lstsq(X, utm_positions, rcond=None)
    return A


def compute_utm_bounds(A):
    meta = pq.read_table(META_PATH).to_pydict()
    q = np.array([meta['center_city_qw'][0], meta['center_city_qx'][0], meta['center_city_qy'][0], meta['center_city_qz'][0]], dtype=np.float64)
    qw, qx, qy, qz = q
    rot = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    trans = np.array([meta['center_city_tx_m'][0], meta['center_city_ty_m'][0], meta['center_city_tz_m'][0]], dtype=np.float64)
    points = pq.read_table(POINTS_PATH, columns=['x','y','z']).to_pandas().values.astype(np.float64)
    city_points = (rot @ points.T).T + trans
    city_xy = city_points[:, :2]
    XY = np.hstack([city_xy, np.ones((city_xy.shape[0], 1))])
    utm_xy = XY @ A
    min_e = utm_xy[:,0].min() - BUFFER_METERS
    max_e = utm_xy[:,0].max() + BUFFER_METERS
    min_n = utm_xy[:,1].min() - BUFFER_METERS
    max_n = utm_xy[:,1].max() + BUFFER_METERS
    return min_e, max_e, min_n, max_n


def select_imagery_tiles(bounds):
    min_e, max_e, min_n, max_n = bounds
    transformer = Transformer.from_crs(TARGET_CRS, CRS.from_epsg(4326), always_xy=True)
    west_lon, south_lat = transformer.transform(min_e, min_n)
    east_lon, north_lat = transformer.transform(max_e, max_n)
    df = pd.read_csv(IMAGERY_BOUNDS)
    mask = (
        (df['east_lon'] >= west_lon) &
        (df['west_lon'] <= east_lon) &
        (df['north_lat'] >= south_lat) &
        (df['south_lat'] <= north_lat)
    )
    subset = df[mask]
    if subset.empty:
        raise RuntimeError("No imagery tiles overlap sample area")
    return subset


def build_imagery(bounds, tiles_df):
    min_e, max_e, min_n, max_n = bounds
    vrts = []
    base_sources = []
    try:
        for _, row in tiles_df.iterrows():
            jp2_dir = IMAGERY_DIR / row['source_dir']
            jp2_name = Path(row['source_xml']).with_suffix('.jp2').name
            src_path = jp2_dir / jp2_name
            if not src_path.exists():
                raise FileNotFoundError(src_path)
            src = rasterio.open(src_path)
            base_sources.append(src)
            vrt = WarpedVRT(src, crs=TARGET_CRS, res=TARGET_RES)
            vrts.append(vrt)
        merged, out_transform = merge(
            vrts,
            bounds=(min_e, min_n, max_e, max_n),
            res=TARGET_RES,
            nodata=0,
        )
    finally:
        for vrt in vrts:
            vrt.close()
        for src in base_sources:
            src.close()
    out_meta = {
        "driver": "GTiff",
        "height": merged.shape[1],
        "width": merged.shape[2],
        "count": merged.shape[0],
        "dtype": merged.dtype,
        "crs": TARGET_CRS.to_wkt(),
        "transform": out_transform,
    }
    with rasterio.open(OUTPUT_TIF, 'w', **out_meta) as dst:
        dst.write(merged)


def build_dsm(bounds, tiles_df):
    min_e, max_e, min_n, max_n = bounds
    laz_paths = []
    for _, row in tiles_df.iterrows():
        tile_id = row['sub_tile_id']
        big_tile = row['big_tile_id']
        laz_dir = DSM_DIR / f"stratmap21-28cm-50cm-bexar-travis_{big_tile}_lpc"
        laz_path = laz_dir / f"stratmap21-28cm_{tile_id}.laz"
        if laz_path.exists():
            laz_paths.append(laz_path)
    if not laz_paths:
        raise RuntimeError("No DSM LAZ tiles found for bounding box")
    transformer = Transformer.from_crs(CRS.from_epsg(6578), TARGET_CRS, always_xy=True)
    z_scale = 0.3048006096012192
    collected_east = []
    collected_north = []
    collected_z = []
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
        raise RuntimeError("DSM tiles present but no points intersect bounding box")
    east = np.concatenate(collected_east)
    north = np.concatenate(collected_north)
    z = np.concatenate(collected_z)
    header = laspy.LasHeader(point_format=2, version="1.4")
    header.offsets = [float(east.min()), float(north.min()), float(z.min())]
    header.scales = [0.01, 0.01, 0.01]
    header.parse_crs = TARGET_CRS
    out_las = laspy.LasData(header)
    out_las.x = east
    out_las.y = north
    out_las.z = z
    try:
        out_las.write(OUTPUT_LAZ)
    except Exception:
        out_las.write(OUTPUT_LAZ.with_suffix('.las'))


def main():
    A = load_city_to_utm_transform()
    bounds = compute_utm_bounds(A)
    tiles_df = select_imagery_tiles(bounds)
    build_imagery(bounds, tiles_df)
    build_dsm(bounds, tiles_df)


if __name__ == '__main__':
    main()
