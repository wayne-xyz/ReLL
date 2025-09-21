"""Combine consecutive LiDAR sweeps into a macro-sweep with reference-centred coordinates."""
from __future__ import annotations

import argparse
import json
from enum import Enum, unique
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
from pyproj import Proj, Transformer




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

def load_city_annotation(log_dir: Path) -> str:
    map_dir = log_dir / "map"
    if not map_dir.exists():
        raise ValueError("Map directory missing; cannot infer city")
    for json_file in map_dir.glob("*.json"):
        stem = json_file.stem
        parts = stem.split("__")
        if len(parts) >= 3:
            city_part = parts[-1]
            if city_part:
                return city_part
    raise ValueError("City annotation not found in map metadata")



def parse_city_enum(city_tag: str) -> CityName:
    if not city_tag:
        raise ValueError("City annotation missing; ensure map metadata is available")
    token = city_tag.split("_")[0].upper()
    try:
        return CityName[token]
    except KeyError as exc:
        raise ValueError(f"Unrecognised city tag {city_tag!r}") from exc

def get_city_reference(city: CityName) -> tuple[str, np.ndarray, Transformer]:
    zone = UTM_ZONE_MAP[city]
    utm_epsg = 32600 + zone
    origin_lat, origin_lon = CITY_ORIGIN_LATLONG_DICT[city]
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    origin_e, origin_n = to_utm.transform(origin_lon, origin_lat)
    offset = np.array([origin_e, origin_n], dtype=np.float64)
    to_wgs = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    return f"{zone}N", offset, to_wgs

def city_xy_to_utm(city_xy: np.ndarray, city: CityName) -> tuple[np.ndarray, str, Transformer]:
    zone_str, offset, to_wgs = get_city_reference(city)
    utm_xy = city_xy.astype(np.float64) + offset
    return utm_xy, zone_str, to_wgs

CITY_ORIGIN_LATLONG_DICT: Dict[CityName, Tuple[float, float]] = {
    CityName.ATX: (30.27464237939507, -97.7404457407424),
    CityName.DTW: (42.29993066912924, -83.17555750783717),
    CityName.MIA: (25.77452579915163, -80.19656914449405),
    CityName.PAO: (37.416065, -122.13571963362166),
    CityName.PIT: (40.44177902989321, -80.01294377242584),
    CityName.WDC: (38.889377, -77.0355047439081),
}

class SE3:
    def __init__(self, *, rotation: np.ndarray, translation: np.ndarray) -> None:
        self.rotation = rotation.astype(np.float64)
        self.translation = translation.astype(np.float64)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return points @ self.rotation.T + self.translation

    def inverse(self) -> "SE3":
        rot_T = self.rotation.T
        return SE3(rotation=rot_T, translation=-rot_T @ self.translation)


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def compose(a: SE3, b: SE3) -> SE3:
    rotation = a.rotation @ b.rotation
    translation = a.rotation @ b.translation + a.translation
    return SE3(rotation=rotation, translation=translation)


def load_pose_map(path: Path) -> Dict[int, SE3]:
    table = feather.read_table(path)
    cols = table.to_pydict()
    tx_col = cols.get("tx_m", cols.get("tx"))
    ty_col = cols.get("ty_m", cols.get("ty"))
    tz_col = cols.get("tz_m", cols.get("tz"))
    result: Dict[int, SE3] = {}
    for ts, qw, qx, qy, qz, tx, ty, tz in zip(
        cols["timestamp_ns"], cols["qw"], cols["qx"], cols["qy"], cols["qz"], tx_col, ty_col, tz_col
    ):
        R = quaternion_to_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        result[int(ts)] = SE3(rotation=R, translation=t)
    return result


def load_sensor_transform(calibration_path: Path, keyword: str = "up_lidar") -> SE3:
    table = feather.read_table(calibration_path)
    cols = table.to_pydict()
    names = cols.get("sensor_name")
    idx = 0
    if names is not None:
        for i, name in enumerate(names):
            if keyword.lower() in name.lower():
                idx = i
                break
    tx_col = cols.get("tx_m", cols.get("tx"))
    ty_col = cols.get("ty_m", cols.get("ty"))
    tz_col = cols.get("tz_m", cols.get("tz"))
    R = quaternion_to_matrix(cols["qw"][idx], cols["qx"][idx], cols["qy"][idx], cols["qz"][idx])
    t = np.array([tx_col[idx], ty_col[idx], tz_col[idx]], dtype=np.float64)
    return SE3(rotation=R, translation=t)


def read_lidar_feather(path: Path) -> Dict[str, np.ndarray]:
    table = feather.read_table(
        path,
        columns=["x", "y", "z", "intensity", "laser_number", "offset_ns"],
    )
    return {
        "x": table["x"].to_numpy(zero_copy_only=False).astype(np.float32),
        "y": table["y"].to_numpy(zero_copy_only=False).astype(np.float32),
        "z": table["z"].to_numpy(zero_copy_only=False).astype(np.float32),
        "intensity": table["intensity"].to_numpy(zero_copy_only=False).astype(np.float32),
        "laser_number": table["laser_number"].to_numpy(zero_copy_only=False).astype(np.int16),
        "offset_ns": table["offset_ns"].to_numpy(zero_copy_only=False).astype(np.int64),
    }



def create_macro_sweep(
    log_dir: Path,
    sweep_indices: Sequence[int],
    center_index: int,
    output_prefix: Path,
    dataset_dir: Optional[Path] = None,
    compression: str = "zstd",
) -> Tuple[Path, Path]:
    _ = dataset_dir  # retained for backwards compatibility
    lidar_dir = log_dir / "sensors" / "lidar"
    feather_files = sorted(lidar_dir.glob("*.feather"))
    pose_path = log_dir / "city_SE3_egovehicle.feather"
    calib_path = log_dir / "calibration" / "egovehicle_SE3_sensor.feather"

    if len(feather_files) < len(sweep_indices):
        raise ValueError("Not enough LiDAR sweeps in the log to satisfy the request")

    poses = load_pose_map(pose_path)
    sensor_in_ego = load_sensor_transform(calib_path)

    selected_files = [feather_files[i] for i in sweep_indices]
    timestamps = [int(f.stem) for f in selected_files]

    city_from_sensor_list: List[SE3] = []
    city_points_list: List[np.ndarray] = []
    intensity_list: List[np.ndarray] = []
    laser_list: List[np.ndarray] = []
    offset_list: List[np.ndarray] = []

    for file_path, ts in zip(selected_files, timestamps):
        if ts not in poses:
            raise KeyError(f"Timestamp {ts} missing in city_SE3_egovehicle")
        ego_in_city = poses[ts]
        city_from_sensor = compose(ego_in_city, sensor_in_ego)
        city_from_sensor_list.append(city_from_sensor)

        raw = read_lidar_feather(file_path)
        points = np.column_stack([raw["x"], raw["y"], raw["z"]])
        points_city = city_from_sensor.transform_points(points.astype(np.float64))
        city_points_list.append(points_city.astype(np.float32))
        intensity_list.append(raw["intensity"])
        laser_list.append(raw["laser_number"])
        offset_list.append(raw["offset_ns"])

    translations = np.stack([tf.translation for tf in city_from_sensor_list], axis=0)
    diffs = np.diff(translations, axis=0)
    motion_length = float(np.linalg.norm(diffs, axis=1).sum()) if len(translations) > 1 else 0.0
    displacement = float(np.linalg.norm(translations[-1] - translations[0])) if len(translations) > 1 else 0.0

    reference_transform = city_from_sensor_list[center_index]
    sensor_from_city = reference_transform.inverse()

    aligned_points: List[np.ndarray] = []
    source_indices: List[np.ndarray] = []
    source_timestamps: List[np.ndarray] = []

    for idx, (pts_city, ts) in enumerate(zip(city_points_list, timestamps)):
        pts_ref = sensor_from_city.transform_points(pts_city.astype(np.float64)).astype(np.float32)
        aligned_points.append(pts_ref)
        count = pts_ref.shape[0]
        source_indices.append(np.full(count, idx, dtype=np.int8))
        source_timestamps.append(np.full(count, ts, dtype=np.int64))

    all_points = np.concatenate(aligned_points, axis=0)
    all_intensity = np.concatenate(intensity_list, axis=0)
    all_laser = np.concatenate(laser_list, axis=0)
    all_offset = np.concatenate(offset_list, axis=0)
    all_sources = np.concatenate(source_indices, axis=0)
    all_source_ts = np.concatenate(source_timestamps, axis=0)

    data_table = pa.Table.from_pydict({
        "x": pa.array(all_points[:, 0], type=pa.float32()),
        "y": pa.array(all_points[:, 1], type=pa.float32()),
        "z": pa.array(all_points[:, 2], type=pa.float32()),
        "intensity": pa.array(all_intensity, type=pa.float32()),
        "laser_number": pa.array(all_laser, type=pa.int16()),
        "offset_ns": pa.array(all_offset, type=pa.int64()),
        "source_index": pa.array(all_sources, type=pa.int8()),
        "source_timestamp_ns": pa.array(all_source_ts, type=pa.int64()),
    })

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    point_path = output_prefix.with_suffix(".parquet")
    pq.write_table(data_table, point_path, compression=compression, use_dictionary=False)

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    extents = maxs - mins

    ref_rotation = reference_transform.rotation
    ref_translation = reference_transform.translation
    ref_quat = matrix_to_quaternion(ref_rotation)

    city_tag = load_city_annotation(log_dir)
    city_enum = parse_city_enum(city_tag)
    utm_xy, utm_zone, utm_to_wgs = city_xy_to_utm(translations[:, :2], city_enum)
    lon_vals, lat_vals = utm_to_wgs.transform(utm_xy[:, 0], utm_xy[:, 1])
    positions_latlon = [[float(lat), float(lon)] for lat, lon in zip(lat_vals, lon_vals)]
    positions_utm = [[utm_zone, float(easting), float(northing)] for easting, northing in utm_xy]

    center_lat = float(lat_vals[center_index])
    center_lon = float(lon_vals[center_index])
    center_easting = float(utm_xy[center_index, 0])
    center_northing = float(utm_xy[center_index, 1])
    center_zone = utm_zone
    center_city_name = city_tag or city_enum.value

    metadata_table = pa.Table.from_pydict({
        "log_id": pa.array([log_dir.name]),
        "point_file": pa.array([point_path.name]),
        "center_timestamp_ns": pa.array([timestamps[center_index]], type=pa.int64()),
        "center_city_tx_m": pa.array([ref_translation[0]], type=pa.float64()),
        "center_city_ty_m": pa.array([ref_translation[1]], type=pa.float64()),
        "center_city_tz_m": pa.array([ref_translation[2]], type=pa.float64()),
        "center_city_qw": pa.array([ref_quat[0]], type=pa.float64()),
        "center_city_qx": pa.array([ref_quat[1]], type=pa.float64()),
        "center_city_qy": pa.array([ref_quat[2]], type=pa.float64()),
        "center_city_qz": pa.array([ref_quat[3]], type=pa.float64()),
        "center_latitude_deg": pa.array([center_lat], type=pa.float64()),
        "center_longitude_deg": pa.array([center_lon], type=pa.float64()),
        "center_utm_zone": pa.array([center_zone]),
        "center_utm_easting_m": pa.array([center_easting], type=pa.float64()),
        "center_utm_northing_m": pa.array([center_northing], type=pa.float64()),
        "city_name": pa.array([center_city_name]),
        "point_count": pa.array([all_points.shape[0]], type=pa.int64()),
        "sweep_count": pa.array([len(timestamps)], type=pa.int32()),
        "duration_ns": pa.array([int(timestamps[-1] - timestamps[0])], type=pa.int64()),
        "duration_s": pa.array([(timestamps[-1] - timestamps[0]) / 1e9], type=pa.float64()),
        "bbox_x_min": pa.array([mins[0]], type=pa.float32()),
        "bbox_x_max": pa.array([maxs[0]], type=pa.float32()),
        "bbox_y_min": pa.array([mins[1]], type=pa.float32()),
        "bbox_y_max": pa.array([maxs[1]], type=pa.float32()),
        "bbox_z_min": pa.array([mins[2]], type=pa.float32()),
        "bbox_z_max": pa.array([maxs[2]], type=pa.float32()),
        "extent_x": pa.array([extents[0]], type=pa.float32()),
        "extent_y": pa.array([extents[1]], type=pa.float32()),
        "extent_z": pa.array([extents[2]], type=pa.float32()),
        "source_timestamps_ns": pa.array([json.dumps([int(ts) for ts in timestamps])]),
        "sensor_motion_length_m": pa.array([motion_length], type=pa.float64()),
        "sensor_displacement_m": pa.array([displacement], type=pa.float64()),
        "sensor_positions_city_m": pa.array([json.dumps(translations.tolist())]),
        "sensor_positions_latlon_deg": pa.array([json.dumps(positions_latlon)]),
        "sensor_positions_utm": pa.array([json.dumps(positions_utm)]),
    })

    meta_path = output_prefix.with_name(output_prefix.name + "_meta").with_suffix(".parquet")
    pq.write_table(metadata_table, meta_path, compression=compression, use_dictionary=True)

    return point_path, meta_path


def main() -> None:
    default_output_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Combine consecutive LiDAR sweeps into a macro-sweep.")
    parser.add_argument("log_dir", type=Path, help="Path to the log directory (e.g. .../val/<log_id>)")
    parser.add_argument("--count", type=int, default=5, help="Number of consecutive sweeps to merge (default: 5)")
    parser.add_argument("--start-index", type=int, default=0, help="Index of the first sweep to use (default: 0)")
    parser.add_argument(
        "--center-index",
        type=int,
        default=None,
        help="Index within the selected sweeps to use as the reference centre (default: middle)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Prefix for the output parquet files (default derived from timestamps)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory to store output parquet files (default: {default_output_dir})",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory containing av2_coor.parquet or av2_coor.feather (default: {default_output_dir})",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        help="Parquet compression codec (default: zstd)",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    lidar_dir = log_dir / "sensors" / "lidar"
    feather_files = sorted(lidar_dir.glob("*.feather"))
    if not feather_files:
        raise RuntimeError(f"No LiDAR feather files found in {lidar_dir}")

    indices = list(range(args.start_index, args.start_index + args.count))
    if args.center_index is None:
        center_index = len(indices) // 2
    else:
        center_index = args.center_index
    if center_index < 0 or center_index >= len(indices):
        raise ValueError("center_index must fall within the selected sweep range")

    timestamps = [int(feather_files[i].stem) for i in indices]
    if args.output_name:
        prefix_name = args.output_name
    else:
        prefix_name = f"macro_{log_dir.name}_{timestamps[0]}_{timestamps[-1]}"

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / prefix_name

    dataset_dir = args.dataset_dir.resolve()

    point_path, meta_path = create_macro_sweep(
        log_dir=log_dir,
        sweep_indices=indices,
        center_index=center_index,
        output_prefix=output_prefix,
        dataset_dir=dataset_dir,
        compression=args.compression,
    )

    print(f"Created point cloud parquet: {point_path}")
    print(f"Created metadata parquet: {meta_path}")


if __name__ == "__main__":
    main()
