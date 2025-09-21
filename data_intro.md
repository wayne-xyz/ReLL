# Argoverse LiDAR Data Overview

This document introduces the datasets stored in this repository:

- the original Argoverse 2 LiDAR preview logs still living under `argverse_data_preview/val`, and
- the derived 0.5 s "macro-sweeps" we materialise as Parquet bundles in this folder.

## 1. Original preview logs (`argverse_data_preview/val`)

Each log folder (e.g. `0dr6jn0kF6YjT9Qr1mtpYrE0ihkGpKsd`) mirrors the public Argoverse 2 structure:

- `sensors/lidar/*.feather`: one file per LiDAR sweep (10 Hz).  Columns are:
  - `x`, `y`, `z` (float32): raw point coordinates in the sensor frame.
  - `intensity` (float32): reflectance.
  - `laser_number` (int16): channel index (0‑63).
  - `offset_ns` (int64): nanosecond offset from the sweep timestamp.
- `city_SE3_egovehicle.feather`: per-sweep city-frame pose (quaternion + translation in metres).
- `calibration/egovehicle_SE3_sensor.feather`: static sensor-to-ego calibration.
- Additional folders such as `map/`, `calibration/`, etc., matching the AV2 release.

Typical log statistics (see `speed_analysis.json`): ~299 sweeps per log covering ~29.8 s of motion.

## 2. Combined macro-sweeps (`combined_*.parquet`)

We merge five consecutive sweeps (≈0.5 s) and express all points in the reference sensor frame (default: the middle sweep).  Each bundle consists of a point file and a metadata file.

### Point parquet (`combined_*.parquet`)
| column | meaning |
| --- | --- |
| `x`, `y`, `z` (float32) | Coordinates in the reference sensor frame (origin at the centre sweep). |
| `intensity` (float32) | Reflectance. |
| `laser_number` (int16) | Channel index. |
| `offset_ns` (int64) | Return time offset from the sweep timestamp. |
| `source_index` (int8) | Which input sweep produced the point (0..N‑1). |
| `source_timestamp_ns` (int64) | Timestamp of that sweep. |

### Metadata parquet (`combined_*_meta.parquet`)
| column | meaning |
| --- | --- |
| `log_id`, `point_file` | Source log id and point parquet name. |
| `center_timestamp_ns` | Timestamp of the reference sweep.
| `center_city_t{x,y,z}_m` | City-frame position of the reference sensor (metres).
| `center_city_q{w,x,y,z}` | Orientation of the reference sensor (quaternion).
| `center_latitude_deg`, `center_longitude_deg` | WGS84 coordinates looked up from the global pose table. |
| `center_utm_zone`, `center_utm_easting_m`, `center_utm_northing_m` | UTM zone string and metric coordinates for the centre pose. |
| `city_name` | City tag from the log metadata / map archive (e.g. `ATX_city_77093`). |
| `point_count`, `sweep_count` | Total number of points and sweeps merged. |
| `duration_ns`, `duration_s` | Time span between the first and last sweeps. |
| `bbox_*`, `extent_{x,y,z}` | Axis-aligned bounding box in the reference frame. |
| `source_timestamps_ns` | JSON list of sweep timestamps. |
| `sensor_motion_length_m` | Sum of consecutive position deltas across the window (actual path length).
| `sensor_displacement_m` | Straight-line distance between first and last sensor positions.
| `sensor_positions_city_m` | JSON list of each sweep’s city-frame sensor position `[x, y, z]`.
| `sensor_positions_latlon_deg` | JSON list of `[lat, lon]` per sweep.
| `sensor_positions_utm` | JSON list of `[zone, easting, northing]` per sweep.

### Motion metrics
- `sensor_motion_length_m` > `sensor_displacement_m` when the vehicle turns or follows a curved path; equal when the path is straight.

## 3. Helper files
- `combined_cloud_gui.py`: Tk/Open3D viewer for point/metadata parquet pairs.
- `combine_lidar_sweeps.py`: script to generate new macro-sweeps (now annotates latitude/longitude and UTM info).
- `speed_analysis.json`, `speed_distribution.png`: speed statistics for the preview logs.

This setup keeps the raw sensor-frame data intact while providing the transforms, WGS84, and UTM summaries needed to relate any macro-sweep back to the city/GNSS frame.

#### Example (`Data-Sample/combined_0p5s*`)

Point parquet (first five rows):

| idx | x (m)   | y (m)   | z (m)   | intensity | laser | offset_ns | source_idx | source_timestamp_ns |
|----:|-------:|-------:|-------:|----------:|------:|----------:|-----------:|--------------------:|
| 0 | -17.147 | 16.118 | 1.365 | 51 | 3 | 559872 | 0 | 315978397259557000 |
| 1 |  -9.221 |  9.207 | 0.339 | 3  | 25 | 566784 | 0 | 315978397259557000 |
| 2 | -17.131 | 16.118 | 1.936 | 48 | 7 | 569088 | 0 | 315978397259557000 |
| 3 | -17.100 | 16.134 | 0.795 | 50 | 19 | 585216 | 0 | 315978397259557000 |
| 4 | -17.085 | 16.180 | 1.365 | 48 | 3 | 615168 | 0 | 315978397259557000 |

Metadata parquet (selected fields):

| field | value |
|:--|:--|
| log_id | 0dr6jn0kF6YjT9Qr1mtpYrE0ihkGpKsd |
| city_name | ATX_city_77093 |
| sweep_count | 5 |
| point_count | 272,792 |
| center_timestamp_ns | 315978397459950000 |
| center_latitude_deg / longitude_deg | 30.256388° / -97.719416° |
| center_utm_zone | 14N |
| center_utm_easting_m / northing_m | 623,195.705 / 3,347,889.585 |
| sensor_motion_length_m | 0.0655 |
| sensor_displacement_m | 0.0650 |
| duration_s | 0.4001 |
| bbox_x_min .. bbox_x_max (m) | -206.26 .. 217.40 |
| bbox_y_min .. bbox_y_max (m) | -93.56 .. 182.16 |
| bbox_z_min .. bbox_z_max (m) | -2.45 .. 45.58 |
| source_timestamps_ns | [315978397259557000, … , 315978397659679000] |

Generated companions:

- `Data-Sample/imagery_utm.tif`: GeoTIFF clipped to the macro-sweep footprint, reprojected to UTM zone 14N.
- `Data-Sample/dsm_utm.laz`: DSM point subset in UTM (coordinates in metres). Use `ArgoverseLidar/imagery_dsm_viewer.py` to preview either file.
