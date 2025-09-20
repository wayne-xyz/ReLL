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

Typical log statistics (from `original_data_summary.json`): ~299 sweeps per log covering ~29.8 s of motion.

## 2. Combined macro-sweeps (`combined_*.parquet`)

We merge five consecutive sweeps (≈0.5 s) and express all points in the reference sensor frame (default: the middle sweep).  Each bundle consists of a point file and a metadata file.

### Point parquet (`combined_*.parquet`)
| column | meaning |
| --- | --- |
| `x`, `y`, `z` (float32) | Coordinates in the reference sensor frame (origin at the centre sweep). |
| `intensity` (float32) | Reflectance. |
| `laser_number` (int16) | Channel index. |
| `offset_ns` (int64) | Return time offset from the sweep timestamp. |
| `source_index` (int8) | Which of the input sweeps produced the point (0..N‑1). |
| `source_timestamp_ns` (int64) | Timestamp of that sweep. |

### Metadata parquet (`combined_*_meta.parquet`)
| column | meaning |
| --- | --- |
| `log_id`, `point_file` | Source log id and point parquet name. |
| `center_timestamp_ns` | Timestamp of the reference sweep.
| `center_city_t{x,y,z}_m` | City-frame position of the reference sensor (metres).
| `center_city_q{w,x,y,z}` | Orientation of the reference sensor (quaternion).
| `point_count`, `sweep_count` | Total number of points and sweeps merged. |
| `duration_ns`, `duration_s` | Time span between the first and last sweeps. |
| `bbox_*`, `extent_{x,y,z}` | Axis-aligned bounding box in the reference frame. |
| `source_timestamps_ns` | JSON list of sweep timestamps. |
| `sensor_motion_length_m` | Sum of consecutive position deltas across the window (actual path length).
| `sensor_displacement_m` | Straight-line distance between first and last sensor positions.
| `sensor_positions_city_m` | JSON list of each sweep’s city-frame sensor position `[x, y, z]`.

### Motion metrics
- `sensor_motion_length_m` > `sensor_displacement_m` when the vehicle turns or follows a curved path; equal when the path is straight.

## 3. Helper files
- `original_data_summary.json`: quick sweep-duration stats for each preview log.
- `combined_cloud_gui.py`: Tk/Leaflet-based viewer for point/metadata parquet pairs.
- `combine_lidar_sweeps.py`: script to generate new macro-sweeps.

This setup keeps the raw sensor-frame data intact while providing the transforms and summaries needed to relate any macro-sweep back to the city/GNSS frame.
