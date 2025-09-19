# Argoverse 2 LiDAR Toolkit

Helpers for pulling down manageable slices of the Argoverse 2 LiDAR dataset and poking at the sweeps/poses without touching the rest of the repo.

## Folder contents
- `download_sample_logs.py` – Enumerates S3 log prefixes and syncs a limited count per split into `../argverse_data_preview`, forwarding concurrency/part-size tweaks to `s5cmd`.
- `preview_lidar_gui.py` – Tkinter picker plus Open3D viewer for LiDAR sweep `.feather` files, with optional root/stride overrides and a pose picker to render sweeps in city coordinates.
- `preview_city_pose_gui.py` – Table view of `city_SE3_egovehicle.feather` data with summary stats (duration, cadence, translation bounds).
- `plot_pose_on_map.py` – Converts pose translations from city ENU to WGS84 via the published anchors and renders the trajectory on an OpenStreetMap basemap.
- `export_pose_coordinates.py` – Streams every pose file from the Argoverse S3 bucket and appends the results to `av2_coor.feather`.
- `aggregate_heatmap_cells.py` – Bins poses into 500m grid cells with counts and per-cell opacity suggestions, writing `city_heatmap_cells.json`.
- `compute_city_bounds.py`, `render_city_bounds_map.py`, `city_bounds.json`, `city_bounds_map.html` – Extract city extents and visualise their rectangles.
- `render_city_heatmap_map.py`, `city_heatmap_map.html` – Overlay the city bounds and heatmap cells with transparency scaling to highlight density.
- `av2_coor.feather` – Materialised pose table (~0.9 GB) with quaternion, translation, UTM, and WGS84 coordinates for every log/timestamp in the LiDAR splits.
- `environment.yml`, `requirements.txt` – Dependency manifests (numpy, pandas, pyarrow, open3d, pyproj, folium, etc.).
- `s5cmd/` – Bundled Windows binary (`s5cmd.exe`) plus upstream CHANGELOG, LICENSE, and README for high-throughput S3 transfers.

## Setup
Create an environment inside this folder with either conda or pip:

```powershell
conda env create -f environment.yml
conda activate argoverse-lidar
```
_or_
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Download sample logs
`download_sample_logs.py` lists the S3 prefixes under `s3://argoverse/datasets/av2/lidar/` and syncs a limited number of logs per split via the bundled `s5cmd` binary. By default it fetches 100 validation logs into `..\argverse_data_preview\val`, auto-creating the destination if needed.

Key flags:
- `--split` repeatable; defaults to just `val`.
- `--count` number of logs per split to mirror (default `100`).
- `--dest` target directory (default `..\argverse_data_preview`).
- `--skip-existing` leaves already-downloaded logs untouched.
- `--concurrency`, `--part-size`, and `--extra-arg` are forwarded to `s5cmd` for tuning throughput.
- `--s5cmd` overrides the path to the executable if you have your own build.

Example command:
```powershell
python download_sample_logs.py --split train --split val --count 25 --skip-existing
```
The script streams the `s5cmd` output so you can keep an eye on progress and abort safely with Ctrl+C.

## Visualisation utilities
- `preview_lidar_gui.py` opens a Tkinter list of `.feather` sweeps and renders the selected cloud in Open3D. Use `--root` to point at a custom folder and `--stride` to subsample points, or supply a log's `city_SE3_egovehicle.feather` via the pose picker to stack sweeps in city coordinates.
- `preview_city_pose_gui.py` loads a `city_SE3_egovehicle.feather` file, summarises timestamps/translations, and shows up to 500 rows in a table. Launch it with `python preview_city_pose_gui.py --file <path>` or use the built-in file picker.
- `plot_pose_on_map.py` converts pose translations to WGS84 using the published anchors and writes an interactive HTML map. Pass `--pose <path>` plus optional `--city`, `--output`, and `--no-open` flags.

![preview_lidar_gui](https://github.com/user-attachments/assets/cf5133f1-fad5-4365-bfc4-10771765e25a)

![Priview-lidar-log-level30s](https://github.com/user-attachments/assets/3f436bf7-b499-4ac9-9f04-338b0262c9ca)


![plot_pose_on_map](https://github.com/user-attachments/assets/77a94507-72cf-4aa3-9c1f-e7bd41d54a91)


## Pose coordinate export
- `export_pose_coordinates.py` streams every `city_SE3_egovehicle.feather` from the public Argoverse bucket and writes a consolidated table to `av2_coor.feather`.
- The exported dataset now resides at `ArgoverseLidar/av2_coor.feather` (~0.9 GB, ~6 M rows) and is ready for downstream analytics.
- Each row represents a timestamped ego pose with both UTM and WGS84 projections derived via the city anchors.
- Per-log pose files (`city_SE3_egovehicle.feather`) keep `timestamp_ns` aligned with a quaternion (`qw`, `qx`, `qy`, `qz`) and translation in metres (`tx_m`, `ty_m`, `tz_m`) for each sweep.

Pose table columns:
- `city`, `split`, `log_id` – Provenance of each measurement.
- `timestamp_ns` – Capture time in nanoseconds.
- `qw`, `qx`, `qy`, `qz` – Ego pose quaternion.
- `tx`, `ty`, `tz` – Ego translation in the city frame (metres).
- `zone_num_hemi` – UTM zone label (all northern hemisphere).
- `easting`, `northing` – Projected UTM coordinates in metres.
- `latitude`, `longitude` – WGS84 coordinates computed via the published city anchors.

Sample rows (first 5):
| city | split | log_id | timestamp_ns | qw | qx | qy | qz | tx | ty | tz | zone_num_hemi | easting | northing | latitude | longitude |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321560119000 | -0.108155 | 0.001635 | -0.006183 | -0.994113 | 599.537892 | -2959.546793 | -2.974099 | 14N | 621749.565372 | 3346930.452772 | 30.247881 | -97.734556 |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321659651000 | -0.108047 | 0.001117 | -0.004756 | -0.994134 | 598.731013 | -2959.370230 | -2.993882 | 14N | 621748.758493 | 3346930.629335 | 30.247883 | -97.734565 |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321759848000 | -0.108156 | 0.000556 | -0.004991 | -0.994121 | 597.926754 | -2959.195747 | -3.002643 | 14N | 621747.954235 | 3346930.803818 | 30.247884 | -97.734573 |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321860044000 | -0.107996 | 0.000077 | -0.004928 | -0.994139 | 597.106269 | -2959.014908 | -3.013438 | 14N | 621747.133749 | 3346930.984657 | 30.247886 | -97.734581 |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321960240000 | -0.107731 | -0.000679 | -0.006384 | -0.994159 | 596.283263 | -2958.833294 | -3.021527 | 14N | 621746.310743 | 3346931.166271 | 30.247888 | -97.734590 |

## Heatmap aggregation & maps
- `aggregate_heatmap_cells.py` buckets the pose table into 500 m x 500 m UTM cells per city and records the point density (`city_heatmap_cells.json`).
- `compute_city_bounds.py` and `render_city_bounds_map.py` create a quick bounds overlay (`city_bounds.json` → `city_bounds_map.html`).
- `render_city_heatmap_map.py` layers the per-city rectangles and the heatmap cells onto OpenStreetMap (`city_heatmap_map.html`), modulating opacity between 0.1–0.8 based on counts.
  
<img width="2342" height="2486" alt="Heatmap Records Lidar R" src="https://github.com/user-attachments/assets/24067b18-44ba-49ce-b101-f15b46a72287" />

## Bundled `s5cmd`
The `s5cmd/` folder contains `s5cmd.exe` (v2.3.0) and upstream documentation so Windows users do not need to install it separately. The downloader script points to this binary by default but accepts a custom path when required.
