# Argoverse 2 LiDAR Toolkit

Utilities for downloading, inspecting, and geolocating Argoverse 2 LiDAR logs without touching the rest of the repo.

## Setup
Use either conda or pip inside this folder:

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

## Key tools
- `s5cmd/` – Bundled `s5cmd.exe` (v2.3.0) plus upstream docs.
- `download_sample_logs.py` – Pull a capped number of logs per split into `../argverse_data_preview`.
- `download_full_dataset.py` – Wrapper around `s5cmd sync` to mirror the entire LiDAR bucket.
- `preview_lidar_gui.py` – Open3D viewer for any LiDAR `.feather` sweep.
- `preview_city_pose_gui.py` – Tkinter table view of `city_SE3_egovehicle.feather` pose files with per-log stats.
- `plot_pose_on_map.py` – Convert a pose file to WGS84 and render the trajectory as small dots and a red path on OpenStreetMap tiles.

## Common commands
Download a manageable preview sample:
```powershell
python download_sample_logs.py --split val --count 100 --dest ..\argverse_data_preview --skip-existing
```

Mirror the full dataset (expect TBs and hours):
```powershell
python download_full_dataset.py --dest ..\Argoverse2LidarData --concurrency 32 --part-size 128
```

Inspect a LiDAR sweep or pose table:
```powershell
python preview_lidar_gui.py --feather ..\argverse_data_preview\val\<log_id>\sensors\lidar\<timestamp>.feather
python preview_city_pose_gui.py --file ..\argverse_data_preview\val\<log_id>\city_SE3_egovehicle.feather
```

Plot a trajectory on OpenStreetMap (opens the HTML by default):
```powershell
python plot_pose_on_map.py --pose ..\argverse_data_preview\val\<log_id>\city_SE3_egovehicle.feather
```
Use `--city` if you need to override the inferred city code, `--output` to control the HTML path, and `--no-open` to skip launching a browser.
