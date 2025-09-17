# Argoverse 2 LiDAR Toolkit

Helpers for pulling down manageable slices of the Argoverse 2 LiDAR dataset and poking at the sweeps/poses without touching the rest of the repo.

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
- `preview_lidar_gui.py` opens a Tkinter list of `.feather` sweeps and renders the selected cloud in Open3D. Use `--root` to point at a custom folder and `--stride` to subsample points.
- `preview_city_pose_gui.py` loads a `city_SE3_egovehicle.feather` file, summarises timestamps/translations, and shows up to 500 rows in a table. Launch it with `python preview_city_pose_gui.py --file <path>` or use the built-in file picker.
- `plot_pose_on_map.py` converts pose translations to WGS84 using the published anchors and writes an interactive HTML map. Pass `--pose <path>` plus optional `--city`, `--output`, and `--no-open` flags.

![preview_lidar_gui](https://github.com/user-attachments/assets/cf5133f1-fad5-4365-bfc4-10771765e25a)

![plot_pose_on_map](https://github.com/user-attachments/assets/77a94507-72cf-4aa3-9c1f-e7bd41d54a91)

## Pose coordinate export
- `export_pose_coordinates.py` streams every `city_SE3_egovehicle.feather` from the public Argoverse bucket and writes a consolidated table to `av2_coor.feather`.
- The exported dataset now resides at `ArgoverseLidar/av2_coor.feather` (~0.9 GB, ~6 M rows) and is ready for downstream analytics.
- Columns: `city`, `split`, `log_id`, `timestamp_ns`, `qw`, `qx`, `qy`, `qz`, `tx`, `ty`, `tz`, `zone_num_hemi`, `easting`, `northing`, `latitude`, `longitude`.
- Each row represents a timestamped ego pose with both UTM and WGS84 projections derived via the city anchors.

## Heatmap aggregation & maps
- `aggregate_heatmap_cells.py` buckets the pose table into 500 m × 500 m UTM cells per city and records the point density (`city_heatmap_cells.json`).
- `compute_city_bounds.py` and `render_city_bounds_map.py` create a quick bounds overlay (`city_bounds.json` → `city_bounds_map.html`).
- `render_city_heatmap_map.py` layers the per-city rectangles and the heatmap cells onto OpenStreetMap (`city_heatmap_map.html`), modulating opacity between 0.1–0.8 based on counts.
  
<img width="2342" height="2486" alt="Heatmap Records Lidar R" src="https://github.com/user-attachments/assets/24067b18-44ba-49ce-b101-f15b46a72287" />

## Bundled `s5cmd`
The `s5cmd/` folder contains `s5cmd.exe` (v2.3.0) and upstream documentation so Windows users do not need to install it separately. The downloader script points to this binary by default but accepts a custom path when required.
