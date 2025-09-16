# ArgoverseLidar Folder Overview

Quick reference for the helper assets that keep the main project untouched while you work with Argoverse 2 LiDAR data.

## Scripts & utilities
- `download_sample_logs.py` – Enumerates S3 log prefixes and syncs a limited count per split into `../argverse_data_preview`, forwarding concurrency/part-size tweaks to `s5cmd`.
- `preview_lidar_gui.py` – Tkinter file picker + Open3D viewer for LiDAR sweep `.feather` files.
- `preview_city_pose_gui.py` – GUI table view of `city_SE3_egovehicle.feather` data with summary stats (duration, cadence, translation bounds).
- `plot_pose_on_map.py` – Converts pose translations from city ENU to WGS84 using the published city anchors and renders the trajectory as tiny dots on an OpenStreetMap basemap.
- `environment.yml` / `requirements.txt` – Dependency manifests (numpy/pandas/pyarrow/open3d/pyproj/folium, etc.).
- `README.md` – Usage quick start for downloads, GUIs, and the mapping utility.

## Bundled `s5cmd`
- `s5cmd/s5cmd.exe` – Windows binary for high-throughput S3 transfers.
- `s5cmd/CHANGELOG.md`, `LICENSE`, `README.md` – Upstream documentation packaged alongside the binary.

## Typical workflow
1. Activate the helper environment (conda or pip) using the manifests above.
2. Pull a preview subset: `python download_sample_logs.py --split val --count 100 --dest ..\argverse_data_preview --skip-existing`.
3. Inspect data: `preview_lidar_gui.py` for sweeps, `preview_city_pose_gui.py` for poses, or `plot_pose_on_map.py --pose <path>` to see the trajectory on OSM.
4. For full dataset mirrors, call `s5cmd` directly with your preferred arguments.
