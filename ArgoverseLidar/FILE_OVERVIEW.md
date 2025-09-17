# ArgoverseLidar Folder Overview

Quick reference for the helper assets that keep the main project untouched while you work with Argoverse 2 LiDAR data.

## Scripts & utilities
- `download_sample_logs.py` – Enumerates S3 log prefixes and syncs a limited count per split into `../argverse_data_preview`, forwarding concurrency/part-size tweaks to `s5cmd`.
- `preview_lidar_gui.py` – Tkinter file picker + Open3D viewer for LiDAR sweep `.feather` files.
- `preview_city_pose_gui.py` – GUI table view of `city_SE3_egovehicle.feather` data with summary stats (duration, cadence, translation bounds).
- `plot_pose_on_map.py` – Converts pose translations from city ENU to WGS84 using the published city anchors and renders the trajectory as tiny dots on an OpenStreetMap basemap.
- `environment.yml` / `requirements.txt` – Dependency manifests (numpy/pandas/pyarrow/open3d/pyproj/folium, etc.).
- `README.md` – Usage quick start for downloads, GUIs, and the mapping utility.
- `export_pose_coordinates.py` – Streams every pose file directly from the Argoverse S3 bucket and appends the results to `av2_coor.feather`.
- `av2_coor.feather` – Materialised pose table (~0.9 GB) with quaternion, translation, UTM, and WGS84 coordinates for every log/timestamp in the LiDAR splits.
- `compute_city_bounds.py` / `city_bounds.json` / `city_bounds_map.html` – Extract city extents and plot their rectangles on OSM.
- `aggregate_heatmap_cells.py` / `city_heatmap_cells.json` – Bin poses into 500 m grid cells with counts and per-cell opacity suggestions.
- `render_city_heatmap_map.py` / `city_heatmap_map.html` – Overlay the bounds and heatmap cells with transparency scaling (0.1–0.8) to visualise density.

## Bundled `s5cmd`
- `s5cmd/s5cmd.exe` – Windows binary for high-throughput S3 transfers.
- `s5cmd/CHANGELOG.md`, `LICENSE`, `README.md` – Upstream documentation packaged alongside the binary.

## Pose table columns
- `city`, `split`, `log_id` – provenance of each measurement.
- `timestamp_ns` – capture time in nanoseconds.
- `qw`, `qx`, `qy`, `qz` – ego pose quaternion.
- `tx`, `ty`, `tz` – ego translation in the city frame (metres).
- `zone_num_hemi` – UTM zone label (all northern hemisphere).
- `easting`, `northing` – projected UTM coordinates in metres.
- `latitude`, `longitude` – WGS84 coordinates computed via the published city anchors.

### Sample rows (first 5)
| city | split | log_id | timestamp_ns | qw | qx | qy | qz | tx | ty | tz | zone_num_hemi | easting | northing | latitude | longitude |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321560119000 | -0.108155 | 0.001635 | -0.006183 | -0.994113 | 599.537892 | -2959.546793 | -2.974099 | 14N | 621749.565372 | 3346930.452772 | 30.247881 | -97.734556 |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321659651000 | -0.108047 | 0.001117 | -0.004756 | -0.994134 | 598.731013 | -2959.370230 | -2.993882 | 14N | 621748.758493 | 3346930.629335 | 30.247883 | -97.734565 |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321759848000 | -0.108156 | 0.000556 | -0.004991 | -0.994121 | 597.926754 | -2959.195747 | -3.002643 | 14N | 621747.954235 | 3346930.803818 | 30.247884 | -97.734573 |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321860044000 | -0.107996 | 0.000077 | -0.004928 | -0.994139 | 597.106269 | -2959.014908 | -3.013438 | 14N | 621747.133749 | 3346930.984657 | 30.247886 | -97.734581 |
| ATX | train | 000VFSWWAAkobywItdrErpC6fedKDWg4 | 315980321960240000 | -0.107731 | -0.000679 | -0.006384 | -0.994159 | 596.283263 | -2958.833294 | -3.021527 | 14N | 621746.310743 | 3346931.166271 | 30.247888 | -97.734590 |
