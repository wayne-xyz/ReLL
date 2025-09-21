# Data Preparation Overview

The current workflow lives under `Data_pipeline/` and is split into two stages:
1. `Data_pipeline/combine_lidar_sweeps.py` fuses raw AV2 sweeps into macro-sweep Parquet bundles.
2. `Data_pipeline/sample_pipeline_cli.py` / `sample_pipeline_gui.py` transform those bundles into co-registered imagery and DSM products.

The notes below show the inputs, processing steps, and outputs so new contributors can reproduce the samples in `Data-Sample/` (or any custom folder).

## Stage 1 — LiDAR macro-sweeps (`Data_pipeline/combine_lidar_sweeps.py`)
### Inputs
- `argverse_data_preview/val/<log_id>/sensors/lidar/*.feather`: one 10 Hz sweep per file in the sensor frame.
- `argverse_data_preview/val/<log_id>/city_SE3_egovehicle.feather`: ego poses in the city frame for each timestamp.
- `argverse_data_preview/val/<log_id>/calibration/egovehicle_SE3_sensor.feather`: static LiDAR-to-ego calibration.

Everything needed comes with the log; we no longer depend on `av2_coor.*` because the city→UTM transform is rebuilt directly from the log metadata.

### Processing steps
1. **Select a sweep window.**  The script expects consecutive indexes (default 5 ≈ 0.5 s).  The CLI/GUI picks the first available chunk automatically when not specified.
2. **Build sensor→city transforms.**  For each sweep timestamp we compose the ego pose with the LiDAR calibration to obtain `city_from_sensor`.
3. **Transform raw returns.**  Points (`x,y,z,intensity,laser_number,offset_ns`) are lifted into the city frame and cached.
4. **Measure vehicle motion.**  The script records both cumulative path length and straight-line displacement over the window for diagnostics.
5. **Re-express in a reference frame.**  We invert the middle sweep’s transform (or user-supplied `center_index`) so every point cloud is expressed in that sensor frame.
6. **Concatenate points with provenance.**  Arrays are stacked and labelled with `source_index` and `source_timestamp_ns` before writing to parquet.  No downsampling or filtering occurs.
7. **Populate metadata.**  The script now derives UTM offsets from the city annotation (`map/...city_XXXX.json`) plus per-sweep city positions.  It stores `center_utm_zone`, `center_utm_easting_m`, `center_utm_northing_m`, and per-sweep `sensor_positions_{city_m,latlon_deg,utm}` alongside the usual bounding boxes, timestamps, and motion metrics.
8. **Write outputs.**  Points and metadata are written side-by-side (`<prefix>.parquet`, `<prefix>_meta.parquet`) with zstd compression.

### Results
- `<output>/<prefix>.parquet`: points expressed in the reference sensor frame.
- `<output>/<prefix>_meta.parquet`: metadata with full city/UTM alignment details.

Use the CLI:
```
python Data_pipeline/sample_pipeline_cli.py argverse_data_preview/val/<log_id> Argoverse2-geoalign/ATX <output_dir>
```
or the GUI (`python Data_pipeline/sample_pipeline_gui.py`) to generate the stage-2 outputs in one shot.

## Stage 2 — Imagery & DSM companions (`Data_pipeline/sample_pipeline_cli.py` / `sample_pipeline_gui.py`)
### Source data
- Macro-sweep point & metadata parquets from Stage 1.
- A city folder under `Argoverse2-geoalign/<CITY>/` containing:
  * `Imagery/<subdirectories>/<*.jp2>` with footprints in `imagery_tile_bounds.csv`.
  * `DSM/stratmap.../*.laz` tiles in EPSG:6578 (US survey feet).

### Processing pipeline
1. **Read UTM offsets from metadata.**  Instead of solving an affine fit, the pipeline reuses the stored `center_utm_*` and `sensor_positions_utm` to anchor the macro-sweep footprint directly in UTM.
2. **Derive LiDAR bounds.**  Points are rotated/translated into the city frame, offset to UTM, and buffered (±2 m) to ensure imagery/DSM coverage.
3. **Select imagery tiles.**  The script filters `imagery_tile_bounds.csv` by projecting the UTM bounds back to lon/lat.  Missing tiles trigger a descriptive error.
4. **Reproject imagery.**  Each JPEG2000 tile is read and reprojected into a mosaic grid (0.3 m UTM pixels).  The CLI uses Rasterio’s `reproject`; the GUI shares the same core function.
5. **Clip DSM tiles.**  LAZ files are read with `laspy`, transformed from EPSG:6578 to the target UTM CRS, scaled into metres, and clipped to the buffered bounds.  Points are concatenated and written to `*.laz` (with `.las` fallback if compression isn’t available).
6. **Write results.**  Imagery and DSM outputs inherit the macro-sweep prefix (e.g. `combined_0p5s_imagery_utm.tif`, `combined_0p5s_dsm_utm.laz`).

### Outputs
- `<output>/<prefix>_imagery_utm.tif`: RGB GeoTIFF aligned to the macro-sweep footprint.
- `<output>/<prefix>_dsm_utm.laz`: DSM sample in the same UTM frame.

### Optional: rasterized LiDAR height/intensity
`ArgoverseLidar/rasterize_macro_sweep.py` now offers a CLI/GUI to convert any macro-sweep (`points`, `meta`) into a 2-band GeoTIFF (height + intensity).  Handy for quick inspection or ML baselines:
```
python ArgoverseLidar/rasterize_macro_sweep.py --gui
```

## Preview & QA helpers
- `Data_sample_helper/imagery_dsm_viewer.py`: simple GUI to preview imagery (Matplotlib), DSM (Open3D), and LiDAR parquets (Open3D) independently.
- `Data_pipeline/sample_pipeline_cli.py`: scripted way to regenerate the full pipeline for any log (useful for batch processing or testing).

With these pieces aligned in `Data_pipeline/`, generating new samples is just a matter of pointing the CLI/GUI at a log folder and the matching `Argoverse2-geoalign/<CITY>` assets.
