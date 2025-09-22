# Data Preparation Overview

The preparation workflow in `Data_pipeline/` now produces a consistent UTM footprint from LiDAR through imagery/DSM. It runs in two stages:
1. `Data_pipeline/combine_lidar_sweeps.py` merges consecutive sweeps into a “macro sweep”, exporting both sensor-frame and UTM-frame point clouds plus rich metadata.
2. `Data_pipeline/sample_pipeline_cli.py` / `sample_pipeline_gui.py` consume that macro sweep and build co-registered imagery and DSM rasters.

The notes below summarise inputs, processing, and outputs so you can recreate the samples in `Data-Sample/` (or your own target directory).

## Stage 1 — Macro LiDAR sweep (`Data_pipeline/combine_lidar_sweeps.py`)
### Inputs
- `.../<log_id>/sensors/lidar/*.feather` – raw LiDAR sweeps (one per timestamp).
- `.../<log_id>/city_SE3_egovehicle.feather` – ego poses in the city/world frame.
- `.../<log_id>/calibration/egovehicle_SE3_sensor.feather` – LiDAR-to-ego calibration.

These three files ship with every AV2 log. No external `av2_coor.*` dependency remains; UTM alignment is rebuilt from the log’s map annotation.

### Processing outline
1. **Sweep window selection** – Choose consecutive indices (default 5 ≈ 0.5 s). CLI/GUI pick the first valid window unless explicitly set.
2. **Sensor → city transforms** – Compose the ego pose with the LiDAR calibration to obtain `city_from_sensor` for every sweep.
3. **Lift returns into the city frame** – Cache `x,y,z,intensity,laser_number,offset_ns` in double precision before converting back to float32.
4. **Reference-frame alignment** – Invert the chosen centre sweep so every point cloud is expressed in that sensor frame; provenance columns (`source_index`, `source_timestamp_ns`) travel with the points.
5. **Optional cropping** – Squares can be applied in both sensor and UTM frames. Each crop records the applied size (`crop_square_m_sensor`, `crop_square_m_utm`) and keeps track of the retained points.
6. **UTM projection** – Translate city coordinates into UTM using per-city offsets derived from `map/*.json` metadata. A dedicated UTM point parquet is written alongside the sensor-frame parquet.
7. **Metadata capture** – All outputs share a single metadata parquet containing: reference pose/quaternion, per-sweep sensor positions (city, lat/lon, UTM), footprint extents (`extent_x/y`, `utm_extent_x/y`), motion metrics, timestamps, and crop settings.
8. **Write artefacts** – Results are compressed with zstd and returned via a `MacroSweepResult` named tuple so callers have immediate access to paths, applied crop sizes, and footprint sizes.

### Outputs
- `<prefix>.parquet` – macro sweep in the reference sensor frame.
- `<prefix>_utm.parquet` – matching macro sweep in UTM.
- `<prefix>_meta.parquet` – metadata/summary record.

CLI example:
```bash
python Data_pipeline/sample_pipeline_cli.py \
    path/to/log_dir \
    Argoverse2-geoalign/ATX \
    path/to/output_dir [--crop-size 64]
```
The GUI (`python Data_pipeline/sample_pipeline_gui.py`) wraps the same pipeline with file pickers.

## Stage 2 — Imagery + DSM companions (`sample_pipeline_cli.py` / `_gui.py`)
### Additional inputs
- Stage‑1 outputs (sensor parquet + metadata; the CLI also reads the UTM parquet for footprint logging).
- City assets under `Argoverse2-geoalign/<CITY>/`:
  * `Imagery/.../*.jp2` tiles plus `imagery_tile_bounds.csv`.
  * `DSM/.../*.laz` tiles provided in EPSG:6578 (US survey feet).

### Processing outline
1. **Footprint reconstruction** – Use metadata UTM offsets (no affine fitting) to rebuild the LiDAR footprint, applying a configurable buffer when no crop is set.
2. **Tile discovery** – Filter `imagery_tile_bounds.csv` by projecting UTM bounds back to lon/lat; the GUI/CLI report if any coverage is missing.
3. **Imagery reprojection** – Mosaic the necessary JP2 tiles into a single RGB GeoTIFF at the requested resolution (default 0.3 m UTM pixels).
4. **DSM reprojection** – Load the stratmap LAZ tiles with `laspy`, convert from EPSG:6578 to the LiDAR’s UTM zone, crop to the footprint, and rewrite as LAZ (or LAS fallback).
5. **Result packaging** – File paths and coverage extents are returned via a `StageTwoResult` named tuple so callers can log footprint sizes or post-process further.

### Outputs
- `<prefix>_imagery_utm.tif` – RGB orthomosaic aligned with the LiDAR footprint.
- `<prefix>_dsm_utm.laz` – DSM point sample in the same UTM coordinates.

### Crop & extent logging
Both CLI and GUI announce the sensor and UTM extents for each artefact. If a requested crop is adjusted, the applied size is echoed for both frames.

## Optional helpers
- `ArgoverseLidar/rasterize_macro_sweep.py` – converts any macro sweep to a 2-band (height, intensity) GeoTIFF for quick inspection.
- `analysis/pointcloud_viewer.py` – previews LiDAR + DSM together. It now understands UTM parquets and handles LAS headers without explicit EPSG tags.
- `analysis/data_shift_gicp.py` – aligns UTM LiDAR parquets to DSM via vertical offset + optional GICP (metadata optional when UTM columns exist).

With these updates, the entire pipeline works off the macro sweep exports: produce the sweep via Stage 1, then feed it (plus the city asset bundle) into Stage 2 to obtain aligned imagery & DSM outputs ready for modelling or QA.
