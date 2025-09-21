# Data Preparation Overview

This note walks through the two conversion stages we run on top of the Argoverse 2 preview logs:
1. `ArgoverseLidar/combine_lidar_sweeps.py` fuses raw LiDAR sweeps into macro-sweep Parquet bundles.
2. `Argoverse2-geoalign/prepare_imagery_dsm.py` turns those macro-sweeps into co-registered imagery and DSM products.

Everything below aims to be a step-by-step description so newcomers can trace how the original files become the light-weight samples in `Data-Sample/`.

## Stage 1 — LiDAR macro-sweeps (`combine_lidar_sweeps.py`)
### Inputs
- `argverse_data_preview/val/<log_id>/sensors/lidar/*.feather`: one 10 Hz sweep per file in the sensor frame.
- `argverse_data_preview/val/<log_id>/city_SE3_egovehicle.feather`: ego poses in the city frame for every timestamp.
- `argverse_data_preview/val/<log_id>/calibration/egovehicle_SE3_sensor.feather`: static LiDAR-to-ego calibration.
- Optional pose catalog (`av2_coor.parquet` or `.feather`) with latitude/longitude samples for GNSS enrichment.

### Processing steps
1. **Pick consecutive sweeps.**  We index into the LiDAR directory, choose a fixed-length window (default 5 files ≈ 0.5 s), and name the output prefix from the first/last timestamps.
2. **Load motion and extrinsics.**  Poses from `city_SE3_egovehicle` give ego→city transforms per timestamp.  The calibration file supplies the rigid transform from the LiDAR sensor to the ego frame.  Multiplying them yields each sweep’s sensor→city transform.
3. **Lift raw points into the city frame.**  For every selected feather file we read `x,y,z,intensity,laser_number,offset_ns`, apply the sensor→city transform, and cache both the transformed points and original attributes.
4. **Measure vehicle motion.**  Translational deltas between consecutive poses produce `sensor_motion_length_m` (path length) and `sensor_displacement_m` (straight-line distance) stored later in metadata.
5. **Choose a reference sweep.**  Either the middle index or a user-specified one becomes the anchor frame.  We invert its sensor→city transform so all city-frame points can be re-expressed back into that sweep’s sensor frame.  This keeps the combined cloud internally consistent.
6. **Stack data and annotate provenance.**  After re-transforming the points we concatenate the arrays, track where each sample came from (`source_index`, `source_timestamp_ns`), and create a Parquet table with columns `[x, y, z, intensity, laser_number, offset_ns, source_index, source_timestamp_ns]` in `float32`/`int16`/`int64` formats.
7. **Summarise metadata.**  Bounding boxes, extents, sweep count, timestamps, and centre-frame pose information are captured.  If GNSS data exists, we attach the centre latitude/longitude, compute the UTM zone/easting/northing, and store per-sweep `sensor_positions_{city_m,latlon_deg,utm}` as JSON strings.  We also attempt to infer the city tag from the `map/` folder.
8. **Write outputs.**  Points go to `combined_...parquet`; metadata goes to the sibling `_meta.parquet`.  Compression defaults to `zstd` to keep files compact without losing precision.

### Results
- `Data-Sample/combined_0p5s.parquet`: macro-sweep points expressed in the chosen reference sensor frame.
- `Data-Sample/combined_0p5s_meta.parquet`: companion metadata enabling conversions back to city/WGS84/UTM coordinates.

These two files are the sole inputs for the imagery/DSM stage, so keeping their coordinate bookkeeping accurate is critical.

## Stage 2 — Imagery & DSM companions (`prepare_imagery_dsm.py`)
### Source data and directories
- `Data-Sample/combined_0p5s.parquet` and `_meta.parquet`: produced by Stage 1.
- `Argoverse2-geoalign/ATX/Imagery`: airborne RGB tiles (`*.jp2`) with footprints listed in `imagery_tile_bounds.csv`.
- `Argoverse2-geoalign/ATX/DSM`: StratMap DSM tiles (`*.laz`) stored in EPSG:6578 (US survey feet).

All paths live at the top of the script so outputs land back in `Data-Sample/` as `imagery_utm.tif` and `dsm_utm.laz`.

### Processing pipeline
1. **Derive a city→UTM transform.**  The metadata parquet lists sensor positions in both the city frame and UTM coordinates.  Dropping null entries, we solve a least-squares affine map `X_city → X_utm`, smoothing over any small discrepancies between the frames.
2. **Project LiDAR points into UTM space.**  Using the centre sweep’s quaternion/translation, every point from the macro-sweep is rotated into the city frame, translated to absolute coordinates, then pushed through the affine map into UTM zone 14N.  A ±2 m buffer widens the min/max corners to guarantee coverage.
3. **Select overlapping imagery tiles.**  We convert the buffered UTM bounds back to lon/lat, filter `imagery_tile_bounds.csv` for intersecting footprints, and collect the matching JPEG2000 imagery rows.  Missing tiles trigger a clear error before any heavy processing.
4. **Reproject and mosaic RGB imagery.**  Each tile is opened, wrapped in a `WarpedVRT` targeting UTM 14N at 0.3 m resolution, and merged with Rasterio’s `merge`.  The clipped mosaic is written to `imagery_utm.tif`, keeping the original band count and noting CRS/transform for downstream use.
5. **Clip DSM point clouds.**  DSM filenames are derived from the same tile table (`sub_tile_id`/`big_tile_id`).  We read each LAZ with `laspy`, transform coordinates from EPSG:6578 into UTM 14N, rescale heights from US survey feet to metres (× 0.3048006096), drop out-of-bounds samples, and concatenate the rest.  The result is saved as `dsm_utm.laz` (falling back to `.las` if compression is unavailable) with centimetre-scale quantisation.
6. **Handle PROJ data.**  At import time we set `PROJ_LIB` so rasterio/GDAL can resolve grid definitions on fresh machines.

### Outputs and coordinate conventions
- `Data-Sample/imagery_utm.tif`: RGB GeoTIFF aligned to UTM zone 14N, 0.3 m pixels, covering the buffered LiDAR footprint.
- `Data-Sample/dsm_utm.laz`: DSM subset in the same UTM frame and metres, ready for overlay or meshing.

With both stages complete, LiDAR, imagery, and DSM now share the same UTM coordinates, which makes inspection (`ArgoverseLidar/imagery_dsm_viewer.py`) or downstream modelling straightforward.
