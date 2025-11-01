# Argoverse 2 LiDAR Dataset - Original Data Structure

This document provides a comprehensive introduction to the **original Argoverse 2 LiDAR dataset** structure and format.

## Dataset Overview

**Argoverse 2 Sensor Dataset** is a large-scale autonomous driving dataset containing:
- Multi-modal sensor data (LiDAR, cameras, IMU, etc.)
- High-definition maps
- Vehicle trajectory annotations
- Multiple cities across the United States

**Our focus:** LiDAR point cloud data and associated poses for point cloud processing and alignment.

## 1. Dataset Organization

### Top-Level Structure

```
argoverse2/
├── train/              # Training split
│   ├── {log_id_1}/
│   ├── {log_id_2}/
│   └── ...
├── val/                # Validation split
│   ├── {log_id_1}/
│   ├── {log_id_2}/
│   └── ...
└── test/               # Test split
    ├── {log_id_1}/
    ├── {log_id_2}/
    └── ...
```

**Key Points:**
- **3 splits**: `train`, `val`, `test`
- **Log IDs**: Each recording session has a unique UUID identifier
  - Example: `0000a0e0-a333-4a84-b645-74b4c2a96bda`
- **Multiple cities**: Logs from different cities are mixed within each split

### Cities Covered

| City Code | City Name | UTM Zone | Coverage |
|-----------|-----------|----------|----------|
| ATX | Austin, Texas | 14N | Urban and suburban |
| DTW | Detroit, Michigan | 17N | Urban |
| MIA | Miami, Florida | 17N | Urban and coastal |
| PAO | Palo Alto, California | 10N | Urban and suburban |
| PIT | Pittsburgh, Pennsylvania | 17N | Urban and hilly terrain |
| WDC | Washington DC | 18N | Urban |

## 2. Log Directory Structure

Each log directory follows this standard structure:

```
{log_id}/
├── sensors/
│   └── lidar/
│       ├── {timestamp_1}.feather
│       ├── {timestamp_2}.feather
│       ├── {timestamp_3}.feather
│       └── ...                      # ~100-300 files per log
├── city_SE3_egovehicle.feather      # Vehicle poses in city frame
├── calibration/
│   └── egovehicle_SE3_sensor.feather # Sensor extrinsic calibration
└── map/
    └── log_map_archive_{log_id}__Summer____{city}_city_{id}.json  # HD map data
```

### Log Statistics

**Typical log characteristics:**
- **Duration**: 15-30 seconds of driving
- **LiDAR sweeps**: 150-300 sweeps (at ~10 Hz)
- **File count**: 150-300 `.feather` files in `sensors/lidar/`
- **Total size**: 50-500 MB per log (uncompressed)

## 3. LiDAR Data Format

### File Naming

LiDAR files are named by their **nanosecond timestamp**:
```
sensors/lidar/315967736019990908.feather
               └─────────────────┘
                  Timestamp (ns)
```

- **Sorted chronologically**: Filenames sort lexicographically = chronologically
- **Frequency**: ~10 Hz (one file every ~100 milliseconds)
- **Format**: Apache Feather (optimized columnar format)

### LiDAR Point Cloud Schema

Each `.feather` file contains a point cloud with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `x` | float32 | X coordinate in sensor frame (meters) |
| `y` | float32 | Y coordinate in sensor frame (meters) |
| `z` | float32 | Z coordinate in sensor frame (meters) |
| `intensity` | float32 | Reflectance intensity (0-255 range) |
| `laser_number` | int16 | Laser channel/ring ID (0-127) |
| `offset_ns` | int64 | Nanosecond offset from sweep timestamp |

**Coordinate System:**
- **Origin**: Sensor center
- **Axes**: Right-handed coordinate system
- **Units**: Meters
- **Range**: Typically 0-200m radius

**Point Cloud Size:**
- **Points per sweep**: ~50,000 - 120,000 points
- **Dense 360° coverage**: Full panoramic scan
- **Vertical channels**: 128 channels (high-density LiDAR)

### Example LiDAR Data

```python
# Reading a LiDAR sweep
import pyarrow.feather as feather

sweep = feather.read_table("sensors/lidar/315967736019990908.feather")
print(sweep.schema)

# Output:
# x: float
# y: float
# z: float
# intensity: float
# laser_number: int16
# offset_ns: int64

# Sample points:
#     x        y        z      intensity  laser_number  offset_ns
# -17.147   16.118   1.365      51           3          559872
#  -9.221    9.207   0.339       3          25          566784
# -17.131   16.118   1.936      48           7          569088
```

## 4. Vehicle Pose Data

### File: `city_SE3_egovehicle.feather`

Contains vehicle poses (position + orientation) for each LiDAR timestamp.

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_ns` | int64 | Timestamp (nanoseconds) matching LiDAR files |
| `qw` | float64 | Quaternion W component (rotation) |
| `qx` | float64 | Quaternion X component |
| `qy` | float64 | Quaternion Y component |
| `qz` | float64 | Quaternion Z component |
| `tx_m` | float64 | Translation X in city frame (meters) |
| `ty_m` | float64 | Translation Y in city frame (meters) |
| `tz_m` | float64 | Translation Z in city frame (meters) |

**Coordinate Frame:**
- **City Frame**: Local city-centric coordinate system
- **SE(3) Transform**: Rotation (quaternion) + Translation (meters)
- **Mapping**: `city_SE3_egovehicle` means "pose of ego vehicle in city frame"

**Usage:**
- Transform sensor-frame points to city frame
- Track vehicle trajectory
- Compute motion between sweeps

### Example Pose Data

```
timestamp_ns           qw       qx       qy       qz      tx_m      ty_m      tz_m
315967736019990908   0.9998   0.0123  -0.0156   0.0034  1234.56   7890.12   -15.34
315967736119990908   0.9998   0.0124  -0.0156   0.0035  1235.67   7891.23   -15.35
315967736219990908   0.9998   0.0125  -0.0157   0.0036  1236.78   7892.34   -15.36
```

## 5. Sensor Calibration

### File: `calibration/egovehicle_SE3_sensor.feather`

Contains **static** extrinsic calibration for all sensors relative to the ego vehicle frame.

**Schema (for LiDAR sensor):**

| Column | Type | Description |
|--------|------|-------------|
| `sensor_name` | string | Sensor identifier (e.g., "up_lidar") |
| `qw`, `qx`, `qy`, `qz` | float64 | Rotation quaternion |
| `tx_m`, `ty_m`, `tz_m` | float64 | Translation (meters) |

**Key Sensor:**
- **`up_lidar`**: Main roof-mounted LiDAR sensor
- **Static transform**: Does not change during the log
- **Purpose**: Transform from sensor frame to ego vehicle frame

**Transform Chain:**
```
LiDAR points (sensor frame)
    → [sensor_SE3_egovehicle] →
Ego vehicle frame
    → [city_SE3_egovehicle] →
City frame
    → [offset to UTM] →
UTM coordinates
```

## 6. Map Data

### File: `map/log_map_archive_{log_id}__Summer____{city}_city_{id}.json`

Contains HD map data for the area covered by the log.

**Filename Format:**
```
log_map_archive_{LOG_ID}__Summer____{CITY}_city_{MAP_ID}.json
```

**Example:**
```
log_map_archive_0QB8KZQ9HFftSYAPyyIktvRCbbE9oL9r__Summer____ATX_city_77093.json
                └──────────────────┬───────────────┘        └─┬─┘
                            Log UUID                    City Code
```

**Contents:**
- **City identifier**: City code embedded in filename after `____` (4 underscores)
  - Format: `{LOG_ID}__Summer____{CITY}_city_{MAP_ID}`
  - Cities: ATX, MIA, PIT, DTW, WDC, PAO
- **Season marker**: `Summer` (indicates data collection season)
- **Map elements**: Lane boundaries, crosswalks, drivable areas
- **Geographic reference**: Links to city coordinate system

**City Name Extraction:**
```
Filename: log_map_archive_0QB8KZQ9HFftSYAPyyIktvRCbbE9oL9r__Summer____ATX_city_77093.json
                                                                   └─┬─┘
                                                               City code: ATX
Split by "____" (4 underscores) → Take last part before "_city_"
```

**Note:** The filename includes the log UUID and season, making it unique per log.

## 7. Data Access

### Download from S3

Argoverse 2 data is hosted on AWS S3:

```bash
# S3 bucket structure
s3://argoverse/datasets/av2/lidar/
    ├── train/{log_id}/
    ├── val/{log_id}/
    └── test/{log_id}/
```

**Download tools:**
- **s5cmd**: Fast parallel downloads (recommended)
- **AWS CLI**: Standard AWS tool
- **Python SDK**: boto3

### File Sizes

| Component | Size per Log | Notes |
|-----------|-------------|-------|
| LiDAR `.feather` files | 40-400 MB | Depends on log duration |
| Poses | ~50 KB | Small, one row per sweep |
| Calibration | ~5 KB | Static, minimal size |
| Map | 1-10 MB | Varies by area coverage |
| **Total per log** | **50-500 MB** | Uncompressed |

## 8. Coordinate Systems

### Multiple Reference Frames

1. **Sensor Frame** (LiDAR `.feather` files)
   - Origin: LiDAR sensor center
   - Axes: Sensor-aligned
   - Units: Meters
   - **Raw point cloud data**

2. **Ego Vehicle Frame** (calibration)
   - Origin: Vehicle center
   - Axes: Vehicle-aligned (forward = X)
   - Units: Meters
   - **Intermediate frame**

3. **City Frame** (poses)
   - Origin: City-specific reference point
   - Axes: ENU (East-North-Up)
   - Units: Meters
   - **Trajectory frame**

4. **UTM** (derived)
   - Origin: UTM zone origin
   - Projection: Universal Transverse Mercator
   - Units: Meters
   - **Geographic alignment**

5. **WGS84 Lat/Lon** (derived)
   - Geographic coordinates
   - Units: Degrees
   - **GPS/mapping**

## 9. Temporal Continuity

### Timestamp Guarantees

- **Sequential**: Timestamps increase monotonically
- **Sorted filenames**: Lexicographic sort = chronological sort
- **Consistent**: Same timestamps in poses and LiDAR filenames
- **Nanosecond precision**: High-resolution timing

### Sweep Continuity

```python
# Example timestamps (nanoseconds):
315967736019990908   # Sweep 0
315967736119990908   # Sweep 1 (~100ms later)
315967736219990908   # Sweep 2 (~100ms later)
315967736319990908   # Sweep 3 (~100ms later)
315967736419990908   # Sweep 4 (~100ms later)
```

**Interval**: ~100 milliseconds (10 Hz frequency)

## 10. Data Quality

### Characteristics

- **High-density**: 128-channel LiDAR
- **360° coverage**: Full panoramic scans
- **Urban environments**: Complex scenes with buildings, vehicles, pedestrians
- **Motion**: Vehicle in motion during capture
- **GPS-aligned**: Poses aligned to geographic coordinates

### Known Patterns

- **Variable log duration**: 15-30 seconds typical
- **Urban density**: Point count varies by environment
  - Open areas: ~50K points/sweep
  - Dense urban: ~120K points/sweep
- **No gaps**: Continuous recording within each log

## Summary

The Argoverse 2 LiDAR dataset provides:

✅ **Organized structure**: Clear train/val/test splits
✅ **Multi-city coverage**: 6 cities across the US
✅ **High-quality LiDAR**: 128-channel, 10 Hz, dense point clouds
✅ **Precise poses**: SE(3) transforms with nanosecond timestamps
✅ **Standard formats**: Feather files for efficient access
✅ **Geographic alignment**: City frame + map data for localization

This rich dataset enables:
- LiDAR processing and segmentation
- Point cloud registration and alignment
- Vehicle trajectory analysis
- Multi-modal sensor fusion
- Geographic mapping and localization

---

## Raster data (training-ready)

In addition to the raw LiDAR/pose/map formats described above, we also produce "rasterized" training data used by the models in this repository. The script and helpers that perform this conversion live in `Data-pipeline-fetch/raster.py`.

What the raster preparation does (high-level):
- Reads processed point-cloud samples, pose and calibration data produced by the pipeline.
- Transforms sensor-frame points into a common geographic/city frame using the calibration and `city_SE3_egovehicle` poses.
- Reprojects the 3D points into a 2D image grid (raster) at a chosen spatial resolution (meters/pixel). Each pixel represents a small ground area.
- Rasterizes point attributes into per-pixel channels (for example: height/elevation, intensity, point count, or DSM/DEM layers). A simple height-channel often stores the height value for the highest (or average) point that falls into that pixel.
- Writes out tiles/arrays (numpy, image, or TFRecord depending on config) together with metadata describing coordinate transforms, resolution and any offsets applied.

How to use `Data-pipeline-fetch/raster.py` (overview):
- The file contains helpers and a small CLI to convert processed sample folders into raster tiles. You can either run it as a script on a directory of processed samples or import the module and call the helper functions from Python to customize parameters like tile size, resolution, and output format.
- Typical parameters you will choose: input sample directory, output directory, tile size (meters), raster resolution (meters/pixel), and whether to aggregate using max/min/mean for the height channel.

Why we apply a vertical shift when projecting heights into the raster image
- Coordinate reference differences: LiDAR points after transformation to the city frame have Z values in the city coordinate (often meters relative to some city/UTM origin). Raster products (DSM/DEM or image storage formats) may assume a different baseline (surface elevation, zero at sea-level, or a cropped tile origin). Directly storing raw Z values can lead to negative numbers or a large dynamic range that complicates on-disk formats and model training.
- Numerical stability and discretization: Many image formats and training pipelines expect non-negative values (e.g., uint8 or normalized floats). Adding a constant vertical shift (and recording it in metadata) moves all heights into a stable positive range and keeps quantization consistent across tiles.
- Removing sensor mounting bias: The LiDAR Z includes vehicle mounting height and vehicle motion. For some tasks we want heights relative to ground or relative to a digital surface model — applying a vertical offset lets us align the point heights with the raster reference surface (DSM/DEM) before storing.
- Occlusion handling / z-buffering: When multiple points fall into the same pixel, the rasterization rule (max, min, mean) interacts with vertical offsets. Using a fixed vertical shift consistently ensures the chosen aggregator behaves predictably across tiles and logs.

Simple formula (conceptual):

projected_pixel_value = (z_point - z_tile_base) + vertical_shift

Where:
- `z_point` is the point Z in the city frame (meters).
- `z_tile_base` is the base elevation or reference for the output tile (for example the DEM value or tile origin elevation).
- `vertical_shift` is a constant added so stored values are positive and numerically stable for the chosen output type.

Practical example:
- If LiDAR heights in a neighborhood are around -15 m (city frame origin placed above the area) and we want uint8 storage, adding +20 m moves values into the 0–255-ish range after any scaling. The exact shift value and scaling factor must be stored in tile metadata so loaders can convert back to real-world meters for evaluation.

Edge cases and best-practices
- Always store the vertical shift, resolution and coordinate transform inside the tile metadata (JSON sidecar or header) so training and inference code can invert the transformation.
- Choose the raster aggregator (max/min/mean) according to the downstream task. For ground height estimation, min/percentile is often more robust; for surface objects (cars, trees) max is common.
- Validate by overlaying transformed 3D points on the produced raster (color points by stored pixel height) — visual inspection catches sign/shift errors quickly.

Verification steps
- Visualize a few tiles and overlay original (transformed) points colored by height.
- Check histograms of stored height-channel values to ensure they fall in the expected range and that the vertical shift was applied.
- Confirm metadata keys (resolution, vertical_shift, tile_origin) are present for each tile.


**Next Steps:**
- See `Data_pipeline/README.md` for processing pipeline
- See `Data-pipeline-fetch/README.md` for downloading and processing automation
- See `GICP_intro.md` for point cloud alignment workflows
