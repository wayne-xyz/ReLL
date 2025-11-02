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

### Training Data Format (Post-Rasterization)

After `Data-pipeline-fetch/raster.py` processes the raw data, each training sample is stored as a **directory** containing multiple `.npy` files and metadata. These processed samples are used directly by the training pipeline in `Train/data.py`.

#### Sample Directory Structure

Each training sample directory contains:

```
{sample_id}/
├── gicp_height.npy              # GICP-aligned LiDAR height raster
├── gicp_intensity.npy           # GICP-aligned LiDAR intensity raster
├── non_aligned_height.npy       # Non-aligned (raw) LiDAR height raster
├── non_aligned_intensity.npy    # Non-aligned (raw) LiDAR intensity raster
├── dsm_height.npy               # Digital Surface Model height raster
├── imagery.npy                  # RGB or multispectral imagery
├── resolution.pkl               # Raster resolution (meters/pixel)
├── transform.pkl                # Rasterio Affine transform (pixel↔world coords)
├── profile.pkl                  # (Optional) Rasterio profile metadata
└── metadata.pkl                 # (Optional) Additional sample metadata
```

#### Raster Array Specifications

**LiDAR Rasters** (`gicp_height.npy`, `non_aligned_height.npy`):
- **Shape**: `(H, W)` - 2D height map
- **Dtype**: `float32`
- **Units**: Meters (elevation)
- **Missing data**: `NaN` for pixels with no LiDAR returns
- **Projection**: Already rasterized at specified resolution (default: 0.2 m/px)

**LiDAR Intensity** (`gicp_intensity.npy`, `non_aligned_intensity.npy`):
- **Shape**: `(H, W)` - 2D intensity map
- **Dtype**: `float32`
- **Range**: Typically 0-255 (reflectance values)
- **Missing data**: `NaN` for pixels with no LiDAR returns

**DSM Height** (`dsm_height.npy`):
- **Shape**: `(H, W)` - 2D Digital Surface Model
- **Dtype**: `float32`
- **Units**: Meters (elevation)
- **Source**: Reference elevation data (e.g., from imagery/photogrammetry)
- **Missing data**: `NaN` for unavailable regions

**Imagery** (`imagery.npy`):
- **Shape**: `(H, W, C)` or `(C, H, W)` - RGB or multispectral
- **Channels**: Typically 3 (RGB), can be more for multispectral
- **Dtype**: `uint8` (0-255) or `float32` (normalized)
- **Order**: HWC (Height-Width-Channel) or CHW (Channel-Height-Width)
- **Note**: Training code automatically converts HWC→CHW if needed

#### Metadata Files

**resolution.pkl**:
- **Type**: `float`
- **Value**: Spatial resolution in meters per pixel (e.g., `0.2`)
- **Usage**: Scale factor for geometric operations

**transform.pkl**:
- **Type**: `rasterio.Affine`
- **Purpose**: Maps pixel coordinates to world coordinates (UTM/city frame)
- **Example**:
  ```python
  Affine(0.2, 0.0, 500000.0,
         0.0, -0.2, 4000000.0)
  # Translation(x_min, y_max) * Scale(res, -res)
  ```

#### Training Data Loading Pipeline

The training pipeline (`Train/data.py`) processes these files as follows:

**1. Load rasters from sample directory:**
```python
rasters = raster_builder_from_processed_dir(sample_dir)
# Returns dict with keys: gicp_height, gicp_intensity, non_aligned_height,
#                         non_aligned_intensity, dsm_height, imagery, resolution
```

**2. Select LiDAR variant** (GICP or non-aligned):
```python
if lidar_variant == "gicp":
    lidar_height = rasters["gicp_height"]
    lidar_intensity = rasters["gicp_intensity"]
else:  # non_aligned
    lidar_height = rasters["non_aligned_height"]
    lidar_intensity = rasters["non_aligned_intensity"]
```

**3. Fill NaN gaps and create validity masks:**
```python
def replace_nan_with_zero(tensor):
    mask = torch.isfinite(tensor)
    cleaned = torch.where(mask, tensor, torch.zeros_like(tensor))
    return cleaned, mask.float()

lidar_height, lidar_mask = replace_nan_with_zero(lidar_height)
lidar_intensity, _ = replace_nan_with_zero(lidar_intensity)
dsm_height, dsm_mask = replace_nan_with_zero(dsm_height)
```

**Key point**: Missing pixels (NaN) are filled with **zeros**, and a binary mask channel tracks which pixels had valid data.

**4. Construct multi-channel tensors:**

**LiDAR Tensor** (3 channels):
```python
lidar_tensor = torch.stack([
    lidar_height,              # Channel 0: Height (meters, NaN→0)
    lidar_intensity / 255.0,   # Channel 1: Intensity (normalized 0-1)
    lidar_mask,                # Channel 2: Validity mask (1=valid, 0=empty)
], dim=0)
# Shape: (3, H, W)
```

**Map Tensor** (5 channels):
```python
map_tensor = torch.cat([
    imagery[:3] / 255.0,  # Channels 0-2: RGB (normalized 0-1)
    dsm_height.unsqueeze(0),    # Channel 3: DSM height (meters, NaN→0)
    dsm_mask.unsqueeze(0),      # Channel 4: DSM validity mask
], dim=0)
# Shape: (5, H, W)
```

**5. Apply augmentation** (rotation + translation):
```python
# Random geometric augmentation
warped_lidar = affine_warp(lidar_tensor, angle_deg=dtheta, translate_px=(dx, dy))

# Target pose offset (what the model should predict)
target_mu = torch.tensor([-dx_m, -dy_m, -dtheta_deg], dtype=torch.float32)
```

#### Training Batch Format

Each training batch contains:

```python
{
    "lidar": Tensor,        # Shape: (B, 3, H, W) - Warped LiDAR with mask
    "map": Tensor,          # Shape: (B, 5, H, W) - RGB + DSM + mask
    "pose_mu": Tensor,      # Shape: (B, 3) - Target [dx_m, dy_m, dtheta_deg]
    "resolution": Tensor,   # Shape: (B,) - Meters per pixel
    "sample_idx": Tensor,   # Shape: (B,) - Sample index in dataset
}
```

#### Gap Filling Strategy

**Why fill with zeros instead of interpolation?**
1. **Explicit missing data tracking**: The mask channel (channel 2 for LiDAR, channel 4 for DSM) explicitly tells the model which pixels are valid vs. filled.
2. **No hallucination**: Zero-filling doesn't create fake height values — the model learns to recognize zero+mask=0 as "no data here."
3. **Computational efficiency**: Simple, deterministic, no expensive interpolation.
4. **Training stability**: Consistent behavior across samples.

**Alternative method (data pipeline visualization)**:
- For **visualization** purposes (not training), `Data-pipeline-fetch/raster.py` uses a shifted baseline:
  ```python
  shift = np.percentile(finite_values, 0.5)  # 0.5th percentile
  shifted = height - shift
  shifted = np.maximum(shifted, 0.0)  # Clamp to ≥0
  ```
- This makes visualizations easier to interpret but is **NOT** used during training.

#### Viewing Training Samples

Two utilities are provided for visualizing processed samples:

**1. `utilities/viewer_raster.py`** - View all channels in a sample:
```bash
python utilities/viewer_raster.py PATH/TO/SAMPLE_FOLDER
```
Shows: DSM, GICP height/intensity, RGB imagery, non-aligned height/intensity, plus original vs. shifted histograms.

**2. `utilities/projection_compare.py`** - Compare processing methods:
```bash
python utilities/projection_compare.py PATH/TO/FILE.parquet --crop --crop-size 30.0
```
Compares gap-filled (training method) vs. shifted+filled (visualization method) side-by-side with distributions.

#### Summary: From Raw Data to Training Tensors

```
Raw Argoverse 2 LiDAR (.feather)
    ↓ [Data-pipeline-fetch/raster.py]
    ↓ • Transform to city/UTM frame
    ↓ • Rasterize to 2D grid (0.2 m/px)
    ↓ • Save as .npy arrays
    ↓
Sample directory (gicp_height.npy, imagery.npy, etc.)
    ↓ [Train/data.py: raster_builder_from_processed_dir()]
    ↓ • Load .npy arrays
    ↓ • Fill NaN → 0 and create masks
    ↓
Multi-channel tensors
    ↓ • LiDAR: [height, intensity/255, mask] - 3 channels
    ↓ • Map: [R, G, B, dsm_height, dsm_mask] - 5 channels
    ↓ [Train/data.py: __getitem__()]
    ↓ • Apply random rotation/translation to LiDAR
    ↓ • Compute target pose offset
    ↓
Training batch ready for model
```

This pipeline ensures:
- ✅ Consistent spatial resolution across all modalities
- ✅ Aligned coordinate frames (all data in same grid)
- ✅ Explicit missing data handling (masks + zero-fill)
- ✅ Geometric augmentation for robust training
- ✅ Metadata preservation (resolution, transform) for inference


