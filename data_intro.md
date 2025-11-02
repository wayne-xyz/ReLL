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
- City identifier embedded in filename (format: `{LOG_ID}__Summer____{CITY}_city_{MAP_ID}`)
- Map elements: Lane boundaries, crosswalks, drivable areas
- Geographic reference: Links to city coordinate system

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

## Raster Data (Training-Ready)

The `Data-pipeline-fetch/raster.py` script converts processed point clouds into rasterized training data:

**Process:**
1. Transform sensor-frame points to geographic/city frame using calibration and poses
2. Reproject 3D points to 2D grid at specified resolution (default: 0.2 m/pixel)
3. Rasterize attributes (height, intensity) into per-pixel channels
4. Save as numpy arrays (.npy) with metadata (transform, resolution)

**Vertical Shift for Height Rasters:**
- **Purpose**: Align LiDAR heights with DSM reference surface and ensure non-negative values
- **Formula**: `pixel_value = (z_point - z_tile_base) + vertical_shift`
- **Why needed**: City frame Z can be negative; shift moves values to stable positive range
- **Metadata**: Shift value stored in metadata for inverse transformation during evaluation

## Data Processing Pipeline Stages

The `Data-pipeline-fetch/fetch_and_process_pipeline.py` orchestrates the complete data processing workflow. Understanding this pipeline is crucial as it produces the training-ready samples.

### Pipeline Overview (4 Stages)

The pipeline processes raw Argoverse 2 LiDAR logs through four sequential stages:

1. **Stage 1: LiDAR Processing** - Create macro sweep from multiple LiDAR frames
2. **Stage 2: Imagery & DSM Processing** - Fetch aligned imagery and Digital Surface Model
3. **Stage 3: DSM Extraction (Filtering)** ⭐ - Filter DSM points near LiDAR coverage
4. **Stage 4: GICP Alignment** - Align LiDAR to DSM using GICP registration

### Stage 3: DSM Extraction - Spatial Filtering Rule

**Purpose**: Before running GICP alignment, we filter the DSM point cloud to keep only points that are spatially near the LiDAR coverage. This dramatically reduces computational cost and improves GICP quality.

**Implementation**: `Data-pipeline-fetch/lib/dsm_extraction.py::extract_dsm_near_lidar()`

#### Filtering Method

**Algorithm** (XY-plane spatial filtering):
1. Load LiDAR points (UTM coordinates): Extract XY positions only
2. Load DSM points (LAZ file): Full XYZ point cloud
3. Build KD-Tree from LiDAR XY positions
4. For each DSM point: Find distance to nearest LiDAR point (XY only, ignore Z)
5. Keep DSM points where: `distance_xy <= max_distance`
6. Save filtered DSM as parquet file

**Key Parameters**:
```python
max_distance = 0.5  # meters (default threshold)
```

**Code Implementation**:
```python
# Build KDTree of LiDAR points (XY only)
lidar_tree = cKDTree(lidar_xy)

# For each DSM point, find distance to nearest LiDAR point
distances, _ = lidar_tree.query(dsm_points[:, :2], k=1)

# Filter DSM points
mask = distances <= max_distance
extracted_dsm = dsm_points[mask]
```

#### Why This Rule?

**1. Computational Efficiency**
- DSM point clouds are typically **very dense** (millions of points covering large areas)
- LiDAR coverage is **spatially limited** (vehicle's sensor footprint)
- Most DSM points are far from LiDAR coverage and irrelevant for alignment
- **Result**: Filtering reduces DSM by 80-95% while keeping all relevant points

**2. GICP Quality Improvement**
- GICP (Generalized Iterative Closest Point) is sensitive to point density balance
- Too many DSM points vs. LiDAR points → biased correspondence matching
- Filtering ensures **balanced point densities** in the overlap region
- Removes outlier DSM points that could degrade alignment

**3. Memory Constraints**
- Full DSM + LiDAR in memory can exceed available RAM
- Filtered DSM is manageable for real-time processing

**4. Spatial Relevance**
- Only DSM points within `max_distance` can meaningfully contribute to alignment
- Points beyond 0.5m in XY plane are unlikely to correspond to the same surface

#### How `max_distance = 0.5m` Was Chosen

**Empirical reasoning**:
- **LiDAR raster resolution**: Default is 0.2 m/pixel
- **Buffer for uncertainty**:
  - GPS/IMU positioning errors: ~0.1-0.3m typical
  - DSM georeferencing errors: ~0.1-0.2m typical
  - LiDAR measurement noise: ~0.02-0.05m
- **Total uncertainty budget**: ~0.3-0.5m
- **Chosen threshold**: 0.5m provides comfortable margin

**Validation**:
- Distance statistics are logged for each sample:
  ```
  Distance stats: min=0.001m, max=0.498m, mean=0.234m
  ```
- **p95 (95th percentile)** typically < 0.45m, confirming 0.5m captures the relevant region

#### Typical Filtering Results

Example output from pipeline:
```
[Stage 3/4] DSM Extraction...
✓ Extracted DSM: sample_001_extract_dsm_utm.parquet
✓ DSM points: 2,458,392 → 312,847 (87.3% reduction)
✓ Distance stats: min=0.002m, max=0.499m, mean=0.187m
✓ Stage 3 completed in 3.2s
```

**Interpretation**:
- **Original DSM**: 2.5M points (entire city block)
- **Filtered DSM**: 313K points (only LiDAR coverage area)
- **Reduction**: 87.3% fewer points → 7-10x faster GICP
- **Coverage**: All points within 0.5m of LiDAR → no information loss for alignment

#### Design Choices

**XY-only filtering (ignore Z):**
- LiDAR and DSM measure different surfaces (LiDAR: tops of objects; DSM: bare earth)
- Z difference can be 5-20m for vehicles/buildings → 3D distance filtering would reject valid points

**Fixed 0.5m threshold:**
- Consistent across all samples (no adaptive complexity)
- Works well empirically for diverse urban environments
- KD-Tree distance filtering captures irregular coverage shapes better than bounding boxes

### Stage 4: GICP Alignment (Uses Filtered DSM)

After DSM extraction, the filtered DSM is used for GICP alignment:

```python
gicp_result = align_lidar_to_dsm(
    lidar_parquet_path=macro.utm_point_path,
    dsm_parquet_path=extracted_dsm_path,  # Filtered DSM from Stage 3
    params=GICPParams(
        voxel_size=0.3,
        max_corr_dist=0.8,
        max_iter=60,
    ),
)
```

**Benefits of using filtered DSM**:
- ✅ Faster convergence (fewer points → fewer correspondences)
- ✅ Better alignment quality (balanced densities)
- ✅ Lower memory usage
- ✅ More stable optimization (no distant outliers)

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

| File | Shape | Dtype | Description |
|------|-------|-------|-------------|
| `gicp_height.npy` / `non_aligned_height.npy` | (H, W) | float32 | LiDAR height map in meters, NaN for gaps |
| `gicp_intensity.npy` / `non_aligned_intensity.npy` | (H, W) | float32 | LiDAR intensity (0-255), NaN for gaps |
| `dsm_height.npy` | (H, W) | float32 | Digital Surface Model elevation |
| `imagery.npy` | (C, H, W) or (H, W, C) | uint8/float32 | RGB imagery (auto-converted to CHW) |
| `resolution.pkl` | scalar | float | Spatial resolution (m/pixel), e.g., 0.2 |
| `transform.pkl` | Affine | - | Pixel↔world coordinate mapping |

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


