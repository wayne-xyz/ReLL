# Data Pipeline Fetch

An integrated pipeline that fetches Argoverse2 LiDAR data from S3 and processes it into aligned point cloud segments with imagery and DSM.

## Overview

This is a **standalone, self-contained pipeline** that fetches Argoverse2 LiDAR data from S3 and processes it into aligned point cloud segments with imagery and DSM. All processing code is included in the `lib/` module - no external dependencies on other project folders.

### Key Features

✅ **Smart city filtering** - Checks city BEFORE downloading (saves bandwidth)
✅ **Skip processed logs** - Automatically skips logs that already have output
✅ **Configurable downloads** - Choose city (Austin) and exact number of samples
✅ **No storage of raw data** - Downloads, processes, and deletes original files
✅ **5-second time segments** - Divides long sequences into manageable chunks
✅ **No train/val/test split** - All output in single directory structure
✅ **Complete alignment** - Generates LiDAR + imagery + DSM for each segment
✅ **Vertical alignment** - Ground-based elevation alignment between LiDAR and DSM
✅ **DSM extraction** - Extracts DSM points within 0.5m of LiDAR for efficient alignment
✅ **GICP alignment** - Generalized ICP alignment of LiDAR to DSM reference
✅ **Quality metrics** - Comprehensive alignment quality assessment (RMSE, distances, percentiles)
✅ **Incremental CSV** - Appends to existing summary files, never overwrites
✅ **Detailed logging** - Track progress and time cost for each sample
✅ **Automatic cleanup** - Removes temporary files after processing

## Project Structure

```
Data-pipeline-fetch/
├── fetch_and_process_pipeline.py  # Main pipeline script
├── config.yaml                     # Main configuration file
├── config_test.yaml                # Quick test configuration
├── README.md                       # This file
└── lib/                            # Self-contained processing library
    ├── __init__.py                 # Module exports
    ├── lidar_processing.py         # LiDAR sweep combining & transformations
    └── imagery_processing.py       # Imagery & DSM alignment
```

**Standalone Design:** All processing code is in the `lib/` module, making this folder fully self-contained. No dependencies on external `Data_pipeline` or `ArgoverseLidar` folders.

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pyarrow pandas rasterio pyproj laspy scipy pyyaml open3d
```

**Required packages:**
- `numpy`, `pyarrow`, `pandas` - Core data processing
- `rasterio`, `pyproj` - Geospatial transformations
- `laspy` - LAZ/LAS point cloud I/O
- `scipy` - KD-tree for DSM extraction and vertical alignment
- `pyyaml` - Configuration file parsing
- `open3d` - GICP alignment (Generalized ICP)

### 2. Configure the Pipeline

Edit `config.yaml` to set your parameters:

**Important:** `sample_count` specifies the number of **Austin logs** to download, not total logs. The pipeline automatically filters by city before downloading.

```yaml
# Basic configuration
target_city: "ATX"           # Currently only Austin supported
sample_count: 5              # Number of AUSTIN logs to download (or "all")
                             # Pipeline filters by city BEFORE downloading
fetch_splits:                # Which splits to fetch from
  - "val"

# Output paths
output_dir: "../processed_samples_austin"
city_geoalign_root: "../Argoverse2-geoalign/ATX"
```

See [Configuration](#configuration) section for all available options.

### 3. Run the Pipeline

```bash
python fetch_and_process_pipeline.py
```

Or specify a custom config file:

```bash
python fetch_and_process_pipeline.py --config my_config.yaml
```

## How It Works

### City Filtering (Smart & Efficient)

The pipeline uses **smart city filtering** to avoid wasting time and bandwidth:

1. **Lists log directories** in S3 for the selected split(s)
2. **For each log**, checks `map/log_map_archive_{CITY}_city_*.json` filename
3. **Extracts city code** from filename (e.g., `ATX`, `MIA`, `PIT`)
4. **Skips non-target cities** WITHOUT downloading the full log
5. **Downloads only** logs from target city

**Result:** `sample_count: 2` means "download exactly 2 Austin logs", not "download 2 logs and hope they're Austin"

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: FETCH (Smart City Filtering)                         │
│  Downloads logs from Argoverse2 S3                              │
├─────────────────────────────────────────────────────────────────┤
│  • Lists ALL log directories in split (val/train/test)          │
│  • Checks city from map files (WITHOUT downloading full log)    │
│  • Skips non-Austin logs (only lists map directory)             │
│  • Downloads exactly N Austin logs using s5cmd                  │
│  • Stores temporarily in temp_download_dir                      │
│  • Efficiency: Only downloads what you need!                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: SEGMENT                                               │
│  Divides each log into 5-second segments                        │
├─────────────────────────────────────────────────────────────────┤
│  • Reads all LiDAR sweeps in the log                            │
│  • Creates non-overlapping segments (default: 5 sweeps each)    │
│  • Each segment becomes an independent sample                   │
│  • Naming: {original_log_id}_{segment_index}                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: PROCESS LIDAR                                         │
│  Creates macro-sweep point clouds                               │
├─────────────────────────────────────────────────────────────────┤
│  • Combines sweeps in sensor frame                              │
│  • Transforms to city frame and UTM coordinates                 │
│  • Optional square cropping (default: 100m)                     │
│  • Generates metadata (poses, timestamps, extents)              │
│  • Outputs:                                                     │
│    - segment.parquet (sensor frame)                             │
│    - segment_utm.parquet (UTM frame)                            │
│    - segment_meta.parquet (metadata)                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: ALIGN IMAGERY & DSM + VERTICAL ALIGNMENT              │
│  Generates aligned imagery and DSM tiles, aligns elevations     │
├─────────────────────────────────────────────────────────────────┤
│  • Queries imagery tile bounds                                  │
│  • Reprojects and mosaics imagery to UTM                        │
│  • Extracts DSM points matching LiDAR footprint                 │
│  • Vertical alignment (ground-based):                           │
│    - Divides area into 10m grid cells                           │
│    - Identifies ground points (bottom 5% per cell)              │
│    - Matches LiDAR ground to DSM elevations                     │
│    - Computes median offset (robust to outliers)                │
│    - Applies offset to all LiDAR UTM elevations                 │
│  • Outputs:                                                     │
│    - segment_imagery_utm.tif (GeoTIFF)                          │
│    - segment_dsm_utm.laz (LAZ point cloud)                      │
│    - segment_utm.parquet (UPDATED with aligned elevations)      │
│    - segment_meta.parquet (UPDATED with z_offset_m)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: EXTRACT DSM NEAR LIDAR                                │
│  Extracts DSM points within 0.5m of LiDAR points                │
├─────────────────────────────────────────────────────────────────┤
│  • Builds KD-tree from LiDAR UTM points (XY only)               │
│  • For each DSM point, computes nearest LiDAR distance          │
│  • Filters DSM to points within 0.5m threshold                  │
│  • Computes distance statistics (min, max, mean, percentiles)   │
│  • Outputs:                                                     │
│    - segment_extract_dsm_utm.parquet (extracted DSM points)     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 6: GICP ALIGNMENT                                        │
│  Aligns LiDAR to DSM reference using Generalized ICP            │
├─────────────────────────────────────────────────────────────────┤
│  • Source: segment_utm.parquet (LiDAR)                          │
│  • Target: segment_extract_dsm_utm.parquet (extracted DSM)      │
│  • Local frame anchor: Uses metadata center point              │
│    (center sweep's vehicle position from original pose data)    │
│  • Shifts both clouds to anchor-centered local frame            │
│  • Downsampling: 0.3m voxel size                                │
│  • Estimates normals with k=20 neighbors                        │
│  • Runs GICP with max correspondence distance 0.8m              │
│  • Composes transform back to global UTM frame                  │
│  • Computes alignment quality metrics:                          │
│    - GICP fitness and RMSE                                      │
│    - Nearest-neighbor RMSE and mean absolute distance           │
│    - Distance percentiles (p50, p75, p90, p95, p99)             │
│  • Outputs:                                                     │
│    - segment_gicp_utm.parquet (GICP-aligned LiDAR)              │
│    - segment_gicp_metrics.json (transform + quality metrics)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 7: CLEANUP                                               │
│  Removes temporary downloaded data                              │
├─────────────────────────────────────────────────────────────────┤
│  • Deletes original downloaded log (if cleanup_after_processing)│
│  • Keeps only processed segments in output_dir                  │
│  • Removes entire temp directory after completion (optional)    │
└─────────────────────────────────────────────────────────────────┘
```

### Output Structure

Each segment is saved in its own directory:

```
processed_samples_austin/
├── 0000a0e0-a333-4a84-b645-74b4c2a96bda_000/
│   ├── segment.parquet                      # Sensor-frame point cloud
│   ├── segment_utm.parquet                  # UTM-frame point cloud (vertically aligned)
│   ├── segment_meta.parquet                 # Metadata (poses, timestamps, extents)
│   ├── segment_imagery_utm.tif              # Aligned imagery (GeoTIFF)
│   ├── segment_dsm_utm.laz                  # Aligned DSM (LAZ)
│   ├── segment_extract_dsm_utm.parquet      # Extracted DSM points (within 0.5m of LiDAR)
│   ├── segment_gicp_utm.parquet             # GICP-aligned LiDAR
│   └── segment_gicp_metrics.json            # GICP metrics and transform matrix
├── 0000a0e0-a333-4a84-b645-74b4c2a96bda_001/
│   ├── segment.parquet
│   ├── segment_utm.parquet
│   ├── ...
└── ...
```

**Naming convention:**
- Directory: `{original_log_id}_{segment_index:03d}`
- Example: `0000a0e0-a333-4a84-b645-74b4c2a96bda_000`

**File contents:**
- `segment.parquet`: Point cloud in sensor frame (x, y, z, intensity, laser_number, offset_ns, source_index, source_timestamp_ns)
- `segment_utm.parquet`: Point cloud in UTM frame with **vertically aligned elevations** (utm_e, utm_n, elevation, intensity, laser_number, offset_ns, source_index, source_timestamp_ns)
- `segment_meta.parquet`: Rich metadata including:
  - Original log ID
  - Source timestamps for all sweeps in the segment
  - Center pose (position + orientation)
  - UTM zone and coordinates
  - Bounding boxes and extents
  - Crop information
  - Motion statistics
- `segment_imagery_utm.tif`: Aligned overhead imagery in GeoTIFF format
- `segment_dsm_utm.laz`: Digital Surface Model points in LAZ format
- `segment_extract_dsm_utm.parquet`: **Extracted DSM points** within 0.5m of any LiDAR point (utm_e, utm_n, elevation)
  - Used as reference for GICP alignment
  - Typically 30-60% of original DSM points
- `segment_gicp_utm.parquet`: **GICP-aligned LiDAR** point cloud (utm_e, utm_n, elevation, intensity, laser_number, offset_ns, source_index, source_timestamp_ns)
  - LiDAR aligned to DSM reference using Generalized ICP
  - Preserves all original attributes
- `segment_gicp_metrics.json`: GICP alignment metrics including:
  - Local frame anchor (center sweep's vehicle position from metadata)
  - Transformation matrix (4x4 homogeneous in global UTM frame)
  - Translation vector (XYZ in meters)
  - Yaw angle (degrees)
  - GICP fitness and inlier RMSE
  - Nearest-neighbor RMSE and mean absolute distance
  - Distance percentiles (p50, p75, p90, p95, p99)

## Configuration

### Essential Parameters

#### Data Fetching

```yaml
target_city: "ATX"              # Target city code
                                # Currently only "ATX" (Austin) supported
                                # due to geoalign data availability

sample_count: 5                 # Number of TARGET CITY logs to download
                                # Use integer (e.g., 5) for exactly 5 Austin logs
                                # Use "all" to download all Austin logs
                                # Pipeline checks city BEFORE downloading (efficient)

fetch_splits:                   # Which Argoverse2 splits to fetch from
  - "val"                       # Options: "train", "val", "test"
                                # Can specify multiple (all merged in output)
                                # Non-Austin logs are skipped automatically

skip_existing_downloads: true   # Skip downloading if log exists in temp_dir
                                # Useful when resuming interrupted pipeline

skip_processed_logs: true       # Skip logs that already have output segments
                                # Checks for {log_id}_* folders in output_dir
                                # Avoids re-processing and re-downloading
                                # Set to false to force reprocessing
```

#### Time Segmentation

```yaml
sweeps_per_segment: 5           # Number of LiDAR sweeps per segment
                                # At ~10Hz: 5 sweeps ≈ 0.5 seconds
                                #           10 sweeps ≈ 1.0 second
                                #           50 sweeps ≈ 5.0 seconds
                                # Actual duration calculated from timestamps

segment_overlap_sweeps: 0       # Overlap between consecutive segments
                                # Set to 0 for non-overlapping (as requested)
```

#### Spatial Parameters

```yaml
crop_size_meters: 100.0         # Square crop size in meters
                                # Minimum: 32.0
                                # Default: 100.0 (recommended)
                                # Options: 32.0, 64.0, 100.0, null (no crop)

buffer_meters: 2.0              # Buffer around LiDAR footprint
                                # Only used if crop_size_meters is null
```

**Imagery Resolution:**
- **Auto-detected from source** - No resampling or downsampling applied
- Typical source resolution: **0.05-0.15 meters per pixel**
- Preserves original imagery quality
- File sizes will be larger than with downsampling (but highest quality)

#### Paths

```yaml
temp_download_dir: "../temp_argoverse_download"
                                # Temporary storage for downloads
                                # Will be deleted after processing

output_dir: "../processed_samples_austin"
                                # Final output directory
                                # Each segment in separate subfolder

city_geoalign_root: "../Argoverse2-geoalign/ATX"
                                # Path to city imagery/DSM data
                                # Must contain Imagery/ and DSM/ subdirs

s5cmd_path: "../ArgoverseLidar/s5cmd/s5cmd.exe"
                                # Path to s5cmd executable
                                # Used for fast S3 downloads
```

#### Performance

```yaml
s5cmd_concurrency: 16           # Parallel download threads
                                # Higher = faster but more resources

s5cmd_part_size_mb: 128         # Multipart chunk size for downloads
                                # Larger may improve speed for big files

compression: "zstd"             # Parquet compression
                                # Options: snappy, gzip, zstd, lz4
```

#### Cleanup

```yaml
cleanup_after_processing: true  # Delete temp data after each log
                                # Recommended: true (saves disk space)

cleanup_temp_dir_on_completion: true
                                # Delete entire temp dir when done
                                # Recommended: true
```

#### Logging

```yaml
enable_detailed_logging: true   # Enable DEBUG level logs (console)
                                # Useful for troubleshooting

log_file: "pipeline"            # Summary CSV filename prefix
                                # Saved as {output_dir}/{log_file}_summary.csv
                                # Contains: segment_name, times, metadata
                                # Set to null to skip summary generation
```

**Summary CSV Format:**

The pipeline generates a clean CSV summary with segment metadata:

| Column | Description |
|--------|-------------|
| `segment_name` | Segment folder name (e.g., `log_abc_000`) |
| `output_path` | Full path to segment folder |
| `point_count` | Total number of points in the segment |
| `duration_s` | Time span of segment from metadata |
| `sensor_motion_length_m` | Vehicle path length during segment |
| `sensor_displacement_m` | Straight-line displacement |
| `z_offset_m` | Vertical alignment offset applied to LiDAR elevations |

**Note:**
- Console output shows detailed progress with processing times
- Summary CSV is **updated incrementally** after each segment completes
- **Incremental writes**: New runs append to existing CSV without overwriting
- If pipeline crashes, you keep data for all completed segments
- Summary file contains only segment metadata (no processing times)

**Example CSV Output:**

```csv
segment_name,output_path,point_count,duration_s,sensor_motion_length_m,sensor_displacement_m,z_offset_m
log_abc_000,../processed_samples_austin/log_abc_000,272792,0.4001,0.0655,0.0650,145.2341
log_abc_001,../processed_samples_austin/log_abc_001,268341,0.4002,0.0658,0.0652,145.1987
log_abc_002,../processed_samples_austin/log_abc_002,275183,0.3998,0.0661,0.0655,145.2156
```

This CSV can be easily imported into Excel, Python (pandas), or any data analysis tool for further processing.

## City Filtering Details

### How City Detection Works

The pipeline detects city efficiently without downloading entire logs:

**Step 1: List map directory**
```
S3 path: datasets/av2/lidar/val/{log_id}/map/
```

**Step 2: Find and parse map archive file**

Filename format:
```
log_map_archive_{LOG_ID}__Summer____{CITY}_city_{MAP_ID}.json
```

Example:
```
log_map_archive_0QB8KZQ9HFftSYAPyyIktvRCbbE9oL9r__Summer____ATX_city_77093.json
                └────────────Log UUID───────────┘        └┬┘
                                                      City Code
```

Extraction process:
```
1. Remove "log_map_archive_" prefix
2. Remove "_city_{MAP_ID}.json" suffix
3. Split by "____" (4 underscores)
4. Take last part → "ATX"
```

**Step 3: Compare with target**
```
ATX == ATX? → Download this log ✓
MIA == ATX? → Skip this log (no download) ✗
```

### Console Output Example

```
Listing and filtering logs for split 'val' (target city: ATX)...
    0a1b2c3d-1234-5678-90ab-cdef01234567: MIA (skip)
    1e2f3g4h-5678-90ab-cdef-123456789012: ATX ✓
  Found 1/2 ATX logs (checked 2 total)
    2i3j4k5l-90ab-cdef-1234-56789abcdef0: PIT (skip)
    3m4n5o6p-cdef-1234-5678-90abcdef0123: ATX ✓
  Found 2/2 ATX logs (checked 4 total)
  Collected 2 ATX logs from split 'val' (checked 4 total)

Total logs to process: 2
```

**Interpretation:**
- Checked 4 logs total
- Found 2 Austin logs (50% efficiency in this example)
- Skipped 2 non-Austin logs WITHOUT downloading them
- Ready to download only the 2 Austin logs

### Efficiency Gains

**Without filtering (old way):**
- Download 5 logs (500 MB each) = 2.5 GB downloaded
- Check city after download
- Only 2 are Austin → wasted 1.5 GB bandwidth and time

**With filtering (new way):**
- List 10+ log directories (tiny metadata)
- Find 5 Austin logs
- Download only 5 Austin logs = 2.5 GB downloaded
- 100% efficiency!

## Understanding the Output

### Imagery Resolution

The pipeline automatically detects and preserves the **original source imagery resolution**:

**How it works:**
1. Pipeline reads the first available imagery tile
2. Extracts the native resolution (typically 0.05-0.15 m/pixel)
3. Uses that resolution for the entire mosaic
4. **No resampling** - preserves maximum quality

**What you get:**
- **0.05 m/pixel** (5cm): Extremely high detail, see individual features
- **0.10 m/pixel** (10cm): High detail, good for most analysis
- **0.15 m/pixel** (15cm): Good detail, still very usable

**Example for 100m × 100m crop:**
- At 0.05 m/pixel: 2000×2000 = 4M pixels (~50-80 MB)
- At 0.10 m/pixel: 1000×1000 = 1M pixels (~10-30 MB)
- At 0.15 m/pixel: 667×667 = 445K pixels (~5-15 MB)

The exact resolution depends on the source imagery tiles available for your area.

### Point Cloud Files

**Sensor Frame (`segment.parquet`):**
- Coordinates relative to reference sweep sensor
- Origin at center sweep's LiDAR position
- Useful for sensor-centric analysis

**UTM Frame (`segment_utm.parquet`):**
- Coordinates in UTM projection
- **Vertically aligned elevations** matching DSM vertical datum
- Aligned with geo-referenced data
- Enables cross-dataset comparison
- Alignment details:
  - Ground points identified using bottom 5% percentile per 10m grid cell
  - Matched to nearest DSM elevations (within 2m radius)
  - Median offset computed for robustness
  - Applied to all point elevations in `segment_utm.parquet`
  - Offset value saved in metadata as `z_offset_m`

### Metadata File

The `segment_meta.parquet` contains:

| Field | Description |
|-------|-------------|
| `log_id` | Original Argoverse2 log ID |
| `point_file` | Name of sensor-frame parquet |
| `utm_point_file` | Name of UTM-frame parquet |
| `center_timestamp_ns` | Timestamp of center sweep |
| `source_timestamps_ns` | JSON array of all sweep timestamps in segment |
| `center_city_*` | Center pose in city frame (tx, ty, tz, qw, qx, qy, qz) |
| `center_latitude_deg` | Center position latitude |
| `center_longitude_deg` | Center position longitude |
| `center_utm_zone` | UTM zone (e.g., "14N") |
| `center_utm_easting_m` | Center easting in UTM |
| `center_utm_northing_m` | Center northing in UTM |
| `city_name` | City code (e.g., "ATX") |
| `point_count` | Total points in segment |
| `sweep_count` | Number of sweeps merged |
| `duration_ns` | Actual time span from timestamps (nanoseconds) |
| `duration_s` | Actual time span from timestamps (seconds) |
| `crop_square_m_*` | Applied crop size (sensor and UTM) |
| `bbox_*` | Bounding box coordinates |
| `extent_*` | Extent dimensions |
| `sensor_motion_length_m` | Total path length during segment |
| `sensor_displacement_m` | Straight-line displacement |
| `sensor_positions_*` | JSON array of all sweep positions |
| `z_offset_m` | Vertical alignment offset applied to UTM elevations (meters) |

### Imagery and DSM

**Imagery (`segment_imagery_utm.tif`):**
- Multi-band GeoTIFF (typically RGB)
- UTM-projected and mosaicked from source tiles
- **Original source resolution preserved** (no resampling)
- Typical resolution: 0.05-0.15 meters per pixel
- Aligned with LiDAR footprint
- ZSTD compressed

**DSM (`segment_dsm_utm.laz`):**
- LAZ/LAS point cloud format
- Elevation data matching LiDAR coverage
- UTM coordinates
- Useful for terrain analysis and GICP alignment

### Vertical Alignment

The pipeline automatically performs **ground-based vertical alignment** to ensure LiDAR elevations match the DSM vertical datum. This is critical because the raw LiDAR data is in a local city frame with arbitrary Z=0, while the DSM uses an absolute vertical datum (NAD83/WGS84 ellipsoid).

#### Why Alignment is Needed

- **LiDAR Z-axis**: City frame Z-coordinates are relative to an arbitrary city origin
- **DSM Z-axis**: Absolute elevations referenced to a proper geodetic datum
- **Problem**: Without alignment, there's a constant vertical offset (typically 100-200m)
- **Solution**: Compute and apply the offset so LiDAR and DSM elevations match

#### How Alignment Works

**Algorithm: Ground-Based Median Offset**

1. **Grid Subdivision**: Divide the 100×100m area into 10m×10m grid cells
2. **Ground Detection**: For each cell, identify ground points as bottom 5% percentile in Z
3. **DSM Matching**: Build KD-tree from DSM points, find nearest DSM point for each LiDAR ground point (within 2m radius)
4. **Offset Computation**: Calculate differences `DSM_z - LiDAR_z` for all matched pairs
5. **Robust Estimation**: Use median (not mean) to reject outliers and get final offset
6. **Apply Offset**: Add `z_offset_m` to all elevations in `segment_utm.parquet`
7. **Record**: Save `z_offset_m` in metadata and CSV for traceability

**Key Properties:**
- ✅ **Robust**: Median handles outliers (trees, buildings, noise)
- ✅ **Local**: Grid-based ground detection works on varied terrain
- ✅ **Automatic**: No manual intervention required
- ✅ **Traceable**: Offset value recorded in metadata
- ✅ **Consistent**: Same datum as DSM enables direct comparison

**Typical Offset Values:**
- Austin, TX: ~145-150 meters (depends on local ellipsoid height)
- Values are consistent within a city (±1m variation across segments)

**Failure Handling:**
- If alignment fails (no ground points, no DSM overlap, etc.), offset defaults to 0.0m
- Warning message logged, processing continues with unaligned data
- CSV and metadata still record the 0.0 offset for transparency

#### Alignment Validation

After processing, you can verify alignment quality:

```python
import pandas as pd
import pyarrow.parquet as pq

# Read metadata
meta = pq.read_table('segment_meta.parquet').to_pandas()
z_offset = meta['z_offset_m'].iloc[0]
print(f"Applied vertical offset: {z_offset:.4f} m")

# Read aligned LiDAR
lidar = pq.read_table('segment_utm.parquet').to_pandas()
lidar_ground = lidar.nsmallest(int(len(lidar)*0.05), 'elevation')

# Read DSM
import laspy
dsm = laspy.read('segment_dsm_utm.laz')

# Compare elevation ranges
print(f"LiDAR ground elevation range: {lidar_ground['elevation'].min():.2f} - {lidar_ground['elevation'].max():.2f} m")
print(f"DSM elevation range: {dsm.z.min():.2f} - {dsm.z.max():.2f} m")
# Should be similar after alignment
```

## Examples

### Example 1: Quick Test with 2 Samples

```yaml
target_city: "ATX"
sample_count: 2              # Get exactly 2 Austin logs
fetch_splits: ["val"]
sweeps_per_segment: 5
crop_size_meters: 100.0
cleanup_after_processing: true
```

```bash
python fetch_and_process_pipeline.py
```

Expected output:
- Checks city from S3 map files (no wasted downloads)
- Downloads **exactly 2 Austin logs** from validation set
- Skips non-Austin logs automatically
- Creates ~10-20 segments total (depends on log duration)
- Each segment has 5 sweeps in 100m crop
- Cleans up temp data automatically

### Example 2: Full Austin Dataset

```yaml
target_city: "ATX"
sample_count: "all"          # Download ALL Austin logs
fetch_splits: ["train", "val", "test"]
sweeps_per_segment: 50
crop_size_meters: 64.0
```

```bash
python fetch_and_process_pipeline.py
```

Expected output:
- Scans all splits for Austin logs
- Downloads **ALL Austin logs** (hundreds of logs)
- Automatically skips logs from other cities
- Creates 5-second segments (50 sweeps @ 10Hz)
- 64m crop for larger coverage
- May take several hours to complete

### Example 3: No Cropping, Original Resolution

```yaml
target_city: "ATX"
sample_count: 10             # Get exactly 10 Austin logs
crop_size_meters: null       # No spatial cropping
buffer_meters: 5.0           # 5m buffer around footprint
```

```bash
python fetch_and_process_pipeline.py
```

Expected output:
- Downloads exactly 10 Austin logs (city-filtered)
- Full spatial extent preserved
- Original source imagery resolution (~0.05-0.15m/pixel)
- Largest file sizes (highest quality)

## Troubleshooting

### Issue: "s5cmd executable not found"

**Solution:** Ensure s5cmd is installed in `ArgoverseLidar/s5cmd/s5cmd.exe`

Download from: https://github.com/peak/s5cmd/releases

### Issue: "City geoalign root not found"

**Solution:** Download Argoverse2-geoalign data for Austin:

```bash
# TODO: Add download instructions for geoalign data
```

Ensure directory structure:
```
Argoverse2-geoalign/ATX/
├── Imagery/
├── DSM/
└── imagery_tile_bounds.csv
```

### Issue: "No imagery tiles overlap the requested bounding box"

**Cause:** LiDAR footprint outside available imagery coverage

**Solutions:**
1. Check that log is actually from Austin (city filter working?)
2. Verify imagery tile bounds CSV is correct
3. Try smaller crop size or different log

### Issue: "DSM tiles found but no points intersect the LiDAR footprint"

**Cause:** DSM tiles exist but don't cover LiDAR area

**Solutions:**
1. Check DSM directory structure matches expected format
2. Verify UTM coordinate transformations are correct
3. Try different log or check DSM coverage maps

### Issue: Pipeline very slow

**Solutions:**
1. Increase `s5cmd_concurrency` (16-32 for fast connections)
2. Decrease `sweeps_per_segment` for faster processing
3. Use `crop_size_meters: 100.0` (default) or smaller for faster processing
4. Enable `skip_existing_downloads: true` if rerunning
5. Enable `skip_processed_logs: true` to skip already processed logs

### Issue: Out of disk space

**Solutions:**
1. Ensure `cleanup_after_processing: true`
2. Set `cleanup_temp_dir_on_completion: true`
3. Process fewer samples at a time
4. Increase `temp_download_dir` capacity

### Issue: "Vertical alignment failed" warning

**Possible Causes:**
- No ground points detected in LiDAR data
- No DSM points near LiDAR ground points
- LiDAR and DSM don't overlap spatially

**Solutions:**
1. Check console logs for specific error message
2. Verify DSM coverage matches LiDAR footprint
3. Check if segment is in area with good DSM data
4. Processing continues with z_offset = 0.0 (unaligned but usable)
5. For critical alignment, manually inspect problematic segments

**Note:** Alignment failures are rare but can occur at dataset boundaries or areas with limited DSM coverage.

## Performance Considerations

### Download Speed

- Depends on internet connection and S3 region
- `s5cmd_concurrency: 16` typically good for gigabit connections
- Each log is ~50-500 MB depending on duration

### Processing Time

**Per log (approximate):**
- Download: 30 seconds - 2 minutes
- Segment processing: 10-30 seconds per segment
- Imagery/DSM: 20-60 seconds per segment

**Example:**
- 10 logs × 10 segments each × 45 seconds/segment = ~75 minutes

### Disk Space

**Temporary:**
- Each log: 50-500 MB (deleted after processing)
- Peak usage: 1-2 logs at a time

**Final output:**
- Each segment: 20-100 MB (depends on crop size; original resolution imagery is larger)
  - LiDAR parquet files: ~5-10 MB
  - Imagery (original res): ~10-80 MB for 100m×100m crop
  - DSM: ~5-10 MB
- 100 segments: ~2-10 GB total
- Summary CSV: < 100 KB (lightweight, easy to analyze)

**Note:** File sizes are larger than with downsampled imagery (0.3m/pixel would give ~3-5 MB imagery per segment), but you get maximum quality.

## Analyzing the Summary CSV

After pipeline completion, use the summary CSV for analysis:

### Load in Python

```python
import pandas as pd

# Load summary
df = pd.read_csv('processed_samples_austin/pipeline_summary.csv')

# Basic statistics
print(f"Total segments: {len(df)}")
print(f"Total points: {df['point_count'].sum():,}")
print(f"Average points per segment: {df['point_count'].mean():.0f}")
print(f"Average segment duration: {df['duration_s'].mean():.4f}s")

# Find largest segments (by point count)
largest = df.nlargest(10, 'point_count')
print("\nLargest segments:")
print(largest[['segment_name', 'point_count']])

# Motion statistics
print(f"\nAverage motion length: {df['sensor_motion_length_m'].mean():.4f}m")
print(f"Average displacement: {df['sensor_displacement_m'].mean():.4f}m")

# Alignment statistics
print(f"\nAverage vertical offset: {df['z_offset_m'].mean():.4f}m")
print(f"Offset std deviation: {df['z_offset_m'].std():.4f}m")
print(f"Offset range: {df['z_offset_m'].min():.4f} - {df['z_offset_m'].max():.4f}m")
```

### Filter Segments

```python
# Find segments with high point count
dense_segments = df[df['point_count'] > 300000]

# Find segments with high motion
moving_segments = df[df['sensor_displacement_m'] > 0.1]

# Find segments where vehicle turned (path length > displacement)
turning_segments = df[df['sensor_motion_length_m'] > df['sensor_displacement_m'] * 1.1]

# Find sparse segments (low point count)
sparse_segments = df[df['point_count'] < 200000]
```

### Export Analysis

```python
# Save filtered results
dense_segments.to_csv('dense_segments.csv', index=False)

# Generate summary report
with open('dataset_report.txt', 'w') as f:
    f.write(f"Total segments: {len(df)}\n")
    f.write(f"Total points: {df['point_count'].sum():,}\n")
    f.write(f"Average points per segment: {df['point_count'].mean():.0f}\n")
    f.write(f"Average segment duration: {df['duration_s'].mean():.4f}s\n")
    f.write(f"Average motion length: {df['sensor_motion_length_m'].mean():.4f}m\n")

# Point density analysis
df['points_per_second'] = df['point_count'] / df['duration_s']
print(f"Average point density: {df['points_per_second'].mean():.0f} points/second")
```

## Resuming Interrupted Pipeline

If the pipeline crashes or is interrupted, you can resume without reprocessing:

**Scenario 1: Pipeline crashed after processing 5 logs**

```yaml
skip_processed_logs: true  # Enabled by default
sample_count: 10           # Original target
```

When you restart:
- Checks output directory for existing `{log_id}_*` folders
- Skips the 5 already-processed logs
- Continues with next logs until 10 total

**Scenario 2: Re-run with more samples**

```yaml
skip_processed_logs: true
sample_count: 20  # Increased from 10
```

Result:
- Keeps existing 10 processed logs
- Downloads and processes 10 more

**Scenario 3: Force reprocessing**

```yaml
skip_processed_logs: false  # Disable skip
sample_count: 10
```

Result:
- Reprocesses all logs even if output exists
- Overwrites existing segments

## Advanced Usage

### Custom Configuration File

Create multiple config files for different scenarios:

```bash
python fetch_and_process_pipeline.py --config configs/test_small.yaml
python fetch_and_process_pipeline.py --config configs/production_full.yaml
```

### Processing Specific Logs

If you already know which logs you want, you can modify the pipeline to accept a list of log IDs instead of discovering them automatically.

### Different Time Windows

Adjust `sweeps_per_segment` based on your needs:

| Sweeps | Duration @ ~10Hz | Use Case |
|--------|-----------------|----------|
| 5 | ~0.5 seconds | Quick tests, small windows |
| 10 | ~1.0 second | Short sequences |
| 50 | ~5.0 seconds | Standard segments (recommended) |
| 100 | ~10.0 seconds | Long sequences |

**Note:** Actual duration varies slightly based on real LiDAR timestamps and is automatically recorded in the metadata (`duration_s` field).

## Coordinate Systems

The pipeline handles multiple coordinate reference systems:

1. **Sensor Frame**: LiDAR-centric (origin at reference sweep)
2. **City Frame**: Argoverse2 city-SE3 coordinates
3. **UTM**: Universal Transverse Mercator (Austin = Zone 14N)
4. **WGS84**: Latitude/longitude (EPSG:4326)

All transformations are handled automatically.

## Limitations

### Current Limitations

1. **City Support**: Only Austin (ATX) currently supported
   - Other cities require corresponding geoalign data
   - Would need imagery and DSM tiles for each city

2. **City Filtering**: Applied after download
   - Mixed-city splits download all, filter during processing
   - Future: Could filter before download using metadata API

3. **Fixed Segment Structure**: Non-overlapping segments only
   - Can't currently create sliding windows
   - Would require modifying `process_log_segments()`

4. **No Incremental Processing**: Processes entire logs at once
   - Can't resume partial processing
   - Would need checkpointing mechanism

### Future Enhancements

- [ ] Multi-city support (when geoalign data available)
- [ ] Pre-download city filtering
- [ ] Sliding window segments
- [ ] Incremental/resumable processing
- [ ] Parallel log processing
- [ ] Custom imagery sources
- [ ] Configurable point cloud attributes

## Summary

This pipeline provides an end-to-end solution for:

✅ Downloading Argoverse2 LiDAR data with smart city filtering
✅ Processing into aligned point cloud segments
✅ Generating co-registered imagery and DSM
✅ **Vertical alignment** of LiDAR elevations to DSM datum
✅ Managing storage efficiently with automatic cleanup
✅ **Incremental CSV tracking** for resumable processing
✅ Logging detailed progress and metadata

**Key Advantages:**

- **No manual data management**: Automatic download and cleanup
- **Consistent output structure**: Easy to integrate into downstream pipelines
- **Flexible configuration**: Adapt to different requirements via YAML
- **Production-ready**: Robust error handling and logging
- **Elevation-aligned data**: LiDAR and DSM share the same vertical datum
- **Resumable processing**: Incremental CSV updates survive crashes
- **Traceable alignment**: All offsets recorded in metadata

---

For questions or issues, check the [Troubleshooting](#troubleshooting) section or review the pipeline logs.
