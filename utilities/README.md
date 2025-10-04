# Utilities

Visualization and analysis tools for point cloud and geospatial data processing.

## Overview

This folder contains utility scripts for visualizing and analyzing processed LiDAR data, DSM, and imagery from the data pipeline.

## Tools

### 1. Point Cloud Viewer (`viewer.py`)

Interactive 3D viewer for comparing up to three point cloud files side-by-side.

**Supported Formats:**
- Parquet files (`.parquet`) - LiDAR data from the pipeline
- LAZ/LAS files (`.laz`, `.las`) - DSM and compressed point clouds

**Features:**
- **Multi-file comparison**: Load up to 3 files simultaneously
- **Color-coded display**: Red, Blue, Green for Files 1, 2, 3
- **Format auto-detection**: Automatically detects file type
- **Flexible coordinate support**: 
  - UTM coordinates: `utm_e`, `utm_n`, `elevation`
  - Sensor frame: `x`, `y`, `z`
- **Interactive 3D navigation**: Pan, rotate, zoom with Open3D viewer
- **Toggle visibility**: Show/hide individual files

**Usage:**

```bash
# GUI mode
python viewer.py

# The viewer will open a GUI where you can:
#   1. Browse and select up to 3 point cloud files
#   2. Toggle visibility for each file
#   3. Click "View" to open the 3D visualization
```

**Common Use Cases:**

```bash
# Compare LiDAR with DSM
File 1 (Red):   segment_utm.parquet     # LiDAR data
File 2 (Blue):  segment_dsm_utm.laz     # DSM data

# Compare before/after alignment
File 1 (Red):   segment_utm_before.parquet
File 2 (Blue):  segment_utm_after.parquet

# Compare multiple segments
File 1 (Red):   segment_000_utm.parquet
File 2 (Blue):  segment_001_utm.parquet
File 3 (Green): segment_002_utm.parquet
```

**Dependencies:**
```bash
pip install numpy pandas pyarrow laspy open3d
```

**Color Scheme:**
- **File 1**: Red (RGB: 1.0, 0.3, 0.3)
- **File 2**: Blue (RGB: 0.2, 0.5, 1.0)
- **File 3**: Green (RGB: 0.3, 0.8, 0.3)

---

### 2. GeoTIFF Overlay Viewer (`geotiff_overlay_viewer.py`)

Projects point cloud data onto GeoTIFF imagery using UTM coordinates to visualize alignment.

**Features:**
- **Image + Point Cloud Overlay**: Display point clouds on top of satellite/aerial imagery
- **Elevation colormapping**: Point colors represent elevation (darker = lower, lighter = higher)
- **UTM coordinate matching**: Uses UTM coordinates to align data
- **Interactive display**: Pan and zoom within matplotlib canvas

**Usage:**

```bash
# GUI mode
python geotiff_overlay_viewer.py

# Command line mode
python geotiff_overlay_viewer.py --image segment_imagery_utm.tif --points segment_utm.parquet
```

**Dependencies:**
```bash
pip install numpy pandas pyarrow rasterio matplotlib pillow
```

---

## Installation

### Required Dependencies

Install all dependencies for both viewers:

```bash
pip install numpy pandas pyarrow laspy open3d rasterio matplotlib pillow
```

---

## Integration with Data Pipeline

These utilities work directly with output from the `Data-pipeline-fetch` pipeline:

```
processed_samples_austin/
├── segment_000/
│   ├── segment.parquet              # Sensor frame (viewer.py)
│   ├── segment_utm.parquet          # UTM frame (viewer.py + geotiff_overlay_viewer.py)
│   ├── segment_meta.parquet         # Metadata
│   ├── segment_imagery_utm.tif      # Imagery (geotiff_overlay_viewer.py)
│   └── segment_dsm_utm.laz          # DSM (viewer.py)
```

### Workflow Examples

#### 1. Verify Vertical Alignment

Check if LiDAR and DSM are vertically aligned:

```bash
# Load both in point cloud viewer
python viewer.py
# File 1: segment_utm.parquet (LiDAR)
# File 2: segment_dsm_utm.laz (DSM)
# Look for vertical offset - should be minimal after alignment
```

#### 2. Check Spatial Registration

Verify LiDAR points align with imagery:

```bash
# Overlay LiDAR on imagery
python geotiff_overlay_viewer.py --image segment_imagery_utm.tif --points segment_utm.parquet
# Points should align with visible features (buildings, roads, etc.)
```

---

## Keyboard Controls

### viewer.py (Open3D)

- **Mouse Left**: Rotate view
- **Mouse Right**: Translate view
- **Mouse Wheel**: Zoom in/out
- **R**: Reset viewpoint
- **Q/ESC**: Close viewer

### geotiff_overlay_viewer.py (Matplotlib)

- **Mouse Wheel**: Zoom in/out
- **Left Click + Drag**: Pan image
- **Home**: Reset to original view

---

## Troubleshooting

### Issue: "module 'laspy' not found"

**Solution:**
```bash
pip install laspy
```

### Issue: "module 'open3d' not found"

**Solution:**
```bash
pip install open3d
```

### Issue: Point clouds don't align in viewer

**Solution:**
- Always compare files in the same coordinate frame
- Use `*_utm.parquet` files for UTM comparison
- Use `segment.parquet` files for sensor frame comparison

---

## Related Documentation

- **Data Pipeline**: See `../Data-pipeline-fetch/README.md` for data processing
- **GICP Analysis**: See `../gicp_analysis_example/` for alignment verification
- **Project Overview**: See main `../README.md` for project structure
