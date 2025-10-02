# LiDAR-DSM GICP Alignment Verification

A complete, user-friendly pipeline for aligning real-world LiDAR and DSM point clouds using Generalized ICP (GICP). Designed for 100m×100m data tiles with interactive GUI support for all operations.

## Key Features

✅ **Full GUI Support** - All scripts have interactive file browsers and parameter editors
✅ **Local Frame Transforms** - Readable transform matrices centered at metadata point
✅ **Vertical Alignment** - Automatic Z-shift using center point comparison
✅ **Smart DSM Extraction** - Crops DSM to match LiDAR coverage (0.5m threshold)
✅ **Comprehensive Metrics** - JSON output with diagnostics and quality measures
✅ **Interactive Viewer** - Compare original, shifted, and aligned point clouds

## Table of Contents

- [Key Features](#key-features)
- [Pipeline Overview](#pipeline-overview)
- [Required Files](#required-files)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Workflow Details](#workflow-details)
  - [Step 1: Data Preprocessing](#step-1-data-preprocessing)
  - [Step 2: GICP Alignment](#step-2-gicp-alignment)
  - [Step 3: Interactive Visualization](#step-3-interactive-visualization)
- [GUI Features](#gui-features)
- [File Structure](#file-structure)
- [Parameters](#parameters)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)
- [Understanding Transform Frames](#understanding-transform-frames)
- [Color Legend](#color-legend)
- [Tips & Best Practices](#tips--best-practices)
- [Summary](#summary)

## Pipeline Overview

**Three-step workflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: PREPROCESSING                                          │
│  Script: preprocess_lidar_dsm.py                                │
├─────────────────────────────────────────────────────────────────┤
│  Inputs:                                                        │
│    • data3_utm.parquet (LiDAR point cloud)                      │
│    • data3_dsm_utm.laz (DSM surface)                            │
│                                                                 │
│  Operations:                                                    │
│    1. Vertical shift: Align LiDAR Z to DSM at center point     │
│    2. DSM extraction: Crop DSM to LiDAR coverage (0.5m)         │
│                                                                 │
│  Outputs:                                                       │
│    • lidar_shifted.parquet                                      │
│    • dsm_extracted.parquet                                      │
│    • preprocessing_metrics.json                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: GICP ALIGNMENT                                         │
│  Script: align_lidar_dsm.py                                     │
├─────────────────────────────────────────────────────────────────┤
│  Inputs:                                                        │
│    • lidar_shifted.parquet                                      │
│    • data3_meta.parquet (center coordinates)                    │
│    • dsm_extracted.parquet                                      │
│                                                                 │
│  Operations:                                                    │
│    1. Load center point from metadata                           │
│    2. Run GICP in local frame (centered at metadata point)      │
│    3. Compute transform matrices and quality metrics            │
│                                                                 │
│  Outputs:                                                       │
│    • lidar_aligned.parquet                                      │
│    • alignment_metrics.json                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: VISUALIZATION                                          │
│  Script: viewer.py                                              │
├─────────────────────────────────────────────────────────────────┤
│  Inputs (select any combination):                              │
│    • data3_utm.parquet (original LiDAR - red)                   │
│    • lidar_shifted.parquet (shifted - orange)                   │
│    • lidar_aligned.parquet (aligned - green)                    │
│    • data3_dsm_utm.laz (original DSM - blue)                    │
│    • dsm_extracted.parquet (extracted - purple)                 │
│                                                                 │
│  Output:                                                        │
│    • Interactive Open3D 3D viewer                               │
└─────────────────────────────────────────────────────────────────┘
```

## Required Files

Before starting, ensure you have these input files in the folder:

| File | Type | Description | Required For |
|------|------|-------------|--------------|
| `data3_utm.parquet` | LiDAR data | Point cloud with UTM coordinates | Preprocessing |
| `data3_dsm_utm.laz` | DSM data | Digital Surface Model (LAZ/LAS format) | Preprocessing |
| `data3_meta.parquet` | Metadata | Contains center coordinates for local frame | GICP Alignment |

**Generated files** (created by the pipeline):
- `lidar_shifted.parquet` - Vertically aligned LiDAR
- `dsm_extracted.parquet` - Cropped DSM matching LiDAR coverage
- `lidar_aligned.parquet` - Final GICP-aligned LiDAR
- `preprocessing_metrics.json` - Diagnostics from step 1
- `alignment_metrics.json` - GICP transform and quality metrics

## Dependencies

Install required Python packages:

```bash
pip install numpy pandas scipy laspy open3d
```

**Package versions:**
- `numpy` - Array operations
- `pandas` - Parquet file I/O
- `scipy` - Spatial indexing (KDTree)
- `laspy` - LAZ/LAS file reading
- `open3d` - GICP and visualization
- `tkinter` - GUI (usually included with Python)

## Workflow Details

### Step 1: Data Preprocessing

The preprocessing script performs two key operations:

1. **Vertical Shift**: Aligns LiDAR Z values to DSM using the center point as anchor
   - Computes the center of the 100x100 square
   - Queries DSM height at center point
   - Shifts all LiDAR Z values to match DSM at center
   - Produces: `lidar_shifted.parquet`

2. **DSM Extraction**: Crops DSM to match LiDAR coverage
   - Keeps only DSM points within 0.5m of any LiDAR point
   - Creates DSM with same spatial coverage as LiDAR
   - Produces: `dsm_extracted.parquet`

**Run preprocessing:**

*Option 1: GUI (recommended for beginners)*
```bash
python preprocess_lidar_dsm.py --gui
# Or simply run without arguments:
python preprocess_lidar_dsm.py
```

*Option 2: Command line*
```bash
python preprocess_lidar_dsm.py \
  --lidar data3_utm.parquet \
  --dsm data3_dsm_utm.laz \
  --max-distance 0.5 \
  --output-dir .
```

**Outputs:**
- `lidar_shifted.parquet` – Vertically shifted LiDAR
- `dsm_extracted.parquet` – Extracted DSM points
- `preprocessing_metrics.json` – Diagnostics (shift amount, point counts, reduction ratio)

### Step 2: GICP Alignment

The alignment script runs GICP between shifted LiDAR (source) and extracted DSM (target) in a **local coordinate frame** centered at the true center point of the original 100x100 square.

**Why use metadata center point?**
- After extraction, the DSM loses the original square shape
- The metadata preserves the true center of the 100x100 area
- Using this fixed center ensures consistent transforms across different extractions
- Produces readable transform matrices with small values (close to identity)
- Avoids numerical issues from large UTM coordinates

**Run alignment:**

*Option 1: GUI (recommended)*
```bash
python align_lidar_dsm.py --gui
# Or simply run without arguments:
python align_lidar_dsm.py
```

*Option 2: Command line*
```bash
python align_lidar_dsm.py \
  --shifted-lidar lidar_shifted.parquet \
  --meta data3_meta.parquet \
  --extracted-dsm dsm_extracted.parquet \
  --voxel-size 0.3 \
  --normal-k 20 \
  --max-corr-dist 0.8 \
  --max-iter 60 \
  --output-dir .
```

**Outputs:**
- `lidar_aligned.parquet` – GICP-aligned LiDAR
- `alignment_metrics.json` – Transform matrices, GICP fitness/RMSE, alignment quality

**Metrics JSON structure:**
```json
{
  "inputs": {
    "shifted_lidar": "lidar_shifted.parquet",
    "meta_data": "data3_meta.parquet",
    "extracted_dsm": "dsm_extracted.parquet"
  },
  "local_frame_anchor_utm": {
    "utm_e": 622124.285,
    "utm_n": 3348302.746,
    "z": -16.179,
    "note": "Center point from metadata - all transforms in local frame centered here"
  },
  "transform_local_frame": {
    "matrix_4x4": [...],
    "translation_m": [x, y, z],
    "yaw_deg": 0.123,
    "note": "Small, readable values for analysis"
  },
  "transform_global_frame": {
    "matrix_4x4": [...],
    "note": "For applying to actual UTM data"
  },
  "alignment_quality": {
    "rmse_to_target_m": 0.05,
    "mean_abs_distance_m": 0.04
  }
}
```

### Step 3: Interactive Visualization

The viewer provides a GUI to compare different datasets.

**Run viewer:**
```bash
python viewer.py
```

**LiDAR options:**
- **Original (red)** – Raw LiDAR from `data3_utm.parquet`
- **Shifted (orange)** – After vertical alignment
- **Aligned (green)** – After GICP alignment

**DSM options:**
- **Original (blue)** – Full DSM from `data3_dsm_utm.laz`
- **Extracted (purple)** – Cropped to LiDAR coverage

**Usage:**
1. Launch viewer with `python viewer.py`
2. Select which LiDAR version to view (radio buttons)
3. Select which DSM version to view (radio buttons)
4. Toggle visibility checkboxes for each dataset
5. Click "View" button to open Open3D 3D visualizer
6. Use mouse to rotate, pan, zoom in the 3D view

## GUI Features

All scripts support interactive GUI mode with file browsers:

### Preprocessing GUI (`preprocess_lidar_dsm.py`)
- Browse button for LiDAR parquet file
- Browse button for DSM LAZ/LAS file
- Browse button for output directory
- Editable max distance parameter (default: 0.5m)
- "Run Preprocessing" button executes the pipeline
- Success dialog shows output file paths

### Alignment GUI (`align_lidar_dsm.py`)
- Browse buttons for all 3 input files:
  - Shifted LiDAR parquet
  - Metadata parquet
  - Extracted DSM parquet
- Browse button for output directory
- Editable GICP parameters:
  - Voxel Size (default: 0.3m)
  - Normal K (default: 20)
  - Max Correspondence Distance (default: 0.8m)
  - Max Iterations (default: 60)
- "Run GICP Alignment" button executes alignment
- Success dialog shows metrics file path

### Viewer GUI (`viewer.py`)
- Radio buttons to select LiDAR version (original/shifted/aligned)
- Radio buttons to select DSM version (original/extracted)
- Checkboxes to toggle visibility
- Color legend display
- "View" button opens Open3D visualization

**How to launch GUI mode:**
- Simply run the script without arguments: `python script_name.py`
- Or explicitly request GUI: `python script_name.py --gui`

## File Structure

```
gicp-veri-lidar-dsm/
├── README.md                          # This file
├── data3_utm.parquet                  # Input: LiDAR point cloud
├── data3_meta.parquet                 # Input: Metadata with center coordinates
├── data3_dsm_utm.laz                  # Input: DSM point cloud
├── preprocess_lidar_dsm.py            # Script 1: Data preprocessing
├── align_lidar_dsm.py                 # Script 2: GICP alignment
├── viewer.py                          # Script 3: Interactive viewer
├── lidar_shifted.parquet              # Output: Vertically shifted LiDAR
├── dsm_extracted.parquet              # Output: Extracted DSM
├── lidar_aligned.parquet              # Output: GICP-aligned LiDAR
├── preprocessing_metrics.json         # Output: Preprocessing diagnostics
└── alignment_metrics.json             # Output: GICP metrics
```

## Quick Start

### GUI Mode (Recommended)

All scripts support GUI mode for easy file selection:

```bash
# 1. Preprocess data - GUI will open automatically
python preprocess_lidar_dsm.py

# 2. Run GICP alignment - GUI will open automatically
python align_lidar_dsm.py

# 3. Visualize results - GUI included
python viewer.py
```

### Command Line Mode

Run the complete pipeline via command line:

```bash
# 1. Preprocess data
python preprocess_lidar_dsm.py \
  --lidar data3_utm.parquet \
  --dsm data3_dsm_utm.laz

# 2. Run GICP alignment
python align_lidar_dsm.py \
  --shifted-lidar lidar_shifted.parquet \
  --meta data3_meta.parquet \
  --extracted-dsm dsm_extracted.parquet

# 3. Visualize results
python viewer.py
```

## Parameters

### Preprocessing
- `--max-distance`: Maximum distance for DSM extraction (default: 0.5m)
  - Smaller values → more aggressive cropping
  - Larger values → more DSM points retained

### GICP Alignment
- `--voxel-size`: Downsampling voxel size (default: 0.3m)
  - Smaller values → finer detail, slower
  - Larger values → coarser, faster
- `--normal-k`: Neighbors for normal estimation (default: 20)
- `--max-corr-dist`: Max correspondence distance (default: 0.8m)
- `--max-iter`: Maximum GICP iterations (default: 60)

## Expected Results

For well-aligned data with ~100m×100m coverage:

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Preprocessing vertical shift | 0-5 m | Depends on initial LiDAR-DSM offset |
| DSM extraction reduction | 60-80% | Depends on LiDAR coverage density |
| GICP fitness | > 0.95 | Higher is better (max 1.0) |
| GICP RMSE | < 0.2 m | Lower is better |
| Final alignment RMSE | < 0.1 m | Nearest-neighbor error after alignment |
| Transform translation | < 1 m | In local frame (small values) |
| Transform yaw | < 1° | Should be small for similar datasets |

## Troubleshooting

**"LiDAR file not found"**
- Ensure `data3_utm.parquet` exists in the folder
- Or specify path with `--lidar path/to/file.parquet`

**"DSM file not found"**
- Ensure `data3_dsm_utm.laz` exists
- Install laspy: `pip install laspy`
- Or specify path with `--dsm path/to/file.laz`

**"Metadata file not found"**
- Ensure `data3_meta.parquet` exists in the folder
- This file contains the true center point coordinates
- Or specify path with `--meta path/to/meta.parquet`

**"Downsampled cloud empty"**
- Voxel size too large → reduce `--voxel-size`
- Not enough overlap between LiDAR and DSM

**"GICP fitness very low (<0.5)"**
- Poor initial alignment → check vertical shift
- Try adjusting `--max-corr-dist` (increase to 1.5-2.0)
- Try smaller `--voxel-size` for finer detail

**Viewer shows no data**
- Check file paths in terminal output
- Ensure preprocessing and alignment completed successfully
- Files should exist: `lidar_shifted.parquet`, `dsm_extracted.parquet`, `lidar_aligned.parquet`

## Understanding Transform Frames

This pipeline uses **local coordinate frames** for all transform calculations:

1. **Global frame**: Original UTM coordinates (e.g., E=622124 m, N=3348302 m)
2. **Local frame**: Centered at the 100x100 square's true center from metadata (origin = metadata center)

**Why metadata center?**
- The metadata (`data3_meta.parquet`) contains `center_utm_easting_m` and `center_utm_northing_m`
- This is the true center of the original 100x100 square before any processing
- After DSM extraction, the shape is no longer a perfect square, so computing centroid would give wrong results
- Using the fixed metadata center ensures all transforms are in a consistent reference frame

**Benefits of local frame:**
- Transform matrices have small, interpretable values
- Translation components show actual movement in meters
- Rotation angles are accurate (no numerical issues from large coordinates)
- Easy to verify: identity matrix = perfect alignment

**Both transforms are saved** in `alignment_metrics.json`:
- `transform_local_frame`: For analysis and interpretation
- `transform_global_frame`: For applying to actual UTM point clouds

## Color Legend

When using the viewer:

| Color | Dataset |
|-------|---------|
| Red | LiDAR Original |
| Orange | LiDAR Shifted |
| Green | LiDAR Aligned |
| Blue | DSM Original |
| Purple | DSM Extracted |

## Tips & Best Practices

**General Workflow:**
- Start with default parameters, then tune if needed
- Always run the full pipeline in order: preprocess → align → visualize
- Check metrics JSON files after each step to verify quality

**Visual Verification:**
Use the viewer to check alignment at each stage:
- **Original LiDAR (red) vs Original DSM (blue)** → Shows initial misalignment (expect large Z offset)
- **Shifted LiDAR (orange) vs Extracted DSM (purple)** → Shows vertical alignment (Z should match, XY may differ)
- **Aligned LiDAR (green) vs Extracted DSM (purple)** → Shows final GICP result (should overlap closely)

**Quality Checks:**
- `preprocessing_metrics.json` - Check vertical shift amount (0-5m typical)
- `alignment_metrics.json` - Check GICP fitness (>0.95 good), RMSE (<0.1m excellent)
- Large translation errors (>2m in local frame) suggest poor initial alignment or insufficient overlap

**Troubleshooting Tips:**
- If GICP fails: Try increasing `--max-corr-dist` to 1.5-2.0m
- If alignment is slow: Increase `--voxel-size` to 0.5m
- If alignment is imprecise: Decrease `--voxel-size` to 0.2m
- If DSM extraction removes too much: Increase `--max-distance` to 1.0m

## Summary

This pipeline provides a complete workflow for aligning LiDAR and DSM point clouds:

1. **Easy to use** - GUI mode requires no command-line knowledge
2. **Robust** - Handles vertical misalignment and varying coverage
3. **Accurate** - Uses metadata center for consistent local frame transforms
4. **Transparent** - Comprehensive metrics and visual verification tools
5. **Flexible** - All parameters are adjustable via GUI or command line

**Typical workflow time:**
- Preprocessing: 1-2 minutes (depends on point count)
- GICP Alignment: 2-5 minutes (depends on voxel size and point density)
- Visualization: Instant

**Expected accuracy:**
- Vertical alignment: <0.5m error
- GICP alignment: <0.1m RMSE for well-aligned data
- Transform precision: Sub-millimeter in local frame

For questions or issues, check the [Troubleshooting](#troubleshooting) section or verify your input files match the [Required Files](#required-files) specification.
