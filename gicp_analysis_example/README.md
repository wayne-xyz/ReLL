# GICP Example Collection

This folder contains a collection of GICP (Generalized Iterative Closest Point) implementations, verification tools, and experimental results for aligning LiDAR and DSM (Digital Surface Model) point clouds.

## Overview

The examples demonstrate different approaches to point cloud alignment using GICP, ranging from simple verification playgrounds to production-ready pipelines with GUI support. Each subdirectory represents a different implementation strategy or experimental result.

## Directory Structure

### 📁 [gicp-veri-lidar-dsm/](gicp-veri-lidar-dsm/)
**Production-Ready LiDAR-DSM Alignment Pipeline**

A complete, user-friendly pipeline for aligning real-world LiDAR and DSM point clouds with full GUI support.

**Key Features:**
- Three-step workflow: Preprocessing → GICP Alignment → Visualization
- Interactive file browsers and parameter editors
- Vertical alignment using center point comparison
- Smart DSM extraction (crops to match LiDAR coverage)
- 3D viewer for comparing up to 3 point clouds simultaneously
- GeoTIFF overlay viewer for visual verification on satellite imagery

**Typical Use Case:** Production alignment of 100m×100m data tiles with comprehensive metrics and visual verification.

**Quick Start:**
```bash
cd gicp-veri-lidar-dsm
python preprocess_lidar_dsm.py  # Step 1: Preprocess
python align_lidar_dsm.py        # Step 2: GICP alignment
python viewer.py                 # Step 3: Visualize results
```

---

### 📁 [gicp-veri/](gicp-veri/)
**GICP Verification Playground**

A minimal environment for sanity-checking GICP algorithms on synthetic test cases.

**Key Features:**
- Generates artificially perturbed copies of reference point clouds
- Applies known transformations (translations + rotations)
- Runs GICP to recover the ground-truth transform
- Compares recovered vs. ground-truth transforms with detailed error metrics
- Interactive 3D viewer with color-coded layers (Target/Perturbed/Aligned)

**Typical Use Case:** Validating GICP implementations and tuning parameters on controlled test cases before applying to real data.

**Quick Start:**
```bash
cd gicp-veri
python gicp_verify.py --target data3_utm.parquet --output-dir outputs --visualize
python cloud_viewer.py  # View results
```

---

### 📁 [gicp-non-extracted-core-op1-op2/](gicp-non-extracted-core-op1-op2/)
**Multi-Strategy GICP Analysis Pipeline**

An advanced pipeline supporting multiple GICP strategies with different preprocessing approaches.

**Available Strategies:**
- **Op1 (Default):** Vertical gating with downsampling
  - Gates LiDAR points by vertical distance to nearest DSM point
  - Applies voxel downsampling before GICP
  - Best for ground-level alignment with varied terrain

- **Op2:** DSM-style filtering without downsampling
  - Extracts highest point per horizontal grid cell (mimics DSM structure)
  - Uses full DSM without cropping
  - Best for surface-to-surface matching (buildings, canopy)

- **Core:** Baseline Open3D GICP
  - Minimal preprocessing
  - Standard GICP implementation

**Key Features:**
- Supports both UTM and sensor-frame LiDAR inputs
- Metadata-driven coordinate transformations
- Z down-weighting to reduce vertical dominance
- Post-correction for vertical drift prevention
- Comprehensive diagnostics and gating overlays in viewer

**Quick Start:**
```bash
cd gicp-non-extracted-core-op1-op2
python data_shift_gicp.py --lidar data.parquet --meta meta.parquet --dsm dsm.laz --output-dir results
python pointcloud_viewer.py --lidar data.parquet --dsm dsm.laz --meta meta.parquet
```

---

### 📁 gicp_try/
**Sample GICP Results**

Contains example outputs from running the GICP alignment pipeline.

**Contents:**
- `data3_utm_shifted.parquet` - LiDAR point cloud after vertical shift
- `data3_utm_shifted_gicp.parquet` - Final GICP-aligned LiDAR
- `data3_utm_alignment_metrics.json` - Alignment diagnostics (fitness, RMSE, transforms)
- `data3_utm_alignment_results.npz` - Numerical results archive

**Typical Metrics:**
- Vertical offset applied: ~157m
- GICP registration fitness: 0.369
- GICP RMSE: 0.587m
- Method: Median vertical difference

---

### 📁 no-gate-gicp-result/
**GICP Results Without Gating**

Results from running GICP alignment without vertical gating or preprocessing filters.

**Purpose:** Baseline comparison to demonstrate the impact of gating strategies on alignment quality.

---

### 📁 op2-dsmsty-filter-gicp-result/
**GICP Results with DSM-Style Filtering**

Results from running the Op2 strategy (DSM-style filtering without downsampling).

**Purpose:** Demonstrates surface-to-surface matching approach where LiDAR is filtered to mimic DSM topology.

---

## Choosing the Right Tool

| Use Case | Recommended Tool | Why |
|----------|------------------|-----|
| Production alignment with GUI | `gicp-veri-lidar-dsm/` | User-friendly, comprehensive workflow |
| Testing GICP with synthetic data | `gicp-veri/` | Controlled environment with ground truth |
| Advanced preprocessing strategies | `gicp-non-extracted-core-op1-op2/` | Multiple strategies, detailed diagnostics |
| Ground-level terrain alignment | Op1 strategy | Vertical gating selects ground points |
| Building/canopy surface matching | Op2 strategy | DSM-style filtering preserves surfaces |

## Common Workflows

### 1. Quick Visual Alignment
```bash
cd gicp-veri-lidar-dsm
python preprocess_lidar_dsm.py  # GUI mode
python align_lidar_dsm.py        # GUI mode
python viewer.py                 # Compare before/after
```

### 2. Algorithm Validation
```bash
cd gicp-veri
python gicp_verify.py --target reference.parquet --visualize
# Check metrics: should recover ground-truth transform within tolerance
```

### 3. Strategy Comparison
```bash
cd gicp-non-extracted-core-op1-op2
# Run Op1 (vertical gating)
python data_shift_gicp.py --gicp-strategy op1 --lidar data.parquet --dsm dsm.laz --output-dir op1_results

# Run Op2 (DSM-style filtering)
python data_shift_gicp.py --gicp-strategy op2 --lidar data.parquet --dsm dsm.laz --output-dir op2_results

# Compare metrics and visual results
```

## Dependencies

All tools require:
```bash
pip install numpy pandas scipy laspy open3d
```

Optional (for GeoTIFF overlay viewer):
```bash
pip install rasterio matplotlib pillow
```

## Understanding GICP Strategies

**Vertical Gating (Op1):**
- Selects LiDAR points close to DSM surface vertically (±0.5m threshold)
- Works well for ground-level features
- Reduces outliers from vegetation/buildings
- Faster due to downsampling

**DSM-Style Filtering (Op2):**
- Extracts highest point per horizontal grid cell
- Mimics DSM topology for surface-to-surface matching
- Preserves building tops and canopy structure
- No downsampling (uses all filtered points)

**Baseline (Core):**
- Minimal preprocessing
- Standard Open3D GICP
- Good for clean, well-aligned data

## Tips & Best Practices

1. **Start with the GUI tools** (`gicp-veri-lidar-dsm/`) for initial exploration
2. **Validate with synthetic data** (`gicp-veri/`) before trusting results on real data
3. **Compare strategies** using the multi-strategy pipeline when alignment quality is critical
4. **Check metrics JSON files** after each run to verify alignment quality:
   - GICP fitness > 0.95 is excellent
   - RMSE < 0.1m indicates precise alignment
   - Large translation errors (>2m) suggest poor initial alignment
5. **Use the viewers** to visually confirm alignment before trusting metrics

## Troubleshooting

**GICP fitness very low (<0.5):**
- Try different strategies (Op1 vs Op2)
- Increase max correspondence distance
- Check vertical shift was applied correctly

**Alignment looks wrong visually:**
- Verify input coordinate systems match (both UTM)
- Check metadata file contains correct center coordinates
- Try baseline strategy to rule out preprocessing issues

**Empty/missing output files:**
- Check terminal output for errors
- Verify input file paths exist and are readable
- Ensure output directory has write permissions

---

For detailed documentation on each tool, see the README files in the respective subdirectories.
