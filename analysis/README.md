# Analysis Pipeline Overview

## Purpose
The scripts in this directory convert raw LiDAR sweeps and DSM tiles into an aligned point-cloud pair that is ready for inspection or downstream modelling. The workflow has two major stages:

1. **Pre-GICP alignment** – transform the LiDAR into the DSM frame and apply a vertical shift so that the clouds are roughly co-registered.
2. **Generalized ICP refinement** – run the OP1-enhanced GICP pipeline to estimate a precise rigid transform between the LiDAR and DSM.

## Inputs
- **LiDAR parquet** (--lidar): Contains either UTM columns (utm_e, utm_n, elevation) or sensor-frame XYZ coordinates.  
- **Metadata parquet** (--meta, optional): Required when the LiDAR file lacks UTM coordinates. Provides the quaternion/translation to convert sensor-frame points into the city frame and the affine parameters to reach UTM.  
- **DSM LAS/LAZ** (--dsm): Dense DSM point cloud covering the LiDAR sweep.  
- **Output directory** (--output-dir): Target folder for shifted clouds, GICP results, and diagnostics.

## Stage 1 – Pre-GICP Processing
1. **Frame harmonisation** (load_inputs in data_shift_gicp.py)
   - If UTM columns are present, they are used directly.
   - Otherwise the metadata quaternion/translation maps sensor-frame XYZ into the city frame, and an affine fit infers UTM easting/northing.
2. **DSM subsetting**
   - The DSM tile is trimmed to the LiDAR XY extent plus a margin for efficiency.
3. **Vertical offset estimation** (evaluate, query_dsm_height)
   - When metadata is present, the DSM height at the sensor centre is compared to the recorded centre-city Z value.
   - Otherwise the median vertical difference between LiDAR points and nearest DSM neighbours is used.
4. **Shifted cloud export** (save_parquet)
   - Apply the vertical offset to produce a coarse-aligned LiDAR cloud and save it as <name>_shifted.parquet.
   - A companion .npz file stores the shifted coordinates and metrics for reproducibility.

At this point, the LiDAR and DSM clouds are aligned well enough for visual inspection. The vertical shift is also what drives the gating preview in pointcloud_viewer.py when the GICP dependencies are available.

## Stage 2 – GICP Refinement
The optional refinement supports multiple strategies via `--gicp-strategy`:
- **op1** (default): Vertical gating with downsampling, implemented in `gicp_core_op1.py`
- **op2**: DSM-style filtering with no downsampling, implemented in `gicp_core_op2.py`
- **core**: Baseline Open3D GICP without gating

### Op1 Strategy – Vertical Gating with Downsampling

**Key parameters (Op1Config defaults)**
- `vertical_gate_m = 0.5`: absolute |dz| tolerance when comparing LiDAR Z to the nearest DSM Z.
- `min_points_after_gate = 1000`: minimum LiDAR samples required after gating before fallback is triggered.
- `fallback_fraction = 0.25`: when the gate keeps too few points, retain the best |dz|-ranked max(`min_points_after_gate`, `fallback_fraction × N`).
- `target_xy_margin_m = 10.0`: metres added to the LiDAR XY bounds when cropping the DSM ROI.
- `target_min_points_after_gate = 5000`: minimum DSM points expected after DSM gating.
- `z_downweight_factor = 2.0`: optional scaling applied to Z during ICP to reduce vertical dominance.
- `voxel_size = 0.5 m`, `normal_k = 20`, `max_corr_dist = 0.8 m`, `max_iter = 60`: downsampling and ICP settings.

**Workflow**
1. **Source gating** (`_gate_ground_like`)
   - For each shifted LiDAR point, find the nearest DSM neighbour and compute `dz = z_dsm - z_lidar` along with lateral distance.
   - Keep points with `|dz| ≤ vertical_gate_m` and record diagnostics (mean/std/p05/p50/p95 for dz and XY distance).
   - If fewer than `min_points_after_gate` survive, fall back to the lowest-|dz| points sized at `max(min_points_after_gate, fallback_fraction × total_points)`.
   - Diagnostics include `preselection_total`, `preselection_kept_by_gate`, `preselection_final_kept`, `preselection_final_rejected`, and `z_summary_kept`.
2. **Target crop and gating** (`_crop_and_gate_target_by_source`)
   - Build a DSM ROI by expanding the kept LiDAR XY bounds by `target_xy_margin_m` metres.
   - Apply the same vertical gate; require at least `target_min_points_after_gate` DSM points. If not met, reuse the 25% fallback strategy.
   - Diagnostics store crop counts (`original_dsm_count`, `roi_count`, `crop_rejected_count`) and gating results (`preselection_kept_by_gate`, `final_kept`, `final_rejected`).
3. **Z down-weighting (optional)**
   - Scale the Z coordinate of both gate-kept clouds by `1 / z_downweight_factor` prior to ICP to prevent steep terrain from dominating the solution.
4. **Downsampling & normals** (`prepare_downsampled_clouds`)
   - Apply voxel downsampling (`0.5 m`) around a shared anchor origin and estimate normals with `normal_k = 20`, enforcing consistent orientation.
5. **Generalized ICP** (`run_gicp`)
   - Execute Open3D’s `registration_generalized_icp` with `max_corr_dist = 0.8 m` and `max_iter = 60`, producing a transform in the scaled coordinate space.
6. **Post-correction**
   - Undo any Z scaling and decompose the transform into yaw plus XY translation.
   - Re-estimate the Z translation via the median `dz` between DSM and LiDAR after applying yaw+XY, which guards against vertical drift.
7. **Outputs and diagnostics**
   - Apply the final transform to the shifted LiDAR cloud (`<name>_shifted_gicp.parquet`).
   - Persist metrics (`_alignment_metrics.json`, `_alignment_results.npz`) including ICP fitness/RMSE, gate statistics, point counts, and post-correction summaries.

### Op2 Strategy – DSM-Style Filtering without Downsampling

**Key parameters (Op2Config defaults)**
- `vertical_cell_size_m = 0.05`: horizontal grid cell size for extracting highest LiDAR points (mimics DSM structure).
- `disable_downsampling = True`: skips voxel downsampling before GICP (uses filtered points directly).
- `voxel_size = 0.5 m`, `normal_k = 20`, `max_corr_dist = 0.8 m`, `max_iter = 60`: ICP settings.

**Workflow**
1. **LiDAR DSM-style filtering** (`_extract_highest_per_vertical_cell`)
   - Partition shifted LiDAR points into horizontal grid cells of size `vertical_cell_size_m`.
   - For each cell, extract only the point with the maximum Z value (highest point).
   - This mimics DSM topology by creating a pseudo-surface from the LiDAR data.
   - Diagnostics include `original_count`, `filtered_count`, `cells_created`, and `reduction_ratio`.
2. **Use full DSM**
   - All DSM points are kept for GICP (no gating or cropping).
   - This provides maximum surface coverage for registration.
3. **No downsampling (default)**
   - When `disable_downsampling = True`, skip voxel downsampling and use the filtered LiDAR and full DSM points directly.
   - Normals are still estimated with `normal_k = 20`.
4. **Generalized ICP** (`run_gicp`)
   - Execute Open3D's GICP on the filtered LiDAR surface and full DSM.
5. **Post-correction**
   - Extract yaw + XY translation from GICP transform.
   - Re-estimate Z translation using median `dz` after applying yaw+XY to guard against vertical drift.
6. **Outputs and diagnostics**
   - Apply final transform to shifted LiDAR cloud (`<name>_shifted_gicp.parquet`).
   - Persist metrics including DSM-style filter statistics, ICP fitness/RMSE, and post-correction summaries.

**When to use Op2**
- When LiDAR data has dense vertical structure (e.g., buildings, vegetation) and you want to match DSM topology.
- When you want to preserve the full DSM coverage without gating or cropping.
- When surface-to-surface matching is preferred over ground-point selection.
- When you prefer a simpler pipeline: filter LiDAR → GICP with full DSM.

## Visualisation
pointcloud_viewer.py consumes the shifted or GICP-refined clouds:
- It uses infer_lidar_points to reproduce the frame harmonisation.
- When the gating helpers are importable, compute_gating_overlay supports both Op1 and Op2 strategies and displays kept vs rejected LiDAR/DSM points with distinct colours.
- The GUI includes a checkbox to toggle gating overlay and radio buttons to select between Op1 (vertical gate) and Op2 (DSM-style filter).
- Without GICP dependencies, the viewer gracefully falls back to the basic two-colour rendering.

## Running the Pipeline
`
python analysis/data_shift_gicp.py \
  --lidar path/to/lidar.parquet \
  --meta path/to/meta.parquet \
  --dsm path/to/dsm.laz \
  --output-dir path/to/output
`
Add `--skip-gicp` to stop after the vertical shift, or use `--gicp-strategy` to choose between:
- `op1`: Vertical gating with downsampling (default)
- `op2`: DSM-style filtering without downsampling
- `core`: Baseline Open3D GICP implementation

Launch the viewer directly or after alignment:
`
python analysis/pointcloud_viewer.py --lidar path/to/lidar.parquet --dsm path/to/dsm.laz [--meta path/to/meta.parquet]
`

## Key Files
- data_shift_gicp.py: End-to-end alignment pipeline with strategy selection.
- gicp_core.py: Core Open3D GICP helpers.
- gicp_core_op1.py: Op1 strategy with vertical gating, downsampling, and post-correction.
- gicp_core_op2.py: Op2 strategy with DSM-style filtering, no downsampling, and post-correction.
- pointcloud_viewer.py: Open3D visualisation with gating overlays for both Op1 and Op2.




