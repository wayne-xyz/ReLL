# GICP Verification Playground

This folder isolates a small workflow for sanity-checking Generalized ICP (GICP) on our LiDAR/DSM extracts without running the full alignment pipeline. It contains:

- **gicp_verify.py** – generate a perturbed copy of a reference cloud, run GICP, and dump metrics plus three parquet files (`target_reference`, `source_offset`, `source_aligned`). The synthetic offset is applied in a local frame centered on the cloud centroid, and all transforms are reported in this local coordinate system for readable, interpretable values.
- **cloud_viewer.py** – visualise any trio of parquet point clouds with interactive colour toggles (Target = blue, Perturbed Source = red, GICP Aligned = green). File paths are configured directly in the script via the `FILE_PATHS` dictionary; the viewer logs load status and keeps your current viewpoint when layers are switched on/off.
- **outputs/** – default destination for the generated verification artefacts.

## Quick Start

1. Place a reference parquet (e.g., `data3_utm.parquet`) in this folder.
2. Run the verifier:
   ```
   python gicp_verify.py --target data3_utm.parquet --output-dir outputs --visualize
   ```
   Adjust the CLI flags or provide a JSON config if you want different synthetic offsets or GICP parameters.
3. Set the three file paths inside `cloud_viewer.py` to the parquet files of interest and launch the viewer:
   ```
   python cloud_viewer.py
   ```

Both scripts print diagnostic information to the terminal (counts, column names, load errors) to make troubleshooting easier.

## Example Metrics

The table below captures a typical run stored in `outputs/gicp_metrics.json`.

| Metric | Value | Notes |
| --- | --- | --- |
| Input points | 494,065 | Rows taken from reference parquet |
| Local anchor (UTM) | Computed centroid | All transforms below are in local frame centered here |
| Synthetic offset (m) | (0.30, -0.25, 0.15) | Applied in local frame |
| Synthetic rotation (deg) | (roll 0.30, pitch -0.20, yaw 0.50) | Small 6-DoF perturbation |
| GICP fitness | 1.0000 | Reported by Open3D registration |
| GICP RMSE (m) | 0.0925 | Inlier RMSE from GICP |
| NN RMSE to target (m) | 0.00257 | Post-alignment nearest-neighbour RMSE |
| NN mean abs distance (m) | 0.00254 | Post-alignment mean nearest-neighbour distance |
| Translation error (m) | ~0.003 | Error between GICP and ground truth (local frame) |
| Yaw error (deg) | ~0.001 | Δ between recovered and ground-truth yaw |

**Transform Frames**: All transforms are reported in a **local coordinate frame** centered at the cloud's centroid. This produces readable matrix values close to identity with small offsets, rather than huge UTM coordinates. The metrics JSON includes both:
- **Local frame transforms**: For analysis and interpretation (small, meaningful values)
- **Global frame GICP transform**: For applying to actual UTM point clouds if needed

**Ground Truth**: Since the source is artificially perturbed with a known offset, the `ground_truth_transform_local_frame` is simply the inverse of the applied offset. GICP's goal is to recover this transform, and the `transform_error_local_frame` section quantifies how accurately it does so.

## Tips & Troubleshooting

- The viewer uses `defaultUnlit` materials, so normals are not required. If your Open3D build complains about missing attributes, upgrade to 0.17+ or change the shader to `defaultLit` and add normals before rendering.
- `gicp_verify.py` falls back to a minimal Open3D implementation if the shared `analysis.gicp_core` module is unavailable, so it remains usable even outside the main project environment.
- Edit `FILE_PATHS` and press the corresponding “Reload …” button to swap datasets without restarting the viewer.
- Both scripts log successes and failures to stdout; keep the terminal open when debugging.
