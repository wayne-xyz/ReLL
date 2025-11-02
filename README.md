# ReLL: Reproduce Learned Localization with GICP Registration

## Overview

This repository implements a full data pipeline and training code to reproduce the learned localization approach from the paper:

**📄 [Evaluating Global Geo-alignment for Precision Learned Autonomous Vehicle Localization using Aerial Data](https://arxiv.org/abs/2503.13896)** (arXiv:2503.13896)

For a detailed walkthrough of the implementation, results, and challenges, see the accompanying blog post:

**📖 [Reproduction Blog Post](https://rongweiji.github.io/localization/2025/10/10/Reproduce-Learned-Localization-with-GICP-Registration.html)** — Implementation notes, figures, and insights

---

## What This Repo Does

- **GICP alignment**: Registers LiDAR point clouds to DSM to improve geo-alignment between modalities
- **Learned localization**: Trains an encoder to produce embeddings for LiDAR/height and map/imagery
- **Cross-correlation matching**: Uses a cost volume (sliding window over feature embeddings) to measure similarity
- **Sub-pixel refinement**: Refines integer-pixel peaks using Gaussian fitting for sub-pixel accuracy
- **Finds fill-rate correlation**: Shows that LiDAR point coverage (fill rate) strongly affects localization quality

## Key Results

- Achieves **~0.1 m RMS translation error** on 0.2 m resolution dataset (after Gaussian refinement)
- Demonstrates that careful preprocessing (GICP + filtering, height normalization) is critical
- Validates the trade-off between raster resolution, coverage, and localization accuracy

## Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

All dependencies are listed with notes about optional packages for data preprocessing and utilities.

### 2. Data Preparation

Refer to `Data_intro.md` for dataset layout and structure.  
For GICP-based alignment and preprocessing, see scripts in:
- `Data-pipeline-fetch/` — Main pipeline for fetching and processing raster data
- `Argoverse2-geoalign/` — GICP alignment and geo-registration

### 3. Train the Model

```powershell
python train.py \
  --data-root Rell-sample-raster-0p2 \
  --save-dir .\model-save\ \
  --plot-metrics \
  --epochs 200
```

**Optional arguments:**
- `--batch-size 16` — Batch size (default: from config)
- `--lr 1e-4` — Learning rate (default: from config)
- `--device cuda` — Compute device (default: auto-detect)
- `--subset-frac 0.5` — Use only 50% of data for quick experiments

### 4. Run Inference & Visualization

```powershell
# Infer on a single sample
python .\Train\infer_sample_vis.py \
  --sample <SAMPLE_PATH> \
  --checkpoint .\model-save\best_1000_0p3.ckpt

# Infer on entire dataset
python .\Train\infer_dataset_static.py \
  --dataset <DATASET_PATH> \
  --checkpoint .\model-save\best_1000_0p3.ckpt
```

## Architecture & Design

The pipeline follows this sequence:

1. **Input**: Rasterized LiDAR heights/intensities + DSM + aerial imagery (all co-registered)
2. **Encoders**: Dual pyramid encoders extract embeddings for LiDAR and map modalities
3. **Projection**: L2-normalized projection layers map embeddings to a shared space
4. **Cross-correlation**: Sliding window correlation computes a 2D cost volume (translation search)
5. **Rotation search**: Separate rotation similarity scores across angle candidates
6. **Softmax loss** (training): Uses differentiable softmax expectation for sub-pixel accuracy
7. **Gaussian refinement** (inference): Advanced peak fitting (centroid + quadratic + Newton steps) for improved sub-pixel precision

**Key insight**: The model learns to find peaks in the cost volume; training uses softmax (differentiable), inference uses Gaussian fitting (non-differentiable but more accurate).

## Illustration

GICP alignment improves LiDAR-to-DSM registration:

![GICP alignment example](https://rongweiji.github.io/img/GICP%20align%20comapre.png)

## Repository Structure

### Core Training & Inference

- `train.py` — Main training entrypoint (configurable hyperparameters, device detection, early stopping)
- `Train/config.py` — Configuration system (loads from YAML + CLI overrides)
- `Train/engine.py` — Training loop, evaluation, checkpointing, learning rate scheduling
- `Train/model.py` — PyramidEncoder, LocalizationModel, LocalizationCriterion
- `Train/data.py` — GeoAlignRasterDataset, data augmentation (rotation/translation), dataloader
- `Train/gaussian_peak_refine.py` — Advanced Gaussian peak refinement (multi-strategy blended approach)
- `Train/theta_peak_refine.py` — Rotation angle refinement using softmax expectation
- `Train/infer_sample_vis.py` — Visualize inference results on a single sample
- `Train/infer_dataset_static.py` — Run inference on entire dataset

### Data & Preprocessing

- `Data-pipeline-fetch/` — Main pipeline for dataset preparation
  - `raster.py` — Raster I/O (LAS/LAZ, GeoTIFF), resampling, coordinate transforms
  - `lib/gicp_alignment.py` — GICP registration (Open3D)
  - `lib/imagery_processing.py` — Aerial imagery and DSM processing
  - `lib/lidar_processing.py` — LiDAR point cloud handling
  - `lib/dsm_extraction.py` — DSM extraction and rasterization
- `Argoverse2-geoalign/` — Argoverse 2 dataset specific utilities
- `ArgoverseLidar/` — Visualization and exploration tools
- `utilities/` — Miscellaneous tools (projection compare, viewer, etc.)

### Configuration & Documentation

- `Train/default.yaml` — Default training config (batch size, learning rate, model depth, etc.)
- `Data_intro.md` — Dataset structure and layout documentation
- `requirements.txt` — Python dependencies (core + optional data-processing)

## References

- **Original Paper**: [Evaluating Global Geo-alignment for Precision Learned Autonomous Vehicle Localization using Aerial Data](https://arxiv.org/abs/2503.13896) (arXiv:2503.13896)
- **Reproduction Blog**: [Implementation notes, results, and challenges](https://rongweiji.github.io/localization/2025/10/10/Reproduce-Learned-Localization-with-GICP-Registration.html)
- **Datasets**:
  - LiDAR: [Argoverse 2](https://www.argoverse.org/av2.html)
  - DSM: [Bexar & Travis Counties LiDAR (2021)](https://data.geographic.texas.gov/collection/?c=447db89a-58ee-4a1b-a61f-b918af2fb0bb)
  - Imagery: [Capital Area Council of Governments (2022)](https://data.geographic.texas.gov/collection/?c=a15f67db-9535-464e-9058-f447325b6251), 0.3047 m resolution

## Contributing & License

This repository is open source. See repository files for licensing details.  
Contributions are welcome — open an issue or pull request with any improvements, bug fixes, or extensions.


