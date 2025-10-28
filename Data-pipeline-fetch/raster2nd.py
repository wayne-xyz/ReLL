"""
Shift selected rasters so that their 0.5th percentile becomes zero.

Usage:
    python Data-pipeline-fetch/raster2nd.py --src_dir Rell-sample-raster --output_dir Rell-sample-raster2

Only the following rasters are altered:
    - gicp_height.npy
    - dsm_height.npy
    - non_aligned_height.npy

All other files in each sample directory are copied verbatim.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

TARGET_RASTERS = (
    "gicp_height.npy",
    "dsm_height.npy",
    "non_aligned_height.npy",
)


def compute_shift(array: np.ndarray) -> float:
    """Return the 0.5th percentile (ignoring NaNs)."""
    finite = np.isfinite(array)
    if not np.any(finite):
        return 0.0
    return float(np.percentile(array[finite], 0.5))


def shift_raster(path: Path, output_path: Path) -> None:
    """Load a raster, subtract the 0.5th percentile, save to output."""
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array for {path}, got shape {data.shape}")

    shift = compute_shift(data)
    shifted = data.astype(np.float32, copy=False) - shift
    np.maximum(shifted, 0.0, out=shifted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, shifted)


def process_sample(sample_dir: Path, output_dir: Path) -> None:
    """Shift target rasters and copy other artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in sample_dir.iterdir():
        dst = output_dir / item.name
        if item.name in TARGET_RASTERS:
            if not item.exists():
                print(f"[warning] {item} missing; skipping.")
                continue
            shift_raster(item, dst)
        elif item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shift rasters so p0.5 -> 0.")
    parser.add_argument("--src_dir", type=Path, required=True, help="Source dataset directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Destination directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.output_dir

    if not src_dir.exists():
        print(f"[error] Source directory does not exist: {src_dir}")
        return 1
    if dst_dir.exists() and any(dst_dir.iterdir()):
        print(f"[error] Output directory already exists and is not empty: {dst_dir}")
        return 1

    dst_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = [p for p in src_dir.iterdir() if p.is_dir()]
    if not sample_dirs:
        print(f"[error] No sample folders found in {src_dir}")
        return 1

    for sample in sample_dirs:
        out_sample = dst_dir / sample.name
        try:
            process_sample(sample, out_sample)
            print(f"[ok] processed {sample.name}")
        except Exception as exc:
            print(f"[error] Failed processing {sample.name}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
