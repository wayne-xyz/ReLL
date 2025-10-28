"""
Investigation utility for ReLL rasters.

Scans a directory of sample folders (e.g. Rell-sample-raster) and collects
per-sample statistics for the GICP LiDAR height and DSM height rasters.
The script plots histograms for the minimum and 1st percentile values of each
dataset so that anomalies (e.g. large gaps between min and p01) can be inspected.

Usage:
    python Data-pipeline-fetch/invesitation_v.py --root Rell-sample-raster
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

TARGET_FILES: Dict[str, str] = {
    "gicp_height.npy": "GICP Height",
    "dsm_height.npy": "DSM Height",
}


def _finite_values(arr: np.ndarray) -> np.ndarray:
    """Return flattened finite values."""
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.array([], dtype=np.float32)
    return arr[finite].astype(np.float32, copy=False)


def collect_stats(sample_dir: Path) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    """
    For each target raster in a sample directory, compute (min, p01) if data exists.
    Returns mapping from target name to tuple(min, p01); missing rasters map to (None, None).
    """
    stats: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for filename in TARGET_FILES:
        path = sample_dir / filename
        if not path.exists():
            stats[filename] = (None, None, None)
            continue
        try:
            arr = np.load(path)
        except Exception as exc:
            print(f"[warning] Failed to load {path}: {exc}")
            stats[filename] = (None, None, None)
            continue
        values = _finite_values(arr)
        if values.size == 0:
            stats[filename] = (None, None, None)
        else:
            stats[filename] = (
                float(np.min(values)),
                float(np.percentile(values, 1)),
                float(np.percentile(values, 0.5)),
            )
    return stats


def scan_directory(root: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Traverse sample folders and aggregate statistics.
    Returns structure mapping each raster to per-sample measurements for:
        min, p01, p01-min, p005, and p005-min.
    """
    aggregates: Dict[str, Dict[str, List[float]]] = {
        key: {"min": [], "p01": [], "p01_delta": [], "p005": [], "p005_delta": []}
        for key in TARGET_FILES
    }

    sample_dirs = [p for p in root.iterdir() if p.is_dir()]
    sample_dirs.sort()
    if not sample_dirs:
        raise FileNotFoundError(f"No sample folders found in {root}")

    for sample in sample_dirs:
        stats = collect_stats(sample)
        for key, (min_val, p01_val, p005_val) in stats.items():
            if min_val is not None:
                aggregates[key]["min"].append(min_val)
            if p01_val is not None:
                aggregates[key]["p01"].append(p01_val)
            if min_val is not None and p01_val is not None:
                aggregates[key]["p01_delta"].append(p01_val - min_val)
            if p005_val is not None:
                aggregates[key]["p005"].append(p005_val)
            if min_val is not None and p005_val is not None:
                aggregates[key]["p005_delta"].append(p005_val - min_val)
    return aggregates


def plot_distributions(aggregates: Dict[str, Dict[str, List[float]]]) -> None:
    """Render histograms for min, p01, p005 and their offsets for each raster type."""
    rows = [
        ("min", "Minimum", "#ef5350"),
        ("p01", "1st percentile", "#1e88e5"),
        ("p01_delta", "p01 - min", "#43a047"),
        ("p005", "0.5th percentile", "#9c27b0"),
        ("p005_delta", "p005 - min", "#fb8c00"),
    ]

    fig, axes = plt.subplots(len(rows), len(TARGET_FILES), figsize=(12, 18), constrained_layout=True)

    for col, filename in enumerate(TARGET_FILES):
        title = TARGET_FILES[filename]
        for row_idx, (key, label, color) in enumerate(rows):
            ax = axes[row_idx][col] if len(TARGET_FILES) > 1 else axes[row_idx]
            values = np.asarray(aggregates[filename][key])
            ax.set_title(f"{title} – {label}")
            if values.size:
                ax.hist(values, bins=50, color=color, alpha=0.75)
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    plt.suptitle("Distribution of Min / p01 / p005 and Offsets for LiDAR and DSM Heights")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate min vs. p01 distributions in ReLL rasters.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing sample folders (e.g., Rell-sample-raster)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"[error] Root directory not found: {root}")
        return 1
    try:
        aggregates = scan_directory(root)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    plot_distributions(aggregates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
