"""
Raster preview tool for ReLL samples.

Usage:
    python utilities/viewer_raster.py PATH/TO/SAMPLE_FOLDER

The viewer loads the saved `.npy` rasters inside a sample folder and renders
six panels (DSM, GICP height/intensity, RGB imagery, non-aligned height/intensity).
It also prints useful metadata from the accompanying pickle files.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


PLOT_ITEMS: Tuple[Tuple[str, str, Optional[str]], ...] = (
    ("dsm_height.npy", "DSM Height", "terrain"),
    ("gicp_height.npy", "GICP Height", "viridis"),
    ("gicp_intensity.npy", "GICP Intensity", "gray"),
    ("imagery.npy", "Imagery (RGB)", None),
    ("non_aligned_height.npy", "Non-aligned Height", "magma"),
    ("non_aligned_intensity.npy", "Non-aligned Intensity", "gray"),
    ("gicp_height.npy", "GICP Height (Original)", None),
    ("gicp_height.npy", "GICP Height (Shifted)", None),
    ("dsm_height.npy", "DSM Height (Original)", None),
    ("dsm_height.npy", "DSM Height (Shifted)", None),
)


def _load_optional(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        print(f"[missing] {path.name}")
        return None
    try:
        arr = np.load(path)
        print(f"[loaded] {path.name} -> shape={arr.shape}, dtype={arr.dtype}")
        return arr
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[error] {path.name}: {exc}")
        return None


def _ensure_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Imagery array must be 3D, got shape {arr.shape}")
    if arr.shape[-1] in (1, 3, 4, 5, 6, 7, 8) and arr.shape[0] not in (1, 3, 4, 5, 6, 7, 8):
        return np.transpose(arr, (2, 0, 1))
    return arr


def _read_pickle(path: Path) -> Optional[object]:
    if not path.exists():
        print(f"{path.name}: <missing>")
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        print(f"{path.name}: {data}")
        return data
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"{path.name}: <error> ({exc})")
        return None


def _extent_from_transform(transform, width: int, height: int):
    # Rasterio-style Affine transform expected
    if transform is None:
        return None
    try:
        x0, y0 = transform * (0, 0)
        x1, y1 = transform * (width, height)
        return (x0, x1, y1, y0)  # left, right, bottom, top
    except Exception:
        return None




def _prepare_rgb(chw: np.ndarray) -> np.ndarray:
    data = chw[:3].astype(np.float32)
    rgb = np.transpose(data, (1, 2, 0))
    finite = np.isfinite(rgb)
    if not np.any(finite):
        return np.zeros(rgb.shape, dtype=np.float32)
    values = rgb[finite]
    p1, p99 = np.percentile(values, [1, 99])
    if np.isclose(p1, p99):
        scale = p99 if p99 != 0 else 1.0
        rgb = rgb / scale
    else:
        rgb = (rgb - p1) / (p99 - p1)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb[~finite] = 0.0
    return rgb


def _summarize_array(title: str, arr: np.ndarray) -> str:
    if arr.ndim == 3:
        return f"{title:<24} shape={arr.shape}, dtype={arr.dtype}"
    finite = np.isfinite(arr)
    if np.any(finite):
        vmin = float(np.nanmin(arr[finite]))
        vmax = float(np.nanmax(arr[finite]))
        mean = float(np.nanmean(arr[finite]))
    else:
        vmin = vmax = mean = float("nan")
    return (
        f"{title:<24} shape={arr.shape}, dtype={arr.dtype}, "
        f"min={vmin:.3f}, max={vmax:.3f}, mean={mean:.3f}"
    )


def preview_sample(sample_dir: Path) -> None:
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample folder not found: {sample_dir}")

    print(f"Previewing sample: {sample_dir}")
    arrays: Dict[str, Optional[np.ndarray]] = {}
    raw_arrays: Dict[str, Optional[np.ndarray]] = {}
    for filename, _, _ in PLOT_ITEMS:
        if filename in arrays:
            continue

        arr = _load_optional(sample_dir / filename)
        if arr is None:
            arrays[filename] = None
            raw_arrays[filename] = None
            continue

        if np.issubdtype(arr.dtype, np.number):
            raw_arrays[filename] = arr.copy()
            arr = arr.astype(np.float32, copy=False)
            finite = np.isfinite(arr)
            if np.any(finite):
                if filename != "imagery.npy":
                    min_val = float(np.nanmin(arr[finite]))
                    arr = np.where(finite, arr - min_val, 0.0)
                else:
                    arr = np.where(finite, arr, 0.0)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
        else:
            raw_arrays[filename] = arr
        arrays[filename] = arr

    # Convert imagery to CHW if needed
    imagery = arrays.get("imagery.npy")
    if imagery is not None:
        try:
            arrays["imagery.npy"] = _ensure_chw(imagery)
        except ValueError as exc:
            print(f"[warning] imagery.npy skipped: {exc}")
            arrays["imagery.npy"] = None

    print("\n" + "=" * 80)
    print("Metadata")
    print("=" * 80)
    resolution = _read_pickle(sample_dir / "resolution.pkl")
    transform = _read_pickle(sample_dir / "transform.pkl")
    _read_pickle(sample_dir / "profile.pkl")
    _read_pickle(sample_dir / "metadata.pkl")

    grid_shape = None
    for key in ("imagery.npy", "gicp_height.npy", "dsm_height.npy"):
        arr = arrays.get(key)
        if arr is None:
            continue
        if arr.ndim == 3:
            grid_shape = (arr.shape[1], arr.shape[2])
        else:
            grid_shape = arr.shape
        if grid_shape:
            break

    extent = _extent_from_transform(transform, grid_shape[1], grid_shape[0]) if grid_shape and transform else None

    print("Original height ranges:")
    def _describe(name: str, arr: np.ndarray) -> None:
        finite = np.isfinite(arr)
        if not np.any(finite):
            print(f"  {name}: no finite values")
            return
        vals = arr[finite]
        stats = {"min": float(np.nanmin(vals)),
                 "p01": float(np.nanpercentile(vals, 1)),
                 "p05": float(np.nanpercentile(vals, 5)),
                 "p95": float(np.nanpercentile(vals, 95)),
                 "p99": float(np.nanpercentile(vals, 99)),
                 "max": float(np.nanmax(vals))}
        print("  {name}: min={min:.3f}, p01={p01:.3f}, p05={p05:.3f}, p95={p95:.3f}, p99={p99:.3f}, max={max:.3f}".format(name=name, **stats))

    gicp_raw = raw_arrays.get("gicp_height.npy")
    if gicp_raw is not None:
        _describe("GICP Height (original)", gicp_raw)
    dsm_raw = raw_arrays.get("dsm_height.npy")
    if dsm_raw is not None:
        _describe("DSM Height (original)", dsm_raw)

    if raw_arrays.get("gicp_height.npy") is not None:
        raw = raw_arrays["gicp_height.npy"]
        finite = np.isfinite(raw)
        if np.any(finite):
            print(f"  GICP Height (original): min={float(np.nanmin(raw[finite])):.3f}, max={float(np.nanmax(raw[finite])):.3f}")
    if raw_arrays.get("dsm_height.npy") is not None:
        raw = raw_arrays["dsm_height.npy"]
        finite = np.isfinite(raw)
        if np.any(finite):
            print(f"  DSM Height (original): min={float(np.nanmin(raw[finite])):.3f}, max={float(np.nanmax(raw[finite])):.3f}")

    print("\nArray statistics:")
    for filename, title, _ in PLOT_ITEMS:
        if "(Original)" in title or "(Shifted)" in title:
            continue
        arr = arrays.get(filename)
        if arr is None:
            continue
        print("  " + _summarize_array(title, arr))

    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14), constrained_layout=True)
    axes = axes.flatten()

    for ax, (filename, title, cmap) in zip(axes, PLOT_ITEMS):
        arr = arrays.get(filename)
        ax.set_title(title)

        if "(Original)" in title or "(Shifted)" in title:
            ax.set_axis_on()
            base_key = filename
            if "(Original)" in title:
                source = raw_arrays.get(base_key)
                label = "Original"
                color = "#ef5350"
            else:
                source = arrays.get(base_key)
                label = "Shifted"
                color = "#1e88e5"

            if source is None or not np.issubdtype(source.dtype, np.number):
                ax.text(0.5, 0.5, f"No {label.lower()} data", ha="center", va="center", transform=ax.transAxes)
            else:
                values = source[np.isfinite(source)]
                if values.size == 0:
                    ax.text(0.5, 0.5, f"No {label.lower()} data", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.hist(values.ravel(), bins=60, color=color, alpha=0.65)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            continue

        ax.set_axis_off()
        if arr is None:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
            continue
        if arr.ndim == 3:
            chw = arr.astype(np.float32)
            if chw.shape[0] >= 3:
                rgb = _prepare_rgb(chw)
            else:
                expanded = np.repeat(chw[:1, :, :], 3, axis=0)
                rgb = _prepare_rgb(expanded)
            ax.imshow(rgb, extent=extent, origin="upper")
        else:
            finite = np.isfinite(arr)
            if np.any(finite):
                vmin = float(np.nanpercentile(arr[finite], 1))
                vmax = float(np.nanpercentile(arr[finite], 99))
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-6
            else:
                vmin, vmax = 0.0, 1.0
            im = ax.imshow(arr, cmap=cmap or "viridis", extent=extent, origin="upper", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for ax in axes[len(PLOT_ITEMS):]:
        ax.set_axis_off()

    title = f"Raster Preview: {sample_dir.name}"
    if isinstance(resolution, (float, int)):
        title += f" | resolution ≈ {float(resolution):.3f} m/px"
    plt.suptitle(title)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview ReLL raster sample contents.")
    parser.add_argument("sample_folder", type=Path, help="Path to a preprocessed sample directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        preview_sample(args.sample_folder)
        return 0
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"[error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
