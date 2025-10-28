import os
import time
import argparse
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import pickle  # Need pickle to save non-numpy objects like transform and profile
import torch


def _fmt_seconds(sec: float) -> str:
    """Return H:MM:SS for a duration in seconds."""
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def raster_builder_from_sample_dir(sample_dir: Path,
    *,
    target_m_per_px: float | None = None,
    splat_mode: str = "none",
    sampling: float = 1.0,
    coarsen_factor: int | None = None,
    preview: bool = False,
):
    """
    Builds raster data for a single sample directory using build_training_rasters.

    Looks for specific files within the sample_dir with known filenames:
      - LiDAR (GICP): segment_gicp_utm.parquet
      - LiDAR (non-aligned): segment_utm.parquet
      - DSM points: segment_dsm_utm.laz
      - Imagery (orthophoto): segment_imagery_utm.tif
    """
    print(f"Attempting to build rasters for sample: {sample_dir}")
    try:
        gicp_path = sample_dir / "segment_gicp_utm.parquet"
        non_path  = sample_dir / "segment_utm.parquet"
        dsm_path  = sample_dir / "segment_dsm_utm.laz"
        img_path  = sample_dir / "segment_imagery_utm.tif"

        # Check if files exist before proceeding
        if not gicp_path.exists():
            raise FileNotFoundError(f"Missing GICP parquet file: {gicp_path}")
        if not non_path.exists():
            raise FileNotFoundError(f"Missing non-aligned parquet file: {non_path}")
        if not dsm_path.exists():
            raise FileNotFoundError(f"Missing DSM laz file: {dsm_path}")
        if not img_path.exists():
            raise FileNotFoundError(f"Missing imagery tif file: {img_path}")

        # Call YOUR builder (already defined earlier in the notebook)
        ras = build_training_rasters(
            gicp_parquet=gicp_path,
            non_aligned_parquet=non_path,
            dsm_points_path=dsm_path,
            imagery_path=img_path,
            # ---- optional knobs you can tune:
            splat_mode=splat_mode,            # "none" | "bilinear" | "gaussian"
            target_m_per_px=target_m_per_px, # set (e.g., 0.2) to force output resolution; else keep source
            sampling=sampling,                 # scale factor if you want quick low-res tests (e.g., 2.0 downsample 2x)
            coarsen_factor=coarsen_factor,    # extra integer coarsening
            preview=preview,                  # set True to visualize once
        )
        print(f"Successfully built rasters for sample: {sample_dir}")
        return ras
    except FileNotFoundError as e:
        print(f"Skipping sample {sample_dir} due to missing file: {e}")
        return None  # Return None or raise a specific error if you prefer to stop
    except Exception as e:  # Catch any other exceptions during raster building
        print(f"Skipping sample {sample_dir} due to error during raster building: {e}")
        return None


def save_all_rasters_to_disk(source_folder, target_folder):
    """
    Iterates through sample directories in source_folder, builds rasters,
    and saves the results to corresponding subfolders in target_folder.
    Includes logging of raster information, per-sample timing, and ETA.
    """
    source_path = Path(source_folder)
    target_path = Path(target_folder)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting raster preparation and saving from {source_path} to {target_path}")

    sample_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    if not sample_dirs:
        print(f"No sample subfolders found in {source_folder}. Exiting.")
        return

    total = len(sample_dirs)
    print(f"Found {total} sample subfolders to process.")

    # Timing trackers
    t0 = time.monotonic()
    completed = 0
    cumulative_seconds = 0.0

    for i, sample_dir in enumerate(sample_dirs, start=1):
        sample_name = sample_dir.name
        sample_target_dir = target_path / sample_name
        sample_target_dir.mkdir(parents=True, exist_ok=True)

        start_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{i}/{total}] Processing sample: {sample_name} (started at {start_wall})")
        t_sample = time.monotonic()

        try:
            # Build rasters for this sample
            rasters = raster_builder_from_sample_dir(sample_dir)

            if rasters is None:
                print(f"Skipping sample {sample_name} as raster building failed.")
                continue

            # --- Raster information logging ---
            print(f"  Raster information for {sample_name}:")
            if "resolution" in rasters:
                print(f"    Resolution: {rasters['resolution']:.3f} m/px")
            if "transform" in rasters:
                transform = rasters['transform']
                print(
                    "    Transform (Affine): "
                    f"a={getattr(transform,'a',float('nan')):.3f}, "
                    f"b={getattr(transform,'b',float('nan')):.3f}, "
                    f"c={getattr(transform,'c',float('nan')):.3f}, "
                    f"d={getattr(transform,'d',float('nan')):.3f}, "
                    f"e={getattr(transform,'e',float('nan')):.3f}, "
                    f"f={getattr(transform,'f',float('nan')):.3f}"
                )
            if "imagery" in rasters:
                imagery_shape = rasters['imagery'].shape
                print(f"    Imagery shape (C, H, W): {imagery_shape}")
            if "gicp_height" in rasters:
                height_shape = rasters['gicp_height'].shape
                print(f"    LiDAR/DSM raster shape (H, W): {height_shape}")

            # Define keys for data to save
            tensor_keys = [
                "gicp_height", "gicp_intensity", "gicp_density",
                "non_aligned_height", "non_aligned_intensity", "non_aligned_density",
                "dsm_height", "dsm_density", "imagery",
            ]
            metadata_keys = ["transform", "crs", "profile", "resolution"]

            # Save tensors
            for key in tensor_keys:
                if key in rasters and isinstance(rasters[key], (np.ndarray, torch.Tensor)):
                    save_path = sample_target_dir / f"{key}.npy"
                    data_to_save = rasters[key].numpy() if isinstance(rasters[key], torch.Tensor) else rasters[key]
                    np.save(save_path, data_to_save)
                else:
                    print(f"Warning: Tensor key '{key}' not found or not a supported type in rasters for {sample_name}")

            # Save metadata
            for key in metadata_keys:
                if key in rasters:
                    save_path = sample_target_dir / f"{key}.pkl"
                    with open(save_path, 'wb') as f:
                        pickle.dump(rasters[key], f)
                else:
                    print(f"Warning: Metadata key '{key}' not found in rasters for {sample_name}")

            print(f"Successfully processed and saved rasters for sample: {sample_name}")

        except Exception as e:
            print(f"An error occurred while processing and saving sample {sample_name}: {e}")

        finally:
            # Per-sample timing + ETA
            elapsed_sample = time.monotonic() - t_sample
            completed += 1
            cumulative_seconds += elapsed_sample
            avg_per_sample = cumulative_seconds / completed if completed else float('inf')
            remaining = total - completed
            eta_seconds = avg_per_sample * remaining
            finish_time = datetime.now() + timedelta(seconds=eta_seconds)

            print(
                f"⏱  Time for {sample_name}: {_fmt_seconds(elapsed_sample)} | "
                f"Avg/sample: {_fmt_seconds(avg_per_sample)} | "
                f"Remaining: {remaining} | "
                f"ETA: {_fmt_seconds(eta_seconds)} (finish by {finish_time.strftime('%Y-%m-%d %H:%M:%S')})"
            )

    total_elapsed = time.monotonic() - t0
    print(f"\nFinished preparing and saving all rasters.")
    print(f"Total time: {_fmt_seconds(total_elapsed)} for {total} samples "
          f"(avg {_fmt_seconds(total_elapsed / total)})")




# !pip install "laspy[lazrs,laszip]"

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy, from_origin
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from matplotlib import cm  # colormap utility

from laspy import LazBackend
from laspy.errors import LaspyException


# --------------------------
# IO helpers
# --------------------------

def _read_las_with_backends(path):
    """Read LAS/LAZ file, automatically selecting an available backend."""
    import laspy
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".las":
        with laspy.open(path) as las_file:
            return las_file.read(), las_file.header

    if suffix == ".laz":
        last_exc = None
        for backend in (LazBackend.Lazrs, LazBackend.Laszip):
            try:
                with laspy.open(path, laz_backend=backend) as las_file:
                    return las_file.read(), las_file.header
            except LaspyException as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            "Unable to decompress LAZ file. Install a backend with:\n"
            '  pip install "laspy[lazrs,laszip]"\n'
            "and restart the runtime. Original error: "
            f"{last_exc}"
        )
    raise ValueError(f"Unsupported DSM format: {suffix}")


def _load_point_cloud(path, x_col="utm_e", y_col="utm_n", z_col="elevation", intensity_col="intensity"):
    """Load LiDAR parquet (GICP or non-aligned) and return x/y/z/intensity as float32."""
    cols = [x_col, y_col, z_col, intensity_col]
    df = pd.read_parquet(path, columns=cols).dropna()
    df = df.rename(columns={x_col: "x", y_col: "y", z_col: "z", intensity_col: "intensity"})
    return df.astype({"x": np.float32, "y": np.float32, "z": np.float32, "intensity": np.float32})


def _load_dsm_points(path, x_col="utm_e", y_col="utm_n", z_col="elevation"):
    """Load DSM points from parquet or LAZ/LAS."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=[x_col, y_col, z_col]).dropna()
        return df.rename(columns={x_col: "x", y_col: "y", z_col: "z"}).astype(np.float32)

    points, header = _read_las_with_backends(path)
    scale = np.array(header.scales, dtype=np.float64)
    offset = np.array(header.offsets, dtype=np.float64)
    x = (points.X * scale[0] + offset[0]).astype(np.float32)
    y = (points.Y * scale[1] + offset[1]).astype(np.float32)
    z = (points.Z * scale[2] + offset[2]).astype(np.float32)
    return pd.DataFrame({"x": x, "y": y, "z": z})


# --------------------------
# Imagery resolution control
# --------------------------

def _resampling_from_name(name: str) -> Resampling:
    """
    Map a friendly string to a rasterio Resampling enum.
    """
    name = (name or "").strip().lower()
    table = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubic_spline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "gauss": Resampling.gauss,
        "mode": Resampling.mode,
        "min": Resampling.min,
        "max": Resampling.max,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
        "sum": Resampling.sum,
        "rms": Resampling.rms,
    }
    if name not in table:
        raise ValueError(
            f"Unknown resampling method '{name}'. "
            f"Choose one of: {', '.join(table.keys())}"
        )
    return table[name]


def _sampling_for_target_res(transform, target_m_per_px: float) -> float:
    """
    Given the original imagery transform, compute a sampling factor so that
    the output pixel size is target_m_per_px.
    sampling = target_m_per_px / orig_res
    """
    orig_res = float(transform.a)  # meters per pixel (assumes square px)
    if target_m_per_px <= 0:
        raise ValueError("target_m_per_px must be > 0")
    return target_m_per_px / orig_res


def _read_imagery_template(path, sampling=1.0, downsample=None,
                           upsample_method="lanczos", downsample_method="gauss",
                           target_m_per_px=None, coarsen_factor=None):
    """
    Read imagery and resample to a target resolution factor or explicit meters-per-pixel.

    Priority of resolution control:
      1) target_m_per_px (explicit)
      2) downsample (legacy alias; sets 'sampling')
      3) sampling (factor; 0.5 = upsample 2x; 2.0 = downsample 2x)

    After resampling, you may coarsen again by an integer 'coarsen_factor' (>=2)
    using average pooling (anti-aliased).
    """
    if downsample is not None:
        sampling = float(downsample)

    with rasterio.open(path) as src:
        if target_m_per_px is not None:
            sampling = _sampling_for_target_res(src.transform, float(target_m_per_px))

        sampling = float(sampling)
        if sampling <= 0:
            raise ValueError("sampling must be > 0")

        out_height = int(round(src.height / sampling))
        out_width  = int(round(src.width  / sampling))

        if sampling == 1.0:
            imagery   = src.read()
            transform = src.transform
        else:
            resamp = _resampling_from_name(upsample_method if sampling < 1.0 else downsample_method)
            imagery = src.read(out_shape=(src.count, out_height, out_width), resampling=resamp)
            transform = src.transform * rasterio.Affine.scale(sampling, sampling)

        # Optional extra coarsening (fast)
        if coarsen_factor and int(coarsen_factor) >= 2:
            cf = int(coarsen_factor)
            new_h = max(1, imagery.shape[1] // cf)
            new_w = max(1, imagery.shape[2] // cf)
            pooled = []
            for b in range(imagery.shape[0]):
                band = imagery[b]
                # trim to multiple of cf
                hh = band[:new_h*cf, :new_w*cf].reshape(new_h, cf, new_w, cf)
                pooled.append(hh.mean(axis=(1,3)))
            imagery = np.stack(pooled, axis=0)
            transform = transform * rasterio.Affine.scale(cf, cf)

        profile = src.profile
        profile.update({"height": imagery.shape[1], "width": imagery.shape[2], "transform": transform})

    return imagery, transform, profile


# --------------------------
# Rasterization (with splats)
# --------------------------

def _extent_from_transform(transform, width, height):
    x_min = transform.c
    x_max = x_min + transform.a * width
    y_max = transform.f
    y_min = y_max + transform.e * height
    return (x_min, x_max, y_min, y_max)


def _print_raster_info(name, transform, width, height):
    res_x = transform.a
    res_y = abs(transform.e)
    center_row = height / 2.0
    center_col = width / 2.0
    center_x, center_y = xy(transform, center_row, center_col)
    print(
        f"{name:>24s} | resolution: ({res_x:.3f} m, {res_y:.3f} m) "
        f"| center (x, y): ({center_x:.3f}, {center_y:.3f}) "
        f"| size: {height} x {width} px"
    )


def _voxel_thin_xy(df, voxel_xy_m: float | None):
    """
    Optional 2D voxel thinning in (x,y): keeps one point per voxel cell (mean z/intensity).
    Set voxel_xy_m=None to skip.
    """
    if not voxel_xy_m or voxel_xy_m <= 0:
        return df
    vx = np.floor(df["x"].values / voxel_xy_m).astype(np.int64)
    vy = np.floor(df["y"].values / voxel_xy_m).astype(np.int64)
    g = pd.DataFrame({
        "vx": vx, "vy": vy,
        "z": df["z"].values,
        "intensity": df.get("intensity", pd.Series(np.nan, index=df.index)).values
    })
    agg = g.groupby(["vx", "vy"], sort=False).agg({"z": "mean", "intensity": "mean"}).reset_index()
    agg["x"] = (agg["vx"].astype(np.float32) + 0.5) * voxel_xy_m
    agg["y"] = (agg["vy"].astype(np.float32) + 0.5) * voxel_xy_m
    return agg[["x", "y", "z", "intensity"]].astype(np.float32)


def rasterize_points_to_grid(
    df,
    transform,
    width,
    height,
    value_col,
    agg="mean",
    *,
    splat_mode="none",         # "none" | "bilinear" | "gaussian"
    splat_radius_px=0,         # used for gaussian
    splat_sigma_px=0.75,       # used for gaussian
    voxel_xy_m=None,           # optional pre-thinning in XY (meters)
    return_density: bool = False,
):
    """
    Rasterize with optional splatting modes:
      - none: assign to nearest pixel (fast; supports agg mean/max/min)
      - bilinear: distribute to 4 neighbors using fractional row/col (agg forced to mean)
      - gaussian: distribute within a (2r+1)x(2r+1) window using a Gaussian kernel (agg forced to mean)
    """
    if df.empty:
        return np.full((height, width), np.nan, dtype=np.float32)

    if splat_mode not in {"none", "bilinear", "gaussian"}:
        raise ValueError("splat_mode must be one of {'none','bilinear','gaussian'}")

    if splat_mode != "none" and agg != "mean":
        raise ValueError("When splat_mode is not 'none', agg must be 'mean' (weighted).")

    # Optional 2D voxel thinning for speed
    if voxel_xy_m:
        df = _voxel_thin_xy(df, voxel_xy_m)
        if df.empty:
            return np.full((height, width), np.nan, dtype=np.float32)

    xs = df["x"].to_numpy(np.float64)
    ys = df["y"].to_numpy(np.float64)
    vals = df[value_col].to_numpy(np.float32)

    inv = ~transform
    cols_f, rows_f = inv * (xs, ys)  # fractional col/row

    # keep only in-bounds (with a margin for gaussian radius)
    margin = 0 if splat_mode != "gaussian" else max(int(splat_radius_px), 0)
    mask = (
        (rows_f >= -margin) & (rows_f < height + margin) &
        (cols_f >= -margin) & (cols_f < width  + margin) &
        np.isfinite(vals)
    )
    rows_f = rows_f[mask].astype(np.float32)
    cols_f = cols_f[mask].astype(np.float32)
    vals   = vals[mask]

    raster = np.full((height, width), np.nan, dtype=np.float32)
    density = np.zeros((height, width), dtype=np.float32) if return_density else None

    if splat_mode == "none":
        rows = rows_f.round().astype(np.int32)
        cols = cols_f.round().astype(np.int32)
        inb = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows, cols, vals = rows[inb], cols[inb], vals[inb]

        if agg == "max":
            raster[:] = -np.inf
            np.maximum.at(raster, (rows, cols), vals)
            raster[raster == -np.inf] = np.nan
            if return_density:
                np.add.at(density, (rows, cols), 1.0)
        elif agg == "min":
            raster[:] = np.inf
            np.minimum.at(raster, (rows, cols), vals)
            raster[raster == np.inf] = np.nan
            if return_density:
                np.add.at(density, (rows, cols), 1.0)
        elif agg == "mean":
            sums   = np.zeros((height, width), dtype=np.float32)
            counts = np.zeros((height, width), dtype=np.uint32)
            np.add.at(sums,   (rows, cols), vals)
            np.add.at(counts, (rows, cols), 1)
            valid = counts > 0
            raster[valid] = sums[valid] / counts[valid]
            if return_density:
                density = counts.astype(np.float32)
        else:
            raise ValueError("agg must be one of {'mean','max','min'}")
        return (raster, density) if return_density else raster

    # Weighted mean accumulators for bilinear/gaussian
    sums   = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    if splat_mode == "bilinear":
        r0 = np.floor(rows_f).astype(np.int32)
        c0 = np.floor(cols_f).astype(np.int32)
        dr = rows_f - r0
        dc = cols_f - c0

        # neighbors and weights
        nbrs = [
            (r0,     c0,     (1-dr)*(1-dc)),
            (r0+1,   c0,     dr*(1-dc)),
            (r0,     c0+1,   (1-dr)*dc),
            (r0+1,   c0+1,   dr*dc),
        ]
        for rr, cc, w in nbrs:
            inb = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width) & (w > 0)
            if not np.any(inb):
                continue
            np.add.at(sums,   (rr[inb], cc[inb]), vals[inb] * w[inb])
            np.add.at(weights,(rr[inb], cc[inb]), w[inb])

    elif splat_mode == "gaussian":
        R = int(max(splat_radius_px, 0))
        if R <= 0:
            # if radius=0, just nearest with mean (degenerate gaussian)
            rows = rows_f.round().astype(np.int32)
            cols = cols_f.round().astype(np.int32)
            inb = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            np.add.at(sums,   (rows[inb], cols[inb]), vals[inb])
            np.add.at(weights,(rows[inb], cols[inb]), 1.0)
        else:
            # precompute kernel grid offsets
            oy, ox = np.mgrid[-R:R+1, -R:R+1]
            dist2 = (ox**2 + oy**2).astype(np.float32)
            two_sigma2 = 2.0 * (float(splat_sigma_px)**2)
            kernel = np.exp(-dist2 / max(two_sigma2, 1e-6)).astype(np.float32)

            # splat each point
            r_center = np.round(rows_f).astype(np.int32)
            c_center = np.round(cols_f).astype(np.int32)
            for rc, cc, v in zip(r_center, c_center, vals):
                r0 = rc - R
                c0 = cc - R
                r1 = rc + R + 1
                c1 = cc + R + 1

                # clip kernel to image bounds
                kr0 = max(0, -r0); kc0 = max(0, -c0)
                kr1 = kernel.shape[0] - max(0, r1 - height)
                kc1 = kernel.shape[1] - max(0, c1 - width)

                rr0 = max(r0, 0); rr1 = min(r1, height)
                cc0 = max(c0, 0); cc1 = min(c1, width)
                if rr0 >= rr1 or cc0 >= cc1:
                    continue

                wpatch = kernel[kr0:kr1, kc0:kc1]
                sums[rr0:rr1, cc0:cc1]    += v * wpatch
                weights[rr0:rr1, cc0:cc1] += wpatch

    valid = weights > 0
    raster[valid] = sums[valid] / weights[valid]
    if return_density:
        density = weights
    return (raster, density) if return_density else raster


# --------------------------
# Viz helpers
# --------------------------

def _normalize_channel(arr, mask, lower=1, upper=99):
    """Normalize an array to 0..1 using percentile clipping."""
    if not np.any(mask):
        return np.zeros_like(arr, dtype=np.float32)
    valid_vals = arr[mask]
    vmin = np.percentile(valid_vals, lower)
    vmax = np.percentile(valid_vals, upper)
    if np.isclose(vmax, vmin):
        return np.zeros_like(arr, dtype=np.float32)
    normalized = (arr - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def compose_height_intensity_rgb(
    height,
    intensity,
    *,
    colormap="cividis",
    height_percentiles=(1, 99),
    intensity_percentiles=(1, 99),
):
    """
    Blend LiDAR height and intensity into an RGB image on black background.
    Height -> colormap (hue), intensity -> brightness.
    """
    rgb = np.zeros((*height.shape, 3), dtype=np.float32)
    mask = np.isfinite(height) & np.isfinite(intensity)
    if not np.any(mask):
        return rgb

    # Use plt.get_cmap as cm.get_cmap is deprecated
    cmap = plt.get_cmap(colormap)
    h_norm = _normalize_channel(height, mask, *height_percentiles)
    i_norm = _normalize_channel(intensity, mask, *intensity_percentiles)

    rgb_masked = cmap(h_norm[mask])[:, :3]         # apply colormap to height
    rgb_masked = (rgb_masked.T * i_norm[mask]).T   # modulate brightness by intensity
    rgb[mask] = rgb_masked
    rgb[~mask] = 0.0
    return rgb.astype(np.float32)


def _preview_rasters(
    gicp_display,
    non_display,
    dsm_display,
    imagery,
    transform,
    *,
    gicp_height,
    non_height,
):
    """Render four panels: two LiDAR composites, DSM, imagery."""
    height, width = gicp_height.shape
    extent = _extent_from_transform(transform, width, height)

    print("\nRaster metadata:")
    _print_raster_info("GICP height", transform, width, height)
    _print_raster_info("Non-aligned height", transform, width, height)
    _print_raster_info("DSM height", transform, width, height)
    _print_raster_info("Imagery", transform, width, height)

    panels = [
        ("GICP (height+intensity)", gicp_display),
        ("Non-aligned (height+intensity)", non_display),
        ("DSM elevation", dsm_display),
        ("Imagery (RGB)", None),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    for ax, (title, data) in zip(axes, panels):
        ax.set_title(title)
        ax.set_axis_off()
        if title == "Imagery (RGB)":
            if imagery.shape[0] >= 3:
                rgb = np.moveaxis(imagery[:3], 0, -1).astype(np.float32)
                rgb_min = np.nanpercentile(rgb, 1)
                rgb_max = np.nanpercentile(rgb, 99)
                rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-6), 0, 1)
            else:
                rgb = imagery.squeeze()
            ax.imshow(rgb, extent=extent, origin="upper")
        elif data.ndim == 3:
            ax.imshow(data, extent=extent, origin="upper")
        else:
            im = ax.imshow(data, cmap="terrain", extent=extent, origin="upper")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Elevation (m)")

    plt.show()


def _preview_channels(channels: dict[str, np.ndarray], transform) -> None:
    """
    Display each raster channel in its own subplot.
    """
    if not channels:
        print("No channels to preview.")
        return

    names = list(channels.keys())
    arrays = [channels[name] for name in names]
    height, width = arrays[0].shape
    extent = _extent_from_transform(transform, width, height)

    fig, axes = plt.subplots(
        1,
        len(arrays),
        figsize=(5 * len(arrays), 5),
        constrained_layout=True,
    )
    if len(arrays) == 1:
        axes = [axes]

    for ax, name, data in zip(axes, names, arrays):
        ax.set_title(name)
        ax.set_axis_off()
        cmap = "cividis" if "intensity" in name.lower() else "terrain"
        mask = np.isfinite(data)
        if np.any(mask):
            vmin = np.nanpercentile(data[mask], 1)
            vmax = np.nanpercentile(data[mask], 99)
            if np.isclose(vmax, vmin):
                vmax = vmin + 1e-3
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(
            data,
            cmap=cmap,
            extent=extent,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    plt.show()


# --------------------------
# Main entry
# --------------------------

def build_training_rasters(
    gicp_parquet,
    non_aligned_parquet,
    dsm_points_path,
    imagery_path,
    *,
    agg_height="mean",
    agg_intensity="mean",
    # resolution controls
    sampling=1.0,                  # factor (0.5=up2x, 1=orig, 2=down2x)
    downsample=None,               # legacy alias; overrides `sampling`
    target_m_per_px=None,          # explicit meters-per-px (overrides sampling if set)
    coarsen_factor=None,           # optional integer >=2 to further coarsen
    upsample_method="lanczos",
    downsample_method="gauss",
    # rasterization controls
    splat_mode="none",             # "none" | "bilinear" | "gaussian"
    splat_radius_px=1,
    splat_sigma_px=0.9,
    voxel_xy_m=None,
    # viz / preview
    preview=False,
    height_percentiles=(1, 99),
    intensity_percentiles=(1, 99),
    lidar_colormap="cividis",
):
    """
    Rasterize training data on a grid derived from imagery, with configurable resolution & resamplers.

    You can control resolution by:
      - target_m_per_px (explicit meters-per-px), or
      - sampling / downsample (scale factor), plus optional coarsen_factor.

    Rasterization supports 'none' (nearest), 'bilinear', or 'gaussian' splats.
    When splat_mode != 'none', agg_* must be 'mean'.
    """
    imagery, transform, profile = _read_imagery_template(
        imagery_path,
        sampling=sampling,
        downsample=downsample,
        upsample_method=upsample_method,
        downsample_method=downsample_method,
        target_m_per_px=target_m_per_px,
        coarsen_factor=coarsen_factor,
    )
    _, height, width = imagery.shape

    # Load data
    gicp_df = _load_point_cloud(gicp_parquet)
    non_df  = _load_point_cloud(non_aligned_parquet)
    dsm_df  = _load_dsm_points(dsm_points_path)

    # LiDAR (GICP)
    gicp_height, gicp_density = rasterize_points_to_grid(
        gicp_df, transform, width, height, "z", agg=agg_height,
        splat_mode=splat_mode, splat_radius_px=splat_radius_px,
        splat_sigma_px=splat_sigma_px, voxel_xy_m=voxel_xy_m,
        return_density=True,
    )
    gicp_intensity, _ = rasterize_points_to_grid(
        gicp_df, transform, width, height, "intensity", agg=agg_intensity,
        splat_mode=splat_mode, splat_radius_px=splat_radius_px,
        splat_sigma_px=splat_sigma_px, voxel_xy_m=voxel_xy_m,
        return_density=True,
    )

    # LiDAR (non-aligned)
    non_height, non_density = rasterize_points_to_grid(
        non_df, transform, width, height, "z", agg=agg_height,
        splat_mode=splat_mode, splat_radius_px=splat_radius_px,
        splat_sigma_px=splat_sigma_px, voxel_xy_m=voxel_xy_m,
        return_density=True,
    )
    non_intensity, _ = rasterize_points_to_grid(
        non_df, transform, width, height, "intensity", agg=agg_intensity,
        splat_mode=splat_mode, splat_radius_px=splat_radius_px,
        splat_sigma_px=splat_sigma_px, voxel_xy_m=voxel_xy_m,
        return_density=True,
    )

    # DSM (height only)
    dsm_height, dsm_density = rasterize_points_to_grid(
        dsm_df, transform, width, height, "z", agg=agg_height,
        splat_mode=splat_mode, splat_radius_px=splat_radius_px,
        splat_sigma_px=splat_sigma_px, voxel_xy_m=voxel_xy_m,
        return_density=True,
    )

    # Fancy LiDAR composites for previewing only
    gicp_rgb = compose_height_intensity_rgb(
        gicp_height, gicp_intensity,
        colormap=lidar_colormap,
        height_percentiles=height_percentiles,
        intensity_percentiles=intensity_percentiles,
    )
    non_rgb = compose_height_intensity_rgb(
        non_height, non_intensity,
        colormap=lidar_colormap,
        height_percentiles=height_percentiles,
        intensity_percentiles=intensity_percentiles,
    )

    # Geospatial stack (imagery + DSM)
    dsm_imagery_stack = np.concatenate([imagery, dsm_height[np.newaxis, ...]], axis=0)

    # Calculate resolution from transform
    resolution = float(abs(transform.a)) # Assuming square pixels

    if preview:
        _preview_rasters(
            gicp_rgb, non_rgb, dsm_height, imagery, transform,
            gicp_height=gicp_height, non_height=non_height,
        )

    return {
        "gicp_height": gicp_height,
        "gicp_intensity": gicp_intensity,
        "gicp_rgb": gicp_rgb,
        "non_aligned_height": non_height,
        "non_aligned_intensity": non_intensity,
        "non_aligned_rgb": non_rgb,
        "dsm_height": dsm_height,
        "gicp_density": gicp_density,
        "non_aligned_density": non_density,
        "dsm_density": dsm_density,
        "imagery": imagery,
        "dsm_imagery_stack": dsm_imagery_stack,
        "transform": transform,
        "crs": profile.get("crs"),
        "profile": profile,
        "resolution": resolution, # Include resolution in the output dictionary
    }


def _resolve_point_path(path: Path) -> Path:
    """
    Ensure the provided path points to an existing point cloud file.
    """
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    if path.is_dir():
        raise ValueError("Please pass a specific point cloud file, not a directory.")
    return path




    def preview_sample_folder(
        sample_dir: Path,
        *,
        target_m_per_px: float = 0.1,
        splat_mode: str = "none",
        sampling: float = 1.0,
        coarsen_factor: int | None = None,
        voxel_xy_m: float | None = None,
    ):
        """Preview preprocessed rasters stored in a sample directory."""
        sample_dir = Path(sample_dir)
        if not sample_dir.exists() or not sample_dir.is_dir():
            raise FileNotFoundError(f"Sample folder not found: {sample_dir}")

        def _load_array(name: str):
            path_local = sample_dir / name
            if not path_local.exists():
                print(f"[missing] {name}")
                return None
            try:
                arr_local = np.load(path_local)
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"[error] Failed to load {name}: {exc}")
                return None
            print(f"[loaded] {name} -> shape={arr_local.shape}, dtype={arr_local.dtype}")
            return arr_local

        dsm_height = _load_array("dsm_height.npy")
        gicp_height = _load_array("gicp_height.npy")
        gicp_intensity = _load_array("gicp_intensity.npy")
        imagery = _load_array("imagery.npy")
        non_height = _load_array("non_aligned_height.npy")
        non_intensity = _load_array("non_aligned_intensity.npy")

        arrays = [
            ("DSM Height", dsm_height, "terrain"),
            ("GICP Height", gicp_height, "viridis"),
            ("GICP Intensity", gicp_intensity, "gray"),
            ("Imagery (RGB)", imagery, None),
            ("Non-aligned Height", non_height, "magma"),
            ("Non-aligned Intensity", non_intensity, "gray"),
        ]

        transform = None
        resolution = None
        print("\n" + "=" * 80)
        print(f"Sample: {sample_dir}")
        print("=" * 80)
        for name in ["resolution.pkl", "transform.pkl", "profile.pkl", "metadata.pkl"]:
            path_local = sample_dir / name
            if not path_local.exists():
                print(f"{name}: <missing>")
                continue
            try:
                with open(path_local, "rb") as fh:
                    data = pickle.load(fh)
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"{name}: <error reading pickle> ({exc})")
                continue
            print(f"{name}: {data}")
            if name == "resolution.pkl":
                try:
                    resolution = float(data)
                except Exception:
                    resolution = None
            if name == "transform.pkl":
                transform = data

        extent = None
        grid_shape = None
        for _, arr, _ in arrays:
            if arr is None:
                continue
            if arr.ndim == 3:
                grid_shape = (arr.shape[1], arr.shape[2])
            elif arr.ndim == 2:
                grid_shape = arr.shape
            if grid_shape is not None:
                break
        if transform is not None and grid_shape is not None:
            extent = _extent_from_transform(transform, grid_shape[1], grid_shape[0])

        print("\nArray statistics:")
        for title, arr, _ in arrays:
            if arr is None:
                continue
            info = f"shape={arr.shape}, dtype={arr.dtype}"
            if arr.ndim == 2:
                finite = np.isfinite(arr)
                if np.any(finite):
                    vmin = float(np.nanmin(arr[finite]))
                    vmax = float(np.nanmax(arr[finite]))
                    mean = float(np.nanmean(arr[finite]))
                else:
                    vmin = vmax = mean = float('nan')
                info += f", min={vmin:.3f}, max={vmax:.3f}, mean={mean:.3f}"
            print(f"  {title:<24} {info}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        axes = axes.flatten()

        for ax, (title, arr, cmap) in zip(axes, arrays):
            ax.set_title(title)
            ax.set_axis_off()
            if arr is None:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
                continue
            if arr.ndim == 3:
                chw = arr
                if chw.shape[0] >= 3:
                    rgb = np.transpose(chw[:3, :, :], (1, 2, 0))
                else:
                    rgb = np.transpose(np.repeat(chw[0:1, :, :], 3, axis=0), (1, 2, 0))
                rgb = np.clip(rgb, 0.0, 1.0)
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

        for ax in axes[len(arrays):]:
            ax.set_axis_off()

        suptitle = f"Raster Preview: {sample_dir.name}"
        if resolution is not None:
            suptitle += f" | resolution ≈ {resolution:.3f} m/px"
        plt.suptitle(suptitle)
        plt.show()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview rasters generated from a single point cloud file."
    )
    parser.add_argument(
        "--data",
        required=False,
        default=None,
        help="Path to a point cloud file (Parquet or LAS/LAZ).",
    )
    parser.add_argument(
        "--splat-mode",
        choices=("none", "bilinear", "gaussian"),
        default="none",
        help="Point splatting mode when rasterizing.",
    )
    parser.add_argument(
        "--splat-radius",
        type=int,
        default=1,
        help="Gaussian splat radius in pixels (only used for gaussian mode).",
    )
    parser.add_argument(
        "--splat-sigma",
        type=float,
        default=0.9,
        help="Gaussian splat sigma in pixels (only used for gaussian mode).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.3047,
        help="Raster cell size in meters.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Extra margin (meters) added around the point cloud extent.",
    )
    parser.add_argument(
        "--voxel-xy-m",
        type=float,
        default=None,
        help="Optional XY voxel thinning size (meters) before rasterization.",
    )
    parser.add_argument(
        "--sampling",
        type=float,
        default=1.0,
        help="Sampling factor to apply when reading imagery (0.5=upsample 2x, 2=downsample 2x).",
    )
    parser.add_argument(
        "--sample-folder",
        type=str,
        default=None,
        help="Path to a sample folder (e.g., Rell-sample-raster/<sample>). If provided, previews rasters from that folder at the requested target resolution.",
    )
    parser.add_argument(
        "--preview-target-m-per-px",
        type=float,
        default=0.1,
        help="Target meters-per-pixel for previewing imagery and rasters (default 0.1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # If --sample-folder provided, preview rasters from that sample at the requested resolution
    if getattr(args, "sample_folder", None):
        sample_path = Path(args.sample_folder)
        print(f"Previewing sample folder: {sample_path} at {args.preview_target_m_per_px} m/px")
        preview_sample_folder(
            sample_path,
            target_m_per_px=args.preview_target_m_per_px,
            splat_mode=args.splat_mode,
            sampling=args.sampling,
            coarsen_factor=None,
            voxel_xy_m=args.voxel_xy_m,
        )
        return

    # Fallback: original single-point preview mode
    point_path = _resolve_point_path(Path(args.data))

    print(f"Using point cloud: {point_path}")
    print(f"Rasterizing with splat_mode='{args.splat_mode}'")

    if args.splat_mode != "gaussian":
        splat_radius = 0
        splat_sigma = args.splat_sigma
    else:
        splat_radius = max(args.splat_radius, 0)
        splat_sigma = args.splat_sigma

    ext = point_path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        df = _load_point_cloud(point_path)
    elif ext in {".laz", ".las"}:
        df = _load_dsm_points(point_path)
        df["intensity"] = np.nan
    else:
        raise ValueError("Unsupported point cloud format. Use Parquet or LAS/LAZ.")

    transform, width, height = _compute_grid_from_points(
        df,
        resolution=args.resolution,
        margin=args.margin,
    )

    extra = {}
    if "intensity" in df and np.isfinite(df["intensity"].to_numpy()).any():
        intensity_raster, intensity_density = rasterize_points_to_grid(
            df,
            transform,
            width,
            height,
            "intensity" if "intensity" in df.columns else "z",
            agg="mean",
            splat_mode=args.splat_mode,
            splat_radius_px=splat_radius,
            splat_sigma_px=splat_sigma,
            voxel_xy_m=args.voxel_xy_m,
            return_density=True,
        )
        extra["intensity"] = intensity_raster
        extra["intensity_density"] = intensity_density

    height_raster, height_density = rasterize_points_to_grid(
        df,
        transform,
        width,
        height,
        "z",
        agg="mean",
        splat_mode=args.splat_mode,
        splat_radius_px=splat_radius,
        splat_sigma_px=splat_sigma,
        voxel_xy_m=args.voxel_xy_m,
        return_density=True,
    )

    channels = {"height": height_raster, "height_density": height_density}
    if "intensity" in extra:
        channels["intensity"] = extra["intensity"]
        channels["intensity_density"] = extra["intensity_density"]

    _preview_channels(channels, transform)


if __name__ == "__main__":
    main()
# Example:
# save_all_rasters_to_disk(source_folder, target_folder)
