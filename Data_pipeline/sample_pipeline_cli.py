from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Data_pipeline.combine_lidar_sweeps import create_macro_sweep
from Data_pipeline.sample_pipeline_gui import (
    run_stage_two,
    DEFAULT_BUFFER_METERS,
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_CROP_MIN,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sample pipeline without the GUI")
    parser.add_argument("log_dir", type=Path, help="Path to the AV2 log directory")
    parser.add_argument("city_root", type=Path, help="Path to the Argoverse2-geoalign city folder (contains Imagery/DSM)")
    parser.add_argument("output_dir", type=Path, help="Directory to store generated samples")
    parser.add_argument("--sweep-count", type=int, default=5, help="Number of consecutive sweeps to merge (default: 5)")
    parser.add_argument("--center-index", type=int, default=None, help="Reference sweep index within the window")
    parser.add_argument("--prefix", type=str, default="combined_0p5s", help="Prefix for output files")
    parser.add_argument("--crop-size", type=float, default=None, help="Optional square crop size in metres (e.g. 32)")
    parser.add_argument("--crop-32", action="store_true", help="Shortcut for --crop-size 32")
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    city_root = args.city_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_size = args.crop_size
    if args.crop_32:
        crop_size = DEFAULT_CROP_MIN
    if crop_size is not None and crop_size < DEFAULT_CROP_MIN:
        parser.error(f"--crop-size must be at least {DEFAULT_CROP_MIN} meters")

    sweep_indices = list(range(args.sweep_count))
    center_index = args.center_index if args.center_index is not None else len(sweep_indices) // 2

    point_path, meta_path, applied_crop = create_macro_sweep(
        log_dir=log_dir,
        sweep_indices=sweep_indices,
        center_index=center_index,
        output_prefix=output_dir / args.prefix,
        compression="zstd",
        crop_square_m=crop_size,
    )
    print(f"Point parquet: {point_path}")
    print(f"Meta parquet: {meta_path}")
    if crop_size is not None:
        if applied_crop is not None and abs(applied_crop - crop_size) > 1e-6:
            print(f"Crop request {crop_size:.2f} m adjusted to {applied_crop:.2f} m")
        elif applied_crop is None:
            print("Crop request skipped; using full extent.")
        else:
            print(f"Applied crop size: {applied_crop:.2f} m")
    crop_for_stage2 = applied_crop if applied_crop is not None else None

    tif_path, laz_path = run_stage_two(
        points_path=point_path,
        meta_path=meta_path,
        city_root=city_root,
        output_dir=output_dir,
        buffer_m=DEFAULT_BUFFER_METERS,
        target_res=DEFAULT_GRID_RESOLUTION,
        base_name=args.prefix,
        crop_square_m=crop_for_stage2,
    )
    print(f"Imagery: {tif_path}")
    print(f"DSM: {laz_path}")


if __name__ == "__main__":
    main()
