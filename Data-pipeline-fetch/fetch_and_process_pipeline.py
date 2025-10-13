"""Integrated pipeline: Fetch Argoverse2 data and process into 5-second segments.

This script combines:
1. Data fetching from Argoverse2 S3
2. LiDAR segment processing (lib.lidar_processing)
3. Imagery and DSM alignment (lib.imagery_processing)

Key features:
- Configurable city filter (Austin only for now)
- Configurable sample count
- Processes entire time sequences into 5-second non-overlapping segments
- No train/val/test division in output
- Logs time cost for each sample
- Doesn't save original downloaded data (processes and deletes)
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import yaml

# Import local processing modules
from lib import (
    create_macro_sweep,
    run_stage_two,
    extract_dsm_near_lidar,
    align_lidar_to_dsm,
    GICPParams,
)

# Argoverse2 S3 configuration
BASE_URL = "https://s3.amazonaws.com/argoverse"
S3_URI_ROOT = "s3://argoverse"
LIDAR_PREFIX = "datasets/av2/lidar/"
XML_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# City code mapping
CITY_CODE_MAP = {
    "ATX": "austin",
    "DTW": "detroit",
    "MIA": "miami",
    "PAO": "palo_alto",
    "PIT": "pittsburgh",
    "WDC": "washington_dc",
}


@dataclass
class PipelineConfig:
    """Configuration for the fetch-and-process pipeline."""

    # Fetching config
    target_city: str
    sample_count: int | str
    fetch_splits: List[str]
    skip_existing_downloads: bool
    skip_processed_logs: bool

    # Processing config
    sweeps_per_segment: int
    segment_overlap_sweeps: int
    crop_size_meters: Optional[float]
    buffer_meters: float

    # Path config
    temp_download_dir: Path
    output_dir: Path
    city_geoalign_root: Path
    s5cmd_path: Path

    # S5cmd config
    s5cmd_concurrency: int
    s5cmd_part_size_mb: int

    # Output config
    compression: str
    output_file_prefix: str

    # Logging config
    enable_detailed_logging: bool
    log_file: Optional[str]

    # Cleanup config
    cleanup_after_processing: bool
    cleanup_temp_dir_on_completion: bool

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert paths
        config_dict["temp_download_dir"] = Path(config_dict["temp_download_dir"])
        config_dict["output_dir"] = Path(config_dict["output_dir"])
        config_dict["city_geoalign_root"] = Path(config_dict["city_geoalign_root"])
        config_dict["s5cmd_path"] = Path(config_dict["s5cmd_path"])

        # Handle 'all' for sample_count
        if config_dict["sample_count"] == "all":
            config_dict["sample_count"] = -1

        return cls(**config_dict)


class S3PrefixLister:
    """List S3 bucket prefixes (log directories)."""

    def __init__(self, max_keys: int = 1000) -> None:
        self.max_keys = max_keys

    def _fetch(self, prefix: str, delimiter: str, continuation: Optional[str]) -> ET.Element:
        """Fetch one page of S3 listing."""
        params = {
            "list-type": "2",
            "prefix": prefix,
            "delimiter": delimiter,
            "max-keys": str(self.max_keys),
        }
        if continuation:
            params["continuation-token"] = continuation
        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as response:  # nosec B310
            data = response.read()
        return ET.fromstring(data)

    def iter_common_prefixes(self, prefix: str) -> Iterator[str]:
        """Iterate through all common prefixes (subdirectories) under a prefix."""
        continuation: Optional[str] = None
        while True:
            root = self._fetch(prefix, delimiter="/", continuation=continuation)
            for element in root.findall(f"{XML_NS}CommonPrefixes"):
                prefix_elem = element.find(f"{XML_NS}Prefix")
                if prefix_elem is None or prefix_elem.text is None:
                    continue
                yield prefix_elem.text
            token_elem = root.find(f"{XML_NS}NextContinuationToken")
            if token_elem is None or not token_elem.text:
                break
            continuation = token_elem.text

    def list_objects(self, prefix: str) -> List[str]:
        """List all objects (files) under a prefix."""
        objects: List[str] = []
        continuation: Optional[str] = None
        while True:
            root = self._fetch(prefix, delimiter="", continuation=continuation)
            for element in root.findall(f"{XML_NS}Contents"):
                key_elem = element.find(f"{XML_NS}Key")
                if key_elem is None or key_elem.text is None:
                    continue
                objects.append(key_elem.text)
            token_elem = root.find(f"{XML_NS}NextContinuationToken")
            if token_elem is None or not token_elem.text:
                break
            continuation = token_elem.text
        return objects


def check_log_city_from_s3(log_id: str, split: str, target_city: str,
                          lister: S3PrefixLister, logger: logging.Logger) -> bool:
    """Check if a log is from the target city by examining map files in S3.

    This checks the map directory without downloading the entire log.
    Map files are named: log_map_archive_{CITY}_city_{id}.json

    Returns:
        True if log is from target city, False otherwise
    """
    try:
        # List files in the map directory
        map_prefix = f"{LIDAR_PREFIX}{split}/{log_id}/map/"
        objects = lister.list_objects(map_prefix)

        # Look for log_map_archive_*.json files
        for obj_key in objects:
            if "log_map_archive_" in obj_key and obj_key.endswith(".json"):
                # Extract city code from filename
                # Format: log_map_archive_{LOG_ID}__Summer____{CITY}_city_{ID}.json
                # Example: log_map_archive_0QB8KZQ9HFftSYAPyyIktvRCbbE9oL9r__Summer____ATX_city_77093.json
                filename = obj_key.split("/")[-1]
                if filename.startswith("log_map_archive_"):
                    # Remove prefix and suffix
                    middle = filename.replace("log_map_archive_", "").split("_city_")[0]
                    # Format: {LOG_ID}__Summer____{CITY}
                    # Extract city: split by ____ (4 underscores) and take last part
                    parts = middle.split("____")
                    if len(parts) >= 2:
                        city_part = parts[-1]  # Last part after ____
                    else:
                        # Fallback: might be older format without season
                        city_part = middle

                    if city_part == target_city:
                        logger.debug(f"    [{split}] {log_id}: {city_part} ✓")
                        return True
                    else:
                        logger.debug(f"    [{split}] {log_id}: {city_part} (skip)")
                        return False

        # No map file found - unclear city
        logger.debug(f"    [{split}] {log_id}: no map file found (skip)")
        return False

    except Exception as e:
        logger.debug(f"    [{split}] {log_id}: error checking city ({e})")
        return False


def collect_logs(split: str, target_city: str, limit: int, lister: S3PrefixLister,
                 logger: logging.Logger) -> List[str]:
    """Collect log IDs for a given split, filtered by city.

    This function checks the city BEFORE downloading by examining map files in S3.
    Returns exactly 'limit' logs from the target city (or fewer if not enough exist).
    """
    if limit == 0:
        return []

    prefix = f"{LIDAR_PREFIX}{split}/"
    found: List[str] = []
    checked_count = 0

    logger.info(f"")
    logger.info(f"="*70)
    logger.info(f"Scanning S3 split: '{split}' (target city: {target_city})")
    logger.info(f"S3 prefix: {prefix}")
    logger.info(f"="*70)

    for common in lister.iter_common_prefixes(prefix):
        if not common.startswith(prefix):
            continue
        suffix = common[len(prefix):].strip("/")
        if not suffix:
            continue

        checked_count += 1

        # Check if this log is from target city (without downloading)
        if check_log_city_from_s3(suffix, split, target_city, lister, logger):
            found.append(suffix)
            logger.info(f"  [{split}] Found {len(found)}/{limit if limit > 0 else '?'} {target_city} logs (checked {checked_count} in this split)")

            if limit > 0 and len(found) >= limit:
                logger.info(f"  [{split}] ✓ Target count reached!")
                break

    logger.info(f"")
    logger.info(f"  [{split}] Summary: Collected {len(found)} {target_city} logs (checked {checked_count} logs in '{split}' split)")
    logger.info(f"")

    return found


def build_s5cmd_command(s5cmd: Path, source: str, dest: Path, concurrency: int,
                       part_size: int) -> List[str]:
    """Build s5cmd command for syncing data."""
    command = [str(s5cmd), "--no-sign-request", "sync"]
    if concurrency > 0:
        command += ["--concurrency", str(concurrency)]
    if part_size > 0:
        command += ["--part-size", str(part_size)]
    command += [source, str(dest)]
    return command


def run_command(command: List[str], logger: logging.Logger) -> int:
    """Run a command and stream output to logger."""
    logger.info(f"  Executing: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    assert process.stdout is not None

    try:
        for line in process.stdout:
            logger.debug("    " + line.rstrip())
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected; terminating command...")
        process.terminate()
        raise

    return process.wait()


def download_log(log_id: str, split: str, config: PipelineConfig,
                logger: logging.Logger) -> Path:
    """Download a single log from S3."""
    split_dest = config.temp_download_dir / split
    split_dest.mkdir(parents=True, exist_ok=True)

    log_dest = split_dest / log_id

    if config.skip_existing_downloads and log_dest.exists():
        logger.info(f"  Skipping existing log {split}/{log_id}")
        return log_dest

    logger.info(f"  Downloading {split}/{log_id}...")
    source = f"{S3_URI_ROOT}/{LIDAR_PREFIX}{split}/{log_id}/*"

    command = build_s5cmd_command(
        s5cmd=config.s5cmd_path,
        source=source,
        dest=log_dest,
        concurrency=config.s5cmd_concurrency,
        part_size=config.s5cmd_part_size_mb,
    )

    log_dest.mkdir(parents=True, exist_ok=True)
    result = run_command(command, logger)

    if result != 0:
        raise RuntimeError(f"s5cmd exited with code {result} while downloading {split}/{log_id}")

    return log_dest


def check_log_city(log_dir: Path, target_city: str, logger: logging.Logger) -> bool:
    """Check if a log belongs to the target city."""
    try:
        from lib import load_city_annotation, parse_city_enum

        city_tag = load_city_annotation(log_dir)
        city_enum = parse_city_enum(city_tag)

        if city_enum.value != target_city:
            logger.info(f"  Skipping log from {city_enum.value} (target: {target_city})")
            return False

        return True
    except Exception as e:
        logger.warning(f"  Failed to check city for {log_dir.name}: {e}")
        return False


def get_log_sweep_count(log_dir: Path) -> int:
    """Get the number of LiDAR sweeps in a log."""
    lidar_dir = log_dir / "sensors" / "lidar"
    if not lidar_dir.exists():
        return 0
    feather_files = list(lidar_dir.glob("*.feather"))
    return len(feather_files)


def write_segment_summary(summary_file: Path, record: Dict, write_header: bool = False) -> None:
    """Append a single segment record to the summary CSV.

    Args:
        summary_file: Path to CSV file
        record: Dictionary with segment data
        write_header: If True, write CSV header first
    """
    mode = 'w' if write_header else 'a'

    with open(summary_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'segment_name',
            'output_path',
            'point_count',
            'duration_s',
            'sensor_motion_length_m',
            'sensor_displacement_m',
            'z_offset_m',
        ])

        if write_header:
            writer.writeheader()

        writer.writerow({
            'segment_name': record['segment_name'],
            'output_path': record['output_path'],
            'point_count': record['point_count'] if record['point_count'] is not None else 'N/A',
            'duration_s': f"{record['duration_s']:.4f}" if record['duration_s'] is not None else 'N/A',
            'sensor_motion_length_m': f"{record['sensor_motion_length_m']:.4f}" if record['sensor_motion_length_m'] is not None else 'N/A',
            'sensor_displacement_m': f"{record['sensor_displacement_m']:.4f}" if record['sensor_displacement_m'] is not None else 'N/A',
            'z_offset_m': f"{record['z_offset_m']:.4f}" if record['z_offset_m'] is not None else 'N/A',
        })


def process_log_segments(log_dir: Path, config: PipelineConfig,
                        logger: logging.Logger, summary_file: Optional[Path] = None) -> Tuple[List[Path], List[Dict]]:
    """Process a log into multiple non-overlapping 5-second segments.

    Returns:
        Tuple: (list of output directories, list of timing dicts)
    """
    log_id = log_dir.name
    sweep_count = get_log_sweep_count(log_dir)

    if sweep_count == 0:
        logger.warning(f"  No LiDAR sweeps found in {log_id}")
        return [], []

    logger.info(f"  Log {log_id} has {sweep_count} sweeps")

    # Calculate number of segments
    sweeps_per_segment = config.sweeps_per_segment
    overlap = config.segment_overlap_sweeps
    stride = sweeps_per_segment - overlap

    if stride <= 0:
        raise ValueError("Segment overlap must be less than sweeps per segment")

    # Generate segment start indices
    segment_starts = []
    idx = 0
    while idx + sweeps_per_segment <= sweep_count:
        segment_starts.append(idx)
        idx += stride

    num_segments = len(segment_starts)
    logger.info(f"  Creating {num_segments} segments (each {sweeps_per_segment} sweeps)")

    output_dirs = []
    timing_records = []

    for seg_idx, start_idx in enumerate(segment_starts):
        segment_name = f"{log_id}_{seg_idx:03d}"
        segment_output_dir = config.output_dir / segment_name
        segment_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"")
        logger.info(f"  {'='*70}")
        logger.info(f"  Processing segment {seg_idx + 1}/{num_segments}: {segment_name}")
        logger.info(f"  {'='*70}")
        segment_start_time = time.time()

        try:
            # Generate sweep indices for this segment
            sweep_indices = list(range(start_idx, start_idx + sweeps_per_segment))
            center_index = len(sweep_indices) // 2

            # Stage 1: Create macro sweep
            logger.info(f"    [Stage 1/4] LiDAR Processing...")
            stage1_start = time.time()
            output_prefix = segment_output_dir / config.output_file_prefix

            macro = create_macro_sweep(
                log_dir=log_dir,
                sweep_indices=sweep_indices,
                center_index=center_index,
                output_prefix=output_prefix,
                compression=config.compression,
                crop_square_m=config.crop_size_meters,
            )

            stage1_elapsed = time.time() - stage1_start
            logger.info(f"    ✓ Created point clouds: {macro.point_path.name}")
            logger.info(f"    ✓ Sensor extent: {macro.sensor_extent_xy[0]:.2f} x {macro.sensor_extent_xy[1]:.2f} m")
            logger.info(f"    ✓ UTM extent: {macro.utm_extent_xy[0]:.2f} x {macro.utm_extent_xy[1]:.2f} m")
            logger.info(f"    ✓ Stage 1 completed in {stage1_elapsed:.2f}s")

            # Stage 2: Generate imagery and DSM
            stage2_elapsed = 0.0
            try:
                logger.info(f"    [Stage 2/4] Imagery & DSM Processing...")
                stage2_start = time.time()

                stage_two = run_stage_two(
                    points_path=macro.point_path,
                    meta_path=macro.meta_path,
                    city_root=config.city_geoalign_root,
                    output_dir=segment_output_dir,
                    buffer_m=config.buffer_meters,
                    target_res=None,  # Auto-detect from source imagery
                    base_name=config.output_file_prefix,
                    crop_square_m=macro.applied_crop_sensor,
                )

                stage2_elapsed = time.time() - stage2_start
                logger.info(f"    ✓ Created imagery: {stage_two.tif_path.name}")
                logger.info(f"    ✓ Imagery resolution: {stage_two.imagery_resolution_m:.4f} m/pixel (source)")
                logger.info(f"    ✓ Created DSM: {stage_two.laz_path.name}")
                logger.info(f"    ✓ Vertical alignment: z_offset = {stage_two.z_offset_m:.4f} m")
                logger.info(f"    ✓ Stage 2 completed in {stage2_elapsed:.2f}s")

            except Exception as e:
                logger.warning(f"    ✗ Stage 2 (imagery/DSM) failed: {e}")
                logger.warning(f"    ⚠ Continuing with LiDAR data only")
                stage_two = None

            # Stage 3: Extract DSM points near LiDAR
            stage3_elapsed = 0.0
            dsm_extraction = None
            if stage_two is not None:
                try:
                    logger.info(f"    [Stage 3/4] DSM Extraction...")
                    stage3_start = time.time()

                    extracted_dsm_path = segment_output_dir / (config.output_file_prefix + "_extract_dsm_utm.parquet")
                    dsm_extraction = extract_dsm_near_lidar(
                        lidar_parquet_path=macro.utm_point_path,
                        dsm_laz_path=stage_two.laz_path,
                        output_path=extracted_dsm_path,
                        max_distance=0.5,
                        compression=config.compression,
                    )

                    stage3_elapsed = time.time() - stage3_start
                    logger.info(f"    ✓ Extracted DSM: {dsm_extraction.extracted_dsm_path.name}")
                    logger.info(f"    ✓ DSM points: {dsm_extraction.original_dsm_count:,} → {dsm_extraction.extracted_dsm_count:,} ({dsm_extraction.reduction_ratio*100:.1f}% reduction)")
                    logger.info(f"    ✓ Distance stats: min={dsm_extraction.distance_stats['min']:.3f}m, max={dsm_extraction.distance_stats['max']:.3f}m, mean={dsm_extraction.distance_stats['mean']:.3f}m")
                    logger.info(f"    ✓ Stage 3 completed in {stage3_elapsed:.2f}s")

                except Exception as e:
                    logger.warning(f"    ✗ Stage 3 (DSM extraction) failed: {e}")
                    logger.warning(f"    ⚠ Skipping GICP alignment")
                    dsm_extraction = None
            else:
                logger.info(f"    ⏭ Skipping Stage 3 (DSM extraction) - Stage 2 failed")

            # Stage 4: GICP Alignment
            stage4_elapsed = 0.0
            gicp_result = None
            if dsm_extraction is not None:
                try:
                    logger.info(f"    [Stage 4/4] GICP Alignment...")
                    stage4_start = time.time()

                    aligned_lidar_path = segment_output_dir / (config.output_file_prefix + "_gicp_utm.parquet")
                    metrics_json_path = segment_output_dir / (config.output_file_prefix + "_gicp_metrics.json")

                    gicp_params = GICPParams(
                        voxel_size=0.3,
                        normal_k=20,
                        max_corr_dist=0.8,
                        max_iter=60,
                        enforce_z_up=True,
                    )

                    gicp_result = align_lidar_to_dsm(
                        lidar_parquet_path=macro.utm_point_path,
                        dsm_parquet_path=dsm_extraction.extracted_dsm_path,
                        meta_path=macro.meta_path,
                        output_lidar_path=aligned_lidar_path,
                        output_metrics_path=metrics_json_path,
                        params=gicp_params,
                        compression=config.compression,
                    )

                    stage4_elapsed = time.time() - stage4_start
                    logger.info(f"    ✓ Aligned LiDAR: {gicp_result.aligned_lidar_path.name}")
                    logger.info(f"    ✓ GICP fitness: {gicp_result.fitness:.6f}, RMSE: {gicp_result.inlier_rmse:.6f}m")
                    logger.info(f"    ✓ Transform: translation=[{gicp_result.translation_m[0]:.3f}, {gicp_result.translation_m[1]:.3f}, {gicp_result.translation_m[2]:.3f}]m, yaw={gicp_result.yaw_deg:.3f}°")
                    logger.info(f"    ✓ Alignment quality: NN RMSE={gicp_result.nn_rmse:.6f}m, mean abs dist={gicp_result.nn_mean_abs_distance:.6f}m")
                    logger.info(f"    ✓ Metrics saved: {gicp_result.metrics_json_path.name}")
                    logger.info(f"    ✓ Stage 4 completed in {stage4_elapsed:.2f}s")

                except Exception as e:
                    logger.warning(f"    ✗ Stage 4 (GICP alignment) failed: {e}")
                    logger.warning(f"    ⚠ Continuing without GICP alignment")
                    gicp_result = None
            else:
                logger.info(f"    ⏭ Skipping Stage 4 (GICP alignment) - Stage 3 failed")

            # Read metadata for logging
            meta_duration_s = None
            meta_motion_length_m = None
            meta_displacement_m = None
            meta_point_count = None
            meta_z_offset_m = None
            try:
                meta_df = pd.read_parquet(macro.meta_path)
                if len(meta_df) > 0:
                    meta_duration_s = float(meta_df['duration_s'].iloc[0])
                    meta_motion_length_m = float(meta_df['sensor_motion_length_m'].iloc[0])
                    meta_displacement_m = float(meta_df['sensor_displacement_m'].iloc[0])
                    meta_point_count = int(meta_df['point_count'].iloc[0])
                    # z_offset_m might not exist if stage 2 failed
                    if 'z_offset_m' in meta_df.columns:
                        meta_z_offset_m = float(meta_df['z_offset_m'].iloc[0])
            except Exception as e:
                logger.warning(f"    Could not read metadata for summary: {e}")

            # Record timing
            segment_elapsed = time.time() - segment_start_time
            logger.info(f"")
            logger.info(f"    {'─'*66}")
            logger.info(f"    ✓ Segment {segment_name} COMPLETED")
            logger.info(f"    Total time: {segment_elapsed:.2f}s (Stage1: {stage1_elapsed:.2f}s, Stage2: {stage2_elapsed:.2f}s, Stage3: {stage3_elapsed:.2f}s, Stage4: {stage4_elapsed:.2f}s)")
            logger.info(f"    Output: {segment_output_dir}")
            logger.info(f"    {'─'*66}")

            output_dirs.append(segment_output_dir)

            record = {
                'segment_name': segment_name,
                'output_path': str(segment_output_dir),
                'point_count': meta_point_count,
                'duration_s': meta_duration_s,
                'sensor_motion_length_m': meta_motion_length_m,
                'sensor_displacement_m': meta_displacement_m,
                'z_offset_m': meta_z_offset_m,
                # Keep timing for console display
                'total_time': segment_elapsed,
                'stage1_time': stage1_elapsed,
                'stage2_time': stage2_elapsed,
                'stage3_time': stage3_elapsed,
                'stage4_time': stage4_elapsed,
            }
            timing_records.append(record)

            # Incrementally save to CSV after each segment
            if summary_file is not None:
                try:
                    # Never write header (already written at start)
                    write_segment_summary(summary_file, record, write_header=False)
                except Exception as e:
                    logger.warning(f"    Failed to write to summary CSV: {e}")

        except Exception as e:
            logger.error(f"    Failed to process segment {segment_name}: {e}")
            logger.exception(e)
            continue

    return output_dirs, timing_records


def check_log_already_processed(log_id: str, output_dir: Path, logger: logging.Logger) -> bool:
    """Check if a log has already been processed.

    Looks for any folders in output_dir that start with {log_id}_

    Args:
        log_id: The log ID to check
        output_dir: The output directory to search
        logger: Logger instance

    Returns:
        True if log has already been processed, False otherwise
    """
    if not output_dir.exists():
        return False

    # Look for any folders starting with {log_id}_
    pattern = f"{log_id}_*"
    existing_segments = list(output_dir.glob(pattern))

    if existing_segments:
        logger.info(f"  ✓ Log already processed: found {len(existing_segments)} existing segments")
        logger.info(f"    First segment: {existing_segments[0].name}")
        if len(existing_segments) > 1:
            logger.info(f"    Last segment:  {existing_segments[-1].name}")
        return True

    return False


def process_single_log(log_id: str, split: str, config: PipelineConfig,
                      logger: logging.Logger, summary_file: Optional[Path] = None) -> Tuple[int, List[Dict]]:
    """Download and process a single log.

    Returns:
        Tuple: (number of segments created, list of timing records)
    """
    log_start_time = time.time()
    download_time = 0.0

    logger.info(f"Processing log: {split}/{log_id}")

    # Check if log has already been processed (if enabled)
    if config.skip_processed_logs:
        if check_log_already_processed(log_id, config.output_dir, logger):
            logger.info(f"  ⏭ Skipping log {log_id} (already processed)")
            logger.info("")
            return 0, []

    try:
        # Download log
        download_start = time.time()
        log_dir = download_log(log_id, split, config, logger)
        download_time = time.time() - download_start

        # City is already verified during collection, but double-check
        if not check_log_city(log_dir, config.target_city, logger):
            logger.warning(f"  Log {log_id} city mismatch (should have been filtered earlier)")
            if config.cleanup_after_processing:
                shutil.rmtree(log_dir)
            return 0, []

        # Process log into segments
        output_dirs, timing_records = process_log_segments(log_dir, config, logger, summary_file)

        # Cleanup temp data
        if config.cleanup_after_processing:
            logger.info(f"  Cleaning up temporary download: {log_dir}")
            shutil.rmtree(log_dir)

        log_elapsed = time.time() - log_start_time

        # Print summary for this log
        logger.info(f"")
        logger.info(f"  {'='*70}")
        logger.info(f"  LOG SUMMARY: {log_id}")
        logger.info(f"  {'='*70}")
        logger.info(f"  Segments created: {len(output_dirs)}")
        logger.info(f"  Download time: {download_time:.2f}s")
        logger.info(f"  Processing time: {log_elapsed - download_time:.2f}s")
        logger.info(f"  Total time: {log_elapsed:.2f}s")
        logger.info(f"  {'='*70}")
        logger.info(f"")

        return len(output_dirs), timing_records

    except Exception as e:
        logger.error(f"Failed to process log {split}/{log_id}: {e}")
        logger.exception(e)
        return 0, []


def main(config_path: Optional[Path] = None) -> int:
    """Main pipeline execution."""

    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    config = PipelineConfig.from_yaml(config_path)

    # Setup logging - only console output, summary saved separately
    log_level = logging.DEBUG if config.enable_detailed_logging else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Only console handler for verbose logging
    handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("Starting Argoverse2 Fetch-and-Process Pipeline")
    logger.info("="*80)
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Target city: {config.target_city}")
    logger.info(f"Sample count: {config.sample_count if config.sample_count > 0 else 'all'}")
    logger.info(f"Fetch splits: {', '.join(config.fetch_splits)}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("="*80 + "\n")

    # Validate paths
    if not config.s5cmd_path.exists():
        logger.error(f"s5cmd executable not found: {config.s5cmd_path}")
        return 1

    if not config.city_geoalign_root.exists():
        logger.warning(f"City geoalign root not found: {config.city_geoalign_root}")
        logger.warning("Stage 2 (imagery/DSM) will be skipped")

    # Create directories
    config.temp_download_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup S3 lister
    lister = S3PrefixLister(max_keys=1000)

    # Collect logs from all requested splits
    all_logs: List[tuple[str, str]] = []  # (log_id, split)

    logger.info("")
    logger.info("="*80)
    logger.info(f"COLLECTING LOGS FROM REQUESTED SPLITS: {config.fetch_splits}")
    logger.info("="*80)

    for split in config.fetch_splits:
        logger.info(f"\nProcessing split: '{split}'...")
        logger.info(f"Target: {config.sample_count} {config.target_city} logs from '{split}' split")

        logs = collect_logs(
            split=split,
            target_city=config.target_city,
            limit=config.sample_count,
            lister=lister,
            logger=logger,
        )

        for log_id in logs:
            all_logs.append((log_id, split))

        logger.info(f"✓ Added {len(logs)} logs from '{split}' split to processing queue")

        if config.sample_count > 0 and len(all_logs) >= config.sample_count:
            all_logs = all_logs[:config.sample_count]
            logger.info(f"\n✓ Reached target count ({config.sample_count} logs). Stopping collection.")
            break

    total_logs = len(all_logs)

    if total_logs == 0:
        logger.warning("No logs to process. Exiting.")
        return 0

    logger.info("")
    logger.info("="*80)
    logger.info(f"COLLECTION COMPLETE: {total_logs} logs ready to process")
    logger.info("="*80)
    logger.info("")

    # Setup summary file for incremental writing
    summary_file = None
    csv_header_written = False  # Track if header has been written globally

    if config.log_file:
        log_base = config.log_file.replace('.log', '')
        summary_file = config.output_dir / f"{log_base}_summary.csv"
        logger.info(f"Summary will be saved to: {summary_file}")

        # Create file with header only if it doesn't exist
        try:
            if not summary_file.exists():
                with open(summary_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'segment_name',
                        'output_path',
                        'point_count',
                        'duration_s',
                        'sensor_motion_length_m',
                        'sensor_displacement_m',
                        'z_offset_m',
                    ])
                    writer.writeheader()
                csv_header_written = True
                logger.info(f"✓ Created new CSV file with header")
            else:
                csv_header_written = True
                logger.info(f"✓ Using existing CSV file (will append records)")
        except Exception as e:
            logger.error(f"Failed to setup CSV file: {e}")
            summary_file = None
        logger.info("")

    # Process each log
    pipeline_start_time = time.time()
    total_segments = 0
    all_timing_records = []

    for idx, (log_id, split) in enumerate(all_logs, 1):
        logger.info(f"[{idx}/{total_logs}] " + "="*70)
        segments_created, timing_records = process_single_log(log_id, split, config, logger, summary_file)
        total_segments += segments_created
        all_timing_records.extend(timing_records)

    # Cleanup temp directory
    if config.cleanup_temp_dir_on_completion:
        logger.info(f"Cleaning up temporary directory: {config.temp_download_dir}")
        shutil.rmtree(config.temp_download_dir, ignore_errors=True)

    # Final Summary
    pipeline_elapsed = time.time() - pipeline_start_time
    logger.info("")
    logger.info("="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)
    logger.info(f"Total logs processed: {total_logs}")
    logger.info(f"Total segments created: {total_segments}")
    logger.info(f"Total time: {pipeline_elapsed/60:.2f} minutes ({pipeline_elapsed:.1f}s)")
    logger.info(f"Average time per segment: {pipeline_elapsed/total_segments:.2f}s" if total_segments > 0 else "N/A")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("="*80)

    # Timing breakdown table
    if all_timing_records:
        logger.info("")
        logger.info("DETAILED TIMING BREAKDOWN")
        logger.info("="*110)
        logger.info(f"{'Segment Name':<50} {'Total':>10} {'Stage1':>10} {'Stage2':>10} {'Stage3':>10} {'Stage4':>10}")
        logger.info("-"*110)

        total_time_sum = 0.0
        stage1_time_sum = 0.0
        stage2_time_sum = 0.0
        stage3_time_sum = 0.0
        stage4_time_sum = 0.0

        for record in all_timing_records:
            logger.info(
                f"{record['segment_name']:<50} "
                f"{record['total_time']:>9.2f}s "
                f"{record['stage1_time']:>9.2f}s "
                f"{record['stage2_time']:>9.2f}s "
                f"{record.get('stage3_time', 0.0):>9.2f}s "
                f"{record.get('stage4_time', 0.0):>9.2f}s"
            )
            total_time_sum += record['total_time']
            stage1_time_sum += record['stage1_time']
            stage2_time_sum += record['stage2_time']
            stage3_time_sum += record.get('stage3_time', 0.0)
            stage4_time_sum += record.get('stage4_time', 0.0)

        logger.info("-"*110)
        logger.info(
            f"{'TOTAL':<50} "
            f"{total_time_sum:>9.2f}s "
            f"{stage1_time_sum:>9.2f}s "
            f"{stage2_time_sum:>9.2f}s "
            f"{stage3_time_sum:>9.2f}s "
            f"{stage4_time_sum:>9.2f}s"
        )
        logger.info(
            f"{'AVERAGE':<50} "
            f"{total_time_sum/len(all_timing_records):>9.2f}s "
            f"{stage1_time_sum/len(all_timing_records):>9.2f}s "
            f"{stage2_time_sum/len(all_timing_records):>9.2f}s "
            f"{stage3_time_sum/len(all_timing_records):>9.2f}s "
            f"{stage4_time_sum/len(all_timing_records):>9.2f}s"
        )
        logger.info("="*110)

    logger.info("")
    logger.info("✓ Pipeline completed successfully!")
    logger.info("")

    # Summary file was already saved incrementally
    if summary_file and all_timing_records:
        logger.info(f"✓ Summary CSV updated successfully!")
        logger.info(f"  Total segments: {len(all_timing_records)}")
        logger.info(f"  File: {summary_file}")
        logger.info("")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Argoverse2 data and process into aligned segments"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration YAML file (default: ./config.yaml)"
    )
    args = parser.parse_args()

    sys.exit(main(args.config))
