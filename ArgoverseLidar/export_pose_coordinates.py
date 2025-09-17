"""Aggregate Argoverse 2 city pose files into a single feather table.

The script enumerates the public Argoverse S3 bucket, streams each
`city_SE3_egovehicle.feather` directly from cloud storage, and writes the
resulting rows to a consolidated Arrow file on disk. Processing happens log by
log, so no local dataset mirror or large in-memory buffer is required.

Example:

    python export_pose_coordinates.py --output ..\\av2_coor.feather
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from urllib.error import HTTPError
from pyproj import Proj


# Public S3 bucket endpoints.
BASE_URL = "https://s3.amazonaws.com/argoverse"
LIDAR_PREFIX = "datasets/av2/lidar/"
XML_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Mapping of AV2 city code to origin latitude/longitude (degrees).
CITY_ORIGIN_LATLON: Dict[str, tuple[float, float]] = {
    "ATX": (30.27464237939507, -97.7404457407424),
    "DTW": (42.29993066912924, -83.17555750783717),
    "MIA": (25.77452579915163, -80.19656914449405),
    "PAO": (37.416065, -122.13571963362166),
    "PIT": (40.44177902989321, -80.01294377242584),
    "WDC": (38.889377, -77.0355047439081),
}


# Matching UTM zone numbers (all northern hemisphere).
CITY_UTM_ZONE: Dict[str, int] = {
    "ATX": 14,
    "DTW": 17,
    "MIA": 17,
    "PAO": 10,
    "PIT": 17,
    "WDC": 18,
}


MAP_NAME_PATTERN = re.compile(r"__([A-Z]{3})_city_")


@dataclass
class LogDescriptor:
    split: str
    log_id: str
    city: str
    pose_key: str


class S3Client:
    def __init__(self, max_keys: int = 1000) -> None:
        self.max_keys = max_keys

    def _fetch(self, prefix: str, *, delimiter: Optional[str], continuation: Optional[str]) -> ET.Element:
        params = {
            "list-type": "2",
            "prefix": prefix,
            "max-keys": str(self.max_keys),
        }
        if delimiter is not None:
            params["delimiter"] = delimiter
        if continuation:
            params["continuation-token"] = continuation
        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url) as response:  # nosec B310
            data = response.read()
        return ET.fromstring(data)

    def iter_common_prefixes(self, prefix: str) -> Iterator[str]:
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

    def iter_objects(self, prefix: str) -> Iterator[str]:
        continuation: Optional[str] = None
        while True:
            root = self._fetch(prefix, delimiter=None, continuation=continuation)
            for contents in root.findall(f"{XML_NS}Contents"):
                key_elem = contents.find(f"{XML_NS}Key")
                if key_elem is None or key_elem.text is None:
                    continue
                yield key_elem.text
            token_elem = root.find(f"{XML_NS}NextContinuationToken")
            if token_elem is None or not token_elem.text:
                break
            continuation = token_elem.text

    def fetch_bytes(self, key: str) -> bytes:
        url = f"{BASE_URL}/{urllib.parse.quote(key, safe='/')}"
        with urllib.request.urlopen(url) as response:  # nosec B310
            return response.read()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pose transforms across the AV2 LiDAR corpus and export them as a feather table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("..") / "av2_coor.feather",
        help="Path to the output feather file (default: ..\\av2_coor.feather).",
    )
    parser.add_argument(
        "--remote-prefix",
        type=str,
        default=LIDAR_PREFIX,
        help="Prefix inside the Argoverse bucket to scan (default: datasets/av2/lidar/).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=("train", "val", "test"),
        help="Optional subset of splits to process (defaults to all).",
    )
    parser.add_argument(
        "--limit-logs",
        type=int,
        help="Debug option: stop after processing this many logs across all splits.",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=1000,
        help="How many keys to request per S3 listing page (default: 1000).",
    )
    return parser.parse_args(argv)


def discover_logs(client: S3Client, base_prefix: str, splits: Optional[Iterable[str]]) -> Iterator[LogDescriptor]:
    clean_prefix = base_prefix.rstrip("/") + "/"
    target_splits = tuple(splits) if splits else ("train", "val", "test")
    for split in target_splits:
        split_prefix = f"{clean_prefix}{split}/"
        for common in client.iter_common_prefixes(split_prefix):
            if not common.startswith(split_prefix):
                continue
            log_id = common[len(split_prefix):].strip("/")
            if not log_id:
                continue
            city = infer_city_from_map(client, f"{common}map/")
            if city is None:
                print(f"[warn] {split}/{log_id}: unable to infer city, skipping")
                continue
            pose_key = f"{common}city_SE3_egovehicle.feather"
            yield LogDescriptor(split=split, log_id=log_id, city=city, pose_key=pose_key)


def infer_city_from_map(client: S3Client, map_prefix: str) -> Optional[str]:
    for key in client.iter_objects(map_prefix):
        filename = key.split("/")[-1]
        match = MAP_NAME_PATTERN.search(filename)
        if match:
            city = match.group(1)
            if city in CITY_ORIGIN_LATLON:
                return city
    return None


def load_pose_table(client: S3Client, key: str) -> pa.Table:
    try:
        blob = client.fetch_bytes(key)
    except HTTPError as exc:
        raise FileNotFoundError(f"HTTP error {exc.code} for {key}") from exc
    reader = pa.BufferReader(blob)
    table = feather.read_table(reader)
    required = [
        "timestamp_ns",
        "qw",
        "qx",
        "qy",
        "qz",
        "tx_m",
        "ty_m",
        "tz_m",
    ]
    missing = [name for name in required if name not in table.column_names]
    if missing:
        raise ValueError(f"pose file missing columns: {missing}")
    return table.select(required)


def convert_city_coords(city: str, positions: np.ndarray) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if city not in CITY_ORIGIN_LATLON:
        raise ValueError(f"unknown city code: {city}")
    lat0, lon0 = CITY_ORIGIN_LATLON[city]
    zone = CITY_UTM_ZONE[city]
    projector = Proj(proj="utm", zone=zone, ellps="WGS84", datum="WGS84", units="m")
    origin_easting, origin_northing = projector(lon0, lat0)

    easting = origin_easting + positions[:, 0]
    northing = origin_northing + positions[:, 1]
    longitude, latitude = projector(easting, northing, inverse=True)
    zone_label = f"{zone}N"
    return zone_label, easting, northing, latitude, longitude


def build_schema() -> pa.Schema:
    return pa.schema(
        [
            ("city", pa.string()),
            ("split", pa.string()),
            ("log_id", pa.string()),
            ("timestamp_ns", pa.int64()),
            ("qw", pa.float64()),
            ("qx", pa.float64()),
            ("qy", pa.float64()),
            ("qz", pa.float64()),
            ("tx", pa.float64()),
            ("ty", pa.float64()),
            ("tz", pa.float64()),
            ("zone_num_hemi", pa.string()),
            ("easting", pa.float64()),
            ("northing", pa.float64()),
            ("latitude", pa.float64()),
            ("longitude", pa.float64()),
        ]
    )


def record_batch_from_log(desc: LogDescriptor, table: pa.Table, schema: pa.Schema) -> pa.RecordBatch:
    city = desc.city
    split = desc.split
    log_id = desc.log_id

    timestamps = table.column("timestamp_ns").to_numpy(zero_copy_only=False)
    qw = table.column("qw").to_numpy(zero_copy_only=False)
    qx = table.column("qx").to_numpy(zero_copy_only=False)
    qy = table.column("qy").to_numpy(zero_copy_only=False)
    qz = table.column("qz").to_numpy(zero_copy_only=False)
    tx = table.column("tx_m").to_numpy(zero_copy_only=False)
    ty = table.column("ty_m").to_numpy(zero_copy_only=False)
    tz = table.column("tz_m").to_numpy(zero_copy_only=False)

    positions = np.column_stack([tx, ty])
    zone_label, easting, northing, latitude, longitude = convert_city_coords(city, positions)

    n = len(timestamps)
    arrays = [
        pa.array([city] * n, type=pa.string()),
        pa.array([split] * n, type=pa.string()),
        pa.array([log_id] * n, type=pa.string()),
        pa.array(timestamps, type=pa.int64()),
        pa.array(qw, type=pa.float64()),
        pa.array(qx, type=pa.float64()),
        pa.array(qy, type=pa.float64()),
        pa.array(qz, type=pa.float64()),
        pa.array(tx, type=pa.float64()),
        pa.array(ty, type=pa.float64()),
        pa.array(tz, type=pa.float64()),
        pa.array([zone_label] * n, type=pa.string()),
        pa.array(easting, type=pa.float64()),
        pa.array(northing, type=pa.float64()),
        pa.array(latitude, type=pa.float64()),
        pa.array(longitude, type=pa.float64()),
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    client = S3Client(max_keys=args.max_keys)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = build_schema()
    total_logs = 0
    total_rows = 0

    limit = args.limit_logs if args.limit_logs and args.limit_logs > 0 else None

    descriptors = discover_logs(client, args.remote_prefix, args.splits)

    try:
        with pa.OSFile(str(output_path), "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for desc in descriptors:
                    if limit is not None and total_logs >= limit:
                        break
                    try:
                        pose_table = load_pose_table(client, desc.pose_key)
                    except Exception as exc:
                        print(
                            f"[warn] {desc.split}/{desc.log_id}: unable to read pose file {desc.pose_key} ({exc}), skipping",
                        )
                        continue

                    batch = record_batch_from_log(desc, pose_table, schema)
                    writer.write_batch(batch)

                    total_logs += 1
                    total_rows += batch.num_rows
                    print(
                        f"[{total_logs}] {desc.split}/{desc.log_id} ({desc.city}) -> {batch.num_rows} rows | total {total_rows}",
                        flush=True,
                    )
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Wrote {total_rows} rows across {total_logs} logs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
