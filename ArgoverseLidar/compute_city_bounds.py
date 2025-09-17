"""Compute per-city latitude/longitude bounds from av2_coor.feather.

The script scans the exported pose table, aggregates min/max latitude and
longitude per city, and writes a rectangle (four WGS84 corner points) for each
city as a JSON payload.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pyarrow.dataset as ds


@dataclass
class Bounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def update(self, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
        if latitudes.size == 0:
            return
        self.min_lat = min(self.min_lat, float(np.min(latitudes)))
        self.max_lat = max(self.max_lat, float(np.max(latitudes)))
        self.min_lon = min(self.min_lon, float(np.min(longitudes)))
        self.max_lon = max(self.max_lon, float(np.max(longitudes)))

    @classmethod
    def from_arrays(cls, latitudes: np.ndarray, longitudes: np.ndarray) -> "Bounds":
        if latitudes.size == 0:
            raise ValueError("Cannot build bounds from empty arrays")
        return cls(
            min_lat=float(np.min(latitudes)),
            max_lat=float(np.max(latitudes)),
            min_lon=float(np.min(longitudes)),
            max_lon=float(np.max(longitudes)),
        )

    def to_corners(self) -> Dict[str, Dict[str, float]]:
        # Define corners in clockwise order starting at north-west.
        return {
            "nw": {"latitude": self.max_lat, "longitude": self.min_lon},
            "ne": {"latitude": self.max_lat, "longitude": self.max_lon},
            "se": {"latitude": self.min_lat, "longitude": self.max_lon},
            "sw": {"latitude": self.min_lat, "longitude": self.min_lon},
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-city lat/lon bounds from av2_coor.feather")
    parser.add_argument("--input", type=Path, default=Path("av2_coor.feather"), help="Path to the pose feather file.")
    parser.add_argument("--output", type=Path, default=Path("city_bounds.json"), help="Path to write the bounds JSON.")
    parser.add_argument("--progress", type=int, default=25, help="Print a status line every N record batches (default: 25).")
    return parser.parse_args(argv)


def ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input feather file not found: {path}")
    return path


def compute_bounds(path: Path, progress_interval: int) -> Dict[str, Bounds]:
    dataset = ds.dataset(str(path), format="feather")
    scanner = dataset.scanner(columns=["city", "latitude", "longitude"])
    reader = scanner.to_reader()

    bounds: Dict[str, Bounds] = {}

    total_rows = 0
    for batch_idx, batch in enumerate(reader, start=1):
        cities = batch.column(0).to_numpy(zero_copy_only=False)
        latitudes = batch.column(1).to_numpy(zero_copy_only=False)
        longitudes = batch.column(2).to_numpy(zero_copy_only=False)

        unique_cities = np.unique(cities)
        for city in unique_cities:
            city_key = str(city)
            mask = cities == city
            city_lats = latitudes[mask]
            city_lons = longitudes[mask]
            if city_key not in bounds:
                bounds[city_key] = Bounds.from_arrays(city_lats, city_lons)
            else:
                bounds[city_key].update(city_lats, city_lons)

        total_rows += len(cities)
        if progress_interval > 0 and batch_idx % progress_interval == 0:
            print(f"Processed {batch_idx} batches (~{total_rows:,} rows)")

    return bounds


def serialize_bounds(bounds: Dict[str, Bounds]) -> Dict[str, Dict[str, object]]:
    return {
        city: {
            "min_latitude": bd.min_lat,
            "max_latitude": bd.max_lat,
            "min_longitude": bd.min_lon,
            "max_longitude": bd.max_lon,
            "corners": bd.to_corners(),
        }
        for city, bd in bounds.items()
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = ensure_exists(args.input)
    try:
        bounds = compute_bounds(input_path, args.progress)
    except Exception as exc:
        print(f"Failed to compute bounds: {exc}")
        return 1

    output_payload = serialize_bounds(bounds)
    args.output.write_text(json.dumps(output_payload, indent=2))
    print(f"Wrote bounds for {len(output_payload)} cities to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
