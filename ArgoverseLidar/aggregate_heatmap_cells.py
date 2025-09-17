"""Aggregate av2_coor.feather into 500 m UTM grid cells per city."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pyarrow.dataset as ds
from pyproj import Proj


CELL_SIZE_METERS = 500.0

CITY_UTM_ZONE = {
    "ATX": 14,
    "DTW": 17,
    "MIA": 17,
    "PAO": 10,
    "PIT": 17,
    "WDC": 18,
}


@dataclass
class CellIndex:
    city: str
    cell_x: int
    cell_y: int


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate av2 poses into 500 m grid cells with counts.")
    parser.add_argument("--input", type=Path, default=Path("av2_coor.feather"), help="Pose feather file path")
    parser.add_argument("--output", type=Path, default=Path("city_heatmap_cells.json"), help="Destination JSON file")
    parser.add_argument("--cell-size", type=float, default=CELL_SIZE_METERS, help="Grid cell size in meters (default: 500)")
    parser.add_argument("--progress", type=int, default=100, help="Print progress every N batches (default: 100)")
    return parser.parse_args(argv)


def ensure_input(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Input feather file not found: {path}")
    return path


def aggregate_counts(path: Path, cell_size: float, progress_interval: int) -> Dict[str, Dict[Tuple[int, int], int]]:
    dataset = ds.dataset(str(path), format="feather")
    scanner = dataset.scanner(columns=["city", "easting", "northing"])
    reader = scanner.to_reader()

    counts: Dict[str, Dict[Tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))

    for batch_idx, batch in enumerate(reader, start=1):
        cities = batch.column(0).to_numpy(zero_copy_only=False)
        easting = batch.column(1).to_numpy(zero_copy_only=False)
        northing = batch.column(2).to_numpy(zero_copy_only=False)

        # Convert to cell indices.
        cell_x = np.floor_divide(easting, cell_size).astype(np.int64)
        cell_y = np.floor_divide(northing, cell_size).astype(np.int64)
        city_codes = cities.astype("<U3")

        structured = np.empty(len(city_codes), dtype=[("city", "U3"), ("cell_x", np.int64), ("cell_y", np.int64)])
        structured["city"] = city_codes
        structured["cell_x"] = cell_x
        structured["cell_y"] = cell_y

        unique, frequency = np.unique(structured, return_counts=True)
        for record, count in zip(unique, frequency):
            counts[record["city"]][(int(record["cell_x"]), int(record["cell_y"]))] += int(count)

        if progress_interval > 0 and batch_idx % progress_interval == 0:
            print(f"Processed {batch_idx} batches")

    return counts


def build_projectors() -> Dict[str, Proj]:
    projectors: Dict[str, Proj] = {}
    for city, zone in CITY_UTM_ZONE.items():
        projectors[city] = Proj(proj="utm", zone=zone, ellps="WGS84", datum="WGS84", units="m")
    return projectors


def convert_to_json(
    counts: Dict[str, Dict[Tuple[int, int], int]],
    cell_size: float,
) -> Dict[str, Dict[str, object]]:
    projectors = build_projectors()
    output: Dict[str, Dict[str, object]] = {}

    for city, cells in counts.items():
        if city not in projectors:
            continue
        proj = projectors[city]
        max_count = max(cells.values()) if cells else 0
        result_cells = []
        for (cell_x, cell_y), count in cells.items():
            e_min = cell_x * cell_size
            e_max = e_min + cell_size
            n_min = cell_y * cell_size
            n_max = n_min + cell_size

            corners_utm = [
                (e_min, n_max),  # NW
                (e_max, n_max),  # NE
                (e_max, n_min),  # SE
                (e_min, n_min),  # SW
            ]
            corners = []
            for easting, northing in corners_utm:
                lon, lat = proj(easting, northing, inverse=True)
                corners.append({"latitude": lat, "longitude": lon})

            center_e = e_min + cell_size / 2.0
            center_n = n_min + cell_size / 2.0
            center_lon, center_lat = proj(center_e, center_n, inverse=True)

            opacity = 0.0
            if max_count > 0:
                opacity = 0.1 + 0.7 * (count / max_count)

            result_cells.append(
                {
                    "count": count,
                    "corners": corners,
                    "center": {"latitude": center_lat, "longitude": center_lon},
                    "opacity": round(min(max(opacity, 0.1), 0.8), 4),
                }
            )

        output[city] = {
            "cell_size_m": cell_size,
            "max_count": max_count,
            "cells": result_cells,
        }

    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_path = ensure_input(args.input)
    counts = aggregate_counts(input_path, args.cell_size, args.progress)
    payload = convert_to_json(counts, args.cell_size)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote heatmap grid for {len(payload)} cities to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
