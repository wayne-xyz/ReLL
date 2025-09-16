"""Plot an Argoverse city pose trajectory onto an interactive OpenStreetMap view."""
from __future__ import annotations

import argparse
import re
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import folium
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pyproj import Proj

CITY_ORIGIN_LATLON = {
    "ATX": (30.27464237939507, -97.7404457407424),
    "DTW": (42.29993066912924, -83.17555750783717),
    "MIA": (25.77452579915163, -80.19656914449405),
    "PAO": (37.416065, -122.13571963362166),
    "PIT": (40.44177902989321, -80.01294377242584),
    "WDC": (38.889377, -77.0355047439081),
}

CITY_UTM_ZONE = {
    "ATX": 14,
    "DTW": 17,
    "MIA": 17,
    "PAO": 10,
    "PIT": 17,
    "WDC": 18,
}

MAP_NAME_PATTERN = re.compile(r"__([A-Z]{3})_city_")


@dataclass
class PoseData:
    city: str
    timestamps_ns: np.ndarray
    positions_city: np.ndarray


class PoseLoader:
    def __init__(self, feather_path: Path, explicit_city: Optional[str]) -> None:
        self.feather_path = feather_path
        self.explicit_city = explicit_city

    def load(self) -> PoseData:
        frame = feather.read_table(self.feather_path).to_pandas()
        missing = {col for col in ("timestamp_ns", "tx_m", "ty_m") if col not in frame}
        if missing:
            raise ValueError(f"Pose file is missing required columns: {sorted(missing)}")

        city = self._resolve_city()
        timestamps = frame["timestamp_ns"].to_numpy(np.int64)
        positions_city = frame[["tx_m", "ty_m"]].to_numpy(np.float64)
        return PoseData(city=city, timestamps_ns=timestamps, positions_city=positions_city)

    def _resolve_city(self) -> str:
        if self.explicit_city:
            city = self.explicit_city.upper()
            if city not in CITY_ORIGIN_LATLON:
                raise ValueError(f"Unknown city code '{city}'. Expected one of {sorted(CITY_ORIGIN_LATLON)}")
            return city

        log_root = self.feather_path.parent
        map_dir = log_root / "map"
        if map_dir.exists():
            for candidate in map_dir.iterdir():
                match = MAP_NAME_PATTERN.search(candidate.name)
                if match:
                    city = match.group(1)
                    if city in CITY_ORIGIN_LATLON:
                        return city
        raise ValueError(
            "Could not infer city code. Pass it explicitly via --city or ensure the map filename contains '__<CITY>_city_'."
        )


def convert_city_to_wgs84(points_city: np.ndarray, city: str) -> np.ndarray:
    lat0, lon0 = CITY_ORIGIN_LATLON[city]
    projector = Proj(proj="utm", zone=CITY_UTM_ZONE[city], ellps="WGS84", datum="WGS84", units="m")
    origin_easting, origin_northing = projector(lon0, lat0)
    easting = origin_easting + points_city[:, 0]
    northing = origin_northing + points_city[:, 1]
    lat_lon: list[tuple[float, float]] = []
    for e, n in zip(easting, northing):
        lon, lat = projector(e, n, inverse=True)
        lat_lon.append((lat, lon))
    return np.array(lat_lon)


def build_map(latlon: np.ndarray, timestamps_ns: np.ndarray, city: str, output: Path) -> Path:
    if latlon.size == 0:
        raise ValueError("No pose samples available to plot.")
    center = latlon.mean(axis=0)
    fmap = folium.Map(location=center.tolist(), tiles="OpenStreetMap", zoom_start=17)

    folium.PolyLine(latlon.tolist(), color="red", weight=3, opacity=0.6).add_to(fmap)

    total = len(latlon)
    for idx, (lat, lon) in enumerate(latlon):
        if idx == 0:
            color = "green"
        elif idx == total - 1:
            color = "red"
        else:
            color = "blue"
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"t = {timestamps_ns[idx] / 1e9:.3f} s",
        ).add_to(fmap)

    fmap.save(str(output))
    return output


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project a city_SE3_egovehicle trajectory onto an OSM basemap.")
    parser.add_argument("--pose", type=Path, required=True, help="Path to city_SE3_egovehicle.feather file.")
    parser.add_argument("--city", type=str, help="Optional city code override (ATX, DTW, MIA, PAO, PIT, WDC).")
    parser.add_argument("--output", type=Path, help="HTML file to write. Defaults to <pose_stem>_map.html")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open the generated map in a browser.")
    return parser.parse_args(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pose_path = args.pose
    if not pose_path.exists():
        print(f"Pose file not found: {pose_path}", file=sys.stderr)
        return 1

    loader = PoseLoader(pose_path, args.city)
    try:
        pose_data = loader.load()
    except Exception as exc:
        print(f"Failed to load pose data: {exc}", file=sys.stderr)
        return 1

    try:
        latlon = convert_city_to_wgs84(pose_data.positions_city, pose_data.city)
    except Exception as exc:
        print(f"Conversion to WGS84 failed: {exc}", file=sys.stderr)
        return 1

    output = args.output or pose_path.with_name(pose_path.stem + "_map.html")
    try:
        build_map(latlon, pose_data.timestamps_ns, pose_data.city, output)
    except Exception as exc:
        print(f"Failed to build map: {exc}", file=sys.stderr)
        return 1

    print(f"Map saved to {output}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
