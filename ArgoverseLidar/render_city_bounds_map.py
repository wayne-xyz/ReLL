"""Render per-city bounding rectangles onto an OpenStreetMap HTML page."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import folium


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render city bounds rectangles using folium.")
    parser.add_argument("--bounds", type=Path, default=Path("city_bounds.json"), help="Path to JSON produced by compute_city_bounds.py")
    parser.add_argument("--output", type=Path, default=Path("city_bounds_map.html"), help="Output HTML file path")
    return parser.parse_args(argv)


def load_bounds(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Bounds JSON not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Bounds JSON must contain an object mapping city names to bounds")
    return data


def compute_center(bounds: Dict[str, dict]) -> List[float]:
    lats: List[float] = []
    lons: List[float] = []
    for info in bounds.values():
        corners = info.get("corners", {})
        for corner in corners.values():
            lat = corner.get("latitude")
            lon = corner.get("longitude")
            if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
                lats.append(float(lat))
                lons.append(float(lon))
    if not lats or not lons:
        return [0.0, 0.0]
    return [sum(lats) / len(lats), sum(lons) / len(lons)]


def add_city_overlay(map_obj: folium.Map, city: str, info: dict, color: str) -> None:
    corners = info.get("corners")
    if not isinstance(corners, dict):
        return
    ordered_keys = ["nw", "ne", "se", "sw"]
    coords = []
    for key in ordered_keys:
        corner = corners.get(key)
        if not isinstance(corner, dict):
            continue
        lat = corner.get("latitude")
        lon = corner.get("longitude")
        if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
            coords.append((float(lat), float(lon)))
    if len(coords) < 4:
        return

    coords.append(coords[0])

    folium.Polygon(
        locations=coords,
        color=color,
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.18,
        popup=folium.Popup(f"{city} bounding rectangle"),
    ).add_to(map_obj)

    # Add a marker at the rectangle centroid.
    centroid_lat = sum(lat for lat, _ in coords[:-1]) / (len(coords) - 1)
    centroid_lon = sum(lon for _, lon in coords[:-1]) / (len(coords) - 1)
    tooltip = (
        f"{city}:\n"
        f"lat {info.get('min_latitude'):.6f} → {info.get('max_latitude'):.6f}\n"
        f"lon {info.get('min_longitude'):.6f} → {info.get('max_longitude'):.6f}"
    )
    folium.Marker(location=[centroid_lat, centroid_lon], tooltip=tooltip).add_to(map_obj)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    bounds = load_bounds(args.bounds)

    center = compute_center(bounds)
    fmap = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=11)

    palette = color_cycle()
    for city, info in bounds.items():
        add_city_overlay(fmap, city, info, next(palette))

    fmap.save(str(args.output))
    print(f"Saved map with {len(bounds)} city bounds to {args.output}")
    return 0


def color_cycle() -> Iterable[str]:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    while True:
        for color in colors:
            yield color


if __name__ == "__main__":
    raise SystemExit(main())
