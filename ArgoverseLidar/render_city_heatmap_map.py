"""Render aggregated city heatmap cells as semi-transparent rectangles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import folium


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render per-city heatmap cells and bounds onto an OSM map.")
    parser.add_argument("--bounds", type=Path, default=Path("city_bounds.json"), help="Path to city bounds JSON")
    parser.add_argument("--cells", type=Path, default=Path("city_heatmap_cells.json"), help="Path to aggregated cell JSON")
    parser.add_argument("--output", type=Path, default=Path("city_heatmap_map.html"), help="Output HTML file")
    parser.add_argument("--zoom", type=int, default=11, help="Initial zoom level (default: 11)")
    return parser.parse_args(argv)


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return json.loads(path.read_text())


def compute_center(bounds: Dict[str, dict]) -> list[float]:
    lat_sum = 0.0
    lon_sum = 0.0
    count = 0
    for info in bounds.values():
        corners = info.get("corners", {})
        for corner in corners.values():
            lat = corner.get("latitude")
            lon = corner.get("longitude")
            if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
                lat_sum += float(lat)
                lon_sum += float(lon)
                count += 1
    if count == 0:
        return [0.0, 0.0]
    return [lat_sum / count, lon_sum / count]


def color_cycle() -> Iterable[str]:
    palette = [
        "#ff5722",
        "#2196f3",
        "#4caf50",
        "#9c27b0",
        "#009688",
        "#ffc107",
    ]
    while True:
        for color in palette:
            yield color


def add_city_bounds(map_obj: folium.Map, city: str, info: dict, color: str) -> None:
    corners = info.get("corners")
    if not isinstance(corners, dict):
        return
    ordered = ["nw", "ne", "se", "sw", "nw"]
    points = []
    for key in ordered:
        corner = corners.get(key)
        if not isinstance(corner, dict):
            continue
        lat = corner.get("latitude")
        lon = corner.get("longitude")
        if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
            points.append((float(lat), float(lon)))
    if len(points) < 4:
        return
    folium.PolyLine(points, color=color, weight=3, opacity=0.8, tooltip=f"{city} bounds").add_to(map_obj)


def add_city_cells(map_obj: folium.Map, city: str, cells_info: dict, color: str) -> None:
    cells = cells_info.get("cells", [])
    for cell in cells:
        corners = cell.get("corners", [])
        if len(corners) < 4:
            continue
        polygon = [(corner["latitude"], corner["longitude"]) for corner in corners]
        polygon.append(polygon[0])
        opacity = float(cell.get("opacity", 0.1))
        folium.Polygon(
            locations=polygon,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=f"{city}: count {cell.get('count', 0)}",
        ).add_to(map_obj)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    bounds = load_json(args.bounds)
    cells = load_json(args.cells)

    center = compute_center(bounds)
    fmap = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=args.zoom)

    palette = color_cycle()
    for city in sorted(bounds.keys()):
        color = next(palette)
        add_city_bounds(fmap, city, bounds[city], color)
        if city in cells:
            add_city_cells(fmap, city, cells[city], color)

    fmap.save(str(args.output))
    print(f"Saved heatmap to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
