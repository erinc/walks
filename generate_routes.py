"""Utility script to convert a directory of GPX files into static map assets.

The script walks through every `.gpx` file inside `workout-routes/`, extracts
track points, simplifies the geometry, and writes:

* `dist/routes.geojson` – GeoJSON FeatureCollection ready for Leaflet
* `dist/routes_manifest.json` – lightweight manifest consumed by index.html

Keep the generator simple and dependency-free so it runs on any Python 3.11+
installation without extra packages.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


@dataclass
class Route:
    """Container for the processed data we care about per GPX file."""

    filename: str
    label: str
    features_coordinates: list[list[list[float]]]
    bounds: list[float]
    point_count: int
    start_time: Optional[str]

    def to_feature(self) -> dict:
        """Serialise the route as a GeoJSON feature."""

        return {
            "type": "Feature",
            "geometry": {
                "type": "MultiLineString",
                "coordinates": self.features_coordinates,
            },
            "properties": {
                "filename": self.filename,
                "label": self.label,
                "point_count": self.point_count,
                "start_time": self.start_time,
            },
        }

    def to_manifest_entry(self) -> dict:
        """Emit the minimal metadata needed by the sidebar."""

        return {
            "filename": self.filename,
            "label": self.label,
            "point_count": self.point_count,
            "bounds": self.bounds,
            "start_time": self.start_time,
        }


def parse_args() -> argparse.Namespace:
    """Configure CLI flags for the generator."""
    parser = argparse.ArgumentParser(
        description="Build a GeoJSON summary of GPX walks for the static viewer."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("workout-routes"),
        help="Directory containing GPX files (default: workout-routes)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist"),
        help="Directory where routes.geojson and routes_manifest.json are written (default: dist)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.00005,
        help="Douglas-Peucker tolerance in degrees (~0.00005 ≈ 5-6m). Set to 0 to disable.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000,
        help="Maximum points per segment after simplification.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for generated JSON (default: 2).",
    )
    return parser.parse_args()


def perpendicular_distance(point: Point, start: Point, end: Point) -> float:
    """Return distance of `point` to the line segment defined by `start` → `end`."""
    if start == end:
        return math.dist(point, start)

    (x, y), (x1, y1), (x2, y2) = point, start, end
    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = math.hypot(x2 - x1, y2 - y1)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def douglas_peucker(points: Sequence[Point], tolerance: float) -> List[Point]:
    """Classic Douglas–Peucker line simplification (recursive)."""
    if len(points) <= 2 or tolerance <= 0:
        return list(points)

    start, end = points[0], points[-1]
    max_distance = -1.0
    index = 0

    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_distance:
            index = i
            max_distance = dist

    if max_distance > tolerance:
        left = douglas_peucker(points[: index + 1], tolerance)
        right = douglas_peucker(points[index:], tolerance)
        return left[:-1] + right
    return [start, end]


def thin_points(points: Sequence[Point], max_points: int) -> List[Point]:
    """Down-sample evenly when Douglas–Peucker still leaves too many vertices."""
    if max_points <= 0 or len(points) <= max_points:
        return list(points)

    step = math.ceil(len(points) / max_points)
    thinned = list(points[::step])
    if thinned[-1] != points[-1]:
        thinned.append(points[-1])
    return thinned


def simplify_points(
    points: Sequence[Point], tolerance: float, max_points: int
) -> List[Point]:
    """Apply Douglas–Peucker first, then thin to the requested maximum."""
    simplified = douglas_peucker(points, tolerance) if tolerance > 0 else list(points)
    simplified = thin_points(simplified, max_points)
    return simplified


def route_label_from_filename(path: Path) -> str:
    """Simple normalisation for filenames exported by Apple Fitness."""
    stem = path.stem
    if stem.startswith("route_"):
        stem = stem[len("route_") :]
    return stem.replace("_", " ")


def iter_track_segments(root: ET.Element) -> Iterator[Iterable[ET.Element]]:
    """Yield each `<trkseg>` block; Apple typically uses a single segment."""
    for trkseg in root.findall(".//{*}trkseg"):
        yield trkseg.findall("{*}trkpt")


def extract_point_from_element(elem: ET.Element) -> Optional[Point]:
    """Parse lat/lon attributes, rejecting invalid values."""
    lat_attr = elem.attrib.get("lat")
    lon_attr = elem.attrib.get("lon")
    try:
        lat = float(lat_attr)  # type: ignore[arg-type]
        lon = float(lon_attr)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return (lat, lon)


def parse_start_time(root: ET.Element) -> Optional[str]:
    """Return ISO start time if the GPX embeds one."""
    time_elem = root.find(".//{*}trkpt/{*}time")
    if time_elem is None or not time_elem.text:
        return None
    try:
        dt = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.isoformat()


def parse_gpx(path: Path, tolerance: float, max_points: int) -> Optional[Route]:
    """Load a single GPX file and convert it into a Route.

    Returns None when the file cannot be parsed or contains no usable points.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    segments: list[list[Point]] = []
    min_lat = float("inf")
    min_lon = float("inf")
    max_lat = float("-inf")
    max_lon = float("-inf")

    def register_point(pt: Point) -> Point:
        nonlocal min_lat, min_lon, max_lat, max_lon
        lat, lon = pt
        min_lat = min(min_lat, lat)
        min_lon = min(min_lon, lon)
        max_lat = max(max_lat, lat)
        max_lon = max(max_lon, lon)
        return (round(lat, 6), round(lon, 6))

    for segment in iter_track_segments(root):
        raw_points: list[Point] = []
        for trkpt in segment:
            point = extract_point_from_element(trkpt)
            if point:
                raw_points.append(point)
        if not raw_points:
            continue
        simplified = simplify_points(raw_points, tolerance, max_points)
        segments.append([register_point(pt) for pt in simplified])

    if not segments:
        fallback_points: list[Point] = []
        for trkpt in root.findall(".//{*}trkpt"):
            point = extract_point_from_element(trkpt)
            if point:
                fallback_points.append(point)
        if fallback_points:
            simplified = simplify_points(fallback_points, tolerance, max_points)
            segments.append([register_point(pt) for pt in simplified])

    if not segments or not math.isfinite(min_lat):
        return None

    geojson_coords = [
        [[lon, lat] for lat, lon in segment] for segment in segments if len(segment) >= 2
    ]
    if not geojson_coords:
        return None

    bounds = [round(min_lat, 6), round(min_lon, 6), round(max_lat, 6), round(max_lon, 6)]
    return Route(
        filename=path.name,
        label=route_label_from_filename(path),
        features_coordinates=geojson_coords,
        bounds=bounds,
        point_count=sum(len(segment) for segment in geojson_coords),
        start_time=parse_start_time(root),
    )


def build_routes(
    source_dir: Path, tolerance: float, max_points: int
) -> tuple[list[Route], list[dict]]:
    """Walk every GPX file in `source_dir`, accumulating processed routes."""
    routes: list[Route] = []
    skipped: list[dict] = []

    for path in sorted(source_dir.glob("*.gpx")):
        route = parse_gpx(path, tolerance=tolerance, max_points=max_points)
        if route is None:
            skipped.append({"filename": path.name, "label": route_label_from_filename(path)})
            continue
        routes.append(route)

    return routes, skipped


def write_json(path: Path, payload: dict | list, indent: int) -> None:
    """Write JSON with UTF-8 encoding and trailing newline for POSIX tooling."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")


def main() -> None:
    """Entry point for the CLI."""
    args = parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source directory not found: {args.source}")

    routes, skipped = build_routes(args.source, args.tolerance, args.max_points)
    if not routes:
        raise SystemExit("No GPX routes were parsed successfully.")

    features = [route.to_feature() for route in routes]
    manifest = [route.to_manifest_entry() for route in routes]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output_dir / "routes.geojson",
        {"type": "FeatureCollection", "features": features},
        indent=args.indent,
    )
    write_json(args.output_dir / "routes_manifest.json", manifest, indent=args.indent)

    summary = {
        "routes": len(routes),
        "skipped": len(skipped),
        "total_points": sum(route.point_count for route in routes),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))

    if skipped:
        print("\nSkipped files:")
        for entry in skipped:
            print(f"  - {entry['filename']}")


if __name__ == "__main__":
    main()
