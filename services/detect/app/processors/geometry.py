"""Georeferencing: measure in a metric CRS, publish in EPSG:4326.

Two different coordinate jobs, kept deliberately separate:

* **Measurement.** Area, perimeter, length, width and orientation must be in
  metres. Computing them on lon/lat degrees is the classic SENTINEL-class bug:
  a degree of longitude is not a degree of latitude, so a "km²" derived from
  square degrees is wrong by ~cos(latitude) (≈6 % at 20° S). Measurement
  therefore happens on the WGS84 ellipsoid via :class:`pyproj.Geod` (geodesic
  area and perimeter — exact, no projection distortion) and, for shape
  descriptors, in an azimuthal-equidistant projection centred on the scene.
* **Publication.** Every polygon handed to the API/UI is reprojected from the
  raster CRS to **EPSG:4326**, which is what GeoJSON consumers expect. The
  raster CRS is never assumed to be 4326 — Sentinel-1 GRD tiles arrive in both
  geographic (EPSG:4326) and projected (UTM) flavours.

If the raster CRS cannot be interpreted, this module does not guess: it raises
:class:`CRSUnavailable`, which the pipeline reports as ``invalid_scene``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyproj import CRS, Geod, Transformer
from rasterio.transform import Affine
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform

WGS84_EPSG = 4326
WGS84 = CRS.from_epsg(WGS84_EPSG)

_GEOD = Geod(ellps="WGS84")


class CRSUnavailable(ValueError):
    """The raster has no usable CRS, so nothing may be placed on the globe."""


def as_crs(value: Any) -> CRS | None:
    """Best-effort conversion of a rasterio/str/EPSG value to :class:`pyproj.CRS`."""
    if value is None:
        return None
    try:
        return CRS.from_user_input(value)
    except Exception:  # noqa: BLE001 - an unusable CRS must not crash the run
        return None


def is_geographic(crs: CRS) -> bool:
    return bool(crs.is_geographic)


def measurement_crs(crs: CRS, lon0: float, lat0: float) -> CRS:
    """A metric CRS for shape measurement.

    A projected raster CRS is already metric, so it is used as-is. A geographic
    one (EPSG:4326 etc.) is swapped for an azimuthal-equidistant projection
    centred on the scene, which preserves distance from that centre — the right
    distortion trade for a 10–100 km SAR tile.
    """
    if not is_geographic(crs):
        return crs
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.6f} +lon_0={lon0:.6f} +datum=WGS84 +units=m +no_defs"
    )


@dataclass(frozen=True)
class Measurement:
    """Metric geometry of one detection polygon."""

    area_m2: float
    perimeter_m: float
    length_m: float
    width_m: float
    orientation_deg: float
    centroid_lon: float
    centroid_lat: float
    elongation_rotated_rect: float
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "area_m2": self.area_m2,
            "area_km2": self.area_m2 / 1.0e6,
            "perimeter_m": self.perimeter_m,
            "perimeter_km": self.perimeter_m / 1000.0,
            "length_m": self.length_m,
            "length_km": self.length_m / 1000.0,
            "width_m": self.width_m,
            "width_km": self.width_m / 1000.0,
            "orientation_deg": self.orientation_deg,
            "centroid": [self.centroid_lon, self.centroid_lat],
            # Public key stays `elongation`; the field name records *how* it
            # was measured (rotated-rect aspect), which matters because the
            # elongation gate is applied on this number and a different
            # definition would move the threshold.
            "elongation": self.elongation_rotated_rect,
            "method": self.method,
        }


class SceneGeometry:
    """CRS-aware helper bound to one raster: reprojection + measurement."""

    def __init__(self, crs: Any, transform: Affine, height: int, width: int) -> None:
        resolved = as_crs(crs)
        if resolved is None:
            raise CRSUnavailable(f"raster CRS is missing or unusable: {crs!r}")
        self.crs: CRS = resolved
        self.transform: Affine = transform
        self.height = int(height)
        self.width = int(width)

        centre_x, centre_y = transform * (width / 2.0, height / 2.0)
        if is_geographic(self.crs):
            self.centre_lon, self.centre_lat = float(centre_x), float(centre_y)
        else:
            lon, lat = self._to_wgs84_xy(centre_x, centre_y)
            self.centre_lon, self.centre_lat = lon, lat

        self.metric_crs = measurement_crs(self.crs, self.centre_lon, self.centre_lat)
        self._to_wgs84 = Transformer.from_crs(self.crs, WGS84, always_xy=True)
        self._to_metric = Transformer.from_crs(self.crs, self.metric_crs, always_xy=True)
        self._metric_to_wgs84 = Transformer.from_crs(self.metric_crs, WGS84, always_xy=True)
        self.pixel_size_m = self._pixel_size_m()

    # ── transforms ─────────────────────────────────────────────────────────

    def _to_wgs84_xy(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self._to_wgs84.transform(x, y)
        return float(lon), float(lat)

    def pixel_to_crs(self, col: float, row: float) -> tuple[float, float]:
        x, y = self.transform * (float(col), float(row))
        return float(x), float(y)

    def pixel_to_wgs84(self, col: float, row: float) -> tuple[float, float]:
        return self._to_wgs84_xy(*self.pixel_to_crs(col, row))

    def xy_to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        """Reproject a point given in raster-CRS coordinates to EPSG:4326."""
        return self._to_wgs84_xy(x, y)

    def crs_to_wgs84_polygon(self, poly: Polygon) -> Polygon:
        """Reproject a polygon from the raster CRS to EPSG:4326."""
        if self.crs == WGS84:
            return poly
        projected = shapely_transform(lambda x, y: self._to_wgs84.transform(x, y), poly)
        return _ensure_polygon(projected)

    def crs_to_metric_polygon(self, poly: Polygon) -> Polygon:
        if self.metric_crs == self.crs:
            return poly
        projected = shapely_transform(lambda x, y: self._to_metric.transform(x, y), poly)
        return _ensure_polygon(projected)

    # ── measurement ────────────────────────────────────────────────────────

    def _pixel_size_m(self) -> tuple[float, float]:
        """Ground size of one pixel in metres (x, y), measured geodesically."""
        if not is_geographic(self.crs):
            return abs(float(self.transform.a)), abs(float(self.transform.e))
        # Geographic CRS: measure one pixel step geodesically at the scene centre.
        x0, y0 = self.transform * (self.width / 2.0, self.height / 2.0)
        x1, y1 = self.transform * (self.width / 2.0 + 1.0, self.height / 2.0)
        x2, y2 = self.transform * (self.width / 2.0, self.height / 2.0 + 1.0)
        _, _, dx = _GEOD.inv(x0, y0, x1, y1)
        _, _, dy = _GEOD.inv(x0, y0, x2, y2)
        return abs(float(dx)), abs(float(dy))

    def pixel_area_m2(self) -> float:
        return float(self.pixel_size_m[0] * self.pixel_size_m[1])

    def measure(self, poly_crs: Polygon) -> Measurement:
        """Measure a polygon given in *raster CRS* coordinates.

        Area and perimeter are geodesic (WGS84 ellipsoid, exact). Length, width
        and orientation come from the minimum-area rotated rectangle in the
        metric CRS, which is what an operator means by "the slick is 4 km long,
        1.2 km wide, oriented NE".
        """
        poly_wgs84 = self.crs_to_wgs84_polygon(poly_crs)
        poly_metric = self.crs_to_metric_polygon(poly_crs)

        area_m2, perimeter_m = self._geodesic_area_perimeter(poly_wgs84)
        length_m, width_m, orientation_deg = _rotated_rect_metrics(poly_metric)
        c_lon, c_lat = self._metric_centroid_to_wgs84(poly_metric)
        elongation = (length_m / width_m) if width_m > 0 else float("inf")

        return Measurement(
            area_m2=abs(float(area_m2)),
            perimeter_m=abs(float(perimeter_m)),
            length_m=length_m,
            width_m=width_m,
            orientation_deg=orientation_deg,
            centroid_lon=c_lon,
            centroid_lat=c_lat,
            elongation_rotated_rect=elongation,
            method="geodesic_area_wgs84+aeqd_shape",
        )

    def _metric_centroid_to_wgs84(self, poly_metric: Polygon) -> tuple[float, float]:
        if poly_metric.is_empty:
            return self.centre_lon, self.centre_lat
        lon, lat = self._metric_to_wgs84.transform(poly_metric.centroid.x, poly_metric.centroid.y)
        return float(lon), float(lat)

    @staticmethod
    def _geodesic_area_perimeter(poly_wgs84: Polygon) -> tuple[float, float]:
        try:
            area, perimeter = _GEOD.geometry_area_perimeter(poly_wgs84)
            return float(area), float(perimeter)
        except Exception:  # noqa: BLE001 - fall back to a projected estimate
            area = float(poly_wgs84.area)
            perimeter = float(poly_wgs84.length)
            return area, perimeter

    def bbox_wsen(self, poly_crs: Polygon) -> list[float]:
        """Bounding box of a raster-CRS polygon, reprojected to EPSG:4326."""
        poly = self.crs_to_wgs84_polygon(poly_crs)
        minx, miny, maxx, maxy = poly.bounds
        return [float(minx), float(miny), float(maxx), float(maxy)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs.to_string(),
            "epsg": self.crs.to_epsg(),
            "is_geographic": is_geographic(self.crs),
            "measurement_crs": self.metric_crs.to_string(),
            "output_crs": "EPSG:4326",
            "pixel_size_m": [float(self.pixel_size_m[0]), float(self.pixel_size_m[1])],
            "pixel_area_m2": self.pixel_area_m2(),
            "centre": [self.centre_lon, self.centre_lat],
            "size": [self.height, self.width],
        }


def _ensure_polygon(geom: Any) -> Polygon:
    """Coerce the result of a reprojection back to a single Polygon."""
    if geom is None or geom.is_empty:
        return Polygon()
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        parts = [g for g in getattr(geom, "geoms", []) if g.area > 0]
        if not parts:
            return Polygon()
        return max(parts, key=lambda g: g.area)  # type: ignore[return-value]
    return Polygon()


def _rotated_rect_metrics(poly_metric: Polygon) -> tuple[float, float, float]:
    """Length, width and orientation from the minimum-area rotated rectangle.

    Orientation is the compass bearing (deg clockwise from north, 0–180) of the
    rectangle's long axis, which is how a slick's heading is reported at sea.
    """
    if poly_metric.is_empty or poly_metric.area <= 0:
        return 0.0, 0.0, 0.0
    rect = poly_metric.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords, dtype=float)
    if coords.shape[0] < 4:
        return 0.0, 0.0, 0.0
    # Four edges of the rotated rectangle (last coord repeats the first).
    edges = []
    for i in range(4):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        edges.append((float(np.hypot(x1 - x0, y1 - y0)), x0, y0, x1, y1))
    # shapely closes the ring, so the last edge is the duplicate closing one.
    edges = edges[:4]
    long_edge = max(edges, key=lambda e: e[0])
    short_edge = min(edges, key=lambda e: e[0])
    length_m = float(long_edge[0])
    width_m = float(short_edge[0])
    dx = long_edge[3] - long_edge[1]
    dy = long_edge[4] - long_edge[2]
    bearing = float(np.degrees(np.arctan2(dx, dy)) % 180.0)
    return length_m, width_m, bearing


def polygon_to_geojson_coords(poly_wgs84: Polygon) -> list[list[float]]:
    """Exterior ring of a WGS84 polygon as GeoJSON coordinates."""
    return [[float(x), float(y)] for x, y in poly_wgs84.exterior.coords]


def polygon_to_geojson_geometry(poly_wgs84: Polygon) -> dict[str, Any]:
    """GeoJSON geometry dict (EPSG:4326) for one polygon."""
    geom = mapping(poly_wgs84)
    if geom["type"] != "Polygon":
        return {"type": "Polygon", "coordinates": [polygon_to_geojson_coords(poly_wgs84)]}
    return geom
