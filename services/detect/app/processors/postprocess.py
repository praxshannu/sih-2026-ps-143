"""Post-processing pipeline: morphological ops, contour extraction, GeoJSON vectorization.

Takes the raw binary mask from the UNet++ model and converts it into
cleaned vector polygons suitable for geospatial analysis.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from loguru import logger
from rasterio.transform import Affine
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid


class PostProcessor:
    """Binary mask post-processing and vectorisation.

    Args:
        confidence_threshold: minimum probability to consider a pixel as spill.
        min_area_m2: minimum spill area in square metres to keep.
        morph_kernel_size: kernel size for morphological operations.
        morph_iterations: number of opening/closing iterations.
        simplify_tolerance: shapely polygon simplification tolerance.
        crs_epsg: EPSG code for coordinate reference system transform.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_area_m2: float = 1000.0,
        morph_kernel_size: int = 5,
        morph_iterations: int = 2,
        simplify_tolerance: float = 1.0,
        crs_epsg: int = 4326,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_area_m2 = min_area_m2
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.simplify_tolerance = simplify_tolerance
        self.crs_epsg = crs_epsg

    def threshold_mask(self, logits: np.ndarray) -> np.ndarray:
        """Apply sigmoid + threshold to produce binary mask."""
        probs = 1.0 / (1.0 + np.exp(-logits))
        mask = (probs >= self.confidence_threshold).astype(np.uint8)
        return mask

    def morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening then closing to clean noise."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        # Opening: remove small bright spots (noise)
        cleaned = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations
        )
        # Closing: fill small holes
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations
        )
        return cleaned

    def extract_contours(
        self, mask: np.ndarray, min_area_px: int = 100
    ) -> list[np.ndarray]:
        """Extract and filter contours from binary mask.

        Returns:
            List of (N, 2) contour arrays, filtered by minimum area.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered = [
            c.squeeze() for c in contours
            if cv2.contourArea(c) >= min_area_px
        ]
        logger.debug(
            f"Contour extraction: {len(contours)} found, "
            f"{len(filtered)} above min_area_px={min_area_px}"
        )
        return filtered

    def contours_to_polygons(
        self,
        contours: list[np.ndarray],
        transform: Optional[Affine] = None,
        pixel_size: float = 1.0,
    ) -> list[dict]:
        """Convert OpenCV contours to GeoJSON-compatible polygons.

        Args:
            contours: list of (N, 2) contour arrays.
            transform: rasterio Affine transform (optional).
            pixel_size: ground sampling distance in metres.

        Returns:
            List of dicts with 'geometry' (Shapely polygon) and 'area_m2'.
        """
        polygons = []
        for contour in contours:
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue

            # Scale pixels to metres if no geo transform
            if transform is not None:
                # Use affine transform to convert pixel to geographic coords
                coords = []
                for pt in contour:
                    x, y = pt[0], pt[1]
                    # rasterio Affine: col, row -> x, y
                    geo_x = transform.c + x * transform.a + y * transform.b
                    geo_y = transform.f + x * transform.d + y * transform.e
                    coords.append((geo_x, geo_y))
            else:
                coords = [(float(pt[0]) * pixel_size, float(pt[1]) * pixel_size)
                          for pt in contour]

            try:
                from shapely.geometry import Polygon
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if poly.is_empty:
                    continue

                area_m2 = poly.area
                if transform is not None:
                    # Area in CRS units (degrees or metres)
                    # For EPSG:4326 this is approximate
                    pass
                else:
                    area_m2 = poly.area  # in pixel² or m² depending on pixel_size

                polygons.append({
                    "geometry": poly,
                    "area_m2": area_m2,
                    "centroid": (poly.centroid.y, poly.centroid.x),
                })
            except Exception as e:
                logger.warning(f"Failed to create polygon from contour: {e}")

        return polygons

    def filter_by_area(
        self, polygons: list[dict], min_area: float
    ) -> list[dict]:
        """Remove polygons below minimum area threshold."""
        filtered = [p for p in polygons if p["area_m2"] >= min_area]
        logger.debug(
            f"Area filter: {len(polygons)} -> {len(filtered)} "
            f"(min_area={min_area:.0f} m²)"
        )
        return filtered

    def simplify_polygons(
        self, polygons: list[dict]
    ) -> list[dict]:
        """Simplify polygon geometries to reduce vertex count."""
        for p in polygons:
            try:
                p["geometry"] = p["geometry"].simplify(
                    self.simplify_tolerance, preserve_topology=True
                )
            except Exception as e:
                logger.warning(f"Simplification failed: {e}")
        return polygons

    def polygons_to_geojson(
        self,
        polygons: list[dict],
        confidence_scores: Optional[list[float]] = None,
        crs_epsg: Optional[int] = None,
    ) -> dict:
        """Convert Shapely polygons to GeoJSON FeatureCollection.

        Args:
            polygons: list of dicts with 'geometry' key.
            confidence_scores: optional per-polygon confidence values.
            crs_epsg: EPSG code for CRS metadata.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        epsg = crs_epsg or self.crs_epsg
        features = []
        for i, p in enumerate(polygons):
            geom = p["geometry"]
            conf = confidence_scores[i] if confidence_scores else 0.0
            feature = {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "id": f"spill_{i:04d}",
                    "confidence": float(conf),
                    "area_m2": float(p.get("area_m2", 0.0)),
                    "centroid_lat": float(p.get("centroid", (0, 0))[0]),
                    "centroid_lon": float(p.get("centroid", (0, 0))[1]),
                    "crs": f"EPSG:{epsg}",
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
            "crs": {
                "type": "name",
                "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"},
            },
        }

    def process(
        self,
        logits: np.ndarray,
        transform: Optional[Affine] = None,
        pixel_size: float = 1.0,
    ) -> tuple[list[dict], dict]:
        """Full post-processing pipeline.

        Args:
            logits: raw model output (1, H, W).
            transform: rasterio Affine transform.
            pixel_size: ground sampling distance in metres.

        Returns:
            Tuple of (polygon_dicts, geojson_feature_collection).
        """
        if logits.ndim == 3:
            logits = logits[0]

        mask = self.threshold_mask(logits)
        cleaned = self.morphological_cleanup(mask)

        min_area_px = max(100, int(self.min_area_m2 / (pixel_size ** 2)))
        contours = self.extract_contours(cleaned, min_area_px=min_area_px)

        polygons = self.contours_to_polygons(contours, transform, pixel_size)
        polygons = self.filter_by_area(polygons, self.min_area_m2)
        polygons = self.simplify_polygons(polygons)

        geojson = self.polygons_to_geojson(
            polygons,
            confidence_scores=[1.0] * len(polygons),
            crs_epsg=self.crs_epsg,
        )

        logger.info(f"Post-processing: {len(polygons)} spills detected")
        return polygons, geojson
