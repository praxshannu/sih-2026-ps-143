"""Deterministic oil-spill detector for SAR GeoTIFFs.

This is the **Tier-A** detector: a defensible, transparent operator that
turns calibrated Sentinel-1 sigma0 into spill polygons. It exists because
the spec calls for a UNet++ that has never been trained on this machine —
shipping that would mean rendering fake probabilities and shipping them as
"real" detections, which violates the no-synthetic-data rule.

Pipeline
--------
1. Lee-sigma speckle filter (7x7) on linear sigma0
2. Convert to dB; mask land using the `dataMask` band (0 = nodata/land)
3. Adaptive local threshold (sliding window mean - k*sigma in dB)
4. Morphological opening + closing with scipy.ndimage
5. Drop blobs below a minimum area
6. Vectorise via `rasterio.features.shapes` -> GeoJSON polygons
7. Per-polygon features: area, perimeter, compactness, elongation, dB contrast
8. Confidence with Wilson 95% CI (Wilson 1927, the correct interval for a
   Bernoulli proportion; never rounded to a meaningless 100%)
9. Fay (1971) spreading-law inverse -> age estimate in hours
10. Wind viability flag: with no wind forcing available (ERA5 licence pending),
    the operator emits detections but marks them LOW_CONFIDENCE rather than
    dropping them. The UI then shows them honestly.

This module has zero heavy dependencies (no torch, no cv2). It runs on the
managed venv with numpy/scipy/rasterio/shapely only.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy import ndimage as ndi
from shapely.geometry import Polygon, mapping, shape
from shapely.validation import make_valid


@dataclass
class DetectionConfig:
    """All knobs in one place so a single change can be tuned per scene."""

    lee_window: int = 7  # 7x7 Lee-sigma speckle filter on linear sigma0
    threshold_k: float = 2.0  # how many local-noise-sigma below local background counts as dark
    local_window: int = 51  # sliding window for the adaptive threshold (pixels)
    big_window: int = 151  # background-window size (pixels)
    baseline_percentile: float = 75  # scene-wide "clean sea" dB percentile
    min_scene_contrast_dB: float = 2.5  # pixel must be ≥ this far below the scene baseline
    open_iter: int = 1  # morphological opening iterations
    close_iter: int = 2  # morphological closing iterations
    min_area_km2: float = 0.02  # drop blobs below this area (~2000 px at this scene scale)
    min_dB_contrast: float = 2.0  # require at least this contrast vs local surroundings
    max_elongation: float = 5.0  # reject very-streak-shaped blobs (likely wind shadows)
    fay_K: float = 3.0e-5  # Fay (1971) spreading coefficient; conservative
    wind_viability: tuple[float, float] = (2.0, 10.0)  # m/s, oil slick detection band
    wind_speed_ms: float | None = None  # set externally once ERA5 licence is accepted


@dataclass
class PolygonFeature:
    """Per-polygon metrics. Serialisable to JSON."""

    area_km2: float
    perimeter_km: float
    compactness: float  # 4πA / P²  -> 1.0 = circle, smaller = more dendritic
    elongation: float  # bbox long side / short side
    mean_dB: float
    contrast_dB: float  # candidate mean dB - local sea mean dB
    scene_contrast_dB: float  # candidate mean dB - scene sea baseline (p75)
    age_hours_fay: float | None
    age_hours_uncertainty_h: float | None
    confidence: float  # 0..1, never 1.0
    confidence_low: float  # Wilson 95% lower
    confidence_high: float  # Wilson 95% upper
    wind_viability: str  # OK | LOW_CONFIDENCE_NO_WIND | OUTSIDE_BAND
    bbox_wsen: list[float] = field(default_factory=list)
    geometry: dict[str, Any] = field(default_factory=dict)


# ── Speckle filtering ─────────────────────────────────────────────────────


def lee_sigma_filter(linear: np.ndarray, win: int = 7, sigma_v: float | None = None) -> np.ndarray:
    """Lee-sigma speckle filter, the standard SAR smoothing operator.

    Reduces multiplicative speckle by replacing each pixel with a weighted
    average between the local mean and the pixel value, with weights set by
    the local coefficient of variation.

    Args:
        linear: linear sigma0.
        win: filter window (odd).
        sigma_v: scene-level speckle coefficient. Left as ``None`` it is
            estimated from the array, which is what the single-pass path does;
            the tiled pipeline passes the *scene* value so every tile is
            filtered with the same coefficient (otherwise a tile's threshold
            would depend on where it was cut).
    """
    pad = win // 2
    padded = np.pad(linear, pad, mode="reflect")
    # Local statistics via box sums.
    s = ndi.uniform_filter(padded, size=win)
    sq = ndi.uniform_filter(padded * padded, size=win)
    # Crop back to the input size *before* mixing with the unpadded input.
    s = s[pad:-pad, pad:-pad]
    sq = sq[pad:-pad, pad:-pad]
    var = np.clip(sq - s * s, 0.0, None)
    if sigma_v is None:
        sigma_v = float(np.nanstd(linear)) or 1e-6  # scene speckle noise coefficient
    cv = np.sqrt(var) / (s + 1e-6)
    k = (cv**2 - sigma_v**2) / (cv**2 * (1 + sigma_v**2) + 1e-6)
    k = np.clip(k, 0.0, 1.0)
    out = s + k * (linear - s)
    return out


# ── Thresholding ──────────────────────────────────────────────────────────


def masked_box_filter(values: np.ndarray, valid: np.ndarray, win: int) -> np.ndarray:
    """Box-filter average of `values` over pixels where `valid` is True, ignoring
    the rest. NaN-aware (does not pull the mean down when land cells are 0)."""
    pad = win // 2
    v = np.where(valid & np.isfinite(values), values, 0.0)
    n = np.where(valid & np.isfinite(values), 1.0, 0.0)
    vp = np.pad(v, pad, mode="reflect")
    np_ = np.pad(n, pad, mode="reflect")
    s = ndi.uniform_filter(vp, size=win)
    cnt = ndi.uniform_filter(np_, size=win) + 1e-6
    out = s / cnt
    return out[pad:-pad, pad:-pad]


def apply_two_gates(
    dB: np.ndarray,
    mask: np.ndarray,
    local_bg: np.ndarray,
    noise_sigma: float,
    sea_baseline: float,
    k: float,
    min_scene_contrast: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Solberg two gates to pixels, given pre-computed statistics.

    Split out of :func:`adaptive_threshold` so the tiled pipeline can compute
    the *global* scene statistics once and then evaluate the same gates
    tile-by-tile without re-deriving them per tile (which would make a tile's
    threshold depend on where the tile was cut).

    Returns:
        dark — pixels passing BOTH gates
        gate_local — pixels passing the local ``k·sigma`` gate
        gate_scene — pixels passing the scene-baseline gate
    """
    gate_local = (dB < (local_bg - k * noise_sigma)) & mask & np.isfinite(dB)
    gate_scene = (dB < (sea_baseline - min_scene_contrast)) & mask & np.isfinite(dB)
    return gate_local & gate_scene, gate_local, gate_scene


def adaptive_threshold(
    dB: np.ndarray,
    mask: np.ndarray,
    win: int,
    k: float,
    big_win: int | None = None,
    baseline_pct: float = 75.0,
    min_scene_contrast: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Two-gate adaptive threshold (after Solberg et al., 1999).

    The classic "local mean − k·σ" gate rejects speckle but lets large,
    slightly-dim features (e.g. wind shadows) through. The scene-wide sea
    baseline gate — the 75th percentile of ocean dB — rejects those, because
    wind shadows are typically <2 dB below clean sea while oil spills are
    typically 3–8 dB below.

    Returns:
        dark   — boolean mask of candidate dark pixels (passes BOTH gates)
        local_bg — per-pixel local-mean background (used downstream for contrast)
        noise_sigma — scalar scene-noise dB standard deviation
    """
    big = big_win or max(win, 151)
    local_bg = masked_box_filter(dB, mask, big)
    residual = np.where(mask & np.isfinite(dB), dB - local_bg, np.nan)
    noise_sigma = float(np.nanstd(residual[mask])) or 1.0
    sea_baseline = float(np.nanpercentile(dB[mask], baseline_pct))

    dark, _, _ = apply_two_gates(
        dB,
        mask,
        local_bg,
        noise_sigma,
        sea_baseline,
        k=k,
        min_scene_contrast=min_scene_contrast,
    )
    return dark, local_bg, noise_sigma


# ── Confidence (Wilson 95% CI) ────────────────────────────────────────────


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the correct binomial CI, not the naive +/-sqrt(pq/n)."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _confidence(
    contrast_dB: float, valid_fraction: float, area_km2: float
) -> tuple[float, float, float, int]:
    """Translate raw evidence into a probability of being a real oil slick.

    Combines contrast strength, scene validity, and a minimum blob size; the
    `n` fed to the Wilson interval is the number of "votes" we have for the
    candidate, taken as area_km2 / 0.01 km² so a 1 km² blob is 100 votes.
    """
    contrast_term = min(1.0, max(0.0, (contrast_dB - 1.0) / 5.0))  # 1..6 dB maps 0..1
    validity_term = max(0.0, min(1.0, valid_fraction))  # already 0..1
    size_term = min(1.0, area_km2 / 5.0)  # 5 km² maps to 1
    p = 0.55 * contrast_term + 0.25 * validity_term + 0.20 * size_term
    p = max(0.05, min(0.95, p))  # never claim 100 %
    n = max(8, int(area_km2 / 0.01))
    lo, hi = _wilson_ci(p, n)
    return p, lo, hi, n


# ── Fay spreading ─────────────────────────────────────────────────────────


def fay_age(area_km2: float, K: float = 3.0e-5) -> tuple[float | None, float | None]:
    """Inverse of A(t) ≈ π K t² -> t = sqrt(A / (π K)).

    Returns (hours, ±rough uncertainty) or (None, None) if area is too small.
    """
    if area_km2 <= 0.01:
        return None, None
    A = area_km2 * 1e6  # m²
    t = math.sqrt(A / (math.pi * K))  # seconds
    hours = t / 3600.0
    # ±50% uncertainty is honest for the inverse-Fay-with-no-wind case.
    return hours, hours * 0.5


# ── Vectorisation ─────────────────────────────────────────────────────────


def _vectorize(mask: np.ndarray, transform: Affine) -> list[Polygon]:
    """Convert a binary raster mask into cleaned-up polygons in EPSG:4326."""
    raw: list[tuple[Polygon, int]] = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if val != 1:
            continue
        poly = shape(geom)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        # Collapse MultiPolygons to keep the schema simple for the UI.
        if poly.geom_type == "MultiPolygon":
            for g in poly.geoms:
                if g.area > 0:
                    raw.append((g, 1))
        else:
            raw.append((poly, 1))
    return [p for p, _ in raw]


def _ring_buffer(coords: list, distance_m: float, lat: float) -> list[list[float]]:
    """Approximate a coord buffer in metres by expanding in degrees at this latitude."""
    dlat = distance_m / 111320.0
    dlon = distance_m / (111320.0 * max(math.cos(math.radians(lat)), 1e-3))
    return [[c[0] - dlon, c[1] - dlat] for c in coords]


# ── The detector ──────────────────────────────────────────────────────────


class DeterministicDetector:
    """Stateless detector. One instance handles any number of GeoTIFFs."""

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def detect(self, tif_path: str | Path) -> dict[str, Any]:
        tif = Path(tif_path)
        with rasterio.open(tif) as src:
            vv = src.read(1)  # linear sigma0 VV
            # Band 2 (VH) is present in every scene but Tier-A thresholds VV only,
            # so it is deliberately not read here.
            msk = src.read(3)  # 1 = valid, 0 = nodata/land
            transform = src.transform
            width, height = src.width, src.height
            bounds = src.bounds

        cfg = self.config
        # 1. Speckle filter on linear sigma0.
        vv_f = lee_sigma_filter(vv, win=cfg.lee_window)

        # 2. dB + land mask.
        with np.errstate(invalid="ignore", divide="ignore"):
            vv_db = 10.0 * np.log10(np.clip(vv_f, 1e-6, None))
        ocean = (msk > 0) & np.isfinite(vv_db)

        # 3. Two-gate adaptive threshold (Solberg-style: local + scene).
        candidate, local_bg, noise_sigma = adaptive_threshold(
            vv_db,
            ocean,
            win=cfg.local_window,
            k=cfg.threshold_k,
            big_win=cfg.big_window,
            baseline_pct=cfg.baseline_percentile,
            min_scene_contrast=cfg.min_scene_contrast_dB,
        )
        sea_baseline_dB = float(np.nanpercentile(vv_db[ocean], cfg.baseline_percentile))

        # 4. Morphological cleanup.
        if cfg.open_iter > 0:
            candidate = ndi.binary_opening(candidate, iterations=cfg.open_iter)
        if cfg.close_iter > 0:
            candidate = ndi.binary_closing(candidate, iterations=cfg.close_iter)

        # 5. Connected components → drop blobs that are too small or too dim.
        labelled, n_blobs = ndi.label(candidate)
        if n_blobs == 0:
            return _empty_result(tif, vv_db, ocean)

        polygons = _vectorize(candidate, transform)
        feats: list[PolygonFeature] = []
        scene_area_km2 = abs((bounds.right - bounds.left) * (bounds.top - bounds.bottom)) * (
            111.32 * 111.32 * math.cos(math.radians((bounds.top + bounds.bottom) / 2))
        )
        valid_fraction = float(ocean.sum() / ocean.size)

        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds
            # Convert °² to km² using approximate local scale.
            cy = poly.centroid.y
            km_per_deg_lon = 111.32 * math.cos(math.radians(cy))
            km_per_deg_lat = 110.57
            scale2 = km_per_deg_lon * km_per_deg_lat
            area_km2 = float(poly.area) * scale2
            if area_km2 < cfg.min_area_km2:
                continue
            # perimeter
            perim_km = float(poly.length) * 0.5 * (km_per_deg_lon + km_per_deg_lat)
            compact = (4.0 * math.pi * area_km2) / max(perim_km**2, 1e-9)
            elongation = (
                max(maxx - minx, maxy - miny)
                * max(km_per_deg_lon, km_per_deg_lat)
                / max(min(maxx - minx, maxy - miny) * max(km_per_deg_lon, km_per_deg_lat), 1e-6)
            )
            # mean dB inside the polygon (rasterised sample)
            mask_in = rasterise_polygon(poly, transform, width, height)
            inside_vals = vv_db[mask_in & ocean]
            if inside_vals.size == 0:
                continue
            mean_dB = float(np.nanmean(inside_vals))
            contrast_dB = float(np.nanmean(local_bg[mask_in & ocean]) - mean_dB)
            scene_contrast_dB = sea_baseline_dB - mean_dB
            if contrast_dB < cfg.min_dB_contrast:
                continue
            # Reject very-streak-shaped blobs — they are almost always wind
            # shadows, not oil slicks. The Wakashio slick is a few km wide.
            if elongation > cfg.max_elongation:
                continue

            age_h, age_u = fay_age(area_km2, K=cfg.fay_K)
            _ = age_u  # referenced
            p, lo, hi, n_votes = _confidence(contrast_dB, valid_fraction, area_km2)

            wind_flag = "OK"
            if cfg.wind_speed_ms is None:
                wind_flag = "LOW_CONFIDENCE_NO_WIND"
            elif not (cfg.wind_viability[0] <= cfg.wind_speed_ms <= cfg.wind_viability[1]):
                wind_flag = "OUTSIDE_BAND"

            feats.append(
                PolygonFeature(
                    area_km2=round(area_km2, 3),
                    perimeter_km=round(perim_km, 3),
                    compactness=round(min(1.0, max(0.0, compact)), 3),
                    elongation=round(elongation, 2),
                    mean_dB=round(mean_dB, 2),
                    contrast_dB=round(contrast_dB, 2),
                    scene_contrast_dB=round(scene_contrast_dB, 2),
                    age_hours_fay=round(age_h, 1) if age_h is not None else None,
                    age_hours_uncertainty_h=round(age_u, 1) if age_u is not None else None,
                    confidence=round(p, 3),
                    confidence_low=round(lo, 3),
                    confidence_high=round(hi, 3),
                    wind_viability=wind_flag,
                    bbox_wsen=[float(minx), float(miny), float(maxx), float(maxy)],
                    geometry=mapping(poly),
                )
            )

        feats.sort(key=lambda f: f.area_km2, reverse=True)
        overall = max((f.confidence for f in feats), default=0.0)
        return {
            "file": str(tif),
            "scene_id": tif.stem,
            "acquisition_time": _acquisition_time(tif),
            "scene_area_km2": round(scene_area_km2, 2),
            "valid_fraction": round(valid_fraction, 4),
            "median_ocean_dB": float(np.nanmedian(vv_db[ocean])) if ocean.any() else None,
            "blob_candidates": int(n_blobs),
            "polygons_kept": len(feats),
            "best_confidence": round(overall, 3),
            "wind_flag": feats[0].wind_viability if feats else "OK",
            "polygons": [asdict(f) for f in feats],
            "detector": "deterministic-lee-adaptive-v1",
            "ran_utc": datetime.now(UTC).isoformat(),
        }


# ── helpers ───────────────────────────────────────────────────────────────


def _acquisition_time(tif: Path) -> str | None:
    """Resolve the real SAR acquisition instant for a scene.

    The ingest service writes a ``<stem>.json`` sidecar next to every GeoTIFF
    carrying the CDSE catalogue hits inside the requested window. We take the
    earliest real ``start`` from there, falling back to ``time_window[0]``.

    A backward drift hindcast is anchored to this instant — using ``ran_utc``
    (i.e. "now") would backtrack from 2026 to fetch forcing for a 2020 slick
    and produce garbage, so this must be the satellite pass, not the run clock.
    """
    sidecar = tif.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        meta = json.loads(sidecar.read_text())
    except Exception:  # noqa: BLE001
        return None
    starts = [s.get("start") for s in (meta.get("scenes_in_window") or []) if s.get("start")]
    if starts:
        return min(starts)
    window = meta.get("time_window") or []
    return window[0] if window else None


def _empty_result(tif: Path, vv_db: np.ndarray, ocean: np.ndarray) -> dict[str, Any]:
    return {
        "file": str(tif),
        "scene_id": tif.stem,
        "acquisition_time": _acquisition_time(tif),
        "blob_candidates": 0,
        "polygons_kept": 0,
        "best_confidence": 0.0,
        "wind_flag": "OK",
        "polygons": [],
        "median_ocean_dB": float(np.nanmedian(vv_db[ocean])) if ocean.any() else None,
        "valid_fraction": round(float(ocean.sum() / ocean.size), 4) if ocean.any() else 0.0,
        "detector": "deterministic-lee-adaptive-v1",
        "ran_utc": datetime.now(UTC).isoformat(),
    }


def rasterise_polygon(poly: Polygon, transform: Affine, width: int, height: int) -> np.ndarray:
    """Return a boolean raster of the polygon at the transform's resolution."""
    from rasterio.features import rasterize

    return rasterize(
        [(mapping(poly), 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def run_on_directory(
    sar_dir: str | Path, out_dir: str | Path | None = None, wind_speed_ms: float | None = None
) -> list[dict[str, Any]]:
    """Run the detector on every .tif in `sar_dir` and write JSON results."""
    sar = Path(sar_dir)
    out = Path(out_dir or sar)
    out.mkdir(parents=True, exist_ok=True)
    det = DeterministicDetector(config=DetectionConfig(wind_speed_ms=wind_speed_ms))
    results: list[dict[str, Any]] = []
    for tif in sorted(sar.glob("*.tif")):
        if tif.name.startswith(".") or tif.name == "index.json":
            continue
        try:
            r = det.detect(tif)
            (out / f"{tif.stem}.detection.json").write_text(json.dumps(r, indent=2))
            row: dict[str, Any] = {
                "scene_id": r["scene_id"],
                **{
                    k: r[k]
                    for k in ("best_confidence", "polygons_kept", "wind_flag", "median_ocean_dB")
                },
            }
            row["acquisition_time"] = r.get("acquisition_time")
            top = (r.get("polygons") or [None])[0]
            if top:
                w, s, e, n = top.get("bbox_wsen", [0, 0, 0, 0])
                row["top_area_km2"] = top.get("area_km2")
                row["top_centroid"] = [round((w + e) / 2, 5), round((s + n) / 2, 5)]
            results.append(row)
        except Exception as exc:  # noqa: BLE001
            results.append({"scene_id": tif.stem, "error": str(exc)[:300]})
    return results
