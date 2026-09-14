"""Complete real-scene inference: validate → tile → segment → vectorise → explain.

This is the single entry point for "run detection on this GeoTIFF". It owns the
whole contract and, above all, the **state machine**: every response carries one
of six machine-readable states, so a caller never has to infer what happened
from an empty list.

    ``ok``                    detections found, best confidence is usable
    ``no_detection``          the scene is valid and nothing was found
    ``low_confidence``        something was found, but even the upper Wilson
                              bound is below ``low_confidence_ci_high``
    ``out_of_distribution``   the scene's statistics fall outside the envelope
                              the thresholds were calibrated on; detections are
                              still returned, flagged, and must not be trusted
                              as if they were in-envelope
    ``missing_forcing_data``  wind was required (``require_wind=True``) and none
                              was available, so look-alike discrimination — the
                              thing that separates oil from a wind shadow —
                              could not be performed
    ``invalid_scene``         the scene never got as far as inference; see
                              ``reason_code``

Precedence is: invalid → no_detection → low_confidence →
out_of_distribution → missing_forcing_data → ok. ``no_detection`` outranks the
caveat states because "nothing was found" is the actionable fact; the caveats
are still listed in ``flags`` so nothing is hidden.

Pipeline
--------
1. :func:`validate_scene` — dimensions, CRS, band count, nodata *before* any
   inference. Invalid ⇒ ``invalid_scene`` with a reason code, never a traceback.
2. Tiled read + Lee-sigma speckle filter + dB conversion + land mask
   (:mod:`app.processors.tiling`). Every operator runs per tile with a halo
   sized from its own reach, so the result is identical to an untiled run.
3. Global scene statistics (clean-sea dB baseline, speckle sigma) then the
   Solberg two-gate threshold — again tiled, stitched into one full-scene mask.
4. Morphological clean-up, then connected components over the **whole**
   stitched mask, which is what merges a slick crossing a tile boundary.
5. Per-blob metrics; filter by area / contrast / elongation.
6. Vectorise surviving blobs, measure geodesically, reproject to EPSG:4326.
7. Confidence + Wilson 95 % CI (AGENTS.md: never a point estimate alone).
8. Deterministic evidence map — explicitly *not* Grad-CAM.

The default detector is the deterministic Tier-A operator, whose behaviour on
the real Wakashio GeoTIFFs is recorded in ``docs/VERIFICATION.md`` §7. The
UNet++ tier reports ``model_status: "untrained"`` and emits no probabilities
(see :mod:`app.processors.unet_path`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from loguru import logger
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy import ndimage as ndi
from shapely.geometry import Polygon, mapping, shape
from shapely.validation import make_valid

from app.processors.deterministic import (
    DetectionConfig,
    _acquisition_time,
    _confidence,
    apply_two_gates,
    fay_age,
    lee_sigma_filter,
    masked_box_filter,
)
from app.processors.evidence import (
    EvidenceConfig,
    EvidenceMap,
    build_evidence_map,
    detection_evidence,
    explanation_block,
)
from app.processors.geometry import SceneGeometry
from app.processors.tiling import TilePlan, plan_tiles
from app.processors.unet_path import resolve_unet_status
from app.processors.validation import STATE_INVALID_SCENE, SceneValidation, validate_scene

STATE_OK = "ok"
STATE_NO_DETECTION = "no_detection"
STATE_LOW_CONFIDENCE = "low_confidence"
STATE_OUT_OF_DISTRIBUTION = "out_of_distribution"
STATE_MISSING_FORCING_DATA = "missing_forcing_data"

ALL_STATES = (
    STATE_OK,
    STATE_NO_DETECTION,
    STATE_LOW_CONFIDENCE,
    STATE_OUT_OF_DISTRIBUTION,
    STATE_MISSING_FORCING_DATA,
    STATE_INVALID_SCENE,
)

DETECTOR_ID = "deterministic-tier-a-lee-solberg-v1"
RESPONSE_SCHEMA = "sentinel.detect.inference/v1"

# Scene-statistics envelope the Tier-A thresholds were calibrated on. Outside
# it, the adaptive threshold is still computable but is no longer the operator
# that was verified, so the response must say so.
OOD_MIN_VALID_FRACTION = 0.10
OOD_DB_RANGE = (-32.0, -2.0)  # median ocean sigma0 dB
OOD_MIN_DB_IQR = 1.0  # p90 - p10 of ocean dB; below this the scene is flat
OOD_MAX_DARK_FRACTION = 0.25  # share of ocean passing both gates
OOD_MAX_PIXEL_M = 200.0

DEFAULT_LOW_CONFIDENCE_CI_HIGH = 0.70


@dataclass
class PipelineOptions:
    """Per-run knobs. Everything has a documented default."""

    wind_speed_ms: float | None = None  # ERA5/CMEMS 10-m wind, when available
    require_wind: bool = False  # if True, no wind ⇒ missing_forcing_data
    tiled: bool = True
    tile_size: int = 1024
    overlap: int = 192
    with_evidence: bool = True
    evidence_grid_side: int = 48
    low_confidence_ci_high: float = DEFAULT_LOW_CONFIDENCE_CI_HIGH
    min_valid_fraction: float = 0.01


@dataclass
class Detection:
    """One spill candidate that survived every gate."""

    index: int
    geometry_wgs84: Polygon
    measurement: dict[str, Any]
    metrics: dict[str, Any]
    confidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": f"det_{self.index:04d}",
            "geometry": mapping(self.geometry_wgs84),
            **self.measurement,
            **self.metrics,
            "confidence": self.confidence,
        }


@dataclass
class InferenceResult:
    state: str
    state_reason: str | None
    flags: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    scene: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    tiling: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    detections: list[Detection] = field(default_factory=list)
    explanation: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    inference_ms: float = 0.0
    ran_utc: str = ""

    def as_dict(self) -> dict[str, Any]:
        feats = []
        for det in self.detections:
            feats.append(
                {
                    "type": "Feature",
                    "geometry": mapping(det.geometry_wgs84),
                    "properties": {
                        "id": f"det_{det.index:04d}",
                        "area_km2": det.measurement["area_km2"],
                        "confidence": det.confidence["value"],
                        "confidence_low": det.confidence["low"],
                        "confidence_high": det.confidence["high"],
                        "centroid": det.measurement["centroid"],
                        "length_km": det.measurement["length_km"],
                        "width_km": det.measurement["width_km"],
                        "orientation_deg": det.measurement["orientation_deg"],
                        "mean_db": det.metrics.get("mean_db"),
                        "contrast_db": det.metrics.get("contrast_db"),
                        "scene_contrast_db": det.metrics.get("scene_contrast_db"),
                        "age_hours_fay": det.metrics.get("age_hours_fay"),
                        "wind_viability": det.metrics.get("wind_viability"),
                    },
                }
            )
        return {
            "schema": RESPONSE_SCHEMA,
            "state": self.state,
            "state_reason": self.state_reason,
            "flags": list(self.flags),
            "count": len(self.detections),
            "provenance": self.provenance,
            "validation": self.validation,
            "scene": self.scene,
            "model": self.model,
            "tiling": self.tiling,
            "detector": DETECTOR_ID,
            "confidence": self.confidence,
            "detections": [d.as_dict() for d in self.detections],
            "geojson": {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
                "features": feats,
            },
            "explanation": self.explanation,
            "inference_ms": round(self.inference_ms, 1),
            "ran_utc": self.ran_utc,
        }


class DetectPipeline:
    """Stateless w.r.t. scenes: one instance serves any number of GeoTIFFs.

    Constructed once per process (see the FastAPI lifespan) so the config,
    tiling plan defaults and UNet++ status resolution are done a single time.
    """

    def __init__(
        self,
        config: DetectionConfig | None = None,
        *,
        model_checkpoint: str | Path | None = None,
        evidence_config: EvidenceConfig | None = None,
    ) -> None:
        self.config = config or DetectionConfig()
        self.evidence_config = evidence_config or EvidenceConfig()
        self.model_status = resolve_unet_status(model_checkpoint)

    # ── public API ────────────────────────────────────────────────────────

    def run(self, tif_path: str | Path, options: PipelineOptions | None = None) -> dict[str, Any]:
        """Run complete inference on one scene. Never raises for a bad scene."""
        opts = options or PipelineOptions()
        started = datetime.now(UTC)
        path = Path(tif_path)
        model = self.model_status.as_dict()

        validation = validate_scene(
            path, min_valid_fraction=opts.min_valid_fraction
        )
        if not validation.valid:
            logger.warning("scene rejected: {} ({})", path, validation.reason_code)
            return InferenceResult(
                state=STATE_INVALID_SCENE,
                state_reason=validation.reason,
                flags=["invalid_scene", validation.reason_code or "unspecified"],
                provenance=self._provenance(path, None, None),
                validation=validation.as_dict(),
                model=model,
                ran_utc=started.isoformat(),
            ).as_dict()

        try:
            return self._run_valid(path, opts, validation, started)
        except Exception as exc:  # noqa: BLE001 - a scene must never 500 with a traceback
            logger.exception("inference failed for {}: {}", path, exc)
            return InferenceResult(
                state=STATE_INVALID_SCENE,
                state_reason=f"inference failed: {type(exc).__name__}: {exc}",
                flags=["invalid_scene", "inference_error"],
                provenance=self._provenance(path, None, None),
                validation=validation.as_dict(),
                model=model,
                ran_utc=started.isoformat(),
            ).as_dict()

    # ── internals ─────────────────────────────────────────────────────────

    def _run_valid(
        self,
        path: Path,
        opts: PipelineOptions,
        validation: SceneValidation,
        started: datetime,
    ) -> dict[str, Any]:
        cfg = self.config
        if opts.wind_speed_ms is not None:
            cfg = DetectionConfig(**{**cfg.__dict__, "wind_speed_ms": opts.wind_speed_ms})

        with rasterio.open(path) as src:
            crs = src.crs
            transform: Affine = src.transform
            height, width = int(src.height), int(src.width)
            descriptions = [d or "" for d in (src.descriptions or ())]
            vv = src.read(1).astype(np.float32)
            msk = src.read(3)

        geometry = SceneGeometry(crs, transform, height, width)
        plan = plan_tiles(
            height,
            width,
            tile_size=opts.tile_size,
            overlap=opts.overlap,
            config=cfg,
        )
        active_plan = plan if opts.tiled else _single_tile_plan(plan, height, width)

        # ── 2. Tiled speckle filter + dB + mask ───────────────────────────
        vv_db = np.full((height, width), np.nan, dtype=np.float32)
        local_bg = np.zeros((height, width), dtype=np.float32)
        ocean = np.zeros((height, width), dtype=bool)
        masked = np.isfinite(vv) & (vv > 0) & np.isfinite(msk) & (msk > 0)

        filled = _fill_nan(vv, masked)
        sigma_v = float(np.nanstd(filled[masked])) if masked.any() else 1e-6

        for window in active_plan.windows():
            rs, cs = window.read
            core_r, core_c = window.core_in_read
            tile_filled = filled[rs, cs]
            tile_db = _to_db(lee_sigma_filter(tile_filled, win=cfg.lee_window, sigma_v=sigma_v))
            tile_ocean = masked[rs, cs] & np.isfinite(tile_db)
            tile_bg = masked_box_filter(tile_db, tile_ocean, cfg.big_window)
            vv_db[window.core] = tile_db[core_r, core_c]
            local_bg[window.core] = tile_bg[core_r, core_c]
            ocean[window.core] = tile_ocean[core_r, core_c]

        ocean &= np.isfinite(vv_db)
        valid_fraction = float(ocean.sum() / max(ocean.size, 1))
        if not ocean.any():
            return self._no_detection_result(
                path, opts, validation, geometry, active_plan, started, vv_db, ocean, 0.0,
                reason="no finite, unmasked ocean pixels after speckle filtering",
            )

        sea_baseline = float(np.nanpercentile(vv_db[ocean], cfg.baseline_percentile))
        residual = np.where(ocean, vv_db - local_bg, np.nan)
        noise_sigma = float(np.nanstd(residual[ocean])) or 1.0
        p10, p50, p90 = (float(x) for x in np.nanpercentile(vv_db[ocean], [10, 50, 90]))

        # ── 3. Tiled two-gate threshold, stitched into one mask ───────────
        candidate = np.zeros((height, width), dtype=bool)
        for window in active_plan.windows():
            rs, cs = window.read
            core_r, core_c = window.core_in_read
            dark, _, _ = apply_two_gates(
                vv_db[rs, cs],
                ocean[rs, cs],
                local_bg[rs, cs],
                noise_sigma,
                sea_baseline,
                k=cfg.threshold_k,
                min_scene_contrast=cfg.min_scene_contrast_dB,
            )
            candidate[window.core] = dark[core_r, core_c]

        dark_fraction = float(candidate.sum() / max(int(ocean.sum()), 1))

        # ── 4. Morphology (tiled) + connected components (global) ──────────
        # The tile halo is sized from the morphological reach as well, so
        # opening/closing inside a tile matches an untiled run exactly.
        cleaned = candidate
        if cfg.open_iter > 0 or cfg.close_iter > 0:
            cleaned = np.zeros((height, width), dtype=bool)
            for window in active_plan.windows():
                rs, cs = window.read
                core_r, core_c = window.core_in_read
                tile = candidate[rs, cs]
                if cfg.open_iter > 0:
                    tile = ndi.binary_opening(tile, iterations=cfg.open_iter)
                if cfg.close_iter > 0:
                    tile = ndi.binary_closing(tile, iterations=cfg.close_iter)
                cleaned[window.core] = tile[core_r, core_c]

        labelled, n_blobs = ndi.label(cleaned)
        if n_blobs == 0:
            return self._no_detection_result(
                path, opts, validation, geometry, active_plan, started, vv_db, ocean,
                dark_fraction,
                reason="no connected dark formation passed both threshold gates",
                extra_scene=self._scene_stats(
                    p10, p50, p90, sea_baseline, noise_sigma, valid_fraction, dark_fraction
                ),
            )

        detections, per_detection_evidence = self._build_detections(
            labelled=labelled,
            n_blobs=n_blobs,
            vv_db=vv_db,
            ocean=ocean,
            local_bg=local_bg,
            geometry=geometry,
            transform=transform,
            cfg=cfg,
            valid_fraction=valid_fraction,
            sea_baseline=sea_baseline,
            noise_sigma=noise_sigma,
        )

        # ── 7/8. Evidence map + explanation ───────────────────────────────
        evidence: EvidenceMap | None = None
        grid = None
        if opts.with_evidence:
            evidence = build_evidence_map(
                vv_db,
                ocean,
                local_bg,
                noise_sigma_dB=noise_sigma,
                sea_baseline_dB=sea_baseline,
                k=cfg.threshold_k,
                min_scene_contrast_dB=cfg.min_scene_contrast_dB,
                config=self.evidence_config,
            )
            grid = evidence.grid(side=opts.evidence_grid_side)

        explanation = explanation_block(
            evidence=evidence,
            grid=grid,
            bbox_wsen=_bounds_wsen(geometry, height, width),
            per_detection=per_detection_evidence,
            gradcam_used=False,
        )

        best = max((d.confidence["value"] for d in detections), default=0.0)
        best_high = max((d.confidence["high"] for d in detections), default=0.0)
        best_low = max((d.confidence["low"] for d in detections), default=0.0)
        best_n = max((d.confidence["n_effective"] for d in detections), default=0)

        flags: list[str] = []
        ood_reasons = self._ood_reasons(
            valid_fraction=valid_fraction,
            median_db=p50,
            db_iqr=p90 - p10,
            dark_fraction=dark_fraction,
            pixel_m=max(geometry.pixel_size_m),
        )
        if ood_reasons:
            flags.append(STATE_OUT_OF_DISTRIBUTION)
        if opts.wind_speed_ms is None:
            flags.append("missing_wind_forcing")

        state, state_reason = self._decide_state(
            n_detections=len(detections),
            best_high=best_high,
            ood=bool(ood_reasons),
            require_wind=opts.require_wind,
            wind_missing=opts.wind_speed_ms is None,
            ci_high_threshold=opts.low_confidence_ci_high,
        )
        if ood_reasons:
            flags.extend(f"ood:{r}" for r in ood_reasons)

        confidence_block = {
            "value": round(float(best), 3),
            "low": round(float(best_low), 3),
            "high": round(float(best_high), 3),
            "interval": "wilson_95",
            "n_effective": int(best_n),
            "scale": "probability the top detection is a mineral-oil slick",
        }

        scene_block = self._scene_stats(
            p10, p50, p90, sea_baseline, noise_sigma, valid_fraction, dark_fraction
        )
        scene_block.update(
            {
                "crs": geometry.crs.to_string(),
                "epsg": geometry.crs.to_epsg(),
                "width": width,
                "height": height,
                "pixel_size_m": [float(v) for v in geometry.pixel_size_m],
                "bounds_wsen": _bounds_wsen(geometry, height, width),
                "bands": descriptions,
                "blob_candidates": int(n_blobs),
            }
        )

        result = InferenceResult(
            state=state,
            state_reason=state_reason,
            flags=flags,
            provenance=self._provenance(path, geometry, None),
            scene=scene_block,
            model=self.model_status.as_dict(),
            tiling=active_plan.as_dict(),
            confidence=confidence_block,
            detections=detections,
            explanation=explanation,
            validation=validation.as_dict(),
            inference_ms=(datetime.now(UTC) - started).total_seconds() * 1000.0,
            ran_utc=started.isoformat(),
        )
        return result.as_dict()

    # ── detections ────────────────────────────────────────────────────────

    def _build_detections(
        self,
        *,
        labelled: np.ndarray,
        n_blobs: int,
        vv_db: np.ndarray,
        ocean: np.ndarray,
        local_bg: np.ndarray,
        geometry: SceneGeometry,
        transform: Affine,
        cfg: DetectionConfig,
        valid_fraction: float,
        sea_baseline: float,
        noise_sigma: float,
    ) -> tuple[list[Detection], list[dict[str, Any]]]:
        """Score every connected component, keep the ones that survive every gate."""
        idx = np.arange(1, n_blobs + 1)
        valid_in = ocean & np.isfinite(vv_db)
        counts = ndi.sum(valid_in, labelled, index=idx)
        db_sums = ndi.sum(np.where(valid_in, vv_db, 0.0), labelled, index=idx)
        bg_sums = ndi.sum(np.where(valid_in, local_bg, 0.0), labelled, index=idx)
        counts = np.asarray(counts, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_db = np.where(counts > 0, db_sums / np.maximum(counts, 1.0), np.nan)
            mean_bg = np.where(counts > 0, bg_sums / np.maximum(counts, 1.0), np.nan)

        pixel_area_m2 = geometry.pixel_area_m2()
        px_x, px_y = geometry.pixel_size_m
        slices = ndi.find_objects(labelled)

        rows: list[tuple[float, Detection, dict[str, Any]]] = []
        serial = 0
        for i in range(n_blobs):
            label = i + 1
            npix = float(counts[i])
            if npix <= 0:
                continue
            area_m2 = npix * pixel_area_m2
            area_km2 = area_m2 / 1.0e6
            if area_km2 < cfg.min_area_km2:
                continue
            blob_mean_db = float(mean_db[i])
            blob_mean_bg = float(mean_bg[i])
            if not (math.isfinite(blob_mean_db) and math.isfinite(blob_mean_bg)):
                continue
            contrast_db = blob_mean_bg - blob_mean_db
            if contrast_db < cfg.min_dB_contrast:
                continue
            scene_contrast_db = sea_baseline - blob_mean_db

            sl = slices[i]
            if sl is None:
                continue
            r0, r1 = sl[0].start, sl[0].stop
            c0, c1 = sl[1].start, sl[1].stop
            span_y = max((r1 - r0) * px_y, 1e-6)
            span_x = max((c1 - c0) * px_x, 1e-6)
            elongation = max(span_x, span_y) / min(span_x, span_y)
            if elongation > cfg.max_elongation:
                continue

            poly_crs = _vectorize_label(labelled, label, sl, transform)
            if poly_crs.is_empty:
                continue

            measurement = geometry.measure(poly_crs).as_dict()
            measured_area_km2 = float(measurement["area_km2"])
            if measured_area_km2 <= 0:
                continue

            age_h, age_u = fay_age(measured_area_km2, K=cfg.fay_K)
            p, lo, hi, n_votes = _confidence(contrast_db, valid_fraction, measured_area_km2)
            wind_flag = "OK"
            if cfg.wind_speed_ms is None:
                wind_flag = "LOW_CONFIDENCE_NO_WIND"
            elif not (cfg.wind_viability[0] <= cfg.wind_speed_ms <= cfg.wind_viability[1]):
                wind_flag = "OUTSIDE_BAND"

            pixel_evidence = self._pixel_evidence(labelled, label, sl, vv_db, ocean, local_bg)
            terms = {
                "contrast": 0.55 * min(1.0, max(0.0, (contrast_db - 1.0) / 5.0)),
                "validity": 0.25 * max(0.0, min(1.0, valid_fraction)),
                "size": 0.20 * min(1.0, measured_area_km2 / 5.0),
            }
            ev = detection_evidence(
                contrast_dB=contrast_db,
                scene_contrast_dB=scene_contrast_db,
                noise_sigma_dB=noise_sigma,
                area_km2=measured_area_km2,
                elongation=elongation,
                valid_fraction=valid_fraction,
                wind_viability=wind_flag,
                confidence_terms=terms,
                min_dB_contrast=cfg.min_dB_contrast,
                max_elongation=cfg.max_elongation,
                min_area_km2=cfg.min_area_km2,
                pixel_evidence=pixel_evidence,
            )

            perimeter_m = float(measurement["perimeter_m"])
            compact = (4.0 * math.pi * measurement["area_m2"]) / max(perimeter_m**2, 1e-9)
            metrics = {
                "mean_db": round(blob_mean_db, 2),
                "contrast_db": round(contrast_db, 2),
                "scene_contrast_db": round(scene_contrast_db, 2),
                "compactness": round(min(1.0, max(0.0, compact)), 3),
                "pixel_count": int(npix),
                "age_hours_fay": round(age_h, 1) if age_h is not None else None,
                "age_hours_uncertainty_h": round(age_u, 1) if age_u is not None else None,
                "wind_viability": wind_flag,
                "bbox_wsen": geometry.bbox_wsen(poly_crs),
                "evidence": ev.as_dict(),
            }
            detection = Detection(
                index=serial,
                geometry_wgs84=geometry.crs_to_wgs84_polygon(poly_crs),
                measurement=measurement,
                metrics=metrics,
                confidence={
                    "value": round(float(p), 3),
                    "low": round(float(lo), 3),
                    "high": round(float(hi), 3),
                    "interval": "wilson_95",
                    "n_effective": int(n_votes),
                },
            )
            serial += 1
            rows.append((measured_area_km2, detection, ev.as_dict()))

        rows.sort(key=lambda r: r[0], reverse=True)
        detections: list[Detection] = []
        evidence_rows: list[dict[str, Any]] = []
        for position, (_, det, ev) in enumerate(rows):
            det.index = position
            detections.append(det)
            ev_with_id = {"id": f"det_{position:04d}", **ev}
            evidence_rows.append(ev_with_id)
        return detections, evidence_rows

    @staticmethod
    def _pixel_evidence(
        labelled: np.ndarray,
        label: int,
        sl: tuple[slice, ...],
        vv_db: np.ndarray,
        ocean: np.ndarray,
        local_bg: np.ndarray,
    ) -> dict[str, float | None]:
        """Contrast statistics of the detection's own pixels (for the audit)."""
        sub_label = labelled[sl]
        inside = (sub_label == label) & ocean[sl] & np.isfinite(vv_db[sl])
        if not inside.any():
            return {"n_pixels": 0, "mean_contrast_dB": None, "max_contrast_dB": None}
        contrast = local_bg[sl][inside] - vv_db[sl][inside]
        return {
            "n_pixels": int(inside.sum()),
            "mean_contrast_dB": round(float(np.mean(contrast)), 3),
            "max_contrast_dB": round(float(np.max(contrast)), 3),
            "min_contrast_dB": round(float(np.min(contrast)), 3),
        }

    # ── states ────────────────────────────────────────────────────────────

    @staticmethod
    def _decide_state(
        *,
        n_detections: int,
        best_high: float,
        ood: bool,
        require_wind: bool,
        wind_missing: bool,
        ci_high_threshold: float,
    ) -> tuple[str, str | None]:
        if n_detections == 0:
            return (
                STATE_NO_DETECTION,
                "scene is valid and was fully processed; no formation passed both "
                "threshold gates",
            )
        if best_high < ci_high_threshold:
            return (
                STATE_LOW_CONFIDENCE,
                f"detections exist but even the upper Wilson 95% bound ({best_high:.3f}) "
                f"is below {ci_high_threshold:.2f}; treat as unconfirmed",
            )
        if ood:
            return (
                STATE_OUT_OF_DISTRIBUTION,
                "scene statistics fall outside the envelope the Tier-A thresholds were "
                "calibrated on; detections are returned but must not be read as "
                "in-envelope evidence",
            )
        if require_wind and wind_missing:
            return (
                STATE_MISSING_FORCING_DATA,
                "wind forcing was required but none was supplied; look-alike "
                "discrimination (oil vs wind shadow) could not be performed",
            )
        return STATE_OK, None

    def _ood_reasons(
        self,
        *,
        valid_fraction: float,
        median_db: float,
        db_iqr: float,
        dark_fraction: float,
        pixel_m: float,
    ) -> list[str]:
        reasons: list[str] = []
        if valid_fraction < OOD_MIN_VALID_FRACTION:
            reasons.append(f"valid_fraction {valid_fraction:.3f} < {OOD_MIN_VALID_FRACTION}")
        if not (OOD_DB_RANGE[0] <= median_db <= OOD_DB_RANGE[1]):
            reasons.append(
                f"median ocean sigma0 {median_db:.2f} dB outside "
                f"[{OOD_DB_RANGE[0]}, {OOD_DB_RANGE[1]}]"
            )
        if db_iqr < OOD_MIN_DB_IQR:
            reasons.append(f"ocean dB inter-decile range {db_iqr:.2f} dB — scene is flat")
        if dark_fraction > OOD_MAX_DARK_FRACTION:
            reasons.append(
                f"{dark_fraction:.1%} of ocean pixels pass both gates — the threshold is "
                "not discriminating on this scene"
            )
        if pixel_m > OOD_MAX_PIXEL_M:
            reasons.append(f"pixel size {pixel_m:.1f} m is too coarse to resolve a slick")
        return reasons

    # ── small builders ────────────────────────────────────────────────────

    def _no_detection_result(
        self,
        path: Path,
        opts: PipelineOptions,
        validation: SceneValidation,
        geometry: SceneGeometry,
        plan: TilePlan,
        started: datetime,
        vv_db: np.ndarray,
        ocean: np.ndarray,
        dark_fraction: float,
        *,
        reason: str,
        extra_scene: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        flags = ["no_detection"]
        if opts.wind_speed_ms is None:
            flags.append("missing_wind_forcing")
        scene_block = extra_scene or {}
        scene_block.update(
            {
                "valid_fraction": round(float(ocean.sum() / max(ocean.size, 1)), 4),
                "dark_fraction": round(float(dark_fraction), 4),
                "median_ocean_db": (
                    float(np.nanmedian(vv_db[ocean])) if ocean.any() else None
                ),
                "crs": geometry.crs.to_string(),
                "pixel_size_m": [float(v) for v in geometry.pixel_size_m],
            }
        )
        return InferenceResult(
            state=STATE_NO_DETECTION,
            state_reason=reason,
            flags=flags,
            provenance=self._provenance(path, geometry, None),
            scene=scene_block,
            model=self.model_status.as_dict(),
            tiling=plan.as_dict(),
            confidence={
                "value": 0.0,
                "low": 0.0,
                "high": 0.0,
                "interval": "wilson_95",
                "n_effective": 0,
                "note": "no detection, so there is no proportion to interval-estimate",
            },
            validation=validation.as_dict(),
            inference_ms=(datetime.now(UTC) - started).total_seconds() * 1000.0,
            ran_utc=started.isoformat(),
        ).as_dict()

    @staticmethod
    def _scene_stats(
        p10: float,
        p50: float,
        p90: float,
        sea_baseline: float,
        noise_sigma: float,
        valid_fraction: float,
        dark_fraction: float,
    ) -> dict[str, Any]:
        return {
            "valid_fraction": round(valid_fraction, 4),
            "ocean_db_p10": round(p10, 2),
            "median_ocean_db": round(p50, 2),
            "ocean_db_p90": round(p90, 2),
            "ocean_db_iqr": round(p90 - p10, 2),
            "sea_baseline_db": round(sea_baseline, 2),
            "noise_sigma_db": round(noise_sigma, 2),
            "dark_fraction": round(dark_fraction, 4),
        }

    def _provenance(
        self, path: Path, geometry: SceneGeometry | None, acquisition: str | None
    ) -> dict[str, Any]:
        """Where this scene came from, read from the ingest sidecar when present."""
        sidecar = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if sidecar.exists():
            try:
                loaded = json.loads(sidecar.read_text())
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception as exc:  # noqa: BLE001
                logger.warning("unreadable sidecar {}: {}", sidecar, exc)
        source = meta.get("source") or {}
        products = [s.get("name") for s in (meta.get("scenes_in_window") or [])]
        provenance: dict[str, Any] = {
            "scene_id": path.stem,
            "scene_path": str(path),
            "source_kind": "real_sentinel1_geotiff",
            "synthetic": False,
            "platform": source.get("platform", "unknown"),
            "catalogue": source.get("catalogue"),
            "process_api": source.get("process"),
            "sha256": meta.get("sha256"),
            "bytes": meta.get("bytes"),
            "fetched_utc": meta.get("fetched_utc"),
            "acquisition_time_utc": acquisition
            or _acquisition_time(path)
            or meta.get("time_window", [None])[0],
            "product_ids": [p for p in products if p][:10],
            "bands": meta.get("bands"),
            "detector": DETECTOR_ID,
            "measurement": "geodesic (WGS84) area/perimeter; shape in scene-centred AEQD",
            "output_crs": "EPSG:4326",
        }
        if geometry is not None:
            provenance["raster_crs"] = geometry.crs.to_string()
        return provenance


# ── helpers ───────────────────────────────────────────────────────────────


def _single_tile_plan(plan: TilePlan, height: int, width: int) -> TilePlan:
    """A plan with exactly one tile covering the whole scene (the untiled path)."""
    return TilePlan(
        height=height,
        width=width,
        tile_size=max(height, width),
        overlap=0,
        context=plan.context,
        stride=max(height, width),
        grid_rows=1,
        grid_cols=1,
        single_tile=True,
        note="untiled run requested (tiled=False)",
    )


def _fill_nan(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace non-finite samples with the local mean of their valid neighbours.

    The Lee-sigma filter is a box filter, and one NaN in a window poisons every
    output it touches — on a partly-masked scene that propagates until almost
    the whole scene is NaN. Filling first keeps the filter usable and is a
    no-op on a fully valid scene, so verified results are unchanged.
    """
    out = np.array(values, dtype=np.float32, copy=True)
    bad = ~np.isfinite(out) | ~valid
    if not bad.any():
        return out
    if valid.any():
        local_mean = masked_box_filter(
            np.where(valid, out, np.nan), valid, 15
        )
        fallback = float(np.nanmean(out[valid])) if valid.any() else 0.0
        fill = np.where(np.isfinite(local_mean), local_mean, fallback)
    else:
        fill = np.zeros_like(out)
    out[bad] = fill[bad].astype(np.float32)
    return out


def _to_db(linear: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (10.0 * np.log10(np.clip(linear, 1e-6, None))).astype(np.float32)


def _vectorize_label(
    labelled: np.ndarray,
    label: int,
    sl: tuple[slice, ...],
    transform: Affine,
) -> Polygon:
    """Vectorise one labelled blob into a single Polygon in the raster CRS.

    Only the blob's bounding window is scanned, so this stays cheap even for a
    scene with hundreds of candidates. Holes are dropped: the reported polygon
    is the outer boundary, which is what an operator draws and what the drift
    service seeds from.
    """
    sub = labelled[sl]
    binary = (sub == label).astype(np.uint8)
    if not binary.any():
        return Polygon()
    window_transform = Affine(
        transform.a,
        transform.b,
        transform.c + sl[1].start * transform.a + sl[0].start * transform.b,
        transform.d,
        transform.e,
        transform.f + sl[1].start * transform.d + sl[0].start * transform.e,
    )
    best: Polygon = Polygon()
    for geom, value in shapes(binary, mask=binary.astype(bool), transform=window_transform):
        if int(value) != 1:
            continue
        poly = shape(geom)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            parts = [g for g in poly.geoms if g.area > 0]
            if not parts:
                continue
            poly = max(parts, key=lambda g: g.area)
        if poly.area > best.area:
            best = poly
    return best


def _bounds_wsen(geometry: SceneGeometry, height: int, width: int) -> list[float]:
    """Scene bounds as [west, south, east, north] in EPSG:4326."""
    corners = [geometry.xy_to_wgs84(x, y) for x, y in _corner_xy(geometry.transform, height, width)]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return [float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))]


def _corner_xy(transform: Affine, height: int, width: int) -> list[tuple[float, float]]:
    return [
        transform * (0, 0),
        transform * (width, 0),
        transform * (width, height),
        transform * (0, height),
    ]
