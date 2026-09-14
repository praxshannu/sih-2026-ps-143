"""HTTP routes that wrap the deterministic Tier-A detector.

These endpoints work without torch / UNet++. They are mounted by both the
production ``main.py`` and by the lightweight ``main_no_ml.py`` so the same
detection contract is available whether or not ML weights are loaded.

Endpoints
---------
POST /detect/deterministic
    Run the operator on a GeoTIFF path. Returns the full PolygonFeature list
    (area_km2, contrast_dB, scene_contrast_dB, Wilson 95% CI, Fay age, etc.).
GET  /detect/deterministic/results
    List the on-disk ``*.detection.json`` results that already exist in
    ``data/sar``.
GET  /detect/deterministic/health
    Whether the deterministic pipeline is importable on this box.

The "real" production service additionally loads a UNet++ model and routes
``/detect`` to that pipeline; this router is the honest fallback that ships
with the spec's Tier-A operator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from app.processors.deterministic import (
    DetectionConfig,
    DeterministicDetector,
    _acquisition_time,
    run_on_directory,
)

router = APIRouter(prefix="/detect", tags=["detect-det"])

DATA_DIR = Path(os.getenv("SENTINEL_DATA_DIR", "/app/data"))
SAR_DIR = DATA_DIR / "sar"


class DetectRequest(BaseModel):
    tif_path: str = Field(..., description="Absolute path to a calibrated GeoTIFF (3 bands: VV, VH, dataMask)")
    wind_speed_ms: float | None = Field(None, description="Optional ERA5 wind to clear the LOW_CONFIDENCE_NO_WIND flag")


@router.get("/deterministic/health")
async def deterministic_health() -> dict[str, Any]:
    """Whether the deterministic pipeline imports cleanly on this host."""
    try:
        # Cheap import smoke-test
        DeterministicDetector(DetectionConfig())  # noqa: F841
        return {"ok": True, "detector": "deterministic-lee-adaptive-v1", "ml_available": _ml_available()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


@router.post("/deterministic")
async def detect(req: DetectRequest) -> dict[str, Any]:
    """Run the deterministic detector on a single GeoTIFF."""
    p = Path(req.tif_path)
    if not p.exists():
        raise HTTPException(404, f"GeoTIFF not found: {req.tif_path}")
    det = DeterministicDetector(
        config=DetectionConfig(wind_speed_ms=req.wind_speed_ms),
    )
    try:
        result = det.detect(p)
    except Exception as exc:  # noqa: BLE001
        logger.error("Deterministic detection failed for {}: {}", p, exc)
        raise HTTPException(500, f"detection failed: {str(exc)[:300]}") from exc

    # Persist alongside the SAR GeoTIFF for easy review.
    try:
        (p.with_suffix(".detection.json")).write_text(json.dumps(result, indent=2))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist detection.json: {}", exc)
    return result


@router.get("/deterministic/results")
async def list_results() -> dict[str, Any]:
    """List all *.detection.json files already written in ``data/sar``."""
    if not SAR_DIR.exists():
        return {"count": 0, "results": [], "sar_dir": str(SAR_DIR)}
    out: list[dict[str, Any]] = []
    for jf in sorted(SAR_DIR.glob("*.detection.json")):
        stem = jf.stem.replace(".detection", "")
        try:
            d = json.loads(jf.read_text())
            row: dict[str, Any] = {
                "scene_id": d.get("scene_id", stem),
                "blob_candidates": d.get("blob_candidates", 0),
                "polygons_kept": d.get("polygons_kept", 0),
                "best_confidence": d.get("best_confidence", 0.0),
                "wind_flag": d.get("wind_flag", "OK"),
                "ran_utc": d.get("ran_utc"),
            }
            # Everything the drift service needs to anchor a hindcast. Older
            # detection files predate `acquisition_time`, so we re-resolve from
            # the ingest sidecar rather than forcing a re-run to backfill.
            row["acquisition_time"] = d.get("acquisition_time") or _acquisition_time(
                SAR_DIR / f"{stem}.tif"
            )
            top = (d.get("polygons") or [None])[0]
            if top:
                w, s, e, n = top.get("bbox_wsen", [0, 0, 0, 0])
                row["top_area_km2"] = top.get("area_km2")
                row["top_centroid"] = [round((w + e) / 2, 5), round((s + n) / 2, 5)]
                row["top_confidence"] = top.get("confidence")
                row["top_confidence_low"] = top.get("confidence_low")
                row["top_confidence_high"] = top.get("confidence_high")
                row["top_age_hours_fay"] = top.get("age_hours_fay")
            out.append(row)
        except Exception as exc:  # noqa: BLE001
            out.append({"scene_id": stem, "error": str(exc)[:200]})
    return {"count": len(out), "sar_dir": str(SAR_DIR), "results": out}


@router.post("/deterministic/run_all")
async def run_all(wind_speed_ms: float | None = Query(None)) -> dict[str, Any]:
    """Re-run the deterministic detector on every GeoTIFF under ``data/sar``."""
    if not SAR_DIR.exists():
        raise HTTPException(404, f"SAR data dir not found: {SAR_DIR}")
    results = run_on_directory(SAR_DIR, wind_speed_ms=wind_speed_ms)
    return {"count": len(results), "results": results}


# Lazy import shim — only used by /health to report ML availability without
# requiring torch at module-load time.
def _ml_available() -> bool:
    try:
        import importlib.util as _u
        return bool(_u.find_spec("torch"))
    except Exception:
        return False