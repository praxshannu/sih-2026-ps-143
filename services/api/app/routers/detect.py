"""Gateway proxy for the deterministic Tier-A detector.

The detect service runs the Solberg two-gate adaptive operator on every SAR
GeoTIFF in ``data/sar/``. This router forwards the four exposed endpoints so
the UI talks to the API gateway only (per AGENTS.md).
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/detect", tags=["detect"])

DETECT_SERVICE_URL = os.getenv("DETECT_SERVICE_URL", "http://sentinel-detect:8002")
PROXY_TIMEOUT = float(os.getenv("DETECT_PROXY_TIMEOUT", "300"))

# Per-scene inference can take 20-30 s on the largest scenes, so the timeout
# is generous. Streaming would be nicer, but the JSON payload is small.


class DetectRequest(BaseModel):
    tif_path: str
    wind_speed_ms: float | None = None


async def _safe_pass(method: str, path: str, **kwargs) -> JSONResponse:
    url = f"{DETECT_SERVICE_URL}/{path}"
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as c:
            resp = await c.request(method, url, **kwargs)
    except httpx.RequestError as exc:
        logger.error("Detect proxy {} {} failed: {}", method, path, exc)
        raise HTTPException(503, "Detect service unreachable") from exc
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text[:300])
        except Exception:  # noqa: BLE001
            detail = (resp.text or "upstream error")[:300]
        raise HTTPException(resp.status_code, detail)
    return JSONResponse(
        content=resp.json(),
        headers={"cache-control": "no-store"},
    )


@router.get("/health")
async def detect_health() -> JSONResponse:
    """Detect service health + which variant is running (deterministic | ml)."""
    return await _safe_pass("GET", "health")


@router.get("/deterministic/results")
async def list_detections() -> JSONResponse:
    """List every ``*.detection.json`` written by the deterministic pipeline."""
    return await _safe_pass("GET", "detect/deterministic/results")


@router.post("/deterministic")
async def run_detection(req: DetectRequest) -> JSONResponse:
    """Run the deterministic detector on a single GeoTIFF.

    `tif_path` is absolute. The `wind_speed_ms` parameter removes the
    `LOW_CONFIDENCE_NO_WIND` flag when within the viability band (2..10 m/s).
    """
    return await _safe_pass(
        "POST",
        "detect/deterministic",
        json=req.model_dump(),
    )


@router.post("/deterministic/run_all")
async def run_all(wind_speed_ms: float | None = Query(None)) -> JSONResponse:
    """Re-run the detector on every GeoTIFF under ``data/sar``.

    Returns the same compact summary as ``/results`` so the UI can refresh
    after a bulk run without an extra round-trip.
    """
    params = {"wind_speed_ms": wind_speed_ms} if wind_speed_ms is not None else None
    return await _safe_pass("POST", "detect/deterministic/run_all", params=params)
