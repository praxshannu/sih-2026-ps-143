"""Gateway proxy for the OpenDrift backward-attribution service.

The drift service runs an OpenOil hindcast *backwards in time* from a detection
polygon: it pulls ERA5 10 m wind + CMEMS GLORYS12 currents for the backtrack
window, releases an ensemble in the past, and reports where the surviving
particles cluster (the origin ellipse) plus which vessels were inside it.

This router keeps the UI talking to the gateway only (per AGENTS.md).
It is deliberately separate from ``routers/drift.py`` — that one is the
case-scoped ``/cases/{id}/drift/*`` stub; this one is the live
``/drift/*`` hindcast engine.

Upstream contract (services/drift/app/routers/attribution.py):
    GET  /drift/health
    POST /drift/attribution
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/drift", tags=["drift"])

DRIFT_SERVICE_URL = os.getenv("DRIFT_SERVICE_URL", "http://sentinel-drift:8003")
# An ERA5 + CMEMS fetch plus a 64-member ensemble routinely takes 60-180 s, so
# this has to be generous — a 30 s default would cut off legitimate runs.
ATTRIBUTION_TIMEOUT = float(os.getenv("DRIFT_PROXY_TIMEOUT", "600"))


class ForecastRequest(BaseModel):
    """Mirrors ``services/drift/app/routers/attribution.py:ForecastRequest``."""

    origin_lon: float
    origin_lat: float
    origin_time: str = Field(..., description="ISO-8601 UTC")
    seed_radius_km: float = Field(1.0, ge=0.05, le=100.0)
    duration_h: float = Field(48.0, ge=1, le=168)
    use_era5: bool = True
    use_cmems: bool = True
    forcing: str = Field("auto", description="Wind source: auto | era5 | gfs")
    bbox: tuple[float, float, float, float] | None = None
    # Off => faster and no GSHHG, but stranded_fraction is meaningless.
    use_landmask: bool = True
    n_members: int = Field(64, ge=16, le=2048)
    seed: int = 20200725
    synthetic_wind_ms: float = 5.0
    synthetic_wind_dir_from_deg: float = 110.0
    synthetic_current_ms: float = 0.10
    synthetic_current_dir_deg: float = 220.0


class AttributionRequest(BaseModel):
    """Mirrors ``services/drift/app/routers/attribution.py:AttributionRequest``.

    Kept structurally identical so an upstream contract change is a single
    edit here rather than a silent field drop.
    """

    detection_lon: float
    detection_lat: float
    detection_area_km2: float = Field(..., ge=0.001, le=200.0)
    detection_time: str = Field(..., description="ISO-8601 UTC")

    duration_h: float = Field(48.0, ge=1, le=168)

    vessels: list[dict[str, Any]] | None = Field(
        default=None,
        description="Vessels to score. Each: {mmsi, name, longitude, latitude, flag?, provenance?}",
    )

    use_era5: bool = Field(True, description="Pull 10m wind for the window")
    use_cmems: bool = Field(True, description="Pull CMEMS GLORYS12 currents for the window")
    # "auto" picks GFS for recent windows (no CDS queue, ~20 s) and ERA5 for
    # anything older; GFS only retains ~10 days on NOMADS.
    forcing: str = Field("auto", description="Wind source: auto | era5 | gfs")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="(W,S,E,N) override for the forcing bbox; default = a 5° box around the detection",
    )

    n_members: int = Field(64, ge=16, le=2048)
    seed: int = 20200725

    synthetic_wind_ms: float = 5.0
    synthetic_wind_dir_from_deg: float = 110.0
    synthetic_current_ms: float = 0.10
    synthetic_current_dir_deg: float = 220.0


async def _safe_pass(method: str, path: str, **kwargs) -> JSONResponse:
    url = f"{DRIFT_SERVICE_URL}/{path}"
    try:
        async with httpx.AsyncClient(timeout=ATTRIBUTION_TIMEOUT) as c:
            resp = await c.request(method, url, **kwargs)
    except httpx.TimeoutException as exc:
        logger.error("Drift proxy {} {} timed out after {}s", method, path, ATTRIBUTION_TIMEOUT)
        raise HTTPException(
            504,
            f"Attribution timed out after {ATTRIBUTION_TIMEOUT:.0f}s — "
            "try a shorter backtrack window or fewer ensemble members",
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Drift proxy {} {} failed: {}", method, path, exc)
        raise HTTPException(503, "Drift/attribution service unreachable") from exc
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text[:500])
        except Exception:  # noqa: BLE001
            detail = (resp.text or "upstream error")[:500]
        raise HTTPException(resp.status_code, detail)
    return JSONResponse(content=resp.json(), headers={"cache-control": "no-store"})


@router.get("/health")
async def drift_health() -> JSONResponse:
    """Drift service health — confirms OpenDrift is importable upstream."""
    return await _safe_pass("GET", "drift/health")


@router.post("/attribution")
async def run_attribution(req: AttributionRequest) -> JSONResponse:
    """Backward hindcast from a detection to its most likely origin.

    Returns the origin ellipse (2-sigma PCA axes + p50/p95 radii), the suspect
    vessel list ranked by distance to that ellipse, and the forcing provenance
    (``era5`` vs ``synthetic_constant``) so the UI can label the confidence
    honestly.
    """
    return await _safe_pass("POST", "drift/attribution", json=req.model_dump())


@router.post("/forecast")
async def run_forecast(req: ForecastRequest) -> JSONResponse:
    """Forward projection from a known origin, with shoreline impact.

    Feed it the origin ellipse from ``/drift/attribution`` to get the full
    "where did it come from -> where is it going" chain. Returns an hourly
    cone plus ``stranded_fraction`` and ``first_stranding_h``.

    Note that heavy beaching is normal and is the headline, not an error:
    stranding deactivates particles, so the late cone is fitted to few
    survivors. The runner flags that explicitly in ``notes``.
    """
    return await _safe_pass("POST", "drift/forecast", json=req.model_dump())
