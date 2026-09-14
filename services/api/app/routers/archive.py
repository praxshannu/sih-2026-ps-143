"""Gateway proxy for the in-platform Copernicus archive browser.

The API gateway is the only service the UI talks to (per AGENTS.md: inter-service
calls go through sentinel-api). The real Copernicus client lives in the ingest
service, which owns data sources. This router is a thin, honest proxy — it
forwards upstream status codes and error bodies rather than masking them, so the
UI can show the operator exactly what Copernicus said.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/archive", tags=["archive"])

INGEST_SERVICE_URL = os.getenv("INGEST_SERVICE_URL", "http://sentinel-ingest:8001")
PROXY_TIMEOUT = float(os.getenv("ARCHIVE_PROXY_TIMEOUT", "180"))

# Long enough to cover a 2048² Sentinel Hub render without being indefinite.
_UPSTREAM_ERROR = (
    "Archive service unreachable. Is the ingest service running with "
    "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET set?"
)


class IngestRequest(BaseModel):
    min_lon: float | None = None
    min_lat: float | None = None
    max_lon: float | None = None
    max_lat: float | None = None
    bbox: str | None = Field(
        None, description="deprecated 'west,south,east,north'; use the named axis fields"
    )
    start: str
    end: str
    size: int = Field(2048, ge=256, le=4096)
    label: str = ""
    scene_name: str | None = None


def _bbox_params(
    min_lon: float | None,
    min_lat: float | None,
    max_lon: float | None,
    max_lat: float | None,
    bbox: str | None,
) -> dict[str, object]:
    """Forward an AOI using named axes whenever the caller supplied them.

    The positional ``west,south,east,north`` string is ambiguous — a lat-first
    caller produces four values that are legal in either slot, so it passes
    validation and silently relocates the AOI. Named axes cannot be misordered,
    so they are what we put on the wire.
    """
    if any(v is not None for v in (min_lon, min_lat, max_lon, max_lat)):
        return {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        }
    return {"bbox": bbox}


async def _proxy_get(path: str, params: dict[str, object]) -> Response:
    url = f"{INGEST_SERVICE_URL}/archive/{path}"
    # httpx renders None as a bare `key=`, which upstream then fails to parse
    # as a float/bool. Absent must mean absent, not "empty string".
    clean = {k: v for k, v in params.items() if v is not None}
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            resp = await client.get(url, params=clean)
    except httpx.RequestError as exc:
        logger.error("Archive proxy GET {} failed: {}", path, exc)
        raise HTTPException(503, _UPSTREAM_ERROR) from exc
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, _safe_detail(resp))
    return Response(
        content=resp.content,
        status_code=200,
        media_type=resp.headers.get("content-type", "application/json"),
    )


def _safe_detail(resp: httpx.Response) -> str:
    """Pass the upstream message through, truncated so we never echo a huge body."""
    try:
        payload = resp.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)[:400]
    except Exception:  # noqa: BLE001 - non-JSON error body
        pass
    return (resp.text or "upstream error")[:400]


@router.get("/health")
async def archive_health() -> dict[str, object]:
    """Whether Copernicus credentials are configured upstream."""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{INGEST_SERVICE_URL}/archive/health")
    except httpx.RequestError as exc:
        logger.error("Archive health check failed: {}", exc)
        raise HTTPException(503, _UPSTREAM_ERROR) from exc
    try:
        return resp.json()
    except ValueError:
        # A proxy or a non-FastAPI listener answering for the ingest host
        # returns HTML. Surface that instead of an opaque 500.
        logger.error(
            "Archive health returned non-JSON (status {}): {}",
            resp.status_code,
            resp.text[:200],
        )
        raise HTTPException(
            503,
            f"{_UPSTREAM_ERROR} (upstream replied with non-JSON, status "
            f"{resp.status_code})",
        ) from None


@router.get("/search")
async def search(
    start: str = Query(...),
    end: str = Query(...),
    min_lon: float | None = Query(None, description="west edge (degrees)"),
    min_lat: float | None = Query(None, description="south edge (degrees)"),
    max_lon: float | None = Query(None, description="east edge (degrees)"),
    max_lat: float | None = Query(None, description="north edge (degrees)"),
    bbox: str | None = Query(None, description="deprecated 'west,south,east,north'"),
    product_type: str | None = Query("GRD"),
    platform: str | None = Query(None),
    mode: str | None = Query(None),
    top: int = Query(50, ge=1, le=100),
) -> Response:
    """Real Sentinel-1 catalogue search with GeoJSON footprints."""
    return await _proxy_get(
        "search",
        {
            **_bbox_params(min_lon, min_lat, max_lon, max_lat, bbox),
            "start": start,
            "end": end,
            "product_type": product_type,
            "platform": platform,
            "mode": mode,
            "top": top,
        },
    )


@router.get("/quicklook")
async def quicklook(
    start: str = Query(...),
    end: str = Query(...),
    min_lon: float | None = Query(None),
    min_lat: float | None = Query(None),
    max_lon: float | None = Query(None),
    max_lat: float | None = Query(None),
    bbox: str | None = Query(None, description="deprecated 'west,south,east,north'"),
    size: int = Query(512, ge=128, le=1024),
) -> Response:
    """False-colour SAR preview rendered by Copernicus."""
    return await _proxy_get(
        "quicklook",
        {
            **_bbox_params(min_lon, min_lat, max_lon, max_lat, bbox),
            "start": start,
            "end": end,
            "size": size,
        },
    )


@router.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, object]:
    """Fetch a real GeoTIFF for an AOI into the local catalogue."""
    url = f"{INGEST_SERVICE_URL}/archive/ingest"
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            resp = await client.post(url, json=req.model_dump())
    except httpx.RequestError as exc:
        logger.error("Archive ingest proxy failed: {}", exc)
        raise HTTPException(503, _UPSTREAM_ERROR) from exc
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, _safe_detail(resp))
    return resp.json()


@router.get("/ais/coverage")
async def ais_coverage(
    min_lon: float | None = Query(None, description="west edge (degrees)"),
    min_lat: float | None = Query(None, description="south edge (degrees)"),
    max_lon: float | None = Query(None, description="east edge (degrees)"),
    max_lat: float | None = Query(None, description="north edge (degrees)"),
    bbox: str | None = Query(None, description="deprecated 'west,south,east,north'"),
) -> Response:
    """Pre-flight verdict: does this AOI have real AIS, or will it be synthetic?

    The UI calls this as soon as an AOI is set so the operator is warned before
    any vessel data is requested. Pass the AOI as named axes — a positional
    bbox string can be silently misread as lat-first.
    """
    return await _proxy_get(
        "ais/coverage", _bbox_params(min_lon, min_lat, max_lon, max_lat, bbox)
    )


@router.get("/ais")
async def ais(
    start: str = Query(...),
    end: str = Query(...),
    min_lon: float | None = Query(None, description="west edge (degrees)"),
    min_lat: float | None = Query(None, description="south edge (degrees)"),
    max_lon: float | None = Query(None, description="east edge (degrees)"),
    max_lat: float | None = Query(None, description="north edge (degrees)"),
    bbox: str | None = Query(None, description="deprecated 'west,south,east,north'"),
    n_vessels: int = Query(6, ge=1, le=8),
    seed: int = Query(20200725),
    acknowledge_synthetic: bool = Query(False),
) -> Response:
    """Vessels for the AOI. Real AIS where coverage exists; otherwise a
    labelled synthetic feed (response carries ``provenance="synthetic_mock"``,
    a ``disclaimer`` block, and a ``coverage`` verdict — the UI must render the
    warning banner).
    """
    return await _proxy_get(
        "ais",
        {
            **_bbox_params(min_lon, min_lat, max_lon, max_lat, bbox),
            "start": start,
            "end": end,
            "n_vessels": n_vessels,
            "seed": seed,
            "acknowledge_synthetic": acknowledge_synthetic,
        },
    )
