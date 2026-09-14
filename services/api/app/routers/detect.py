"""Gateway proxy for the deterministic Tier-A detector.

The detect service runs the Solberg two-gate adaptive operator on every SAR
GeoTIFF in ``data/sar/``. This router forwards the four exposed endpoints so
the UI talks to the API gateway only (per AGENTS.md).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/detect", tags=["detect"])

DETECT_SERVICE_URL = os.getenv("DETECT_SERVICE_URL", "http://sentinel-detect:8002")
PROXY_TIMEOUT = float(os.getenv("DETECT_PROXY_TIMEOUT", "300"))

# Uploads are a different shape of request from the JSON proxies: the body can
# be a gigabyte, so it is streamed rather than buffered, and the read timeout
# has to cover inference *after* the transfer, not just the transfer.
UPLOAD_TIMEOUT = float(os.getenv("DETECT_UPLOAD_TIMEOUT", "900"))

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


# ── Scene inference (validate → tile → segment → vectorise → explain) ──────


class SceneRequest(BaseModel):
    """Run the full pipeline on a scene already on disk."""

    path: str
    wind_speed_ms: float | None = None
    require_wind: bool = False
    tiled: bool = True
    with_evidence: bool = True


def _clean(params: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop None values before proxying.

    httpx serialises ``None`` as a bare ``key=``, which the upstream then fails
    to parse as a float. Omitting the parameter is the only correct encoding of
    "not supplied".
    """
    if not params:
        return None
    cleaned = {k: v for k, v in params.items() if v is not None}
    return cleaned or None


@router.get("/scene/health")
async def scene_health() -> JSONResponse:
    """Detector identity plus the UNet++ status the service will report."""
    return await _safe_pass("GET", "detect/scene/health")


@router.get("/scene/list")
async def scene_list() -> JSONResponse:
    """Scenes on disk, each flagged with whether inference already ran."""
    return await _safe_pass("GET", "detect/scene/list")


@router.get("/scene/result")
async def scene_result(
    path: str = Query(..., description="Scene path or name inside data/sar/"),
) -> JSONResponse:
    """Read a persisted inference result without re-running anything."""
    return await _safe_pass("GET", "detect/scene/result", params=_clean({"path": path}))


@router.post("/scene")
async def scene_infer(req: SceneRequest) -> JSONResponse:
    """Run inference on a scene already on disk."""
    return await _safe_pass("POST", "detect/scene", json=req.model_dump())


@router.post("/scene/upload")
async def scene_upload(
    file: UploadFile = File(..., description="Sentinel-1 GeoTIFF (.tif/.tiff)"),
    wind_speed_ms: float | None = Query(None),
    require_wind: bool = Query(False),
) -> JSONResponse:
    """Stream an analyst's GeoTIFF through to the detect service.

    The body is streamed in chunks rather than read into memory: a full IW GRDH
    scene is around a gigabyte, and the gateway has no business holding that.
    """
    url = f"{DETECT_SERVICE_URL}/detect/scene/upload"
    params = _clean({"wind_speed_ms": wind_speed_ms, "require_wind": require_wind})
    filename = file.filename or "upload.tif"

    # The browser sends multipart (a normal <input type="file">), but the
    # upstream takes a raw body. Re-frame rather than buffer: the file is
    # streamed through in 1 MB chunks, so a 1 GB scene never sits in gateway
    # memory. `require_wind=False` is the default and is dropped by _clean.
    try:
        async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT) as client:
            resp = await client.post(
                url,
                params=params,
                content=_stream_upload(file),
                headers={
                    "content-type": "application/octet-stream",
                    "x-sentinel-filename": filename,
                },
            )
    except httpx.RequestError as exc:
        logger.error("Detect upload proxy failed: {}", exc)
        raise HTTPException(503, "Detect service unreachable") from exc
    finally:
        await file.close()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text[:300])
        except Exception:  # noqa: BLE001
            detail = (resp.text or "upstream error")[:300]
        raise HTTPException(resp.status_code, detail)

    return JSONResponse(content=resp.json(), headers={"cache-control": "no-store"})


async def _stream_upload(file: UploadFile):
    """Yield the upload in 1 MB chunks so the gateway never buffers it whole."""
    while chunk := await file.read(1024 * 1024):
        yield chunk


@router.delete("/scene/{name}")
async def scene_delete(name: str) -> JSONResponse:
    """Remove a local scene and its sidecars."""
    return await _safe_pass("DELETE", f"detect/scene/{name}")
