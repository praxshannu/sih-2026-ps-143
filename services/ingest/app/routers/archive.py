"""Archive browser endpoints — the in-platform Copernicus search UI.

Every response is proxied from a live Copernicus call. There is no cache of
"nice looking" results and no synthetic fallback: an empty archive returns an
empty list, and a failed upstream returns an error the UI shows verbatim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from loguru import logger
from pydantic import BaseModel, Field
from rasterio.transform import from_bounds

from ..sources.cdse import get_cdse_client

router = APIRouter(prefix="/archive", tags=["archive"])

DATA_DIR = Path(os.getenv("SENTINEL_DATA_DIR", "/app/data"))
SAR_DIR = DATA_DIR / "sar"
SAR_DIR.mkdir(parents=True, exist_ok=True)

BBOX_RE = re.compile(r"^-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?,-?\d+(\.\d+)?$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def _validate_bbox(w: float, s: float, e: float, n: float) -> tuple[float, float, float, float]:
    """Range-check an already-ordered (west, south, east, north) box."""
    if not (-180 <= w < e <= 180):
        raise HTTPException(422, "bbox longitudes must satisfy -180 <= west < east <= 180")
    if not (-90 <= s < n <= 90):
        raise HTTPException(422, "bbox latitudes must satisfy -90 <= south < north <= 90")
    return (w, s, e, n)


def _parse_bbox(raw: str) -> tuple[float, float, float, float]:
    """Parse the deprecated `west,south,east,north` string.

    .. warning::
       This form is **ambiguous and unsafe**. A lat-first caller sending
       ``south,west,north,east`` produces four numbers that are all legal in
       either slot, so no validator can tell the two apart — the AOI is
       silently relocated and the AIS real/synthetic verdict flips. Prefer the
       named parameters (``min_lon``/``min_lat``/``max_lon``/``max_lat``).
    """
    if not BBOX_RE.match(raw or ""):
        raise HTTPException(422, "bbox must be 'west,south,east,north' in decimal degrees")
    w, s, e, n = (float(x) for x in raw.split(","))
    return _validate_bbox(w, s, e, n)


def _resolve_bbox(
    min_lon: float | None,
    min_lat: float | None,
    max_lon: float | None,
    max_lat: float | None,
    bbox: str | None,
) -> tuple[float, float, float, float]:
    """Resolve an AOI from named axis params, falling back to the legacy string.

    Named parameters win whenever any of them is supplied, because they cannot
    be misordered. Partial sets are rejected rather than guessed at.
    """
    named = (min_lon, min_lat, max_lon, max_lat)
    if any(v is not None for v in named):
        if any(v is None for v in named):
            raise HTTPException(
                422,
                "min_lon, min_lat, max_lon and max_lat must be supplied together",
            )
        return _validate_bbox(
            float(min_lon), float(min_lat), float(max_lon), float(max_lat)  # type: ignore[arg-type]
        )
    if not bbox:
        raise HTTPException(
            422,
            "supply min_lon, min_lat, max_lon and max_lat "
            "(the 'west,south,east,north' bbox string is deprecated: it cannot "
            "be validated against axis-order mistakes)",
        )
    logger.warning("Deprecated 'bbox' string used for an AOI — use named axis params")
    return _parse_bbox(bbox)


def _parse_iso(raw: str, field_name: str) -> str:
    if not ISO_RE.match(raw or ""):
        raise HTTPException(422, f"{field_name} must be ISO-8601 UTC, e.g. 2020-08-10T00:00:00Z")
    return raw


class SearchResponse(BaseModel):
    count: int
    bbox: list[float]
    window: list[str]
    scenes: list[dict[str, Any]]
    source: str = "CDSE OData catalogue"


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


@router.get("/health")
async def archive_health() -> dict[str, Any]:
    """Whether Copernicus credentials are present and the catalogue is reachable."""
    client = get_cdse_client()
    if not client.configured:
        return {
            "configured": False,
            "catalogue": "unreachable",
            "detail": "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET missing from environment",
        }
    try:
        await client._get_token()
        return {"configured": True, "catalogue": "reachable", "detail": None}
    except Exception as exc:  # noqa: BLE001 - surface the real upstream error
        return {"configured": True, "catalogue": "error", "detail": str(exc)[:300]}


@router.get("/search", response_model=SearchResponse)
async def search(
    start: str = Query(..., description="ISO UTC, e.g. 2020-08-01T00:00:00Z"),
    end: str = Query(..., description="ISO UTC, e.g. 2020-08-31T00:00:00Z"),
    min_lon: float | None = Query(None, description="west edge (degrees)"),
    min_lat: float | None = Query(None, description="south edge (degrees)"),
    max_lon: float | None = Query(None, description="east edge (degrees)"),
    max_lat: float | None = Query(None, description="north edge (degrees)"),
    bbox: str | None = Query(None, description="deprecated 'west,south,east,north'"),
    product_type: str | None = Query("GRD"),
    platform: str | None = Query(None, description="S1A | S1B | S1C | S1D"),
    mode: str | None = Query(None, description="IW | EW | S1..S6 | WV"),
    top: int = Query(50, ge=1, le=100),
) -> SearchResponse:
    """Real catalogue search, with footprints as GeoJSON for the map."""
    box = _resolve_bbox(min_lon, min_lat, max_lon, max_lat, bbox)
    start_iso = _parse_iso(start, "start")
    end_iso = _parse_iso(end, "end")
    client = get_cdse_client()
    try:
        scenes = await client.search(
            box, start_iso, end_iso, product_type=product_type,
            platform=platform, mode=mode, top=top,
        )
    except RuntimeError as exc:
        # Credentials absent — say so plainly instead of returning fake scenes.
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("CDSE search failed: {}", exc)
        raise HTTPException(502, f"Copernicus catalogue error: {str(exc)[:300]}") from exc

    return SearchResponse(
        count=len(scenes),
        bbox=list(box),
        window=[start_iso, end_iso],
        scenes=[s.to_dict() for s in scenes],
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
    """False-colour SAR preview (VV, VH, VV−VH) rendered server-side."""
    box = _resolve_bbox(min_lon, min_lat, max_lon, max_lat, bbox)
    start_iso = _parse_iso(start, "start")
    end_iso = _parse_iso(end, "end")
    client = get_cdse_client()
    try:
        png = await client.quicklook(box, start_iso, end_iso, size=size)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("CDSE quicklook failed: {}", exc)
        raise HTTPException(502, f"Copernicus rendering error: {str(exc)[:300]}") from exc
    if not png[:4] == b"\x89PNG":
        raise HTTPException(502, "Copernicus did not return a PNG for this AOI/window")
    return Response(content=png, media_type="image/png")


@router.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, Any]:
    """Fetch a real GeoTIFF into data/sar/ and return its provenance record."""
    box = _resolve_bbox(req.min_lon, req.min_lat, req.max_lon, req.max_lat, req.bbox)
    start_iso = _parse_iso(req.start, "start")
    end_iso = _parse_iso(req.end, "end")
    client = get_cdse_client()

    try:
        tiff = await client.geotiff(box, start_iso, end_iso, size=req.size)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("CDSE geotiff failed: {}", exc)
        raise HTTPException(502, f"Copernicus rendering error: {str(exc)[:300]}") from exc

    if tiff[:2] not in (b"II", b"MM"):
        raise HTTPException(502, "Copernicus did not return a TIFF for this AOI/window")

    raw_path = SAR_DIR / ".tmp_download.tif"
    raw_path.write_bytes(tiff)
    try:
        with rasterio.open(raw_path) as src:
            arr = src.read()
        vv = arr[0]
        valid = vv[np.isfinite(vv) & (vv > 0)]
        if valid.size == 0:
            raise HTTPException(
                422,
                "No Sentinel-1 swath covers this AOI in the requested window — "
                "nothing was saved (SENTINEL never substitutes synthetic pixels).",
            )
        db = 10.0 * np.log10(valid)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        key = re.sub(r"[^a-z0-9_]+", "_", (req.label or "archive_scene").lower()).strip("_") or "scene"
        out_path = SAR_DIR / f"{key}_{stamp}.tif"
        w, s, e, n = box
        with rasterio.open(
            out_path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
            count=arr.shape[0], dtype="float32", crs="EPSG:4326",
            transform=from_bounds(w, s, e, n, arr.shape[2], arr.shape[1]),
            compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512,
        ) as dst:
            dst.write(arr)
            dst.descriptions = ("sigma0_VV_linear", "sigma0_VH_linear", "dataMask")
            dst.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)

        record = {
            "key": out_path.stem,
            "label": req.label or out_path.stem,
            "bbox": list(box),
            "time_window": [start_iso, end_iso],
            "bands": ["sigma0_VV_linear", "sigma0_VH_linear", "dataMask"],
            "width": int(arr.shape[2]),
            "height": int(arr.shape[1]),
            "bytes": out_path.stat().st_size,
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "scene_name": req.scene_name,
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "source": "CDSE Sentinel Hub Process API (Copernicus Data Space Ecosystem)",
            "stats": {
                "vv_db_p05": round(float(np.percentile(db, 5)), 2),
                "vv_db_median": round(float(np.percentile(db, 50)), 2),
                "vv_db_p95": round(float(np.percentile(db, 95)), 2),
                "valid_fraction": round(float(valid.size / vv.size), 4),
            },
        }
        (SAR_DIR / f"{out_path.stem}.json").write_text(json.dumps(record, indent=2))
        logger.info("Archive ingest -> {} ({} bytes)", out_path.name, record["bytes"])
        return record
    finally:
        raw_path.unlink(missing_ok=True)


# ── AIS coverage verdict + synthetic fallback (open Indian Ocean only) ───
from ..sources.ais_synthetic import (  # noqa: E402
    coverage_verdict,
    generate_synthetic_ais,
    should_use_synthetic,
    synthetic_disclaimer,
)


@router.get("/ais/coverage")
async def ais_coverage(
    min_lon: float | None = Query(None, description="west edge (degrees)"),
    min_lat: float | None = Query(None, description="south edge (degrees)"),
    max_lon: float | None = Query(None, description="east edge (degrees)"),
    max_lat: float | None = Query(None, description="north edge (degrees)"),
    bbox: str | None = Query(None, description="deprecated 'west,south,east,north'"),
) -> dict[str, Any]:
    """Decide, before any data is fetched, whether an AOI has real AIS.

    The UI calls this the moment an AOI is set, so the operator is told up
    front — not after the fact — that the vessel layer for an Indian Ocean
    area will be simulated.

    The AOI is taken from the named axis parameters. These — not a positional
    ``west,south,east,north`` string — are the supported form, because a
    lat-first string cannot be distinguished from a correct one and silently
    relocates the AOI, flipping this verdict.
    """
    box = _resolve_bbox(min_lon, min_lat, max_lon, max_lat, bbox)
    verdict = coverage_verdict(box)
    verdict["disclaimer"] = synthetic_disclaimer() if verdict["use_synthetic"] else None
    return verdict


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
    acknowledge_synthetic: bool = Query(
        False,
        description="Operator confirmation that the synthetic-data notice was seen. "
        "The endpoint still returns data when False — it never silently hides the "
        "problem — but the flag is echoed so the UI can log the acknowledgement.",
    ),
) -> dict[str, Any]:
    """Vessel positions for the AOI. Real AIS when coverage exists;
    clearly-labelled synthetic data when it does not (open Indian Ocean).

    The response always carries ``provenance`` and ``disclaimer``. When
    ``provenance == "synthetic_mock"`` the UI must render the disclaimer as a
    blocking banner. There is no quiet path: the notice, the reason, and the
    machine-readable provenance all travel with the payload itself.
    """
    box = _resolve_bbox(min_lon, min_lat, max_lon, max_lat, bbox)
    start_iso = _parse_iso(start, "start")
    end_iso = _parse_iso(end, "end")
    start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    if end_dt <= start_dt:
        raise HTTPException(422, "end must be after start")

    verdict = coverage_verdict(box)

    if not should_use_synthetic(box):
        # In regions with real coverage we do not serve synthetic data at all.
        return {
            "provenance": "live_terrestrial",
            "is_synthetic": False,
            "bbox": list(box),
            "window": [start_iso, end_iso],
            "count": 0,
            "vessels": [],
            "notice": None,
            "disclaimer": None,
            "coverage": verdict,
            "reason": (
                "Real AIS coverage is available for this region; the UI "
                "should subscribe to AISStream directly via the gateway, "
                "not consume this synthetic endpoint."
            ),
        }

    payload = generate_synthetic_ais(
        box, start_dt, end_dt, n_vessels=n_vessels, seed=seed,
    )
    payload["disclaimer"] = synthetic_disclaimer()
    payload["coverage"] = verdict
    payload["acknowledged"] = bool(acknowledge_synthetic)
    logger.warning(
        "Served SYNTHETIC AIS for bbox={} window={}..{} — no real coverage exists",
        box, start_iso, end_iso,
    )
    return payload
