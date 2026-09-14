"""Scene inference over HTTP — the analyst's front door.

``app.pipeline`` owns the science; this module only owns transport. Its job is
to make the pipeline's state machine reachable from the UI without letting
transport concerns leak into the result:

* inference runs in a threadpool (``AGENTS.md``), never on the event loop, so a
  4000x4000 scene cannot stall every other request;
* an uploaded file is validated by *name and size* here and by *content* in
  ``pipeline.validate_scene``. Transport-level checks are cheap guards, not a
  substitute for the real validation, and the response always carries the
  pipeline's own verdict;
* every completed run is written next to the scene as ``<stem>.inference.json``
  so an analyst can reopen a case after a restart without re-running inference
  — the result on disk is the same object the API returned, so the two cannot
  disagree.

The state machine is deliberately not collapsed here. A caller that receives
``no_detection`` must be able to tell it apart from ``invalid_scene`` and from
``missing_forcing_data``; mapping those onto an empty list would destroy the
one distinction this whole service exists to preserve.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.pipeline import DETECTOR_ID, DetectPipeline, PipelineOptions

router = APIRouter(prefix="/detect/scene", tags=["detect:scene"])

# services/detect/app/routers/scene.py -> 5 levels up is the project root.
ROOT = Path(__file__).resolve().parents[4]
SAR_DIR = ROOT / "data" / "sar"

ALLOWED_SUFFIXES = {".tif", ".tiff"}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024  # 512 MB; a full IW GRDH scene is ~1 GB
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

_pipeline: DetectPipeline | None = None


def get_pipeline() -> DetectPipeline:
    """One pipeline per process, built on first use.

    ``DetectPipeline`` resolves the UNet++ status in its constructor, and doing
    that per request would re-stat the checkpoint on every call.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectPipeline()
    return _pipeline


def _safe_stem(name: str) -> str:
    """Reduce an uploaded filename to something that cannot escape SAR_DIR.

    Path traversal via ``../../etc/passwd`` is the obvious attack; the subtler
    one is a name that is legal on the uploader's platform but not here. Only
    ``[A-Za-z0-9._-]`` survives, and the result can never be a bare dot-name.
    """
    stem = Path(name).stem
    cleaned = _UNSAFE.sub("_", stem).strip("._-")
    return cleaned or f"scene_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"


def _result_path(tif_path: Path) -> Path:
    return tif_path.with_suffix(".inference.json")


def _persist(tif_path: Path, payload: dict[str, Any]) -> str | None:
    """Write the inference result beside the scene. Never fail the request.

    The API response is the product; the sidecar is a convenience for reopening
    a case offline. If the disk refuses, the analyst still gets their answer and
    the response says persistence failed rather than silently claiming it
    worked.
    """
    out = _result_path(tif_path)
    try:
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(out)
    except OSError as exc:
        logger.warning("could not persist inference result for {}: {}", tif_path, exc)
        return None


class SceneRequest(BaseModel):
    """Run inference on a scene already on disk."""

    path: str = Field(..., description="Absolute path, or a name inside data/sar/")
    wind_speed_ms: float | None = Field(
        None, description="10 m wind speed. Omit when unknown — do not guess."
    )
    require_wind: bool = Field(
        False, description="Fail with missing_forcing_data rather than running wind-blind"
    )
    tiled: bool = True
    with_evidence: bool = True


def _resolve_scene(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = SAR_DIR / candidate
    return candidate


def _run(tif_path: Path, req: SceneRequest) -> dict[str, Any]:
    options = PipelineOptions(
        wind_speed_ms=req.wind_speed_ms,
        require_wind=req.require_wind,
        tiled=req.tiled,
        with_evidence=req.with_evidence,
    )
    result = get_pipeline().run(tif_path, options)
    result["result_file"] = _persist(tif_path, result)
    return result


@router.get("/health")
async def scene_health() -> dict[str, Any]:
    """What the detector can honestly claim right now."""
    pipeline = get_pipeline()
    return {
        "ok": True,
        "service": "sentinel-detect",
        "detector": DETECTOR_ID,
        "sar_dir": str(SAR_DIR),
        "scene_count": sum(1 for _ in SAR_DIR.glob("*.tif")) if SAR_DIR.is_dir() else 0,
        "model": pipeline.model_status.as_dict(),
    }


@router.get("/list")
async def list_scenes() -> dict[str, Any]:
    """Scenes available on disk, with whether inference has already run.

    ``has_result`` is what lets the UI offer "reopen" instead of silently
    re-running a multi-second inference.
    """
    scenes: list[dict[str, Any]] = []
    if SAR_DIR.is_dir():
        for tif in sorted(SAR_DIR.glob("*.tif")):
            result_file = _result_path(tif)
            entry: dict[str, Any] = {
                "name": tif.name,
                "path": str(tif),
                "bytes": tif.stat().st_size,
                "modified_utc": datetime.fromtimestamp(tif.stat().st_mtime, UTC).isoformat(),
                "has_result": result_file.is_file(),
            }
            if result_file.is_file():
                try:
                    cached = json.loads(result_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    entry["result_state"] = "unreadable"
                else:
                    entry["result_state"] = cached.get("state")
                    entry["result_count"] = cached.get("count")
                    entry["result_ran_utc"] = cached.get("ran_utc")
            scenes.append(entry)
    return {"count": len(scenes), "sar_dir": str(SAR_DIR), "scenes": scenes}


@router.post("")
async def infer_scene(req: SceneRequest) -> dict[str, Any]:
    """Run inference on a scene already on disk."""
    tif_path = _resolve_scene(req.path)
    if not tif_path.is_file():
        # 404 with the resolved path: the caller needs to know where we looked,
        # because "not found" is almost always an AOI/path mix-up, not a bug.
        raise HTTPException(
            status_code=404,
            detail=f"scene not found: {tif_path}",
        )
    return await run_in_threadpool(_run, tif_path, req)


@router.get("/result")
async def cached_result(
    path: str = Query(..., description="Scene path or name inside data/sar/"),
) -> dict[str, Any]:
    """Return the persisted inference result without re-running anything."""
    tif_path = _resolve_scene(path)
    result_file = _result_path(tif_path)
    if not result_file.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"no persisted inference result for {tif_path.name}; POST /detect/scene first",
        )
    try:
        return json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"result unreadable: {exc}") from exc


@router.post("/upload")
async def upload_and_infer(
    request: Request,
    filename: str = Header(
        ...,
        alias="x-sentinel-filename",
        description="Original filename; only its suffix and a sanitised stem are used",
    ),
    wind_speed_ms: float | None = Query(None),
    require_wind: bool = Query(False),
) -> dict[str, Any]:
    """Accept a local GeoTIFF as a **raw body**, validate it, and run inference.

    The body is the file itself, not a multipart envelope. Two reasons:

    1. A full IW GRDH scene is ~1 GB. Multipart would mean the gateway parses
       and re-encodes that body just to pass it through, and a streamed upload
       cannot be re-framed as valid multipart without buffering it.
    2. Raw bytes are the same shape end to end, so the gateway can stream
       straight from the client socket to this service with no intermediate
       copy.

    The filename arrives in the ``x-sentinel-filename`` header, which is the
    only part of the multipart envelope actually used here.

    The scene is stored under ``data/sar/`` so the bytes, the provenance
    sidecar and the inference result stay together — an analyst reopening the
    case tomorrow gets the same bytes that produced the answer.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported file type {suffix or '(none)'!r}; expected a GeoTIFF "
                f"({', '.join(sorted(ALLOWED_SUFFIXES))})"
            ),
        )

    SAR_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tif_path = SAR_DIR / f"{_safe_stem(filename)}_{stamp}.tif"

    written = 0
    try:
        with tif_path.open("wb") as sink:
            async for chunk in request.stream():
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                sink.write(chunk)
    except HTTPException:
        tif_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tif_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"could not store upload: {exc}") from exc

    if written == 0:
        tif_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    logger.info("scene upload accepted: {} ({} bytes)", tif_path.name, written)

    req = SceneRequest(path=str(tif_path), wind_speed_ms=wind_speed_ms, require_wind=require_wind)
    # If inference fails unexpectedly the scene is deliberately left on disk:
    # it is real data the operator chose to upload, and re-uploading a 1 GB
    # GeoTIFF to retry is a worse outcome than a stray file.
    return await run_in_threadpool(_run, tif_path, req)


@router.delete("/{name}")
async def delete_scene(name: str) -> dict[str, Any]:
    """Remove a scene, its inference sidecar and its provenance sidecar."""
    tif_path = SAR_DIR / Path(name).name
    if tif_path.suffix.lower() not in ALLOWED_SUFFIXES or not tif_path.is_file():
        raise HTTPException(status_code=404, detail=f"scene not found: {name}")

    sidecars = [_result_path(tif_path), tif_path.with_suffix(".json")]
    removed = [s.name for s in sidecars if s.is_file()]
    tif_path.unlink()
    for sidecar in sidecars:
        sidecar.unlink(missing_ok=True)

    return {"deleted": tif_path.name, "removed_sidecars": removed}
