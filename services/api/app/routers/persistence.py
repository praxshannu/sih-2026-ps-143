"""Durable case endpoints — persist, list, reopen.

Kept separate from ``routers/cases.py`` on purpose. That router's CRUD is
backed by a module-level dict (see its `# In-memory store` comment) and is
legacy; rewriting it in the same change as introducing durable storage would
mix "make persistence real" with "change every case endpoint's semantics", and
a reviewer could not tell which regression came from which. These routes are
the real thing, they are additive, and they report which backend served them.

The backend is never inferred by the caller. Every response carries
``persistence_backend``, because "it is in the database" and "it is in a JSON
file next to the data" are different operational claims and the difference
matters when someone is deciding whether a case is safe to rely on.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.case_store import CaseRecord, get_case_store

router = APIRouter(prefix="/scenes", tags=["persistence"])


@router.get("/store/health")
async def store_health() -> dict[str, object]:
    """Which backend a write would use, and why.

    Called before persisting so an operator is told *before* the write whether
    their case is going to Postgres or to disk — not after.
    """
    store = get_case_store()
    backend = await store.active_backend()
    return {
        "backend": backend,
        "requested": store.requested_backend,
        "directory": str(store.directory),
        "database_configured": bool(store.database_url),
        "note": (
            "postgres is the configured backend"
            if backend == "postgres"
            else (
                "Postgres is not reachable, so cases are written to durable JSON "
                f"under {store.directory}. They survive a restart, but they are "
                "NOT in the database."
            )
        ),
    }


@router.post("/persist")
async def persist_case(record: CaseRecord) -> dict[str, object]:
    """Persist a processed scene.

    The request body is the whole :class:`CaseRecord`: scene metadata, source
    and checksum, detection, polygon, drift, forcing provenance, AIS
    provenance, suspect scores, model version, confidence and warnings. A
    partial record is accepted and the missing fields are recorded as null
    rather than guessed at.
    """
    store = get_case_store()
    try:
        saved = await store.save(record)
    except Exception as exc:  # noqa: BLE001 - surface the reason, never a bare 500
        logger.exception("case persistence failed for {}", record.case_id)
        raise HTTPException(status_code=503, detail=f"persistence failed: {exc}") from exc

    return {
        "case_id": saved.case_id,
        "persistence_backend": saved.persistence_backend,
        "created_at": saved.created_at,
        "updated_at": saved.updated_at,
    }


@router.get("/cases")
async def list_persisted_cases(
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, object]:
    """Every persisted case, newest first."""
    records = await get_case_store().list()
    return {
        "count": len(records[:limit]),
        "total": len(records),
        "cases": [
            {
                "case_id": r.case_id,
                "scene_id": r.scene_id,
                "title": r.title,
                "state": r.detection.state,
                "coverage": r.coverage,
                "persistence_backend": r.persistence_backend,
                "updated_at": r.updated_at,
            }
            for r in records[:limit]
        ],
    }


@router.get("/cases/{case_id}")
async def reopen_case(case_id: str) -> CaseRecord:
    """Reopen a persisted case after a restart.

    Returns the record exactly as it was stored, including the confidence
    interval and the forcing provenance — the fields that decide whether the
    result is still usable.
    """
    record = await get_case_store().load(case_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no persisted case {case_id!r}")
    return record


@router.delete("/cases/{case_id}")
async def delete_persisted_case(case_id: str) -> dict[str, object]:
    deleted = await get_case_store().delete(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no persisted case {case_id!r}")
    return {"deleted": case_id}
