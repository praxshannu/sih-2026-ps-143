"""Case-scoped result routes served from the shared pipeline cache.

- GET /api/v1/cases/{case_id}/suspects — ranked suspects with fuzzy + XGB + SHAP
- GET /api/v1/drift/forecast/{case_id} — drift hindcast + forecast from cache
- GET /api/v1/llm/status — proxied intel /health/llm for the UI badge

404 when the pipeline has not produced that stage yet (never fabricated).
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException
from loguru import logger

from app.store import get_results

router = APIRouter(tags=["results"])

INTEL_SERVICE_URL = os.getenv("INTEL_SERVICE_URL", "http://sentinel-intel:8005")


@router.get("/cases/{case_id}/suspects")
async def get_case_suspects(case_id: str):
    results = await get_results(case_id)
    suspects = (results.get("attribute") or {}).get("suspects")
    if not suspects:
        raise HTTPException(
            status_code=404,
            detail=f"No suspects cached for case {case_id}; run POST /api/v1/pipeline/trigger first",
        )
    attr = results.get("attribute") or {}
    return {
        "case_id": case_id,
        "suspects": suspects,
        "total": len(suspects),
        "weights": attr.get("weights", attr.get("scoring_weights", {})),
    }


@router.get("/drift/forecast/{case_id}")
async def get_drift_forecast(case_id: str):
    results = await get_results(case_id)
    drift = results.get("drift")
    if not drift:
        raise HTTPException(
            status_code=404,
            detail=f"No drift results cached for case {case_id}; run POST /api/v1/pipeline/trigger first",
        )
    return {"case_id": case_id, **drift}


@router.get("/cases/{case_id}/drift")
async def get_case_drift(case_id: str):
    return await get_drift_forecast(case_id)


@router.get("/llm/status")
async def get_llm_status():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{INTEL_SERVICE_URL}/health/llm")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("LLM status proxy failed: {}", e)
        return {
            "provider": "unknown",
            "model": "unknown",
            "available": False,
            "latency_ms": 0.0,
        }


@router.get("/cases/{case_id}/results")
async def get_all_results(case_id: str):
    return {"case_id": case_id, "stages": await get_results(case_id)}
