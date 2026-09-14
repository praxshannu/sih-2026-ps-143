from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.middleware.auth import get_current_user
from app.schemas import (
    DriftVisualization,
    ForecastPath,
    OriginEllipse,
    ShorelineRisk,
    TokenPayload,
)

router = APIRouter(prefix="/cases/{case_id}/drift", tags=["drift"])

# 8002 is the detect service; the OpenDrift engine lives on 8003.
DRIFT_SERVICE_URL = os.getenv("DRIFT_SERVICE_URL", "http://sentinel-drift:8003")
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 30


class _CircuitState:
    def __init__(self) -> None:
        self.failures: dict[str, int] = {}
        self.open_until: dict[str, float] = {}

    def allow(self, service: str) -> bool:
        import time

        if service in self.open_until and time.time() < self.open_until[service]:
            return False
        return True

    def record_success(self, service: str) -> None:
        self.failures.pop(service, None)
        self.open_until.pop(service, None)

    def record_failure(self, service: str) -> None:
        import time

        self.failures[service] = self.failures.get(service, 0) + 1
        if self.failures[service] >= CIRCUIT_BREAKER_THRESHOLD:
            self.open_until[service] = time.time() + CIRCUIT_BREAKER_TIMEOUT
            logger.warning(
                "Circuit breaker OPEN for drift service for {}s", CIRCUIT_BREAKER_TIMEOUT
            )


_circuit = _CircuitState()


async def _proxy_get(path: str) -> dict:
    if not _circuit.allow("drift"):
        raise HTTPException(status_code=503, detail="Drift service circuit breaker open")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{DRIFT_SERVICE_URL}{path}")
            resp.raise_for_status()
            _circuit.record_success("drift")
            return resp.json()
    except httpx.HTTPStatusError as exc:
        _circuit.record_failure("drift")
        logger.error("Drift service returned {}: {}", exc.response.status_code, exc)
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
    except httpx.RequestError as exc:
        _circuit.record_failure("drift")
        logger.error("Drift service unreachable: {}", exc)
        raise HTTPException(status_code=503, detail="Drift service unreachable") from exc


@router.get("/origin", response_model=OriginEllipse)
async def get_origin_ellipse(
    case_id: str,
    _user: TokenPayload = Depends(get_current_user),
):
    data = await _proxy_get(f"/cases/{case_id}/drift/origin")
    return OriginEllipse(**data)


@router.get("/forecast", response_model=list[ForecastPath])
async def get_forecast_paths(
    case_id: str,
    _user: TokenPayload = Depends(get_current_user),
):
    data = await _proxy_get(f"/cases/{case_id}/drift/forecast")
    paths = data if isinstance(data, list) else data.get("paths", [])
    return [ForecastPath(**p) for p in paths]


@router.get("/shoreline-risk", response_model=list[ShorelineRisk])
async def get_shoreline_risk(
    case_id: str,
    _user: TokenPayload = Depends(get_current_user),
):
    data = await _proxy_get(f"/cases/{case_id}/drift/shoreline-risk")
    risks = data if isinstance(data, list) else data.get("risks", [])
    return [ShorelineRisk(**r) for r in risks]


@router.get("/full", response_model=DriftVisualization)
async def get_full_drift(
    case_id: str,
    _user: TokenPayload = Depends(get_current_user),
):
    data = await _proxy_get(f"/cases/{case_id}/drift/full")
    return DriftVisualization(**data)
