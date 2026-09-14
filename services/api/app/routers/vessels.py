from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app.middleware.auth import get_current_user
from app.schemas import TokenPayload, VesselListResponse, VesselProfile

router = APIRouter(prefix="/cases/{case_id}/vessels", tags=["vessels"])

ATTRIBUTE_SERVICE_URL = os.getenv("ATTRIBUTE_SERVICE_URL", "http://localhost:8003")
INTEL_SERVICE_URL = os.getenv("INTEL_SERVICE_URL", "http://localhost:8004")
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
            logger.warning("Circuit breaker OPEN for {} for {}s", service, CIRCUIT_BREAKER_TIMEOUT)


_circuit = _CircuitState()


async def _proxy_get(service_url: str, path: str, service_name: str) -> dict:
    if not _circuit.allow(service_name):
        raise HTTPException(status_code=503, detail=f"{service_name} circuit breaker open")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{service_url}{path}")
            resp.raise_for_status()
            _circuit.record_success(service_name)
            return resp.json()
    except httpx.HTTPStatusError as exc:
        _circuit.record_failure(service_name)
        logger.error("{} returned {}: {}", service_name, exc.response.status_code, exc)
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc)) from exc
    except httpx.RequestError as exc:
        _circuit.record_failure(service_name)
        logger.error("{} unreachable: {}", service_name, exc)
        raise HTTPException(status_code=503, detail=f"{service_name} unreachable") from exc


@router.get("", response_model=VesselListResponse)
async def list_vessels(
    case_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    _user: TokenPayload = Depends(get_current_user),
):
    data = await _proxy_get(
        ATTRIBUTE_SERVICE_URL,
        f"/cases/{case_id}/vessels?skip={(page - 1) * page_size}&limit={page_size}",
        "attribute",
    )
    vessels = [
        VesselProfile(**v) for v in data.get("vessels", data if isinstance(data, list) else [])
    ]
    return VesselListResponse(vessels=vessels, total=data.get("total", len(vessels)))


@router.get("/search", response_model=VesselListResponse)
async def search_vessels(
    case_id: str,
    mmsi: str | None = Query(None),
    name: str | None = Query(None),
    min_risk_score: float | None = Query(None, ge=0.0, le=1.0),
    _user: TokenPayload = Depends(get_current_user),
):
    params: list[str] = []
    if mmsi:
        params.append(f"mmsi={mmsi}")
    if name:
        params.append(f"name={name}")
    if min_risk_score is not None:
        params.append(f"min_risk_score={min_risk_score}")
    qs = "&".join(params)
    path = f"/cases/{case_id}/vessels/search?{qs}" if qs else f"/cases/{case_id}/vessels"
    data = await _proxy_get(ATTRIBUTE_SERVICE_URL, path, "attribute")
    vessels = [
        VesselProfile(**v) for v in data.get("vessels", data if isinstance(data, list) else [])
    ]
    return VesselListResponse(vessels=vessels, total=data.get("total", len(vessels)))


@router.get("/{vessel_id}", response_model=VesselProfile)
async def get_vessel(
    case_id: str,
    vessel_id: str,
    _user: TokenPayload = Depends(get_current_user),
):
    data = await _proxy_get(
        ATTRIBUTE_SERVICE_URL, f"/cases/{case_id}/vessels/{vessel_id}", "attribute"
    )
    return VesselProfile(**data)


@router.get("/mmsi/{mmsi}", response_model=VesselProfile)
async def get_vessel_by_mmsi(
    case_id: str,
    mmsi: str,
    _user: TokenPayload = Depends(get_current_user),
):
    data = await _proxy_get(ATTRIBUTE_SERVICE_URL, f"/vessels/mmsi/{mmsi}", "attribute")
    return VesselProfile(**data)
