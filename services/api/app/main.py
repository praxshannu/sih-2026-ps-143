from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.middleware.rate_limit import RateLimitMiddleware
from app.routers import alerts, archive, attribution, cases, detect, drift, results, vessels, ws
from app.schemas import HealthResponse, PipelineTrigger, PipelineStatus, ServiceHealth, WSEvent, WSEventType
from app.routers.ws import get_ws_manager

DETECT_SERVICE_URL = os.getenv("DETECT_SERVICE_URL", "http://sentinel-detect:8002")
DRIFT_SERVICE_URL = os.getenv("DRIFT_SERVICE_URL", "http://sentinel-drift:8003")
ATTRIBUTE_SERVICE_URL = os.getenv("ATTRIBUTE_SERVICE_URL", "http://sentinel-attribute:8004")
INTEL_SERVICE_URL = os.getenv("INTEL_SERVICE_URL", "http://sentinel-intel:8005")
API_VERSION = os.getenv("SENTINEL_API_VERSION", "0.1.0")

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()
    logger.info("SENTINEL API Gateway starting — version {}", API_VERSION)
    yield
    logger.info("SENTINEL API Gateway shutting down")


app = FastAPI(
    title="SENTINEL API Gateway",
    description="Real-time maritime oil spill detection and attribution platform",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases.router, prefix="/api/v1")
app.include_router(vessels.router, prefix="/api/v1")
app.include_router(drift.router, prefix="/api/v1")
app.include_router(results.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")
app.include_router(ws.router, prefix="/api/v1")
app.include_router(archive.router, prefix="/api/v1")
app.include_router(detect.router, prefix="/api/v1")
# Live OpenDrift hindcast engine. Separate from `drift.router`, which is the
# case-scoped /cases/{id}/drift/* stub.
app.include_router(attribution.router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on {} {}: {}", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    services = []
    targets = [
        ("detect", DETECT_SERVICE_URL),
        ("drift", DRIFT_SERVICE_URL),
        ("attribute", ATTRIBUTE_SERVICE_URL),
        ("intel", INTEL_SERVICE_URL),
    ]
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in targets:
            try:
                t0 = time.time()
                resp = await client.get(f"{url}/health")
                latency = (time.time() - t0) * 1000
                status_str = "healthy" if resp.status_code == 200 else "degraded"
                services.append(ServiceHealth(name=name, status=status_str, latency_ms=round(latency, 1)))
            except httpx.RequestError as exc:
                services.append(ServiceHealth(name=name, status="unreachable", error=str(exc)[:200]))

    healthy_count = sum(1 for s in services if s.status == "healthy")
    if healthy_count == len(services):
        overall = "healthy"
    elif healthy_count > 0:
        overall = "degraded"
    else:
        overall = "unhealthy"

    uptime = time.time() - _start_time
    return HealthResponse(status=overall, services=services, uptime_seconds=round(uptime, 1))


@app.get("/api/v1/ws/stats", tags=["websocket"])
async def ws_stats():
    mgr = get_ws_manager()
    return {"total_connections": mgr.total_connections}


@app.post("/api/v1/pipeline/trigger", response_model=PipelineStatus, tags=["pipeline"])
async def trigger_pipeline(body: PipelineTrigger):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    try:
        from app.celery_app import celery_app
        celery_app.send_task(
            "app.tasks.run_full_pipeline",
            args=[body.case_id, body.stages, body.payload],
        )
        logger.info("Pipeline triggered for case {} with stages {}", body.case_id, body.stages)
        queued, detail = True, None
    except Exception as exc:
        # Never silently swallow: report loud and clear so callers can retry.
        logger.error("Celery enqueue failed for case {} (Redis down?): {}", body.case_id, exc)
        queued, detail = False, f"Celery enqueue failed: {exc}"

    return PipelineStatus(
        case_id=body.case_id,
        current_stage=body.stages[0] if body.stages else None,
        stages_completed=[],
        stages_failed=[] if queued else (body.stages or []),
        started_at=now,
        updated_at=now,
        queued=queued,
        detail=detail,
    )


@app.post("/api/v1/broadcast", tags=["websocket"])
async def broadcast_event(event: WSEvent):
    await get_ws_manager().broadcast(event.payload.get("case_id", ""), event)
    return {"status": "broadcast", "type": event.type}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=os.getenv("ENV") == "development")
