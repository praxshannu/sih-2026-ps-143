"""SENTINEL Ingest Service - Autonomous Data Ingestion Daemon.

FastAPI application with lifespan-managed APScheduler daemon, health check,
and manual trigger endpoints for Sentinel-1, ocean, wind, and AIS data.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import FastAPI, HTTPException
from loguru import logger

from .models.schemas import (
    AisIngestResult,
    AisParams,
    CmemsParams,
    DataSource,
    Era5Params,
    HealthResponse,
    OceanCurrentResult,
    SchedulerStatus,
    Sentinel1Params,
    Sentinel1Result,
    TriggerRequest,
    WindFieldResult,
)
from .scheduler import IngestScheduler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://sentinel:sentinel_secret@db:5432/sentinel",
)
ASYNC_DSN = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

FETCH_INTERVAL_HOURS = int(os.getenv("INGEST_INTERVAL_HOURS", "6"))
INDIA_EEZ_BBOX = (68.0, 5.0, 88.0, 25.0)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(
    "logs/ingest_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None
_scheduler: IngestScheduler | None = None


async def _init_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(
        DSN=ASYNC_DSN,
        min_size=2,
        max_size=10,
        timeout=30,
        command_timeout=60,
    )


async def _fetch_sentinel1(**kwargs: Any) -> Sentinel1Result:
    """Fetch Sentinel-1 products via the source fetcher."""
    from .sources.sentinel1 import Sentinel1Fetcher

    fetcher = Sentinel1Fetcher(
        storage_path=os.getenv("SENTINEL1_STORAGE", "/data/sentinel1"),
    )
    params = kwargs.get("params") or {}
    return await fetcher.fetch(
        bbox=params.get("bbox", INDIA_EEZ_BBOX),
        product_type=params.get("product_type", "GRD"),
        max_results=params.get("max_results", 50),
        lookback_days=params.get("lookback_days", 6),
    )


async def _fetch_ocean(**kwargs: Any) -> OceanCurrentResult:
    """Fetch CMEMS ocean current data."""
    from .sources.cmems import CmemsFetcher

    fetcher = CmemsFetcher(
        storage_path=os.getenv("OCEAN_STORAGE", "/data/ocean"),
    )
    params = kwargs.get("params") or {}
    return await fetcher.fetch(
        bbox=params.get("bbox", INDIA_EEZ_BBOX),
        depth_range=params.get("depth_range", (0.0, 50.0)),
        lookback_days=params.get("lookback_days", 7),
    )


async def _fetch_wind(**kwargs: Any) -> WindFieldResult:
    """Fetch ERA5 wind field data."""
    from .sources.era5 import Era5Fetcher

    fetcher = Era5Fetcher(
        storage_path=os.getenv("WIND_STORAGE", "/data/wind"),
    )
    params = kwargs.get("params") or {}
    return await fetcher.fetch(
        bbox=params.get("bbox", INDIA_EEZ_BBOX),
        lookback_days=params.get("lookback_days", 7),
    )


async def _fetch_ais(**kwargs: Any) -> AisIngestResult:
    """Fetch AIS position data."""
    from .sources.ais import AisFetcher

    fetcher = AisFetcher(pool=_pool)
    params = kwargs.get("params") or {}
    return await fetcher.fetch(
        bbox=params.get("bbox", INDIA_EEZ_BBOX),
        lookback_hours=params.get("lookback_hours", 6),
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle: init DB pool and APScheduler daemon."""
    global _pool, _scheduler

    logger.info("SENTINEL Ingest Service starting up")
    from pathlib import Path

    # mkdir is idempotent — never let an existing dir (or a read-only FS)
    # abort startup of the whole service.
    try:
        Path("logs").mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # noqa: BLE001
        logger.debug("logs dir not created: {}", exc)

    # Database pool
    try:
        _pool = await _init_pool()
        logger.info("Database pool connected")
    except Exception as e:
        logger.warning("Database pool init failed (will retry): {}", str(e))
        _pool = None

    # Scheduler
    _scheduler = IngestScheduler(interval_hours=FETCH_INTERVAL_HOURS)
    _scheduler.register_fetch_callback(DataSource.SENTINEL1, _fetch_sentinel1)
    _scheduler.register_fetch_callback(DataSource.CMEMS, _fetch_ocean)
    _scheduler.register_fetch_callback(DataSource.ERA5, _fetch_wind)
    _scheduler.register_fetch_callback(DataSource.AIS, _fetch_ais)
    _scheduler.start()

    logger.info(
        "Ingest service ready: interval={}h, DB={}",
        FETCH_INTERVAL_HOURS,
        "connected" if _pool else "disconnected",
    )

    yield

    # Shutdown
    if _scheduler:
        _scheduler.stop()
    if _pool:
        await _pool.close()
    logger.info("SENTINEL Ingest Service shut down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SENTINEL Ingest Service",
    description="Autonomous data ingestion daemon for Sentinel-1, ocean currents, wind fields, and AIS data",
    version="1.0.0",
    lifespan=lifespan,
)

# Archive browser (in-platform Copernicus search). Owned here because this
# service owns data sources; the API gateway proxies it.
from .routers.archive import router as archive_router  # noqa: E402

app.include_router(archive_router)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db_ok = False
    if _pool:
        try:
            async with _pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except Exception:
            pass

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        db_connected=db_ok,
    )


@app.get("/ingest/status", response_model=SchedulerStatus)
async def get_status():
    """Scheduler status and last fetch times."""
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    return _scheduler.get_status()


@app.post("/ingest/trigger")
async def trigger_full_fetch(req: TriggerRequest):
    """Manually trigger full data ingestion cycle."""
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    logger.info("Manual full ingest trigger: sources={}, force={}", req.sources, req.force)

    results = await _scheduler.trigger_full_cycle(
        sources=req.sources,
        force=req.force,
    )

    # Serialize results
    serialized: dict[str, Any] = {}
    for source, result in results.items():
        if isinstance(result, Exception):
            serialized[source.value] = {"status": "error", "error": str(result)}
        elif hasattr(result, "model_dump"):
            serialized[source.value] = {"status": "ok", **result.model_dump()}
        else:
            serialized[source.value] = {"status": "ok"}

    return {"status": "completed", "sources": serialized}


@app.post("/ingest/sentinel1", response_model=Sentinel1Result)
async def trigger_sentinel1_fetch(params: Sentinel1Params | None = None):
    """Fetch latest Sentinel-1 products."""
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    fetch_params = params.model_dump() if params else {}
    result = await _scheduler.trigger_source(
        DataSource.SENTINEL1,
        params={"params": fetch_params},
    )
    return result


@app.post("/ingest/ais", response_model=AisIngestResult)
async def trigger_ais_fetch(params: AisParams | None = None):
    """Fetch latest AIS data."""
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    fetch_params = params.model_dump() if params else {}
    result = await _scheduler.trigger_source(
        DataSource.AIS,
        params={"params": fetch_params},
    )
    return result


@app.post("/ingest/ocean")
async def trigger_ocean_fetch(
    cmems_params: CmemsParams | None = None,
    era5_params: Era5Params | None = None,
):
    """Fetch latest ocean current and wind field data."""
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")

    ocean_result = await _scheduler.trigger_source(
        DataSource.CMEMS,
        params={"params": cmems_params.model_dump() if cmems_params else {}},
    )
    wind_result = await _scheduler.trigger_source(
        DataSource.ERA5,
        params={"params": era5_params.model_dump() if era5_params else {}},
    )

    return {
        "ocean": ocean_result.model_dump() if hasattr(ocean_result, "model_dump") else {},
        "wind": wind_result.model_dump() if hasattr(wind_result, "model_dump") else {},
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def start_server(host: str = "0.0.0.0", port: int = 8002):
    """Run the ingest service."""
    import uvicorn

    logger.info("Starting SENTINEL Ingest Service on {}:{}", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
