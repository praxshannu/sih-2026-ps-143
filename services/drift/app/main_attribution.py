"""SENTINEL drift service — attribution entry point.

A lean FastAPI app that exposes only the OpenDrift backward attribution
route. Use this on M2 where the full UNet++ pipeline isn't available.

    uvicorn app.main_attribution:app --port 8003
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

# Must run before anything reads CDSAPI_KEY / CMEMS credentials. Without it a
# hand-launched uvicorn has no keys and every hindcast silently falls back to
# synthetic forcing — see app/env_bootstrap.py.
from app.env_bootstrap import load_dotenv

_ENV_FILE = load_dotenv()

from app.routers.attribution import router as attribution_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("sentinel-drift (attribution) starting — OpenDrift-only mode")
    logger.info(
        "env file={} CDSAPI_KEY={} CMEMS configured={}",
        _ENV_FILE or "none",
        "present" if os.getenv("CDSAPI_KEY") else "MISSING (wind will be synthetic)",
        "yes"
        if (os.getenv("COPERNICUSMARINE_SERVICE_USERNAME") or os.getenv("CMEMS_USERNAME"))
        else "no",
    )
    if not os.getenv("CDSAPI_KEY"):
        logger.warning(
            "CDSAPI_KEY is not set — hindcasts will use SYNTHETIC constant wind. "
            "Results are physically meaningless; do not present them as real."
        )
    yield
    logger.info("sentinel-drift (attribution) shutting down")


app = FastAPI(
    title="SENTINEL Drift — Attribution",
    description=(
        "OpenDrift-based backward hindcast + vessel suspect scoring. "
        "Pulls ERA5 wind and CMEMS currents when available; falls back to "
        "synthetic constant fields (clearly labelled) when not."
    ),
    version="0.1.0-attr",
)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "ok": True,
        "service": "sentinel-drift-attribution",
        "started_at": time.time(),
    }


app.include_router(attribution_router)
