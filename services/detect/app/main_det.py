"""sentinel-detect: lightweight service exposing the deterministic Tier-A detector.

Use this entry point when torch is unavailable (Apple M2 CPU, no CUDA) or
when no trained UNet++ weights are present. It mounts the same routes the
production ``main.py`` exposes for the deterministic pipeline:

    GET  /health
    POST /detect/deterministic
    GET  /detect/deterministic/results
    POST /detect/deterministic/run_all

For the full ML pipeline (UNet++ + lookalike filter + age estimator), run
``app.main:app`` instead — that path requires ``pip install -r requirements.txt``
plus trained weights under ``app/models/weights/``.
"""

from __future__ import annotations

from fastapi import FastAPI
from loguru import logger

from app.paths import SAR_DIR
from app.routers.deterministic import router as deterministic_router
from app.routers.scene import router as scene_router

app = FastAPI(
    title="SENTINEL Detect — Tier-A (deterministic)",
    description=(
        "Honest Tier-A SAR spill detector (Lee-sigma speckle → Solberg "
        "two-gate adaptive threshold → morphology → Fay age + Wilson 95 % CI). "
        "No ML weights required. Use this when UNet++ is unavailable."
    ),
    version="0.1.0-det",
)


@app.on_event("startup")
async def _startup() -> None:
    # The scene directory is often a read-only mount (an external disk, a
    # mounted archive). Failing to *create* it is not a reason to refuse to
    # start — failing to *read* it is, and /health reports that instead.
    writable = True
    try:
        SAR_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        writable = False
        logger.warning("SAR dir {} is not creatable ({}); continuing read-only", SAR_DIR, exc)
    logger.info(
        "sentinel-det (deterministic) started — SAR dir={} | writable={} | existing scenes={}",
        SAR_DIR,
        writable,
        sum(1 for _ in SAR_DIR.glob("*.tif")) if SAR_DIR.is_dir() else 0,
    )


@app.get("/health")
async def health() -> dict[str, object]:
    """Probe: the service is up, the SAR dir is reachable, and the detector imports."""
    exists = SAR_DIR.is_dir()
    scene_count = sum(1 for _ in SAR_DIR.glob("*.tif")) if exists else 0
    return {
        "ok": True,
        "service": "sentinel-det-det",
        "variant": "deterministic",
        "sar_dir": str(SAR_DIR),
        "sar_dir_exists": exists,
        "scene_count": scene_count,
        "ml_available": False,
        "detector": "deterministic-lee-adaptive-v1",
    }


app.include_router(deterministic_router)
app.include_router(scene_router)
