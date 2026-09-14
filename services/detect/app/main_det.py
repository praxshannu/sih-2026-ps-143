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

from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from app.routers.deterministic import router as deterministic_router
from app.routers.scene import router as scene_router

# services/detect/app/main_det.py → 4 levels up is the project root.
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
SAR_DIR = DATA_DIR / "sar"

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
    SAR_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "sentinel-det (deterministic) started — SAR dir={} | existing scenes={}",
        SAR_DIR,
        len(list(SAR_DIR.glob("*.tif"))),
    )


@app.get("/health")
async def health() -> dict[str, object]:
    """Probe: the service is up, the SAR dir is reachable, and the detector imports."""
    scene_count = sum(1 for _ in SAR_DIR.glob("*.tif"))
    return {
        "ok": True,
        "service": "sentinel-det-det",
        "variant": "deterministic",
        "sar_dir": str(SAR_DIR),
        "scene_count": scene_count,
        "ml_available": False,
        "detector": "deterministic-lee-adaptive-v1",
    }


app.include_router(deterministic_router)
app.include_router(scene_router)
