"""sentinel-detect: ML inference service for oil spill detection.

FastAPI application that loads a UNet++ with SCSE attention model at
startup and provides async endpoints for SAR-based oil spill detection,
post-processing, look-alike filtering, and age estimation.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TypedDict

import numpy as np
import rasterio
import torch
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.models.inference import TiledInference
from app.models.unetpp_scse import UNetPlusPlusSCSE
from app.processors.age_estimator import AgeEstimator
from app.processors.lookalike_filter import LookalikeFilter
from app.processors.multimodal_input import MultimodalInputProcessor
from app.processors.postprocess import PostProcessor
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DetectedSpill,
    DetectRequest,
    DetectResponse,
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    GeoJSONGeometry,
    HealthResponse,
    LookalikeAnalysis,
    MetricsResponse,
    ProcessingStatus,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_model: UNetPlusPlusSCSE | None = None
_tiled_inference: TiledInference | None = None
_input_processor: MultimodalInputProcessor | None = None
_post_processor: PostProcessor | None = None
_lookalike_filter: LookalikeFilter | None = None
_age_estimator: AgeEstimator | None = None
_device: torch.device | None = None

# Metrics counters
class _Metrics(TypedDict):
    """Typed view of the counters served by /metrics.

    A plain dict literal infers ``dict[str, object]``, which makes every
    ``+=`` and ``.append`` below an "unsupported operand"/"no attribute"
    error. Naming each key's real type fixes the reads and the writes.
    """

    requests_total: int
    spills_detected_total: int
    inference_times_ms: list[float]


_metrics: _Metrics = {
    "requests_total": 0,
    "spills_detected_total": 0,
    "inference_times_ms": [],
}

_thread_pool = ThreadPoolExecutor(max_workers=4)

MODEL_NAME = os.getenv("MODEL_NAME", "unetpp_scse_resnet34")
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pth")
ENCODER_NAME = os.getenv("ENCODER_NAME", "resnet34")
IN_CHANNELS = int(os.getenv("IN_CHANNELS", "6"))
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "1"))
DEVICE = os.getenv("DEVICE", "auto")


def _resolve_device() -> torch.device:
    if DEVICE == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and processors at startup."""
    global _model, _tiled_inference, _input_processor, _post_processor
    global _lookalike_filter, _age_estimator, _device

    logger.info("Starting sentinel-detect service")
    _device = _resolve_device()
    logger.info(f"Using device: {_device}")

    # Load model
    if os.path.isfile(MODEL_CHECKPOINT):
        logger.info(f"Loading model from {MODEL_CHECKPOINT}")
        _model = UNetPlusPlusSCSE.from_pretrained(
            MODEL_CHECKPOINT,
            encoder_name=ENCODER_NAME,
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
            device=_device,
        )
    else:
        logger.warning(
            f"Checkpoint not found at {MODEL_CHECKPOINT}, loading pretrained encoder only"
        )
        _model = UNetPlusPlusSCSE(
            encoder_name=ENCODER_NAME,
            encoder_weights="imagenet",
            in_channels=IN_CHANNELS,
            classes=NUM_CLASSES,
        ).to(_device)
        _model.eval()  # type: ignore[union-attr]

    _tiled_inference = TiledInference(
        model=_model,
        tile_size=512,
        overlap=64,
        device=_device,
    )
    _input_processor = MultimodalInputProcessor()
    _post_processor = PostProcessor()
    _lookalike_filter = LookalikeFilter()
    _age_estimator = AgeEstimator()

    logger.info("Model and processors loaded successfully")
    yield
    logger.info("Shutting down sentinel-detect service")
    _thread_pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="sentinel-detect",
    description="ML inference engine for oil spill detection from Sentinel-1 SAR",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Model health and status check."""
    model_loaded = _model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=MODEL_NAME,
        device=str(_device) if _device else "unknown",
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Prometheus-style metrics endpoint."""
    avg_time = np.mean(_metrics["inference_times_ms"]) if _metrics["inference_times_ms"] else 0.0
    return MetricsResponse(
        requests_total=_metrics["requests_total"],
        spills_detected_total=_metrics["spills_detected_total"],
        avg_inference_ms=float(avg_time),
        model_loaded=_model is not None,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """Detect oil spills from Sentinel-1 SAR imagery.

    Returns binary mask predictions, vectorised polygons, and GeoJSON.
    """
    if _model is None or _tiled_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _metrics["requests_total"] += 1

    try:
        # Run inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _thread_pool,
            _run_detection,
            request,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}") from e


@app.post("/detect/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Full pipeline: detect + postprocess + lookalike filter + age estimate.

    Returns vectorised spill polygons with confidence scores, look-alike
    analysis, and estimated spill age.
    """
    if _model is None or _tiled_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    _metrics["requests_total"] += 1

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _thread_pool,
            _run_full_analysis,
            request,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e


# ---------------------------------------------------------------------------
# Core processing functions (run in thread pool)
# ---------------------------------------------------------------------------


def _run_detection(request: DetectRequest) -> DetectResponse:
    """Synchronous detection pipeline (runs in thread pool)."""
    assert _input_processor is not None
    assert _tiled_inference is not None
    assert _post_processor is not None

    inf_start = time.perf_counter()

    # Read and prepare input
    input_array = _input_processor.process(
        image_path=request.image_path,
        band_selection=request.band_selection,
        wind=request.wind,
        current=request.current,
    )

    # Run tiled inference
    logits = _tiled_inference.predict(input_array)

    # Post-process to polygons
    with rasterio.open(request.image_path) as src:
        transform = src.transform
    _post_processor.confidence_threshold = request.confidence_threshold
    polygons, geojson_dict = _post_processor.process(logits, transform=transform)

    inf_time_ms = (time.perf_counter() - inf_start) * 1000
    _metrics["inference_times_ms"].append(inf_time_ms)

    # Build response
    spills = []
    for i, poly in enumerate(polygons):
        geom = poly["geometry"]
        conf = float(request.confidence_threshold)  # placeholder
        spill = DetectedSpill(
            id=f"spill_{i:04d}",
            confidence=conf,
            confidence_level=_confidence_level(conf),
            area_m2=float(poly.get("area_m2", 0.0)),
            centroid_lon=float(poly.get("centroid", (0, 0))[1]),
            centroid_lat=float(poly.get("centroid", (0, 0))[0]),
            geometry=GeoJSONGeometry(
                type="Polygon",
                coordinates=[list(geom.exterior.coords)],
            ),
            pixel_area_ratio=0.0,
        )
        spills.append(spill)

    _metrics["spills_detected_total"] += len(spills)

    geojson_features = GeoJSONFeatureCollection(
        features=[
            GeoJSONFeature(
                geometry=GeoJSONGeometry(
                    type="Polygon",
                    coordinates=[list(p["geometry"].exterior.coords)],
                ),
                properties={"id": s.id, "confidence": s.confidence, "area_m2": s.area_m2},
            )
            for s, p in zip(spills, polygons)
        ]
    )

    return DetectResponse(
        status=ProcessingStatus.SUCCESS if spills else ProcessingStatus.PARTIAL,
        image_path=request.image_path,
        spills=spills,
        total_spills=len(spills),
        geojson=geojson_features,
        inference_time_ms=round(inf_time_ms, 2),
    )


def _run_full_analysis(request: AnalyzeRequest) -> AnalyzeResponse:
    """Synchronous full analysis pipeline (runs in thread pool)."""
    assert _input_processor is not None
    assert _tiled_inference is not None
    assert _post_processor is not None
    assert _lookalike_filter is not None
    assert _age_estimator is not None

    total_start = time.perf_counter()
    inf_start = time.perf_counter()

    # Read and prepare input
    input_array = _input_processor.process(
        image_path=request.image_path,
        band_selection=request.band_selection,
        wind=request.wind,
        current=request.current,
    )

    # Run tiled inference
    logits = _tiled_inference.predict(input_array)

    inf_time_ms = (time.perf_counter() - inf_start) * 1000

    # Post-process
    with rasterio.open(request.image_path) as src:
        transform = src.transform

    _post_processor.confidence_threshold = request.confidence_threshold
    _post_processor.min_area_m2 = request.min_area_m2
    polygons, geojson_dict = _post_processor.process(logits, transform=transform)

    # Read SAR patch for texture analysis
    bands = _input_processor.read_sar_bands(request.image_path, request.band_selection)
    sar_array = _input_processor.normalize_sar(next(iter(bands.values())))

    # Lookalike filtering per polygon
    lookalike_results: list[LookalikeAnalysis] = []
    filtered_polygons = []

    for poly in polygons:
        geom = poly["geometry"]

        # Extract patch around polygon for texture analysis
        sar_patch = _extract_patch(sar_array, geom)
        mask_patch = np.ones_like(sar_patch, dtype=np.uint8)

        lookalike = _lookalike_filter.assess(
            sar_patch=sar_patch,
            spill_mask=mask_patch,
            wind=request.wind,
        )
        lookalike_results.append(lookalike)

        if not lookalike.is_lookalike or lookalike.score < request.max_lookalike_score:
            filtered_polygons.append(poly)

    # Age estimation
    age_estimates = _age_estimator.estimate_multi_scale(
        [p.get("area_m2", 0.0) for p in filtered_polygons],
        wind=request.wind,
        current=request.current,
    )

    # Build filtered GeoJSON
    geojson_features = GeoJSONFeatureCollection(
        features=[
            GeoJSONFeature(
                geometry=GeoJSONGeometry(
                    type="Polygon",
                    coordinates=[list(p["geometry"].exterior.coords)],
                ),
                properties={
                    "id": f"spill_{i:04d}",
                    "area_m2": p.get("area_m2", 0),
                    "age_hours": age_estimates[i].estimated_hours
                    if i < len(age_estimates)
                    else None,
                    "age_category": age_estimates[i].category.value
                    if i < len(age_estimates)
                    else None,
                },
            )
            for i, p in enumerate(filtered_polygons)
        ]
    )

    # Build spill objects
    spills = []
    for i, poly in enumerate(filtered_polygons):
        geom = poly["geometry"]
        conf = 1.0 - (lookalike_results[i].score if i < len(lookalike_results) else 0.0)
        spill = DetectedSpill(
            id=f"spill_{i:04d}",
            confidence=round(conf, 4),
            confidence_level=_confidence_level(conf),
            area_m2=float(poly.get("area_m2", 0.0)),
            centroid_lon=float(poly.get("centroid", (0, 0))[1]),
            centroid_lat=float(poly.get("centroid", (0, 0))[0]),
            geometry=GeoJSONGeometry(
                type="Polygon",
                coordinates=[list(geom.exterior.coords)],
            ),
            pixel_area_ratio=0.0,
        )
        spills.append(spill)

    _metrics["spills_detected_total"] += len(spills)

    total_time_ms = (time.perf_counter() - total_start) * 1000
    _metrics["inference_times_ms"].append(inf_time_ms)

    return AnalyzeResponse(
        status=ProcessingStatus.SUCCESS if spills else ProcessingStatus.PARTIAL,
        image_path=request.image_path,
        spills=spills,
        total_spills=len(spills),
        geojson=geojson_features,
        lookalike_results=lookalike_results,
        age_estimates=age_estimates,
        inference_time_ms=round(inf_time_ms, 2),
        total_time_ms=round(total_time_ms, 2),
    )


def _extract_patch(sar_array: np.ndarray, geometry) -> np.ndarray:
    """Extract a patch from SAR array around a geometry's bounding box."""
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    x_min = max(0, int(bounds[0]) - 50)
    y_min = max(0, int(bounds[1]) - 50)
    x_max = min(sar_array.shape[1], int(bounds[2]) + 50)
    y_max = min(sar_array.shape[0], int(bounds[3]) + 50)

    patch = sar_array[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return np.zeros((64, 64), dtype=np.float32)
    return patch.astype(np.float32)


def _confidence_level(confidence: float):
    """Map confidence score to level enum."""
    from app.schemas import ConfidenceLevel

    if confidence >= 0.75:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENV") == "development",
    )
