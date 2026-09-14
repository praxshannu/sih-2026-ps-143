"""Pydantic v2 schemas for sentinel-detect API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SpillAgeCategory(StrEnum):
    FRESH = "fresh"
    MILD = "mild"
    WEATHERED = "weathered"
    HEAVY = "heavy"


class ProcessingStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class WindData(BaseModel):
    u10: float = Field(..., description="10-m eastward wind component (m/s)")
    v10: float = Field(..., description="10-m northward wind component (m/s)")

    @property
    def speed(self) -> float:
        return (self.u10**2 + self.v10**2) ** 0.5

    @property
    def direction_deg(self) -> float:
        import math

        return math.degrees(math.atan2(-self.u10, -self.v10)) % 360


class CurrentData(BaseModel):
    u: float = Field(default=0.0, description="Eastward ocean current (m/s)")
    v: float = Field(default=0.0, description="Northward ocean current (m/s)")


class DetectRequest(BaseModel):
    model_config = {"strict": False}

    image_path: str = Field(..., description="Local path or URL to Sentinel-1 GeoTIFF")
    band_selection: str = Field(default="VV", description="SAR polarisation band")
    wind: WindData | None = Field(default=None, description="ERA5 wind data")
    current: CurrentData | None = Field(default=None, description="Ocean current data")
    tile_size: int = Field(default=512, ge=128, le=2048)
    overlap: int = Field(default=64, ge=0, le=256)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    model_config = {"strict": False}

    image_path: str
    band_selection: str = Field(default="VV")
    wind: WindData | None = None
    current: CurrentData | None = None
    tile_size: int = Field(default=512, ge=128, le=2048)
    overlap: int = Field(default=64, ge=0, le=256)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_area_m2: float = Field(default=1000.0, ge=0.0)
    max_lookalike_score: float = Field(default=0.6, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# GeoJSON primitives
# ---------------------------------------------------------------------------


class GeoJSONGeometry(BaseModel):
    type: str = "Polygon"
    coordinates: list[list[list[float]]]


class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: dict[str, Any] = Field(default_factory=dict)


class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class DetectedSpill(BaseModel):
    id: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    area_m2: float
    centroid_lon: float
    centroid_lat: float
    geometry: GeoJSONGeometry
    pixel_area_ratio: float = Field(ge=0.0, le=1.0)


class LookalikeAnalysis(BaseModel):
    is_lookalike: bool
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class AgeEstimate(BaseModel):
    estimated_hours: float | None = Field(default=None, ge=0.0)
    category: SpillAgeCategory
    area_at_detection_m2: float
    fay_constant_k: float | None = None


class DetectResponse(BaseModel):
    status: ProcessingStatus
    image_path: str
    spills: list[DetectedSpill] = Field(default_factory=list)
    total_spills: int = Field(default=0)
    geojson: GeoJSONFeatureCollection
    inference_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnalyzeResponse(BaseModel):
    status: ProcessingStatus
    image_path: str
    spills: list[DetectedSpill] = Field(default_factory=list)
    total_spills: int = Field(default=0)
    geojson: GeoJSONFeatureCollection
    lookalike_results: list[LookalikeAnalysis] = Field(default_factory=list)
    age_estimates: list[AgeEstimate] = Field(default_factory=list)
    inference_time_ms: float
    total_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str = "unetpp_scse_resnet34"
    device: str
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    requests_total: int = Field(default=0)
    spills_detected_total: int = Field(default=0)
    avg_inference_ms: float = Field(default=0.0)
    model_loaded: bool = Field(default=False)
