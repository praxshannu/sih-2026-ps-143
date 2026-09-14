from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------- Enums ----------


class CaseStatus(str, Enum):
    NEW = "new"
    DETECTED = "detected"
    DRIFT_ANALYZING = "drift_analyzing"
    ATTRIBUTING = "attributing"
    INTEL_ENRICHING = "intel_enriching"
    REVIEW = "review"
    CLOSED = "closed"


class AlertType(str, Enum):
    SPILL_DETECTED = "spill_detected"
    SUSPECT_IDENTIFIED = "suspect_identified"
    CASE_FILE_READY = "case_file_ready"
    DRIFT_COMPLETE = "drift_complete"
    MANUAL = "manual"


class AlertPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------- Auth ----------


class TokenPayload(BaseModel):
    sub: str
    exp: int
    iat: int = 0
    roles: list[str] = Field(default_factory=list)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


# ---------- Cases ----------


class CaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str = ""
    region: str | None = None
    priority: AlertPriority = AlertPriority.MEDIUM


class CaseUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: CaseStatus | None = None
    priority: AlertPriority | None = None


class AnnotationCreate(BaseModel):
    author: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)


class AnnotationResponse(BaseModel):
    id: str
    case_id: str
    author: str
    text: str
    tags: list[str]
    created_at: datetime


class CaseResponse(BaseModel):
    id: str
    name: str
    description: str
    status: CaseStatus
    priority: AlertPriority
    region: str | None
    annotations: list[AnnotationResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CaseListResponse(BaseModel):
    cases: list[CaseResponse]
    total: int
    page: int
    page_size: int


# ---------- Vessels ----------


class VesselProfile(BaseModel):
    id: str
    mmsi: str
    name: str
    imo: str | None = None
    flag: str | None = None
    vessel_type: str | None = None
    length: float | None = None
    beam: float | None = None
    draft: float | None = None
    ais_track: list[dict[str, Any]] = Field(default_factory=list)
    risk_score: float = Field(0.0, ge=0.0, le=1.0)
    risk_factors: list[str] = Field(default_factory=list)


class VesselListResponse(BaseModel):
    vessels: list[VesselProfile]
    total: int


class VesselSearchParams(BaseModel):
    mmsi: str | None = None
    name: str | None = None
    min_risk_score: float | None = Field(None, ge=0.0, le=1.0)


# ---------- Drift ----------


class EllipsePoint(BaseModel):
    lat: float
    lon: float


class OriginEllipse(BaseModel):
    center_lat: float
    center_lon: float
    semi_major_km: float
    semi_minor_km: float
    orientation_deg: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    area_km2: float


class ForecastPath(BaseModel):
    path_id: str
    timestamps: list[datetime]
    coordinates: list[EllipsePoint]
    speed_knots: list[float]
    wind_factor: float
    current_factor: float


class ShorelineRisk(BaseModel):
    segment_id: str
    shore_name: str
    distance_km: float
    risk_level: AlertPriority
    eta_hours: float | None = None
    geometry_wkt: str | None = None


class DriftVisualization(BaseModel):
    case_id: str
    origin_ellipse: OriginEllipse
    forecast_paths: list[ForecastPath]
    shoreline_risks: list[ShorelineRisk]
    generated_at: datetime


# ---------- Alerts ----------


class AlertCreate(BaseModel):
    case_id: str
    alert_type: AlertType
    priority: AlertPriority = AlertPriority.MEDIUM
    title: str = Field(..., min_length=1, max_length=512)
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlertResponse(BaseModel):
    id: str
    case_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    metadata: dict[str, Any]
    sent: bool
    sent_at: datetime | None
    created_at: datetime


class AlertListResponse(BaseModel):
    alerts: list[AlertResponse]
    total: int


class WebhookTarget(BaseModel):
    id: str
    name: str
    url: str
    secret: str | None = None
    alert_types: list[AlertType] = Field(default_factory=list)
    enabled: bool = True


class WebhookTargetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    url: str
    secret: str | None = None
    alert_types: list[AlertType] = Field(default_factory=list)
    enabled: bool = True


class WebhookTargetResponse(BaseModel):
    id: str
    name: str
    url: str
    alert_types: list[AlertType]
    enabled: bool
    created_at: datetime


# ---------- WebSocket ----------


class WSEventType(str, Enum):
    SPILL_DETECTED = "SPILL_DETECTED"
    DRIFT_COMPLETE = "DRIFT_COMPLETE"
    SUSPECT_RANKED = "SUSPECT_RANKED"
    NARRATIVE_READY = "NARRATIVE_READY"
    CASE_FILE_READY = "CASE_FILE_READY"
    PIPELINE_PROGRESS = "PIPELINE_PROGRESS"


class WSEvent(BaseModel):
    type: WSEventType
    payload: dict[str, Any]


class WSSubscribe(BaseModel):
    case_id: str


# ---------- Health ----------


class ServiceHealth(BaseModel):
    name: str
    status: Literal["healthy", "degraded", "unreachable"]
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    services: list[ServiceHealth]
    uptime_seconds: float


# ---------- Pipeline ----------


class PipelineTrigger(BaseModel):
    case_id: str
    stages: list[str] = Field(default_factory=lambda: ["detect", "drift", "attribute", "intel"])
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional stage inputs (spill geometry, image path, ...). "
        "Defaults target the Wakashio demo when omitted.",
    )


class PipelineStatus(BaseModel):
    case_id: str
    current_stage: str | None
    stages_completed: list[str]
    stages_failed: list[str]
    started_at: datetime
    updated_at: datetime
    queued: bool = True
    detail: str | None = None
