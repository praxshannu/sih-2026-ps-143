from __future__ import annotations

from datetime import UTC, datetime
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


# ---------- Suspects (wire contract: the API is the source of truth) ----------

CoverageState = Literal["real", "synthetic", "none"]


class PipelineSuspect(BaseModel):
    """One ranked suspect as served by ``GET /api/v1/cases/{id}/suspects``.

    Mirrors ``services/attribute/app/schemas.py:PipelineSuspect`` exactly. The
    UI's TypeScript ``PipelineSuspect`` must match this field-for-field;
    where it currently differs (optional lat/lon, no audit fields) the API
    wins and the UI is aligned to it.

    ``latitude``/``longitude`` are nullable, never defaulted to 0: 0°N 0°E is
    a real place, and a silent default would put a vessel in the Gulf of
    Guinea. Wilson 95% CI always accompanies ``composite_score``.
    """

    mmsi: str
    vessel_name: str = "UNKNOWN"
    composite_score: float = Field(ge=0.0, le=1.0)
    confidence_lower: float = Field(ge=0.0, le=1.0)
    confidence_upper: float = Field(ge=0.0, le=1.0)
    confidence_method: str = "wilson_95"
    fuzzy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    xgb_score: float | None = None
    xgb_method: str = "unavailable"
    shap_breakdown: dict[str, float] = Field(default_factory=dict)
    score_proximity: float = Field(default=0.0, ge=0.0, le=1.0)
    score_temporal: float = Field(default=0.0, ge=0.0, le=1.0)
    score_trajectory: float = Field(default=0.0, ge=0.0, le=1.0)
    score_anomaly: float = Field(default=0.0, ge=0.0, le=1.0)
    score_vessel_type: float = Field(default=0.0, ge=0.0, le=1.0)
    score_history: float = Field(default=0.0, ge=0.0, le=1.0)
    ais_gap_minutes: float = Field(default=0.0, ge=0.0)
    is_dark_vessel: bool = False
    rank: int = Field(default=1, ge=1)
    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    speed_knots: float | None = None
    course_deg: float | None = None
    closest_approach_nm: float | None = None
    matched_ping_count: int = Field(default=0, ge=0)
    ais_provenance: str = "unknown"


class PersistenceInfo(BaseModel):
    """Where a result is actually stored — never let a fallback look like a DB.

    ``backend`` is ``postgres`` or ``json_fallback``. The fallback is durable
    on disk but it is NOT the database, and the response says so.
    """

    backend: Literal["postgres", "json_fallback"]
    durable: bool = True
    path: str | None = None
    reason: str | None = None


class CaseSuspectsResponse(BaseModel):
    """Response of ``GET /api/v1/cases/{id}/suspects``.

    When AIS coverage is not ``real`` the suspect list is empty and
    ``coverage_reason`` carries a machine-readable code. Never a ranking
    built on synthetic or unverified AIS.
    """

    case_id: str
    suspects: list[PipelineSuspect] = Field(default_factory=list)
    total: int = 0
    coverage: CoverageState = "none"
    coverage_reason: str | None = None
    coverage_message: str | None = None
    ais_provenance: str = "unknown"
    weights: dict[str, float] = Field(default_factory=dict)
    persistence: PersistenceInfo | None = None


# ---------- Scene persistence ----------


class SceneMetadata(BaseModel):
    """Which scene this result came from, and how to verify the bytes."""

    scene_id: str | None = None
    label: str | None = None
    platform: str | None = None
    product_id: str | None = None
    acquisition_time: datetime | None = None
    source: str | None = None
    checksum: str | None = None


class DetectionResult(BaseModel):
    """Detection stage: method, geometry, confidence, model version."""

    method: str | None = None
    model_version: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    area_km2: float | None = None
    polygon: dict[str, Any] | None = Field(
        default=None, description="GeoJSON geometry of the detected slick"
    )
    payload: dict[str, Any] = Field(default_factory=dict)


class DriftResult(BaseModel):
    """Drift stage: origin ellipse + the raw runner payload."""

    origin_ellipse: dict[str, Any] | None = Field(
        default=None, description="GeoJSON polygon or {center_lon, center_lat, ...}"
    )
    payload: dict[str, Any] = Field(default_factory=dict)


class ForcingProvenance(BaseModel):
    """Which forcing actually drove the drift run (never assumed)."""

    wind_source: str = "unavailable"
    current_source: str = "unavailable"
    synthetic: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)


class AisProvenance(BaseModel):
    """Which AIS the suspects were scored from, and whether it is real."""

    coverage: CoverageState = "none"
    provenance: str = "unknown"
    reason: str | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ProcessedScene(BaseModel):
    """Everything one processed scene must persist.

    Every field here exists so a result can be challenged later: where the
    pixels came from (source + checksum), what was detected (polygon +
    confidence + CI), how the drift was forced, which AIS the suspects were
    scored from, and what the model version was.
    """

    case_id: str
    scene: SceneMetadata = Field(default_factory=SceneMetadata)
    detection: DetectionResult = Field(default_factory=DetectionResult)
    drift: DriftResult = Field(default_factory=DriftResult)
    forcing: ForcingProvenance = Field(default_factory=ForcingProvenance)
    ais: AisProvenance = Field(default_factory=AisProvenance)
    suspects: list[PipelineSuspect] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    model_version: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# Fields that must survive a round-trip. Used by the persistence tests and by
# the store's own completeness check — an incomplete record is a failed
# record, not a partially true one.
REQUIRED_PERSISTED_FIELDS: tuple[str, ...] = (
    "case_id",
    "scene",
    "detection",
    "drift",
    "forcing",
    "ais",
    "suspects",
    "model_version",
    "confidence",
    "warnings",
)


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
