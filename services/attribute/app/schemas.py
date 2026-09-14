"""Pydantic v2 schemas for sentinel-attribute API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class OriginEllipse(BaseModel):
    """95% confidence ellipse from drift backward analysis."""

    center_lon: float = Field(..., description="Ellipse center longitude")
    center_lat: float = Field(..., description="Ellipse center latitude")
    semi_major_km: float = Field(..., gt=0, description="Semi-major axis (km)")
    semi_minor_km: float = Field(..., gt=0, description="Semi-minor axis (km)")
    orientation_deg: float = Field(0.0, description="Orientation from East (deg)")


class SarDetection(BaseModel):
    """A non-AIS SAR return at the origin location."""

    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)
    timestamp: datetime
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    radar_cross_section_db: Optional[float] = None


class AttribRequest(BaseModel):
    """Full attribution pipeline request."""

    model_config = {"strict": False}

    case_id: str = Field(..., description="Investigation case UUID")
    spill_id: str = Field(..., description="Spill detection UUID")
    origin: OriginEllipse
    spill_time: datetime = Field(..., description="Estimated spill origin time")
    search_radius_nm: float = Field(
        default=50.0, gt=0, le=200, description="Search radius (nautical miles)"
    )
    time_window_hours: float = Field(
        default=6.0, gt=0, le=48, description="Time window around origin (hours)"
    )
    sar_targets: list[SarDetection] = Field(
        default_factory=list, description="SAR non-AIS targets at origin"
    )
    vessel_type_risk_map: dict[str, float] = Field(
        default_factory=dict,
        description="MMSI -> vessel type risk score (0-1)",
    )
    historical_violations: dict[str, int] = Field(
        default_factory=dict,
        description="MMSI -> number of past violations",
    )


class ScoreRequest(BaseModel):
    """Score a single suspect vessel."""

    model_config = {"strict": False}

    mmsi: str = Field(..., min_length=1, max_length=9)
    vessel_name: str = Field(default="UNKNOWN")
    min_distance_nm: float = Field(..., ge=0)
    time_delta_minutes: float = Field(..., ge=0)
    trajectory_intersection_score: float = Field(0.0, ge=0.0, le=1.0)
    ais_gap_minutes: float = Field(0.0, ge=0)
    speed_anomaly_sigma: float = Field(0.0, ge=0)
    course_anomaly_sigma: float = Field(0.0, ge=0)
    vessel_type_risk: float = Field(0.0, ge=0.0, le=1.0)
    historical_violations: int = Field(0, ge=0)
    is_dark_sar_target: bool = Field(False)


class RankedSuspect(BaseModel):
    """Pre-scored suspect for ranking."""

    mmsi: str
    vessel_name: str = "UNKNOWN"
    composite_score: float = Field(0.0, ge=0.0, le=1.0)
    score_proximity: float = Field(0.0, ge=0.0, le=1.0)
    score_temporal: float = Field(0.0, ge=0.0, le=1.0)
    score_trajectory: float = Field(0.0, ge=0.0, le=1.0)
    score_anomaly: float = Field(0.0, ge=0.0, le=1.0)
    score_vessel_type: float = Field(0.0, ge=0.0, le=1.0)
    ais_gap_minutes: float = Field(0.0, ge=0)
    is_dark_vessel: bool = False


class RankRequest(BaseModel):
    """Rank a list of pre-scored suspects."""

    model_config = {"strict": False}

    suspects: list[RankedSuspect] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class ScoreResult(BaseModel):
    """Output of the dual scorer for a single suspect.

    composite_score is the weighted average of the fuzzy Cauchy score and
    the XGBoost/DuckDB score (weights via ATTRIBUTION_FUZZY_WEIGHT /
    ATTRIBUTION_XGB_WEIGHT). Wilson 95% CI always accompanies the point
    estimate. shap_breakdown drives the SuspectCard waterfall.
    """

    mmsi: str
    vessel_name: str = "UNKNOWN"
    composite_score: float = Field(ge=0.0, le=1.0)
    confidence_lower: float = Field(ge=0.0, le=1.0)
    confidence_upper: float = Field(ge=0.0, le=1.0)
    score_proximity: float = Field(ge=0.0, le=1.0)
    score_temporal: float = Field(ge=0.0, le=1.0)
    score_trajectory: float = Field(ge=0.0, le=1.0)
    score_anomaly: float = Field(ge=0.0, le=1.0)
    score_vessel_type: float = Field(ge=0.0, le=1.0)
    score_history: float = Field(ge=0.0, le=1.0)
    fuzzy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    xgb_score: float | None = Field(default=None, ge=0.0, le=1.0)
    xgb_method: str = Field(default="unavailable")
    shap_breakdown: dict[str, float] = Field(default_factory=dict)
    ais_gap_minutes: float = Field(ge=0)
    is_dark_vessel: bool = False
    rank: int = Field(ge=1)


class AttribResponse(BaseModel):
    """Full attribution pipeline response."""

    case_id: str
    spill_id: str
    suspects: list[ScoreResult] = Field(default_factory=list)
    total_candidates: int = Field(default=0)
    dark_vessels_found: int = Field(default=0)
    xgb_candidates: int = Field(default=0)
    scoring_weights: dict[str, float] = Field(default_factory=dict)
    pipeline_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RankResponse(BaseModel):
    """Ranked suspects response."""

    suspects: list[ScoreResult] = Field(default_factory=list)
    total: int = Field(default=0)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    engine: str = "sentinel-attribute"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
