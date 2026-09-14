"""Pydantic v2 schemas for sentinel-attribute API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# AIS coverage states. Only "real" permits a suspect ranking (see
# app/engine/ais_coverage.py — no free AIS source covers the open Indian
# Ocean, so an unproven region must never produce a ranking).
CoverageState = Literal["real", "synthetic", "none"]

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
    radar_cross_section_db: float | None = None


class AttribRequest(BaseModel):
    """Full attribution pipeline request.

    AOI is supplied with **named axes**. The positional ``bbox`` string is
    kept only as a deprecated alias: a lat-first string is four numbers that
    are all legal in either slot, so no validator can detect a swap and the
    real/synthetic verdict would flip silently.
    """

    model_config = {"strict": False}

    case_id: str = Field(..., description="Investigation case UUID")
    spill_id: str = Field(..., description="Spill detection UUID")
    origin: OriginEllipse
    spill_time: datetime = Field(..., description="Estimated spill origin time")
    # --- AOI: named axes are the contract; deprecated string is an alias ---
    min_lon: float | None = Field(None, ge=-180, le=180)
    min_lat: float | None = Field(None, ge=-90, le=90)
    max_lon: float | None = Field(None, ge=-180, le=180)
    max_lat: float | None = Field(None, ge=-90, le=90)
    bbox: str | None = Field(
        None,
        description="DEPRECATED 'west,south,east,north'. Use the named axes: this "
        "form cannot be validated against axis-order mistakes.",
    )
    ais_provenance: str | None = Field(
        None,
        description="Provenance label of the AIS being scored, e.g. live_terrestrial "
        "or synthetic_mock. Unrecognised labels are treated as unverified and "
        "suppress the ranking.",
    )
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
    # Provenance of the AIS fix this score is computed from. Optional, but
    # without it the UI cannot say where the number came from.
    timestamp: datetime | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    speed_knots: float | None = Field(default=None, ge=0)
    course_deg: float | None = Field(default=None, ge=0, le=360)
    matched_ping_count: int = Field(default=0, ge=0)
    ais_provenance: str = "unknown"
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
    """Pre-scored suspect for ranking.

    Carries the same provenance fields as ``ScoreResult`` so the audit trail
    survives a re-rank (a number that arrives without its fix is not evidence).
    """

    mmsi: str
    vessel_name: str = "UNKNOWN"
    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    speed_knots: float | None = None
    course_deg: float | None = None
    closest_approach_nm: float | None = None
    matched_ping_count: int = Field(0, ge=0)
    ais_provenance: str = "unknown"
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

    The audit fields (timestamp, latitude, longitude, speed_knots, course_deg,
    closest_approach_nm, matched_ping_count, ais_gap_minutes, ais_provenance)
    are what make a score defensible: they say *which* AIS fix the number came
    from. They are nullable — a missing position is `null`, never 0.0, because
    0°N 0°E is a real place (and a lie).
    """

    mmsi: str
    vessel_name: str = "UNKNOWN"
    composite_score: float = Field(ge=0.0, le=1.0)
    confidence_lower: float = Field(ge=0.0, le=1.0)
    confidence_upper: float = Field(ge=0.0, le=1.0)
    confidence_method: str = Field(default="wilson_95")
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
    # --- audit trail: the AIS fix this score was computed from ---
    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    speed_knots: float | None = None
    course_deg: float | None = None
    closest_approach_nm: float | None = None
    matched_ping_count: int = Field(default=0, ge=0)
    ais_provenance: str = "unknown"


class PipelineSuspect(BaseModel):
    """The wire contract for one ranked suspect (gateway + UI).

    Mirrors ``ScoreResult`` 1:1 and is deliberately declared here, in the
    service that produces it, so the UI's ``PipelineSuspect`` TypeScript type
    has a single source of truth to align to. ``latitude``/``longitude`` are
    nullable on purpose: "no position" and "position 0,0" must stay
    distinguishable.
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
    score_proximity: float = Field(ge=0.0, le=1.0)
    score_temporal: float = Field(ge=0.0, le=1.0)
    score_trajectory: float = Field(ge=0.0, le=1.0)
    score_anomaly: float = Field(ge=0.0, le=1.0)
    score_vessel_type: float = Field(ge=0.0, le=1.0)
    score_history: float = Field(ge=0.0, le=1.0)
    ais_gap_minutes: float = Field(ge=0.0)
    is_dark_vessel: bool = False
    rank: int = Field(ge=1)
    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    speed_knots: float | None = None
    course_deg: float | None = None
    closest_approach_nm: float | None = None
    matched_ping_count: int = Field(default=0, ge=0)
    ais_provenance: str = "unknown"

    @classmethod
    def from_score_result(cls, s: ScoreResult) -> PipelineSuspect:
        return cls(**s.model_dump())


class AttribResponse(BaseModel):
    """Full attribution pipeline response.

    When ``coverage`` is not ``"real"`` the suspects list is **always empty**
    and ``coverage_reason`` carries a machine-readable code. A ranking built
    on synthetic or unverified AIS looks like evidence and is not evidence.
    """

    case_id: str
    spill_id: str
    suspects: list[ScoreResult] = Field(default_factory=list)
    total_candidates: int = Field(default=0)
    dark_vessels_found: int = Field(default=0)
    xgb_candidates: int = Field(default=0)
    scoring_weights: dict[str, float] = Field(default_factory=dict)
    pipeline_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # --- AIS coverage gate ---
    coverage: CoverageState = Field(
        default="none", description="real | synthetic | none — only 'real' is rankable"
    )
    coverage_reason: str | None = Field(
        default=None, description="Machine-readable REASON_* code when coverage != real"
    )
    coverage_message: str | None = Field(default=None, description="Human-readable explanation")
    ais_provenance: str = Field(default="unknown")
    rankable: bool = Field(default=False)


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
