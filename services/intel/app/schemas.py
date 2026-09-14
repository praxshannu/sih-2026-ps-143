"""Pydantic v2 schemas for sentinel-intel API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class DetectionData(BaseModel):
    """Structured detection data from the detect service."""

    spill_id: str = Field(..., description="Spill detection UUID")
    case_id: str = Field(..., description="Investigation case UUID")
    centroid_lon: float = Field(..., ge=-180, le=180)
    centroid_lat: float = Field(..., ge=-90, le=90)
    area_m2: float = Field(..., gt=0, description="Spill area in square metres")
    confidence: float = Field(..., ge=0.0, le=1.0)
    age_hours: float = Field(..., ge=0, description="Estimated spill age in hours")
    image_path: str = Field(..., description="Path to SAR detection image")
    detected_at: datetime = Field(..., description="Detection timestamp (UTC)")
    geojson: dict[str, Any] = Field(default_factory=dict, description="Spill polygon GeoJSON")
    spill_polygon_wkt: str | None = Field(None, description="Spill polygon as WKT")


class DriftData(BaseModel):
    """Drift analysis results from the drift service."""

    origin_lon: float
    origin_lat: float
    origin_lon_sigma: float = 0.01
    origin_lat_sigma: float = 0.01
    semi_major_km: float = Field(default=0.0, ge=0)
    semi_minor_km: float = Field(default=0.0, ge=0)
    orientation_deg: float = 0.0
    regime: str = "markov1"
    forecast_paths: list[list[list[float]]] = Field(
        default_factory=list, description="72-hour forecast trajectory paths"
    )


class SuspectVessel(BaseModel):
    """A single suspect vessel from the attribute service."""

    mmsi: str
    vessel_name: str = "UNKNOWN"
    composite_score: float = Field(ge=0.0, le=1.0)
    score_proximity: float = Field(0.0, ge=0.0, le=1.0)
    score_temporal: float = Field(0.0, ge=0.0, le=1.0)
    score_trajectory: float = Field(0.0, ge=0.0, le=1.0)
    score_anomaly: float = Field(0.0, ge=0.0, le=1.0)
    score_vessel_type: float = Field(0.0, ge=0.0, le=1.0)
    ais_gap_minutes: float = Field(0.0, ge=0)
    is_dark_vessel: bool = False
    rank: int = Field(ge=1)


class AisPosition(BaseModel):
    """Single AIS position report."""

    timestamp: datetime
    lon: float
    lat: float
    sog: float = Field(0.0, ge=0)
    cog: float = Field(0.0, ge=0, le=360)
    heading: float | None = None


class AisExcerpt(BaseModel):
    """AIS track excerpt for a single vessel."""

    mmsi: str
    vessel_name: str = "UNKNOWN"
    positions: list[AisPosition] = Field(default_factory=list)
    gap_start: datetime | None = None
    gap_end: datetime | None = None


class ChainEntry(BaseModel):
    """Single entry in the chain of custody log."""

    stage: str = Field(..., description="Pipeline stage name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_hash: str = Field(default="", description="SHA-256 of stage input")
    output_hash: str = Field(default="", description="SHA-256 of stage output")
    service: str = Field(default="sentinel-intel", description="Service that produced this entry")


class NarrativeRequest(BaseModel):
    """Request to generate an LLM narrative summary."""

    detection: DetectionData
    drift: DriftData | None = None
    suspects: list[SuspectVessel] = Field(default_factory=list)
    evidence_hash: str = Field(default="", description="SHA-256 hash of evidence package")


class CaseFileRequest(BaseModel):
    """Request to generate a PDF evidence case file."""

    case_id: str = Field(..., description="Investigation case UUID")
    detection: DetectionData
    drift: DriftData | None = None
    suspects: list[SuspectVessel] = Field(default_factory=list)
    ais_excerpt: AisExcerpt | None = None
    narrative: str | None = Field(None, description="Pre-generated narrative text")
    chain_of_custody: list[ChainEntry] = Field(default_factory=list)


class HashRequest(BaseModel):
    """Request to compute an evidence chain hash."""

    case_id: str
    stage: str = Field(..., description="Pipeline stage name")
    data: dict[str, Any] = Field(..., description="Data to hash")
    previous_hash: str = Field(default="", description="Hash of previous stage")


class AlertRequest(BaseModel):
    """Request to dispatch a case alert."""

    case_id: str
    spill_id: str = Field(default="")
    alert_priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    summary: str = Field(..., description="Alert summary text")
    recipients: list[str] = Field(default_factory=list, description="Email addresses")
    webhook_urls: list[str] = Field(default_factory=list, description="Webhook endpoints")
    evidence_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class NarrativeResult(BaseModel):
    """LLM-generated narrative output."""

    summary: str = Field(..., description="Main incident summary (max 4 sentences)")
    key_finding: str = Field(..., description="Single most important finding")
    evidentiary_gaps: list[str] = Field(default_factory=list)
    legal_basis: str = Field(default="", description="Applicable legal framework")
    alert_priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    evidence_hash: str = Field(default="")
    model_used: str = Field(default="gpt-4o")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CaseFileResult(BaseModel):
    """Generated PDF case file."""

    case_id: str
    pdf_path: str
    evidence_hash: str
    page_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class HashResult(BaseModel):
    """Evidence chain hash result."""

    case_id: str
    stage: str
    input_hash: str
    output_hash: str
    chain_hash: str = Field(description="Cumulative chain hash including previous stages")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AlertResult(BaseModel):
    """Alert dispatch result."""

    case_id: str
    alert_id: str
    emails_sent: int = 0
    webhooks_sent: int = 0
    failed_recipients: list[str] = Field(default_factory=list)
    failed_webhooks: list[str] = Field(default_factory=list)
    dispatched_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    engine: str = "sentinel-intel"
    llm_provider: str = "openai"
    llm_available: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
