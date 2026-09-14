"""Pydantic v2 schemas for drift inputs/outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# --- Input Schemas ---


class OceanDataRef(BaseModel):
    """Reference to CMEMS/ERA5 data stores.

    Paths are optional: when omitted the service uses the forcing factory
    (live CMEMS/ERA5 if credentials exist, else NOAA GFS + labelled mock)
    and always reports `forcing_source` in the response.
    """

    cmems_base: str = Field(
        default="", description="Base path to CMEMS GLORYS12 NetCDF files"
    )
    era5_base: str = Field(default="", description="Base path to ERA5 NetCDF files")
    time_start: datetime = Field(..., description="Start of analysis window")
    time_end: datetime = Field(..., description="End of analysis window")
    bbox: tuple[float, float, float, float] = Field(
        ..., description="Bounding box [lon_min, lat_min, lon_max, lat_max]"
    )


class BackwardRequest(BaseModel):
    """Request payload for backward drift (origin finding)."""

    spill_lon: float = Field(..., ge=-180, le=180, description="Spill centroid longitude")
    spill_lat: float = Field(..., ge=-90, le=90, description="Spill centroid latitude")
    spill_age_hours: float = Field(
        ..., gt=0, le=720, description="Hours since spill began"
    )
    ocean_data: OceanDataRef
    n_particles: int = Field(1000, ge=10, le=10000, description="Ensemble size")
    random_seed: int | None = Field(None, description="Reproducibility seed")


class ForwardRequest(BaseModel):
    """Request payload for forward forecast."""

    origin_lon: float = Field(..., ge=-180, le=180, description="Origin longitude")
    origin_lat: float = Field(..., ge=-90, le=90, description="Origin latitude")
    origin_lon_sigma: float = Field(
        0.01, ge=0, description="Origin longitude uncertainty (deg)"
    )
    origin_lat_sigma: float = Field(
        0.01, ge=0, description="Origin latitude uncertainty (deg)"
    )
    forecast_hours: float = Field(72, gt=0, le=168, description="Forecast duration")
    ocean_data: OceanDataRef
    n_particles: int = Field(1000, ge=10, le=10000)
    random_seed: int | None = None


class FullPipelineRequest(BaseModel):
    """Full pipeline: backward + forward + shoreline risk."""

    spill_lon: float = Field(..., ge=-180, le=180)
    spill_lat: float = Field(..., ge=-90, le=90)
    spill_age_hours: float = Field(..., gt=0, le=720)
    ocean_data: OceanDataRef
    forecast_hours: float = Field(72, gt=0, le=168)
    n_particles: int = Field(1000, ge=10, le=10000)
    random_seed: int | None = None


# --- Output Schemas ---


class ConfidenceEllipse(BaseModel):
    """95% confidence ellipse from particle distribution."""

    center_lon: float
    center_lat: float
    semi_major_km: float
    semi_minor_km: float
    orientation_deg: float = Field(description="Angle of semi-major axis from East")


class DriftResult(BaseModel):
    """Result of backward drift computation."""

    origin_ellipse: ConfidenceEllipse
    particle_count: int
    regime: Literal["markov1", "redi", "smagorinsky"]
    origin_points_lon: list[float]
    origin_points_lat: list[float]
    mean_dispersion_km: float
    forcing_source: str = Field(
        default="synthetic_mock",
        description="Provenance: live_cmems_era5 | composite_gfs_mock | synthetic_mock | local_files",
    )
    forcing_detail: dict = Field(
        default_factory=dict, description="Per-source origins, never silently mocked"
    )
    xgb_residual_applied: bool = Field(
        default=False, description="Whether XGBoost physics residual correction ran"
    )
    xgb_correction_m: dict = Field(
        default_factory=dict,
        description="{dx, dy, scale} correction in metres + uncertainty scale",
    )
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class TrajectoryPoint(BaseModel):
    """Single trajectory point."""

    lon: float
    lat: float
    time_hours: float


class TrajectoryEnsemble(BaseModel):
    """Ensemble of trajectories with statistics."""

    mean_path: list[TrajectoryPoint]
    quantile_5_path: list[TrajectoryPoint]
    quantile_95_path: list[TrajectoryPoint]
    all_paths_lon: list[list[float]]
    all_paths_lat: list[list[float]]


class ShorelineRisk(BaseModel):
    """Shoreline impact assessment."""

    risk_index: float = Field(ge=0, le=1, description="Normalized risk 0-1")
    nearest_impact_km: float
    impact_probability: float = Field(ge=0, le=1)
    coastline_segments_at_risk: int
    trajectory_landfall_points_lon: list[float]
    trajectory_landfall_points_lat: list[float]


class ForwardResult(BaseModel):
    """Result of forward forecast."""

    trajectories: TrajectoryEnsemble
    regime: Literal["markov1", "redi", "smagorinsky"]
    forecast_hours: float
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class FullPipelineResult(BaseModel):
    """Combined result of backward + forward + shoreline risk."""

    backward: DriftResult
    forward: ForwardResult
    shoreline_risk: ShorelineRisk
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    engine: str = "sentinel-drift"
