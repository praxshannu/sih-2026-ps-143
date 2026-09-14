"""Pydantic v2 schemas for the SENTINEL Ingest service."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# --- Enums ---


class DataSource(str, Enum):
    SENTINEL1 = "sentinel1"
    CMEMS = "cmems"
    ERA5 = "era5"
    AIS = "ais"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Input Schemas ---


class TriggerRequest(BaseModel):
    """Manual trigger for full data ingestion cycle."""

    sources: list[DataSource] | None = Field(
        None, description="Specific sources to fetch (None = all)"
    )
    force: bool = Field(False, description="Force re-fetch even if recently fetched")


class Sentinel1Params(BaseModel):
    """Parameters for Sentinel-1 product search."""

    bbox: tuple[float, float, float, float] = Field(
        default=(68.0, 5.0, 88.0, 25.0),
        description="Bounding box [lon_min, lat_min, lon_max, lat_max]",
    )
    product_type: Literal["GRD", "SLC", "OCN"] = Field(
        default="GRD", description="Sentinel-1 product type"
    )
    max_results: int = Field(50, ge=1, le=500, description="Max products to fetch")
    lookback_days: int = Field(6, ge=1, le=30, description="Search window in days")


class CmemsParams(BaseModel):
    """Parameters for CMEMS ocean current fetch."""

    bbox: tuple[float, float, float, float] = Field(default=(68.0, 5.0, 88.0, 25.0))
    depth_range: tuple[float, float] = Field(default=(0.0, 50.0))
    lookback_days: int = Field(7, ge=1, le=30)


class Era5Params(BaseModel):
    """Parameters for ERA5 wind field fetch."""

    bbox: tuple[float, float, float, float] = Field(default=(68.0, 5.0, 88.0, 25.0))
    lookback_days: int = Field(7, ge=1, le=30)


class AisParams(BaseModel):
    """Parameters for AIS data ingestion."""

    bbox: tuple[float, float, float, float] = Field(default=(68.0, 5.0, 88.0, 25.0))
    lookback_hours: int = Field(6, ge=1, le=168)
    interpolate_gaps: bool = Field(True, description="Interpolate gaps < 5 min")


# --- Sentinel-1 Product Metadata ---


class Sentinel1Product(BaseModel):
    """Metadata for a single Sentinel-1 product."""

    product_id: str
    name: str
    acquisition_time: datetime
    orbit_number: int
    orbit_direction: Literal["ASCENDING", "DESCENDING"]
    polarization: Literal["HH", "HV", "VH", "VV"]
    product_type: str
    footprint: str = Field(description="WKT polygon of the product footprint")
    download_url: str
    file_size_bytes: int | None = None
    status: str = "new"


class Sentinel1Result(BaseModel):
    """Result of Sentinel-1 ingestion."""

    products_found: int
    products_downloaded: int
    products_skipped: int
    products: list[Sentinel1Product]
    fetch_duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- Ocean Data ---


class OceanCurrentResult(BaseModel):
    """Result of CMEMS ocean current fetch."""

    files_downloaded: int
    total_bytes: int
    bbox: tuple[float, float, float, float]
    time_range: tuple[str, str]
    variable_names: list[str]
    fetch_duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WindFieldResult(BaseModel):
    """Result of ERA5 wind field fetch."""

    files_downloaded: int
    total_bytes: int
    bbox: tuple[float, float, float, float]
    time_range: tuple[str, str]
    variable_names: list[str]
    fetch_duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- AIS Data ---


class AisPosition(BaseModel):
    """Normalized AIS position report."""

    mmsi: str = Field(..., min_length=9, max_length=9)
    timestamp: datetime
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)
    sog: float | None = Field(None, ge=0, le=102.3)
    cog: float | None = Field(None, ge=0, le=360)
    heading: int | None = Field(None, ge=0, le=359)
    nav_status: int | None = None
    vessel_name: str | None = None
    vessel_type: int | None = None
    imo_number: str | None = None
    flag_state: str | None = None
    source: str = "marinecadastre"
    is_interpolated: bool = False


class AisIngestResult(BaseModel):
    """Result of AIS data ingestion."""

    records_fetched: int
    records_upserted: int
    unique_vessels: int
    interpolated_gaps: int
    fetch_duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- Scheduler Status ---


class SourceStatus(BaseModel):
    """Status of a single data source."""

    source: DataSource
    last_fetch_time: datetime | None = None
    last_fetch_status: JobStatus = JobStatus.PENDING
    last_fetch_duration_seconds: float | None = None
    last_error: str | None = None
    total_fetches: int = 0


class SchedulerStatus(BaseModel):
    """Overall scheduler status."""

    running: bool
    next_run_time: datetime | None = None
    interval_hours: int = 6
    sources: list[SourceStatus]
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- Health ---


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "1.0.0"
    service: str = "sentinel-ingest"
    db_connected: bool = False
