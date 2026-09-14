"""HTTP routes that wrap the OpenDrift backward attribution.

Mounted by ``app.main_attribution:app`` (the torch-free / OpenDrift-only entry
point for this M2). The endpoint takes a detection polygon (centroid + area),
an acquisition time, and an AOI; it pulls ERA5 wind + CMEMS currents for
the backtrack window, runs the backward ensemble, and returns the origin
ellipse + a vessel suspect list ranked by proximity.

Endpoints
---------
POST /drift/attribution
    One backward hindcast. Returns AttributionResult (origin ellipse,
    suspect vessels, wind/current sources, WMC divergence max, run UTC).
GET  /drift/health
    Confirms OpenDrift is importable + the wind/current sources status.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Allow the FastAPI app to import the sibling ``runner`` module without
# setting PYTHONPATH manually. ``app/main_attribution.py`` adjusts sys.path
# at boot.
from app.runner import (
    AttributionResult,
    EnsembleConfig,
    run_backward_attribution,
    run_forward_forecast,
)
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/drift", tags=["drift"])

DATA_DIR = Path(os.getenv("SENTINEL_DATA_DIR", "/app/data"))

# How far beyond the backtrack window to fetch forcing. See _fetch_era5 for
# why this is not optional.
FORCING_PAD_HOURS = 3


class AttributionRequest(BaseModel):
    # Detection (a polygon centroid + area + acquisition time)
    detection_lon: float
    detection_lat: float
    detection_area_km2: float = Field(..., ge=0.001, le=200.0)
    detection_time: str = Field(..., description="ISO-8601 UTC")

    # Backtrack window
    duration_h: float = Field(48.0, ge=1, le=168)

    # Optional vessel list (or pass an AOI + bbox and we will ask the
    # ingest service for synthetic / live AIS).
    vessels: list[dict[str, Any]] | None = Field(
        default=None,
        description="Vessels to score. Each: {mmsi, name, longitude, latitude, flag?, provenance?}",
    )

    # Forcing
    use_era5: bool = Field(True, description="Pull ERA5 hourly 10m wind for the window")
    use_cmems: bool = Field(True, description="Pull CMEMS GLORYS12 currents for the window")
    # "auto" -> GFS when the window is recent (no CDS queue), ERA5 otherwise.
    # "gfs"/"era5" pin the source. GFS only retains ~10 days on NOMADS.
    forcing: str = Field("auto", description="Wind source: auto | era5 | gfs")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="(W,S,E,N) override for the forcing bbox; default = a 5° box around the detection",
    )

    # OpenDrift knobs
    n_members: int = Field(64, ge=16, le=2048)
    seed: int = 20200725

    # Synthetic-fallback knobs (only used if ERA5/CMEMS fetch fails)
    synthetic_wind_ms: float = 5.0
    synthetic_wind_dir_from_deg: float = 110.0
    synthetic_current_ms: float = 0.10
    synthetic_current_dir_deg: float = 220.0


def _default_bbox(
    det_lon: float, det_lat: float, pad_deg: float = 5.0
) -> tuple[float, float, float, float]:
    return (det_lon - pad_deg, det_lat - pad_deg, det_lon + pad_deg, det_lat + pad_deg)


def _fetch_gfs(bbox, start_dt, end_dt):
    """Pull NOAA GFS 10m wind. Instant, but recent-only — see sources/gfs.py."""
    from app.sources.gfs import fetch_gfs_wind
    from opendrift.readers import reader_netCDF_CF_generic

    ds = fetch_gfs_wind(bbox, start_dt, end_dt)
    ds = ds.rename({"u10": "x_wind", "v10": "y_wind"})
    for var, sname in (("x_wind", "eastward_wind"), ("y_wind", "northward_wind")):
        ds[var].attrs["standard_name"] = sname
        ds[var].attrs["units"] = "m s-1"
    tmp = Path(tempfile.gettempdir()) / f"gfs_{datetime.now().timestamp():.0f}.nc"
    ds.to_netcdf(tmp)
    reader = reader_netCDF_CF_generic.Reader(str(tmp))
    reader.sentinel_source = "gfs"
    return reader


def _resolve_wind(bbox, start_dt, end_dt, forcing: str, notes: list[str]):
    """Pick a wind source. Returns (reader|None, source_name).

    ``auto`` prefers GFS for recent detections (no CDS queue) and falls back
    to ERA5 for anything older — GFS on NOMADS only retains ~10 days, so for
    a 2020 case study it is not merely slower, it is unavailable.
    """
    if forcing == "gfs":
        try:
            return _fetch_gfs(bbox, start_dt, end_dt), "gfs"
        except Exception as exc:
            notes.append(f"GFS unavailable: {str(exc)[:200]}")
            logger.warning("GFS fetch failed: {}", exc)
            return None, "synthetic_constant"

    if forcing == "era5":
        try:
            return _fetch_era5(bbox, start_dt.isoformat(), end_dt.isoformat()), "era5"
        except Exception as exc:
            notes.append(f"ERA5 unavailable: {str(exc)[:200]}")
            logger.warning("ERA5 fetch failed: {}", exc)
            return None, "synthetic_constant"

    # auto
    from app.sources.gfs import supports_window

    ok, reason = supports_window(start_dt, end_dt)
    if ok:
        try:
            return _fetch_gfs(bbox, start_dt, end_dt), "gfs"
        except Exception as exc:
            notes.append(f"GFS failed ({str(exc)[:120]}); falling back to ERA5.")
            logger.warning("GFS fetch failed, falling back to ERA5: {}", exc)
    else:
        notes.append(f"GFS not usable: {reason}")
    try:
        return _fetch_era5(bbox, start_dt.isoformat(), end_dt.isoformat()), "era5"
    except Exception as exc:
        notes.append(f"ERA5 unavailable: {str(exc)[:200]}")
        logger.warning("ERA5 fetch failed: {}", exc)
        return None, "synthetic_constant"


@router.get("/health")
async def health() -> dict[str, Any]:
    """Whether OpenDrift + the forcing fetchers are importable."""
    out = {"ok": True, "service": "sentinel-drift-attribution"}
    try:
        from opendrift.models.openoil import OpenOil  # noqa: F401

        out["opendrift"] = "available"
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["opendrift"] = f"missing: {str(exc)[:120]}"
    return out


def _fetch_era5(bbox, start_iso, end_iso):
    """Pull ERA5 hourly 10m wind for the bbox+window. Returns an OpenDrift
    `reader_netCDF_CF_generic` or raises."""

    # Import lazily so missing cdsapi/eccodes doesn't break the module load.
    from app.sources.era5 import fetch_era5_wind
    from opendrift.readers import reader_netCDF_CF_generic

    # Pad the requested window before hitting CDS.
    #
    # Without this the reader's coverage ends exactly at the detection time.
    # A backward run starts at that instant, and OpenDrift's time
    # interpolation reaches for ``index + 1`` — past the end of the axis.
    # The resulting IndexError surfaces as the deeply misleading
    #   "Missing variables: ['x_wind', 'y_wind', ...]"
    # i.e. a boundary off-by-one masquerading as an ERA5 outage.
    pad = timedelta(hours=FORCING_PAD_HOURS)
    t_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).replace(tzinfo=None) - pad
    t_end = datetime.fromisoformat(end_iso.replace("Z", "+00:00")).replace(tzinfo=None) + pad
    ds = fetch_era5_wind(bbox, t_start.isoformat(), t_end.isoformat())
    # Normalise longitudes to -180..180 so the reader doesn't get confused.
    if "longitude" in ds.coords and float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
        ds = ds.sortby("longitude")
    # Rename u10 -> x_wind, v10 -> y_wind (CF reader convention).
    rename = {}
    if "u10" in ds.data_vars and "x_wind" not in ds.data_vars:
        rename["u10"] = "x_wind"
    if "v10" in ds.data_vars and "y_wind" not in ds.data_vars:
        rename["v10"] = "y_wind"
    if rename:
        ds = ds.rename(rename)
    # Tag with CF standard names so the reader can match.
    if "x_wind" in ds.data_vars:
        ds["x_wind"].attrs["standard_name"] = "eastward_wind"
        ds["x_wind"].attrs["long_name"] = "10-metre U wind component"
        ds["x_wind"].attrs["units"] = "m s-1"
    if "y_wind" in ds.data_vars:
        ds["y_wind"].attrs["standard_name"] = "northward_wind"
        ds["y_wind"].attrs["long_name"] = "10-metre V wind component"
        ds["y_wind"].attrs["units"] = "m s-1"
    # Persist a small tmp netcdf because cfgrib output isn't reloadable.
    tmp = Path(tempfile.gettempdir()) / f"era5_{datetime.now().timestamp():.0f}.nc"
    ds.to_netcdf(tmp)
    reader = reader_netCDF_CF_generic.Reader(str(tmp))
    # Report the PADDED coverage, not the caller's window — the padding is what
    # keeps the last interpolation index in bounds.
    reader.start_time = t_start
    reader.end_time = t_end
    return reader


def _fetch_cmems(bbox, start_iso, end_iso):
    """Pull CMEMS GLORYS12 surface currents for the bbox+window.

    Verified live against `cmems_mod_glo_phy_my_0.083deg_P1D-m` (copernicusmarine
    2.4.1). Three API traps worth keeping in mind:

    * Use ``open_dataset``, not ``subset``. In copernicusmarine 2.x ``subset()``
      returns a ``ResponseSubset`` (a download receipt with ``file_path``), not
      an xarray object — calling ``.load()`` on it raises AttributeError.
    * Depth bounds are validated strictly, even with
      ``coordinates_selection_method='nearest'``. The shallowest coordinate is
      0.49402499... so an exact ``0.494`` is *below* the grid and raises
      CoordinatesOutOfDatasetBounds. Request ``[0.4941, 1.0]`` instead, which
      brackets the surface level and stays inside the grid.
    * GLORYS12 is **daily**. A 24 h window can easily contain a single 00:00
      snapshot, giving a reader whose ``start_time == end_time``. OpenDrift
      then reports *every* variable missing — including wind, from a perfectly
      healthy reader — which reads like an ERA5 outage but is not. Pad the
      request by a day on each side so there are always >=2 snapshots.
    * The result is lazy; ``.load()`` before writing to netCDF.
    """
    import copernicusmarine
    from opendrift.readers import reader_netCDF_CF_generic

    pad_start = (
        datetime.fromisoformat(start_iso.replace("Z", "+00:00")) - timedelta(days=1)
    ).isoformat()
    pad_end = (
        datetime.fromisoformat(end_iso.replace("Z", "+00:00")) + timedelta(days=1)
    ).isoformat()

    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=bbox[0],
        maximum_longitude=bbox[2],
        minimum_latitude=bbox[1],
        maximum_latitude=bbox[3],
        start_datetime=pad_start,
        end_datetime=pad_end,
        minimum_depth=0.4941,
        maximum_depth=1.0,
    ).load()

    rename = {}
    if "uo" in ds.data_vars and "x_sea_water_velocity" not in ds.data_vars:
        rename["uo"] = "x_sea_water_velocity"
    if "vo" in ds.data_vars and "y_sea_water_velocity" not in ds.data_vars:
        rename["vo"] = "y_sea_water_velocity"
    if rename:
        ds = ds.rename(rename)

    # Drop the singleton depth axis — OpenDrift's CF reader wants 2-D (+time)
    # horizontal fields, and a leftover depth dim makes it mis-map the grid.
    if "depth" in ds.dims and ds.sizes["depth"] == 1:
        ds = ds.isel(depth=0, drop=True)

    for var, sname in (
        ("x_sea_water_velocity", "eastward_sea_water_velocity"),
        ("y_sea_water_velocity", "northward_sea_water_velocity"),
    ):
        if var in ds.data_vars:
            ds[var].attrs["standard_name"] = sname
            ds[var].attrs["units"] = "m s-1"

    tmp = Path(tempfile.gettempdir()) / f"cmems_{datetime.now().timestamp():.0f}.nc"
    ds.to_netcdf(tmp)
    reader = reader_netCDF_CF_generic.Reader(str(tmp))
    # Let the reader derive its own time coverage from the file — overriding it
    # with the narrow backtrack window would re-introduce the zero-span bug.
    return reader


@router.post("/attribution")
async def attribute(req: AttributionRequest) -> dict[str, Any]:
    """One backward hindcast. See ``AttributionRequest``."""
    # Parse detection time.
    try:
        det_dt = datetime.fromisoformat(req.detection_time.replace("Z", "+00:00"))
        if det_dt.tzinfo is None:
            det_dt = det_dt.replace(tzinfo=UTC)
    except Exception as exc:
        raise HTTPException(422, f"detection_time invalid: {exc}") from exc

    start_dt = det_dt - timedelta(hours=req.duration_h)
    start_iso = start_dt.isoformat().replace("+00:00", "Z")
    end_iso = req.detection_time

    bbox = req.bbox or _default_bbox(req.detection_lon, req.detection_lat, pad_deg=5.0)
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise HTTPException(422, "bbox invalid")

    wind_reader = None
    current_reader = None
    notes = []
    if req.use_era5:
        wind_reader, _src = _resolve_wind(
            bbox,
            datetime.fromisoformat(start_iso.replace("Z", "+00:00")),
            datetime.fromisoformat(end_iso.replace("Z", "+00:00")),
            req.forcing,
            notes,
        )
    if req.use_cmems:
        try:
            current_reader = _fetch_cmems(bbox, start_iso, end_iso)
        except Exception as exc:
            notes.append(f"CMEMS unavailable: {str(exc)[:200]}")
            logger.warning("CMEMS fetch failed: {}", exc)

    cfg = EnsembleConfig(
        n_members=req.n_members,
        duration_h=req.duration_h,
        seed=req.seed,
    )
    try:
        result: AttributionResult = run_backward_attribution(
            detection_time=det_dt,
            detection_lon=req.detection_lon,
            detection_lat=req.detection_lat,
            detection_area_km2=req.detection_area_km2,
            wind_reader=wind_reader,
            current_reader=current_reader,
            vessels=req.vessels or [],
            cfg=cfg,
            synthetic_wind_ms=req.synthetic_wind_ms,
            synthetic_wind_dir_from_deg=req.synthetic_wind_dir_from_deg,
            synthetic_current_ms=req.synthetic_current_ms,
            synthetic_current_dir_deg=req.synthetic_current_dir_deg,
        )
    except Exception as exc:
        raise HTTPException(500, f"attribution failed: {str(exc)[:300]}") from exc

    if notes:
        result.notes.extend(notes)
    return result.to_dict()


class ForecastRequest(BaseModel):
    """Forward drift projection from a known origin."""

    origin_lon: float
    origin_lat: float
    origin_time: str = Field(..., description="ISO-8601 UTC")
    seed_radius_km: float = Field(1.0, ge=0.05, le=100.0, description="Radius of the seed disc")
    duration_h: float = Field(48.0, ge=1, le=168)

    use_era5: bool = True
    use_cmems: bool = True
    forcing: str = Field("auto", description="Wind source: auto | era5 | gfs")
    bbox: tuple[float, float, float, float] | None = None

    # Shoreline impact needs a landmask. Turning it off makes the run faster
    # and avoids GSHHG, but then stranded_fraction is meaningless (and the
    # result says so).
    use_landmask: bool = True

    n_members: int = Field(64, ge=16, le=2048)
    seed: int = 20200725

    synthetic_wind_ms: float = 5.0
    synthetic_wind_dir_from_deg: float = 110.0
    synthetic_current_ms: float = 0.10
    synthetic_current_dir_deg: float = 220.0


@router.post("/forecast")
async def forecast(req: ForecastRequest) -> dict[str, Any]:
    """Project the slick forward and report shoreline impact.

    Returns a per-hour cone (centroid + p50/p95 radius + stranded count),
    the final centre, the fraction of the ensemble that beached, and the
    hour of first landfall. Use the origin ellipse from
    ``/drift/attribution`` as the seed for an end-to-end
    "where did it come from -> where is it going" chain.
    """
    try:
        org_dt = datetime.fromisoformat(req.origin_time.replace("Z", "+00:00"))
        if org_dt.tzinfo is None:
            org_dt = org_dt.replace(tzinfo=UTC)
    except Exception as exc:
        raise HTTPException(422, f"origin_time invalid: {exc}") from exc

    end_dt = org_dt + timedelta(hours=req.duration_h)
    start_iso = org_dt.isoformat().replace("+00:00", "Z")
    end_iso = end_dt.isoformat().replace("+00:00", "Z")

    bbox = req.bbox or _default_bbox(req.origin_lon, req.origin_lat, pad_deg=5.0)
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise HTTPException(422, "bbox invalid")

    notes: list[str] = []
    wind_reader = None
    current_reader = None
    if req.use_era5:
        wind_reader, _src = _resolve_wind(
            bbox,
            datetime.fromisoformat(start_iso.replace("Z", "+00:00")),
            datetime.fromisoformat(end_iso.replace("Z", "+00:00")),
            req.forcing,
            notes,
        )
    if req.use_cmems:
        try:
            current_reader = _fetch_cmems(bbox, start_iso, end_iso)
        except Exception as exc:
            notes.append(f"CMEMS unavailable: {str(exc)[:200]}")
            logger.warning("CMEMS fetch failed: {}", exc)

    cfg = EnsembleConfig(n_members=req.n_members, duration_h=req.duration_h, seed=req.seed)
    try:
        result = run_forward_forecast(
            origin_time=org_dt,
            origin_lon=req.origin_lon,
            origin_lat=req.origin_lat,
            seed_radius_km=req.seed_radius_km,
            duration_h=req.duration_h,
            wind_reader=wind_reader,
            current_reader=current_reader,
            cfg=cfg,
            synthetic_wind_ms=req.synthetic_wind_ms,
            synthetic_wind_dir_from_deg=req.synthetic_wind_dir_from_deg,
            synthetic_current_ms=req.synthetic_current_ms,
            synthetic_current_dir_deg=req.synthetic_current_dir_deg,
            use_landmask=req.use_landmask,
        )
    except Exception as exc:
        raise HTTPException(500, f"forecast failed: {str(exc)[:300]}") from exc

    if notes:
        result.notes = notes + result.notes
    return result.to_dict()
