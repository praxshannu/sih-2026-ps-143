"""Dask auto-chunked NetCDF loader for CMEMS GLORYS12 ocean currents."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from loguru import logger


def load_cmems_currents(
    base_path: str,
    time_start: np.datetime64,
    time_end: np.datetime64,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """Load CMEMS GLORYS12 uo/vo currents with Dask auto-chunking.

    Parameters
    ----------
    base_path : str
        Base directory containing GLORYS12 NetCDF files (glob pattern accepted).
    time_start, time_end : np.datetime64
        Temporal bounds.
    bbox : (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    xr.Dataset with variables ``uo`` and ``vo`` (m/s) chunked via Dask.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    base = Path(base_path)
    if base.is_dir():
        pattern = str(base / "*.nc")
    else:
        pattern = str(base)

    logger.info("Loading CMEMS data from {}", pattern)

    ds = xr.open_mfdataset(
        pattern,
        combine="by_coords",
        chunks={"time": 1, "depth": 1, "latitude": "auto", "longitude": "auto"},
        parallel=True,
        mask_and_scale=True,
    )

    # Subset to bbox and time window
    ds = ds.sel(
        longitude=slice(lon_min, lon_max),
        latitude=slice(lat_min, lat_max),
        time=slice(time_start, time_end),
    )

    # Rename to standard names if needed
    if "vo" in ds and "u" not in ds:
        ds = ds.rename({"uo": "u", "vo": "v"})
    elif "uo" in ds:
        ds = ds.rename({"uo": "u", "vo": "v"})

    # Select surface layer (depth ~0)
    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)

    logger.info(
        "CMEMS loaded: shape={} chunks={}",
        ds["u"].shape,
        "dask" if ds["u"].chunks else "in-memory",
    )

    return ds


def load_cmems_bathymetry(
    base_path: str,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """Load CMEMS GEBCO bathymetry if available."""
    lon_min, lat_min, lon_max, lat_max = bbox

    base = Path(base_path)
    pattern = str(base / "*bathy*.nc") if base.is_dir() else str(base)

    try:
        ds = xr.open_mfdataset(
            pattern,
            combine="by_coords",
            chunks={"latitude": "auto", "longitude": "auto"},
        )
        ds = ds.sel(
            longitude=slice(lon_min, lon_max),
            latitude=slice(lat_min, lat_max),
        )
        logger.info("Bathymetry loaded: shape={}", ds.dims)
        return ds
    except FileNotFoundError:
        logger.warning("No bathymetry file found at {}", pattern)
        return xr.Dataset()


def get_depth_at_points(
    bathy_ds: xr.Dataset, lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Extract depth values at particle positions via nearest-neighbor."""
    if not bathy_ds.data_vars:
        return np.zeros_like(lons, dtype=np.float64)

    depth_var = list(bathy_ds.data_vars)[0]
    coords = {"longitude": ("z", lons), "latitude": ("z", lats)}
    ds_point = bathy_ds[depth_var].sel(coords, method="nearest")
    return ds_point.values.astype(np.float64)
