"""xarray loader for ERA5 10m wind fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger


def load_era5_winds(
    base_path: str,
    time_start: np.datetime64,
    time_end: np.datetime64,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """Load ERA5 10m u10/v10 wind fields.

    Parameters
    ----------
    base_path : str
        Base directory or file path for ERA5 NetCDF.
    time_start, time_end : np.datetime64
        Temporal bounds.
    bbox : (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    xr.Dataset with ``u10`` and ``v10`` (m/s).
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    base = Path(base_path)
    if base.is_dir():
        pattern = str(base / "*.nc")
    else:
        pattern = str(base)

    logger.info("Loading ERA5 winds from {}", pattern)

    ds = xr.open_mfdataset(
        pattern,
        combine="by_coords",
        chunks={"time": 1, "latitude": "auto", "longitude": "auto"},
        parallel=True,
        mask_and_scale=True,
    )

    ds = ds.sel(
        longitude=slice(lon_min, lon_max),
        latitude=slice(lat_min, lat_max),
        time=slice(time_start, time_end),
    )

    # Rename to standard names
    rename_map = {}
    for old, new in [("u", "u10"), ("v", "v10"), ("U10", "u10"), ("V10", "v10")]:
        if old in ds.data_vars and new not in ds.data_vars:
            rename_map[old] = new
    if rename_map:
        ds = ds.rename(rename_map)

    logger.info("ERA5 loaded: shape={}", ds["u10"].shape)
    return ds


def interpolate_winds_to_points(
    wind_ds: xr.Dataset,
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear interpolation of winds to particle positions.

    Parameters
    ----------
    wind_ds : xr.Dataset
        Must contain ``u10`` and ``v10``.
    lons, lats : (N,) arrays
        Particle positions.
    times : (N,) array of np.datetime64

    Returns
    -------
    u_wind, v_wind : (N,) float64 arrays in m/s.
    """
    if "u10" not in wind_ds:
        logger.warning("ERA5 wind dataset missing u10, returning zeros")
        n = len(lons)
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)

    u10 = wind_ds["u10"]
    v10 = wind_ds["v10"]

    coords = {"longitude": ("z", lons), "latitude": ("z", lats), "time": ("z", times)}

    u_interp = u10.sel(coords, method="nearest").values.astype(np.float64)
    v_interp = v10.sel(coords, method="nearest").values.astype(np.float64)

    return u_interp, v_interp
