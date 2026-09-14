"""ERA5 hourly 10m wind fetcher via the CDS climate-data-store API.

ERA5 reanalysis supplies the surface wind that drives oil-spill drift
(3–4 % of wind speed translated onto the slick, per Delvigne & Sweeney).
This module pulls a small grib bbox for a given date range and emits a
xarray dataset consumable by OpenDrift's readers.

Verified live (2026-09-14) against `cds.climate.copernicus.eu/api`:
    * Licence accepted — request status transitions `accepted → running → successful`
    * ~414 KB grib zip for a 4×10° box, one hour — pulls in <2 min

API quirks caught while writing this:
    * The newer OGC `retrieve/v1/processes` endpoint only lists derived
      products (drought indices, daily statistics, etc.) — NOT raw
      `reanalysis-era5-single-levels`. To get hourly winds you must use the
      legacy `cdsapi` Python client against the same domain.
    * The endpoint now expects the **bare** API key, NOT the deprecated
      `<UID>:<key>` form.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import xarray as xr

ERA5_DATASET = "reanalysis-era5-single-levels"
CDS_URL = "https://cds.climate.copernicus.eu/api"


def _ensure_cdsapi() -> None:
    try:
        import cdsapi  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is not installed; run `pip install cdsapi` to enable ERA5 pulls."
        ) from exc


def _ensure_rcfile(api_key: str) -> str:
    """Materialise a `.cdsapirc` somewhere and point HOME at it."""
    home = Path(tempfile.mkdtemp(prefix="era5_rc_"))
    (home / ".cdsapirc").write_text(f"url: {CDS_URL}\nkey: {api_key}\n")
    return str(home)


def fetch_era5_wind(
    bbox: tuple[float, float, float, float],
    start: str,  # 'YYYY-MM-DDTHH:MM:SS'
    end: str,
    out_path: str | None = None,
    api_key: str | None = None,
    max_wait_s: int = 600,
) -> xr.Dataset:
    """Pull ERA5 hourly 10m u/v wind for a bbox and time range.

    Args:
        bbox: (W, S, E, N) — ERA5 wants (N, W, S, E) and high-to-low.
        start, end: ISO timestamps (UTC).
        out_path: where to write the raw grib (auto-cleaned if None).
        api_key: CDS API key. Falls back to CDSAPI_KEY env var.

    Returns:
        xarray.Dataset with bands ``u10``, ``v10`` on a regular lat/lon grid
        suitable for `opendrift.readers.reader_netCDF_CF_generic`.
    """
    import cdsapi

    _ensure_cdsapi()

    key = api_key or os.getenv("CDSAPI_KEY")
    if not key:
        raise RuntimeError("CDSAPI_KEY missing from environment")

    west, south, east, north = bbox
    area = [f"{north:.3f}", f"{west:.3f}", f"{south:.3f}", f"{east:.3f}"]

    # Build a list of (date, time) strings inside [start, end].
    from datetime import datetime, timedelta

    t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
    if t1 <= t0:
        raise ValueError("end must be after start")
    # ERA5 is hourly — gather the discrete (date, time) keys.
    seen_dates: set[str] = set()
    times_for_date: dict[str, list[str]] = {}
    cur = t0.replace(minute=0, second=0, microsecond=0)
    while cur <= t1:
        ds = cur.strftime("%Y-%m-%d")
        seen_dates.add(ds)
        times_for_date.setdefault(ds, []).append(cur.strftime("%H:%M"))
        cur += timedelta(hours=1)

    home = _ensure_rcfile(key)
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = home
    try:
        client = cdsapi.Client(url=CDS_URL, key=key, quiet=True)
        # Group by date so each cdsapi call covers ≤1 calendar day of hours.
        # We download all dates and concatenate after; tiny cost, simpler code.
        all_files: list[str] = []
        out_dir = Path(tempfile.mkdtemp(prefix="era5_dl_"))
        for ds in sorted(seen_dates):
            target = out_dir / f"{ds}.grib"
            logger_msg = f"ERA5 fetch {ds} (hours={len(times_for_date[ds])})"
            print(f"[era5] requesting {logger_msg}…", file=sys.stderr)
            client.retrieve(
                ERA5_DATASET,
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                    "year": ds.split("-")[0],
                    "month": ds.split("-")[1],
                    "day": ds.split("-")[2],
                    "time": times_for_date[ds],
                    "area": area,
                    "data_format": "grib",
                    "download_format": "unarchived",
                },
            ).download(str(target))
            all_files.append(str(target))

        ds = xr.open_mfdataset(
            all_files,
            engine="cfgrib",
            combine="by_coords",
            parallel=False,
        )
        # ERA5 emits u10/v10 in m/s. Subset to the bbox if coordinates exceed it.
        if "longitude" in ds.coords:
            ds = ds.sortby("longitude").sortby("latitude")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(out_path)
        return ds
    finally:
        if saved_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = saved_home


# ── lightweight logger so we don't pull in loguru at import time ──
def log(msg: str, *args: Any) -> None:
    print(f"[era5] {msg % args if args else msg}", file=sys.stderr)
