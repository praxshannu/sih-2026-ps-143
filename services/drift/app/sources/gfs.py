"""NOAA GFS 10 m wind via the NOMADS server-side subsetting filter.

Why this exists
---------------
ERA5 (via CDS) is the right forcing for *historical* work — it goes back to
1940 and covers the 2020 Wakashio case. But every request goes into a queue;
a single day can take 1-3 minutes. For a slick detected *today*, that latency
is the difference between a useful forecast and a stale one.

GFS is the complement: free, no API key, and effectively instant because
NOMADS subsets by variable AND bounding box server-side — a 5 deg box of
10 m wind comes back as ~1 KB instead of the ~30 MB full global GRIB.

The trade-off is coverage: NOMADS only serves roughly the last 10 days. It
cannot backfill 2020. So this is a **live-operations** source, and callers
must fall back to ERA5 outside its window. `supports_window()` exists so
that decision is explicit rather than discovered as a 404 at run time.

IMPORTANT — the configured GFS_OPENDAP_URL is dead
--------------------------------------------------
`https://nomads.ncep.noaa.gov/dods/gfs_0p25` returns an HTML page saying
"OpenDAP format has been retired" (NOAA Service Change Notice 25-81). Use
the `filter_gfs_0p25.pl` CGI below, or the AWS Open Data mirror at
`https://noaa-gfs-bdp-pds.s3.amazonaws.com/` (full files, no subsetting).
"""

from __future__ import annotations

import math
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

try:  # requests is already a transitive dep, but be explicit at import time.
    import requests
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("GFS source needs `requests`") from exc

FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
CYCLE_HOURS = (0, 6, 12, 18)
# NOMADS keeps a rolling window; beyond this the f-files are gone.
RETENTION_DAYS = 10
TIMEOUT_S = 60


def _utcnow() -> datetime:
    return datetime.now(UTC)


def latest_cycle(at: datetime | None = None) -> datetime:
    """Most recent GFS cycle (00/06/12/18Z) at or before `at`, minus 6 h lag.

    A cycle's files are published a few hours after the nominal time, so we
    step back one cycle to avoid requesting data that isn't up yet.
    """
    t = (at or _utcnow()).astimezone(UTC)
    t = t - timedelta(hours=6)
    cycle_h = max(h for h in CYCLE_HOURS if h <= t.hour)
    return t.replace(hour=cycle_h, minute=0, second=0, microsecond=0)


def supports_window(
    start: datetime, end: datetime | None = None, now: datetime | None = None
) -> tuple[bool, str]:
    """Whether NOMADS still holds GFS data covering this window.

    Returns (ok, reason). Callers should use this to choose GFS vs ERA5
    instead of trying GFS first and swallowing a 404.
    """
    now = now or _utcnow()
    cutoff = now - timedelta(days=RETENTION_DAYS)
    if start < cutoff:
        return False, (
            f"GFS on NOMADS retains ~{RETENTION_DAYS} days; window starts "
            f"{start.date()} which is before the {cutoff.date()} cutoff — use ERA5."
        )
    if end and end > now + timedelta(hours=18):
        return False, "GFS window extends beyond the available forecast range."
    return True, "within NOMADS retention"


def _fetch_one(
    cycle: datetime, fhour: int, bbox: tuple[float, float, float, float]
) -> xr.Dataset | None:
    """Fetch one forecast hour, subset server-side. Returns None if absent."""
    west, south, east, north = bbox
    params = {
        "file": f"gfs.t{cycle.hour:02d}z.pgrb2.0p25.f{fhour:03d}",
        "lev_10_m_above_ground": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "subregion": "",
        "leftlon": f"{west:.4f}",
        "rightlon": f"{east:.4f}",
        "toplat": f"{north:.4f}",
        "bottomlat": f"{south:.4f}",
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle.hour:02d}/atmos",
    }
    resp = requests.get(FILTER_URL, params=params, timeout=TIMEOUT_S)
    if resp.status_code != 200 or not resp.content.startswith(b"GRIB"):
        return None
    # cfgrib needs a real filesystem path — eccodes cannot decode reliably
    # from an in-memory buffer, and hands back an empty dataset instead of
    # raising. Always spool to a temp file.
    tmp = Path(tempfile.mkdtemp(prefix="gfs_")) / f"f{fhour:03d}.grib"
    try:
        tmp.write_bytes(resp.content)
        ds = xr.open_dataset(str(tmp), engine="cfgrib")
        # Force the decode while the file still exists.
        ds = ds.load()
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    return ds


def fetch_gfs_wind(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    max_fhours: int = 60,
) -> xr.Dataset:
    """Pull GFS 10 m wind covering [start, end].

    Args:
        bbox: (W, S, E, N) in degrees.
        start, end: naive or aware UTC datetimes (aware preferred).
        max_fhours: safety cap on the number of hourly HTTP requests.

    Returns:
        xarray.Dataset with ``u10``/``v10`` on a (time, lat, lon) grid,
        longitudes normalised to -180..180, time = valid_time.

    Raises:
        RuntimeError: if the window is outside retention or nothing downloads.
    """
    start = start.replace(tzinfo=UTC) if start.tzinfo is None else start
    end = end.replace(tzinfo=UTC) if end.tzinfo is None else end

    ok, reason = supports_window(start, end)
    if not ok:
        raise RuntimeError(f"GFS unavailable: {reason}")

    cycle = latest_cycle(start)
    f0 = int(math.floor((start - cycle).total_seconds() / 3600))
    f1 = int(math.ceil((end - cycle).total_seconds() / 3600))
    f0 = max(f0, 0)
    if f1 - f0 > max_fhours:
        raise RuntimeError(
            f"GFS window too long: {f1 - f0} forecast hours exceeds cap {max_fhours}"
        )

    frames: list[xr.Dataset] = []
    times: list[np.datetime64] = []
    for fh in range(f0, f1 + 1):
        ds = _fetch_one(cycle, fh, bbox)
        if ds is None or "u10" not in ds.data_vars:
            continue
        # Each file is a single step; valid_time is the physically meaningful
        # stamp, and it is what OpenDrift will interpolate against.
        vt = ds["valid_time"].values if "valid_time" in ds.coords else None
        if vt is None:
            continue
        d = ds[["u10", "v10"]].expand_dims(time=[np.datetime64(vt)])
        frames.append(d)
        times.append(np.datetime64(vt))

    if not frames:
        raise RuntimeError(
            "GFS returned no usable wind fields for this window "
            f"(cycle {cycle:%Y-%m-%d %H}z, f{fhours_str(f0, f1)})"
        )

    out = xr.concat(frames, dim="time").sortby("time")

    # NOMADS serves 0..360 longitudes; OpenDrift expects -180..180.
    if float(out.longitude.max()) > 180:
        out = out.assign_coords(longitude=((out.longitude + 180) % 360) - 180).sortby("longitude")

    out["u10"].attrs.update(standard_name="eastward_wind", units="m s-1")
    out["v10"].attrs.update(standard_name="northward_wind", units="m s-1")
    return out


def fhours_str(f0: int, f1: int) -> str:
    return f"{f0:03d}..{f1:03d}"
