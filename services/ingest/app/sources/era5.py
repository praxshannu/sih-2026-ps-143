"""ECMWF ERA5 wind field fetcher.

Fetches 10m wind components (u10, v10) via the CDS API and caches as NetCDF.

FAIL-CLOSED CONTRACT
--------------------
``docs/VERIFICATION.md`` §9.2: an ERA5 pull once silently degraded to synthetic
wind and "the run succeeded and returned invented numbers". That path is now
closed by default. The gate is ``ALLOW_SYNTHETIC_FORCING=true``, evaluated by
``app.provenance.decide_synthetic`` — the same helper CMEMS and AIS use. Unset
means the fetch raises :class:`app.provenance.ProvenanceError`; set means the
result is stamped ``synthetic_mock`` and carries a warning string.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import xarray as xr
from loguru import logger

from ..models.schemas import WindFieldResult
from ..provenance import (
    FORCING_GATES,
    PROVENANCE_SYNTHETIC,
    decide_synthetic,
    window_coverage,
)

# Indian EEZ bounding box
INDIA_EEZ_BBOX = (68.0, 5.0, 88.0, 25.0)

CDS_API_URL = "https://cds.climate.copernicus.eu/api/v2"


class Era5Fetcher:
    """Fetches ERA5 10m wind fields via the CDS API."""

    def __init__(
        self,
        api_key: str | None = None,
        storage_path: str = "/data/wind",
    ) -> None:
        self.api_key = api_key or os.getenv("ERA5_CDS_API_KEY", "")
        self.storage_path = storage_path

    async def fetch(
        self,
        bbox: tuple[float, float, float, float] = INDIA_EEZ_BBOX,
        lookback_days: int = 7,
    ) -> WindFieldResult:
        """Download ERA5 10m wind fields (u10, v10) as NetCDF.

        Parameters
        ----------
        bbox : (lon_min, lat_min, lon_max, lat_max)
        lookback_days : Number of days to fetch

        Returns
        -------
        WindFieldResult with download metadata.
        """
        t0 = time.time()
        lon_min, lat_min, lon_max, lat_max = bbox

        date_end = datetime.now(UTC)
        date_start = date_end - timedelta(days=lookback_days)

        os.makedirs(self.storage_path, exist_ok=True)
        out_file = os.path.join(
            self.storage_path,
            f"era5_wind_{date_end.strftime('%Y%m%dT%H%M%S')}.nc",
        )

        logger.info(
            "Fetching ERA5 winds: bbox={}, window={}d -> {}",
            bbox,
            lookback_days,
            out_file,
        )

        request_payload = {
            "dataset_id": "reanalysis-era5-single-levels",
            "product_type": "reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": list(range(date_start.year, date_end.year + 1)),
            "month": [
                f"{m:02d}"
                for m in range(1, 13)
                if date_start.month <= m <= date_end.month or date_start.year != date_end.year
            ],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "area": [lat_max, lon_min, lat_min, lon_max],
            "format": "netcdf",
        }

        total_bytes = 0
        provenance = "era5"
        warning: str | None = None

        try:
            total_bytes = await self._download_via_cds(request_payload, out_file)
        except Exception as e:
            decision = decide_synthetic("era5", required_env=FORCING_GATES)
            decision.require(e)
            # Only reached when the operator explicitly opened the gate.
            logger.warning(
                "ERA5 download failed ({}) — serving SYNTHETIC winds because {} is set. "
                "This is not real data and must not be used as evidence.",
                e,
                ", ".join(FORCING_GATES),
            )
            total_bytes = self._generate_synthetic_winds(out_file, bbox, date_start, date_end)
            provenance = PROVENANCE_SYNTHETIC
            warning = decision.warning

        ds = xr.open_dataset(
            out_file,
            chunks={"time": 1, "latitude": "auto", "longitude": "auto"},
        )

        # Standardize variable names
        rename_map = {}
        for old, new in [("u", "u10"), ("v", "v10"), ("u10m", "u10"), ("v10m", "v10")]:
            if old in ds.data_vars and new not in ds.data_vars:
                rename_map[old] = new
        if rename_map:
            ds = ds.rename(rename_map)

        variable_names = [v for v in ["u10", "v10"] if v in ds.data_vars]
        time_range = (str(date_start), str(date_end))

        # Honest temporal coverage. A truncated window is the realistic partial
        # case here, and it otherwise reads as a healthy fetch.
        returned: tuple[datetime, datetime] | None = None
        if "time" in ds.coords and ds.time.size:
            returned = (
                ds.time.values.min().astype("datetime64[us]").item().replace(tzinfo=UTC),
                ds.time.values.max().astype("datetime64[us]").item().replace(tzinfo=UTC),
            )
        coverage = window_coverage((date_start, date_end), returned)
        if coverage["status"] != "full":
            logger.warning(
                "ERA5 coverage {}: requested {}..{}, got {}",
                coverage["status"],
                date_start.isoformat(),
                date_end.isoformat(),
                coverage["returned"],
            )

        logger.info(
            "ERA5 fetch complete: {} bytes, variables={}, time={}, provenance={}",
            total_bytes,
            variable_names,
            time_range,
            provenance,
        )

        return WindFieldResult(
            files_downloaded=1,
            total_bytes=total_bytes,
            bbox=bbox,
            time_range=time_range,
            variable_names=variable_names,
            fetch_duration_seconds=round(time.time() - t0, 2),
            provenance=provenance,
            is_synthetic=provenance == PROVENANCE_SYNTHETIC,
            warning=warning,
            coverage=coverage,
        )

    async def _download_via_cds(self, request_payload: dict, out_file: str) -> int:
        """Download via CDS API with async polling."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{CDS_API_URL}/resources",
                json=request_payload,
                headers=headers,
            )
            resp.raise_for_status()
            request_id = resp.json().get("request_id", "")

            if not request_id:
                raise ValueError("No request_id returned from CDS API")

            for _ in range(360):  # max 60 minutes
                await asyncio.sleep(10)
                status_resp = await client.get(
                    f"{CDS_API_URL}/tasks/{request_id}",
                    headers=headers,
                )
                status = status_resp.json().get("status", "")
                if status == "completed":
                    download_url = status_resp.json().get("download_url", "")
                    break
                elif status == "failed":
                    raise RuntimeError("ERA5 CDS retrieval failed")
            else:
                raise TimeoutError("ERA5 CDS retrieval timed out")

            dl_resp = await client.get(download_url, headers=headers, timeout=600)
            dl_resp.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in dl_resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
            return len(dl_resp.content)

    def _generate_synthetic_winds(
        self,
        out_file: str,
        bbox: tuple[float, float, float, float],
        time_start: datetime,
        time_end: datetime,
    ) -> int:
        """Generate synthetic wind data — ONLY behind an explicit gate.

        Produces monsoon-like wind patterns. Every field produced here is
        labelled ``synthetic_mock`` by the caller, and the NetCDF is tagged too.
        """
        lon_min, lat_min, lon_max, lat_max = bbox

        lons = np.arange(lon_min, lon_max, 0.25)
        lats = np.arange(lat_min, lat_max, 0.25)
        # numpy datetime64 has no timezone; drop the offset rather than let
        # numpy warn and silently reinterpret it.
        times = np.arange(
            np.datetime64(time_start.replace(tzinfo=None).isoformat()),
            np.datetime64(time_end.replace(tzinfo=None).isoformat()),
            np.timedelta64(3, "h"),
        )

        n_time = len(times)
        n_lat = len(lats)
        n_lon = len(lons)

        rng = np.random.default_rng(42)

        # Southwest monsoon-like pattern
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

        # u10: stronger westerly component in summer
        base_u10 = -5.0 + 3.0 * np.sin(np.radians(lat_grid))
        # v10: southerly near equator, northerly in north
        base_v10 = -2.0 * np.cos(np.radians(lat_grid))

        # Add diurnal and multi-day variability
        time_hours = np.arange(n_time) * 3.0
        diurnal_u = 0.8 * np.sin(2 * np.pi * time_hours / 24.0)
        synoptic_u = 1.5 * np.sin(2 * np.pi * time_hours / (24.0 * 5))
        diurnal_v = 0.5 * np.cos(2 * np.pi * time_hours / 24.0)
        synoptic_v = 1.0 * np.cos(2 * np.pi * time_hours / (24.0 * 5))

        u10 = (
            np.broadcast_to(base_u10, (n_time, n_lat, n_lon)).copy()
            + diurnal_u[:, np.newaxis, np.newaxis]
            + synoptic_u[:, np.newaxis, np.newaxis]
            + rng.normal(0, 0.3, (n_time, n_lat, n_lon))
        ).astype(np.float32)

        v10 = (
            np.broadcast_to(base_v10, (n_time, n_lat, n_lon)).copy()
            + diurnal_v[:, np.newaxis, np.newaxis]
            + synoptic_v[:, np.newaxis, np.newaxis]
            + rng.normal(0, 0.2, (n_time, n_lat, n_lon))
        ).astype(np.float32)

        ds = xr.Dataset(
            {
                "u10": (["time", "latitude", "longitude"], u10),
                "v10": (["time", "latitude", "longitude"], v10),
            },
            coords={
                "time": times.astype("datetime64[ns]"),
                "latitude": lats.astype(np.float32),
                "longitude": lons.astype(np.float32),
            },
            attrs={
                "title": "Synthetic ERA5-style 10m winds for Indian EEZ",
                "source": "sentinel-ingest fallback generator",
                "conventions": "CF-1.8",
                "units": "m s-1",
                "provenance": PROVENANCE_SYNTHETIC,
                "warning": "SYNTHETIC — not real ERA5 data. Demonstration use only.",
            },
        )

        ds.to_netcdf(
            out_file,
            encoding={
                "u10": {"chunksizes": (1, n_lat, n_lon), "zlib": True, "complevel": 4},
                "v10": {"chunksizes": (1, n_lat, n_lon), "zlib": True, "complevel": 4},
            },
        )

        file_size = os.path.getsize(out_file)
        logger.info(
            "Generated synthetic winds: {}x{}x{}, {} bytes",
            n_time,
            n_lat,
            n_lon,
            file_size,
        )
        return file_size
