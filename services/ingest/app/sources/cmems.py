"""Copernicus Marine (CMEMS) GLORYS12 ocean current data fetcher.

Downloads latest ocean current u/v components as NetCDF with Dask-compatible chunking.
Covers the Indian Ocean basin.

FAIL-CLOSED CONTRACT
--------------------
A failed upstream never degrades to invented currents on its own. The gate is
``ALLOW_SYNTHETIC_FORCING=true`` and it is evaluated by
``app.provenance.decide_synthetic`` — the same helper ERA5 and AIS use, so the
three sources cannot drift apart. Unset (the default) means the fetch raises
:class:`app.provenance.ProvenanceError` with a machine-readable reason; set
means the result is stamped ``synthetic_mock`` and carries a warning string.
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

from ..models.schemas import OceanCurrentResult
from ..provenance import (
    PROVENANCE_SYNTHETIC,
    FORCING_GATES,
    decide_synthetic,
    window_coverage,
)

# Indian EEZ bounding box
INDIA_EEZ_BBOX = (68.0, 5.0, 88.0, 25.0)

# Extended Indian Ocean basin
INDIAN_OCEAN_BBOX = (30.0, -30.0, 120.0, 30.0)

CMEMS_TOKEN_URL = "https://cds.contribution.copernicus.eu/api/v2"
CMEMS_CATALOGUE_URL = "https://marine.copernicus.eu/api"


class CmemsFetcher:
    """Fetches GLORYS12v1 ocean current data from Copernicus Marine."""

    def __init__(
        self,
        api_key: str | None = None,
        storage_path: str = "/data/ocean",
    ) -> None:
        self.api_key = api_key or os.getenv("CMEMS_API_KEY", "")
        self.storage_path = storage_path
        self.dataset_id = "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i"
        self.variables = ["uo", "vo"]

    async def fetch(
        self,
        bbox: tuple[float, float, float, float] = INDIA_EEZ_BBOX,
        depth_range: tuple[float, float] = (0.0, 50.0),
        lookback_days: int = 7,
    ) -> OceanCurrentResult:
        """Download latest GLORYS12 ocean current data as NetCDF.

        Uses the Copernicus Marine API to download uo (eastward) and vo (northward)
        ocean current components for the Indian Ocean region.

        Returns OceanCurrentResult with download metadata.
        """
        t0 = time.time()
        lon_min, lat_min, lon_max, lat_max = bbox

        date_end = datetime.now(UTC)
        date_start = date_end - timedelta(days=lookback_days)

        os.makedirs(self.storage_path, exist_ok=True)

        out_file = os.path.join(
            self.storage_path,
            f"cmems_currents_{date_end.strftime('%Y%m%dT%H%M%S')}.nc",
        )

        # Build CDS API request payload
        request_payload = {
            "dataset_id": self.dataset_id,
            "variables": self.variables,
            "geo_grid": "full",
            "minimum_longitude": lon_min,
            "maximum_longitude": lon_max,
            "minimum_latitude": lat_min,
            "maximum_latitude": lat_max,
            "minimum_depth": depth_range[0],
            "maximum_depth": depth_range[1],
            "start_datetime": date_start.isoformat() + "Z",
            "end_datetime": date_end.isoformat() + "Z",
            "format": "netcdf",
            "processing_level": "latest_reasonest",
        }

        logger.info(
            "Fetching CMEMS currents: bbox={}, depth={}m, window={}d -> {}",
            bbox,
            depth_range,
            lookback_days,
            out_file,
        )

        total_bytes = 0
        provenance = "cmems"
        warning: str | None = None

        try:
            # Attempt CDS API download
            total_bytes = await self._download_via_cds(request_payload, out_file)
        except Exception as e:
            decision = decide_synthetic("cmems", required_env=FORCING_GATES)
            decision.require(e)
            # Only reached when the operator explicitly opened the gate.
            logger.warning(
                "CMEMS download failed ({}) — serving SYNTHETIC currents because {} is set. "
                "This is not real data and must not be used as evidence.",
                e,
                ", ".join(FORCING_GATES),
            )
            total_bytes = self._generate_synthetic_currents(
                out_file, bbox, date_start, date_end, depth_range
            )
            provenance = PROVENANCE_SYNTHETIC
            warning = decision.warning

        # Reopen and apply Dask chunking
        ds = xr.open_dataset(
            out_file,
            chunks={"time": 1, "depth": 1, "latitude": "auto", "longitude": "auto"},
        )

        time_range = (str(date_start), str(date_end))

        # Honest coverage: compare what was requested against what the file
        # actually holds. A truncated window otherwise looks exactly like a
        # healthy fetch (see VERIFICATION §10 — the same trap was responsible
        # for an "ERA5 outage" that was really an off-by-one).
        returned: tuple[datetime, datetime] | None = None
        if "time" in ds.coords and ds.time.size:
            # NetCDF time is naive UTC; make it explicit so the comparison in
            # window_coverage cannot silently mix aware and naive datetimes.
            returned = (
                ds.time.values.min().astype("datetime64[us]").item().replace(tzinfo=UTC),
                ds.time.values.max().astype("datetime64[us]").item().replace(tzinfo=UTC),
            )
        coverage = window_coverage((date_start, date_end), returned)
        if coverage["status"] != "full":
            logger.warning(
                "CMEMS coverage {}: requested {}..{}, got {}",
                coverage["status"],
                date_start.isoformat(),
                date_end.isoformat(),
                coverage["returned"],
            )

        logger.info(
            "CMEMS fetch complete: {} bytes, shape={}, time={}, provenance={}",
            total_bytes,
            {v: ds[v].shape for v in ds.data_vars if v in self.variables},
            time_range,
            provenance,
        )

        return OceanCurrentResult(
            files_downloaded=1,
            total_bytes=total_bytes,
            bbox=bbox,
            time_range=time_range,
            variable_names=self.variables,
            fetch_duration_seconds=round(time.time() - t0, 2),
            provenance=provenance,
            is_synthetic=provenance == PROVENANCE_SYNTHETIC,
            warning=warning,
            coverage=coverage,
        )

    async def _download_via_cds(self, request_payload: dict, out_file: str) -> int:
        """Download via CDS API (synchronous request, async poll)."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=30) as client:
            # Submit retrieval request
            resp = await client.post(
                f"{CMEMS_CATALOGUE_URL}/api/v2/resources",
                json=request_payload,
                headers=headers,
            )
            resp.raise_for_status()
            request_id = resp.json().get("request_id", "")

            if not request_id:
                raise ValueError("No request_id returned from CDS API")

            # Poll for completion
            for _ in range(300):  # max 50 minutes
                await asyncio.sleep(10)
                status_resp = await client.get(
                    f"{CMEMS_CATALOGUE_URL}/api/v2/tasks/{request_id}",
                    headers=headers,
                )
                status = status_resp.json().get("status", "")
                if status == "completed":
                    download_url = status_resp.json().get("download_url", "")
                    break
                elif status == "failed":
                    raise RuntimeError("CDS retrieval failed")
            else:
                raise TimeoutError("CDS retrieval timed out")

            # Download the file
            dl_resp = await client.get(download_url, headers=headers, timeout=600)
            dl_resp.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in dl_resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
            return len(dl_resp.content)

    def _generate_synthetic_currents(
        self,
        out_file: str,
        bbox: tuple[float, float, float, float],
        time_start: datetime,
        time_end: datetime,
        depth_range: tuple[float, float],
    ) -> int:
        """Generate synthetic ocean current data — ONLY behind an explicit gate.

        Creates time-varying u/v fields with a tidal-like signal. Every field
        produced here is labelled ``synthetic_mock`` by the caller; the NetCDF
        itself is also tagged so a leaked file still identifies itself.
        """
        lon_min, lat_min, lon_max, lat_max = bbox

        lons = np.arange(lon_min, lon_max, 0.25)
        lats = np.arange(lat_min, lat_max, 0.25)
        depths = np.arange(depth_range[0], depth_range[1], 1.0)
        # numpy datetime64 has no timezone; drop the offset rather than let
        # numpy warn and silently reinterpret it.
        times = np.arange(
            np.datetime64(time_start.replace(tzinfo=None).isoformat()),
            np.datetime64(time_end.replace(tzinfo=None).isoformat()),
            np.timedelta64(6, "h"),
        )

        n_time = len(times)
        n_depth = max(len(depths), 1)
        n_lat = len(lats)
        n_lon = len(lons)

        rng = np.random.default_rng(42)

        # Base current pattern with lat-dependent magnitude.
        # Kept as (n_lat, 1) so it broadcasts across longitude — squeezing it to
        # (n_lat,) made broadcast_to fail against the (…, n_lat, n_lon) target.
        lat_grid = np.linspace(lat_min, lat_max, n_lat)[:, np.newaxis]
        base_u = 0.15 * np.sin(np.radians(lat_grid))
        base_v = 0.10 * np.cos(np.radians(lat_grid))

        # Add temporal variability. Four axes (time, depth, lat, lon) — with
        # only three, broadcast_to left-pads to (1, n_time, 1, 1) and then fails
        # against n_depth, which is how this fallback used to raise ValueError
        # instead of ever producing a field.
        time_phase = np.linspace(0, 4 * np.pi, n_time)[:, np.newaxis, np.newaxis, np.newaxis]
        temporal_u = 0.05 * np.sin(time_phase)
        temporal_v = 0.03 * np.cos(time_phase)

        # 4D array: (time, depth, lat, lon)
        uo = (
            np.broadcast_to(base_u, (n_time, n_depth, n_lat, n_lon)).copy()
            + np.broadcast_to(temporal_u, (n_time, n_depth, n_lat, n_lon)).copy()
            + rng.normal(0, 0.01, (n_time, n_depth, n_lat, n_lon))
        ).astype(np.float32)

        vo = (
            np.broadcast_to(base_v, (n_time, n_depth, n_lat, n_lon)).copy()
            + np.broadcast_to(temporal_v, (n_time, n_depth, n_lat, n_lon)).copy()
            + rng.normal(0, 0.01, (n_time, n_depth, n_lat, n_lon))
        ).astype(np.float32)

        ds = xr.Dataset(
            {
                "uo": (["time", "depth", "latitude", "longitude"], uo),
                "vo": (["time", "depth", "latitude", "longitude"], vo),
            },
            coords={
                "time": times.astype("datetime64[ns]"),
                "depth": depths.astype(np.float32),
                "latitude": lats.astype(np.float32),
                "longitude": lons.astype(np.float32),
            },
            attrs={
                "title": "Synthetic GLORYS12-style ocean currents for Indian EEZ",
                "source": "sentinel-ingest fallback generator",
                "conventions": "CF-1.8",
                "provenance": PROVENANCE_SYNTHETIC,
                "warning": "SYNTHETIC — not real CMEMS data. Demonstration use only.",
            },
        )

        # Dask-compatible chunked encoding
        ds.to_netcdf(
            out_file,
            encoding={
                "uo": {"chunksizes": (1, 1, n_lat, n_lon), "zlib": True, "complevel": 4},
                "vo": {"chunksizes": (1, 1, n_lat, n_lon), "zlib": True, "complevel": 4},
            },
        )

        file_size = os.path.getsize(out_file)
        logger.info(
            "Generated synthetic currents: {}x{}x{}x{}, {} bytes",
            n_time,
            n_depth,
            n_lat,
            n_lon,
            file_size,
        )
        return file_size
