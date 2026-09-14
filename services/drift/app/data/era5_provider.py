"""
ECMWF ERA5 Atmospheric Reanalysis Wind Forcing Adapter for SENTINEL.
Adapted from OceanTrace agent2/adapters/era5_adapter.py.

Ingests 10-meter eastward (u10) and northward (v10) surface wind components.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import numpy as np

from .forcing_base import EnvironmentalForcingProvider
from ..errors import EnvironmentalDataError

logger = logging.getLogger("sentinel.drift.ERA5")


def _cdsapi_importable() -> bool:
    try:
        import cdsapi  # noqa: F401
        return True
    except ImportError:
        return False


def _cds_credentials() -> bool:
    """True when the CDS API can be authenticated from env or ~/.cdsapirc."""
    if os.environ.get("CDSAPI_KEY") and os.environ.get("CDSAPI_URL"):
        return True
    import os as _os
    rc = _os.path.expanduser("~/.cdsapirc")
    if _os.path.exists(rc):
        try:
            with open(rc, "r", encoding="utf-8") as f:
                return "url" in f.read() and "key" in f.read()
        except OSError:
            return False
    return False


class ERA5WindProvider(EnvironmentalForcingProvider):
    """
    Adapter for ECMWF ERA5 / HRES 10m surface winds.
    Supports NetCDF / GRIB files or CDS API queries.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.api_key = api_key or os.environ.get("CDSAPI_KEY")
        self.cache_dir = cache_dir or self._default_cache_dir()
        self._dataset = None
        self._live_state: Optional[Dict[str, str]] = None

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                import xarray as xr
                self._dataset = xr.open_dataset(self.dataset_path)
            except Exception as e:
                raise EnvironmentalDataError(
                    f"Failed to open ERA5 NetCDF dataset at {self.dataset_path}: {e}"
                )

    # ------------------------------------------------------------------
    # Live acquisition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_cache_dir() -> str:
        base = os.getenv("OCEANTRACE_CACHE_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache")
        return os.path.join(base, "forcing", "era5")

    def credentials_available(self) -> bool:
        return _cds_credentials()

    def live_available(self) -> bool:
        # Credentials indicate that the provider is configured. The dependency
        # check remains in require_credentials(), so status/diagnostics can be
        # inspected even in a minimal offline installation.
        return self.credentials_available()

    def require_credentials(self) -> None:
        if not self.live_available():
            missing = []
            if not _cdsapi_importable():
                missing.append("`pip install cdsapi`")
            if not self.credentials_available():
                missing.append("CDSAPI_URL + CDSAPI_KEY in .env (or a ~/.cdsapirc file)")
            raise EnvironmentalDataError(
                "ERA5 live wind download unavailable: Missing " + " and ".join(missing) +
                ". Provide a local dataset_path or configure CDS credentials. Per the Sentinel "
                "No-Fabrication Policy, winds will not be synthesized."
            )

    def stage_winds(
        self,
        bbox: Tuple[float, float, float, float],
        t_start: datetime,
        t_end: datetime,
    ) -> str:
        """
        Downloads ERA5 10m wind components via the CDS API into the local cache
        and attaches the resulting dataset. Returns the cache path.
        """
        self.require_credentials()
        os.makedirs(self.cache_dir, exist_ok=True)

        lon_min, lat_min, lon_max, lat_max = bbox
        # CDS uses area=[north, west, south, east]
        area = f"{lat_max},{lon_min},{lat_min},{lon_max}"
        date_range = f"{t_start.strftime('%Y-%m-%d')}/{t_end.strftime('%Y-%m-%d')}"

        fname = (f"era5_u10v10_{lon_min:.2f}_{lat_min:.2f}_{lon_max:.2f}_{lat_max:.2f}_"
                 f"{t_start.strftime('%Y%m%d')}_{t_end.strftime('%Y%m%d')}.nc")
        out_path = os.path.join(self.cache_dir, fname)
        if not os.path.exists(out_path):
            try:
                import cdsapi
                client = cdsapi.Client(
                    url=os.environ.get("CDSAPI_URL"),
                    key=self.api_key or os.environ.get("CDSAPI_KEY"),
                    quiet=True,
                )
                logger.info("ERA5 live download [reanalysis-era5-single-levels] %s area=%s",
                            date_range, area)
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                        "date": date_range,
                        "time": [f"{h:02d}:00" for h in range(0, 24)],
                        "area": area,
                        "format": "netcdf",
                    },
                    out_path,
                )
            except Exception as e:
                raise EnvironmentalDataError(f"ERA5 live download failed: {e}")
            if not os.path.exists(out_path):
                raise EnvironmentalDataError(
                    f"ERA5 download completed but output file missing: {out_path}"
                )

        try:
            import xarray as xr
            self._dataset = xr.open_dataset(out_path)
            self.dataset_path = out_path
            self._live_state = {"source": "live_cdsapi", "cache_path": out_path}
            return out_path
        except Exception as e:
            raise EnvironmentalDataError(f"Failed to open staged ERA5 NetCDF at {out_path}: {e}")

    def prepare_for_region(self, spill: Any) -> None:
        """Stages ERA5 10m winds for the spill's footprint + hindcast window."""
        if self._dataset is not None:
            return
        try:
            bbox = _spill_bbox(spill, padding_deg=0.15)
            t0 = spill.observation_timestamp - timedelta(hours=48)
            t1 = spill.observation_timestamp + timedelta(hours=6)
            # Stage using the loader's bbox logic
            self.stage_winds(bbox, t0, t1)
        except EnvironmentalDataError:
            raise
        except Exception as e:
            raise EnvironmentalDataError(f"ERA5 region staging failed: {e}")

    # ------------------------------------------------------------------
    # EnvironmentalForcingProvider implementation
    # ------------------------------------------------------------------

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise EnvironmentalDataError(
            "ERA5 Wind Provider does not provide ocean currents. "
            "Use CompositeForcingProvider with CMEMSCurrentProvider."
        )

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates 10m eastward (u10) and northward (v10) wind velocity components."""
        if self._dataset is not None:
            try:
                from datetime import timezone as tz
                lookup_time = timestamp
                if getattr(lookup_time, "tzinfo", None) is not None:
                    lookup_time = lookup_time.astimezone(tz.utc).replace(tzinfo=None)
                ds_t = self._dataset.sel(time=lookup_time, method="nearest")
                u_var = "u10" if "u10" in ds_t else "10u"
                v_var = "v10" if "v10" in ds_t else "10v"
                u_interp = ds_t[u_var].interp(longitude=("points", lons), latitude=("points", lats)).values
                v_interp = ds_t[v_var].interp(longitude=("points", lons), latitude=("points", lats)).values
                return np.asarray(u_interp, dtype=np.float64), np.asarray(v_interp, dtype=np.float64)
            except Exception as e:
                raise EnvironmentalDataError(f"Error interpolating ERA5 dataset: {e}")

        raise EnvironmentalDataError(
            "ERA5 wind forcing data is unavailable. No local dataset path provided and "
            "CDS API credentials not configured. In accordance with the "
            "Sentinel No-Fabrication Policy, missing environmental data will not be silently synthesized."
        )

    def coverage_window(self) -> Tuple[Optional[str], Optional[str]]:
        """(start_utc, end_utc) of the staged wind data, else (None, None)."""
        from .forcing_selection import coverage_from_dataset

        return coverage_from_dataset(self._dataset)

    def to_opendrift_readers(self) -> Optional[List[Any]]:
        """Exposes the staged ERA5 NetCDF to OpenDrift as a CF-generic reader."""
        if self._dataset is None or not self.dataset_path:
            return None
        try:
            from opendrift.readers.reader_netCDF_CF_generic import Reader as NetCDFReader
            return [NetCDFReader(self.dataset_path)]
        except Exception as exc:
            logger.warning("ERA5->OpenDrift reader construction failed: %s", exc)
            return None

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "ERA5WindProvider",
            "source": "ECMWF ERA5 Reanalysis",
            "dataset_path": self.dataset_path,
            "is_connected": self._dataset is not None,
            "live_ready": self.live_available(),
            "credentials_configured": self.credentials_available(),
            "live_state": self._live_state,
        }

def _spill_bbox(spill: Any, padding_deg: float = 0.15) -> Tuple[float, float, float, float]:
    """Derives a lon/lat bounding box for a SpillObservation (or duck-typed object)."""
    try:
        xmin, ymin, xmax, ymax = spill.geometry.bounds
    except Exception:
        try:
            lon, lat = spill.centroid
        except Exception:
            lon = getattr(spill, "lon", 73.0)
            lat = getattr(spill, "lat", 15.0)
        return (lon - padding_deg, lat - padding_deg, lon + padding_deg, lat + padding_deg)
    return (xmin - padding_deg, ymin - padding_deg, xmax + padding_deg, ymax + padding_deg)
