"""
Copernicus Marine Service (CMEMS) Ocean Current Forcing Adapter for SENTINEL.
Adapted from OceanTrace agent2/adapters/cmems_adapter.py.

Ingests global/regional ocean physics analysis and forecast products (uo, vo surface currents).
Adheres to the No-Fabrication Policy: raises EnvironmentalDataError if data is unavailable.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import numpy as np

from .forcing_base import EnvironmentalForcingProvider
from ..errors import EnvironmentalDataError

logger = logging.getLogger("sentinel.drift.CMEMS")


def _cmems_credentials() -> Optional[Tuple[str, str]]:
    """Returns (username, password) from env (legacy or official CAS SSO names), or None."""
    u = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") or os.environ.get("COPERNICUS_MARINE_USERNAME")
    p = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD") or os.environ.get("COPERNICUS_MARINE_PASSWORD")
    if u and p:
        return u, p
    return None


def _copernicusmarine_importable() -> bool:
    try:
        import copernicusmarine  # noqa: F401
        return True
    except ImportError:
        return False


class CMEMSCurrentProvider(EnvironmentalForcingProvider):
    """
    Adapter for CMEMS (Copernicus Marine Environment Monitoring Service) ocean current datasets.
    Supports local NetCDF / Zarr files or live Copernicus Marine API queries (staged + cached).
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        product_id: str = "GLOBAL_ANALYSISFORECAST_PHY_001_024",
        cache_dir: Optional[str] = None,
    ):
        self.dataset_path = dataset_path or os.environ.get("OCEANTRACE_CMEMS_DATASET_PATH") or os.environ.get("CMEMS_DATASET_PATH")
        self.product_id = os.environ.get("COPERNICUS_MARINE_PRODUCT_ID") or product_id
        self.username = username or (os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
                                      or os.environ.get("COPERNICUS_MARINE_USERNAME"))
        self.password = password or (os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
                                      or os.environ.get("COPERNICUS_MARINE_PASSWORD"))
        self.cache_dir = cache_dir or self._default_cache_dir()
        self._dataset = None
        self._live_state: Optional[Dict[str, str]] = None  # populated after a successful live stage

        # If a local NetCDF file is specified, attempt to open
        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                import xarray as xr
                self._dataset = xr.open_dataset(self.dataset_path)
            except Exception as e:
                raise EnvironmentalDataError(
                    f"Failed to open CMEMS NetCDF dataset at {self.dataset_path}: {e}"
                )

    # ------------------------------------------------------------------
    # Live acquisition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_cache_dir() -> str:
        for base in (os.getenv("OCEANTRACE_CACHE_DIR"),
                     os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache")):
            if base:
                return os.path.join(base, "forcing", "cmems")
        return os.path.join(os.getcwd(), "data", "cache", "forcing", "cmems")

    def credentials_available(self) -> bool:
        return bool(self.username and self.password)

    def live_available(self) -> bool:
        """True when a live Copernicus Marine download is both importable and configured."""
        return _copernicusmarine_importable() and self.credentials_available()

    def require_credentials(self) -> None:
        if not self.live_available():
            missing = []
            if not _copernicusmarine_importable():
                missing.append("`pip install copernicusmarine`")
            if not self.credentials_available():
                missing.append("COPERNICUS_MARINE_USERNAME/PASSWORD (or COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD) in .env")
            raise EnvironmentalDataError(
                "CMEMS live current download unavailable: Missing " + " and ".join(missing) +
                ". Provide a local dataset_path or configure credentials. Per the Sentinel "
                "No-Fabrication Policy, currents will not be synthesized."
            )

    def stage_currents(
        self,
        bbox: Tuple[float, float, float, float],
        t_start: datetime,
        t_end: datetime,
    ) -> str:
        """
        Downloads the requested spatio-temporal window from Copernicus Marine into
        the local cache and attaches the resulting xarray dataset. Returns the path.
        """
        self.require_credentials()
        os.makedirs(self.cache_dir, exist_ok=True)

        lon_min, lat_min, lon_max, lat_max = bbox
        # Copernicus Marine commonly rotates uo/vo for global products internally;
        # request the standard surface variables of the global ocean physics product.
        variables = self._resolve_variables()
        st = t_start.strftime("%Y-%m-%dT%H:%M:%S")
        en = t_end.strftime("%Y-%m-%dT%H:%M:%S")
        fname = (f"cmems_{self.product_id.replace('/','_')}_"
                 f"{lon_min:.2f}_{lat_min:.2f}_{lon_max:.2f}_{lat_max:.2f}_"
                 f"{t_start.strftime('%Y%m%d%H')}_{t_end.strftime('%Y%m%d%H')}.nc")
        out_path = os.path.join(self.cache_dir, fname)
        if not os.path.exists(out_path):
            try:
                import copernicusmarine
                if self.username and self.password:
                    # Official CAS token exchange; package reads these from env as well.
                    os.environ.setdefault("COPERNICUSMARINE_SERVICE_USERNAME", self.username)
                    os.environ.setdefault("COPERNICUSMARINE_SERVICE_PASSWORD", self.password)
                import inspect
                subset_kwargs = {
                    "dataset_id": self.product_id,
                    # Pass credentials explicitly. Recent copernicusmarine
                    # releases may prompt interactively even when only the
                    # legacy environment variable names are configured.
                    "username": self.username,
                    "password": self.password,
                    "variables": variables,
                    "minimum_longitude": lon_min,
                    "maximum_longitude": lon_max,
                    "minimum_latitude": lat_min,
                    "maximum_latitude": lat_max,
                    "minimum_depth": 0.0,
                    "maximum_depth": 0.0,
                    "start_datetime": st,
                    "end_datetime": en,
                    "output_directory": self.cache_dir,
                    "output_filename": fname,
                }
                sig = inspect.signature(copernicusmarine.subset)
                if "disable_progress_bar" in sig.parameters:
                    subset_kwargs["disable_progress_bar"] = True
                elif "show_progress" in sig.parameters:
                    subset_kwargs["show_progress"] = False
                if "overwrite" in sig.parameters:
                    subset_kwargs["overwrite"] = False
                elif "force_download" in sig.parameters:
                    subset_kwargs["force_download"] = False

                timeout_seconds = max(5, float(os.getenv("OCEANTRACE_CMEMS_TIMEOUT_SECONDS", "45")))
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cmems-subset")
                future = executor.submit(copernicusmarine.subset, **subset_kwargs)
                try:
                    future.result(timeout=timeout_seconds)
                except FuturesTimeoutError as timeout_error:
                    future.cancel()
                    raise EnvironmentalDataError(
                        f"CMEMS subset exceeded {timeout_seconds:.0f}s. "
                        "Check credentials/network and use a smaller request window."
                    ) from timeout_error
                finally:
                    # Do not block the API worker while a third-party SDK call
                    # unwinds after the bounded wait has expired.
                    executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                detail = str(e).strip() or type(e).__name__
                raise EnvironmentalDataError(f"CMEMS live download failed: {detail}")
            if not os.path.exists(out_path):
                raise EnvironmentalDataError(
                    f"CMEMS download completed but output file missing: {out_path}"
                )

        try:
            import xarray as xr
            self._dataset = xr.open_dataset(out_path)
            self.dataset_path = out_path
            self._live_state = {"source": "live_copernicusmarine", "cache_path": out_path}
            return out_path
        except Exception as e:
            raise EnvironmentalDataError(f"Failed to open staged CMEMS NetCDF at {out_path}: {e}")

    def _resolve_variables(self) -> List[str]:
        """Variable names for the configured product (default: uo/vo surface currents)."""
        base = ["uo", "vo"]
        configured = os.environ.get("COPERNICUS_MARINE_VARIABLES")
        if configured:
            return [v.strip() for v in configured.split(",") if v.strip()]
        if "PHY_001_024" in self.product_id:
            return base
        return base

    def prepare_for_region(self, spill: Any) -> None:
        """Stages CMEMS surface currents for the spill's footprint + look-ahead window."""
        if self._dataset is not None:
            return
        try:
            bbox = _spill_bbox(spill, padding_deg=0.15)
            t0 = spill.observation_timestamp - timedelta(hours=6)
            t1 = spill.observation_timestamp + timedelta(hours=48)
            self.stage_currents(bbox, t0, t1)
        except EnvironmentalDataError:
            raise
        except Exception as e:
            raise EnvironmentalDataError(f"CMEMS region staging failed: {e}")

    # ------------------------------------------------------------------
    # EnvironmentalForcingProvider implementation
    # ------------------------------------------------------------------

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates eastward (uo) and northward (vo) current velocity components."""
        if self._dataset is not None:
            try:
                # Interpolate using xarray spatial coordinates
                # Assumes dataset has variables 'uo' and 'vo' and dims ('time', 'latitude', 'longitude')
                # Most downloaded CMEMS NetCDF files encode UTC as a
                # timezone-naive datetime64 coordinate. Normalize the query
                # at the xarray boundary; the public simulation contract stays
                # timezone-aware UTC.
                lookup_time = timestamp
                if getattr(lookup_time, "tzinfo", None) is not None:
                    lookup_time = lookup_time.astimezone(timezone.utc).replace(tzinfo=None)
                ds_t = self._dataset.sel(time=lookup_time, method="nearest")
                query_lons = np.asarray(lons, dtype=np.float64)
                dataset_lons = np.asarray(self._dataset["longitude"].values)
                # CMEMS files may encode longitude in 0..360 or as an
                # unwrapped 0..360+360 grid. Match the dataset convention.
                if float(np.nanmin(dataset_lons)) >= 0 and np.nanmin(query_lons) < 0:
                    query_lons = np.mod(query_lons, 360.0)
                u_interp = ds_t["uo"].interp(longitude=("points", query_lons), latitude=("points", lats)).values
                v_interp = ds_t["vo"].interp(longitude=("points", query_lons), latitude=("points", lats)).values
                # Surface-only CMEMS files can retain a singleton depth axis
                # after interpolation; the forcing contract is one value per
                # requested particle, so remove only singleton dimensions.
                u_result = np.asarray(u_interp, dtype=np.float64).squeeze()
                v_result = np.asarray(v_interp, dtype=np.float64).squeeze()
                u_result = np.atleast_1d(u_result)
                v_result = np.atleast_1d(v_result)
                if not np.all(np.isfinite(u_result)) or not np.all(np.isfinite(v_result)):
                    raise EnvironmentalDataError(
                        "CMEMS local dataset does not cover the requested longitude/latitude/time "
                        f"({float(np.min(lons)):.3f}, {float(np.min(lats)):.3f}, {timestamp.isoformat()})."
                    )
                return u_result, v_result
            except Exception as e:
                raise EnvironmentalDataError(f"Error interpolating CMEMS dataset: {e}")

        # If no local dataset and no valid API connection:
        raise EnvironmentalDataError(
            "CMEMS current forcing data is unavailable. No local dataset path provided and "
            "live Copernicus Marine credentials not configured. In accordance with the "
            "Sentinel No-Fabrication Policy, missing environmental data will not be silently synthesized."
        )

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CMEMS current provider does not supply atmospheric winds."""
        raise EnvironmentalDataError(
            "CMEMS Ocean Current Provider does not supply atmospheric winds. "
            "Use CompositeForcingProvider with ERA5WindProvider."
        )

    def coverage_window(self) -> Tuple[Optional[str], Optional[str]]:
        """(start_utc, end_utc) of the staged current data, else (None, None)."""
        from .forcing_selection import coverage_from_dataset

        return coverage_from_dataset(self._dataset)

    def to_opendrift_readers(self) -> Optional[List[Any]]:
        """Exposes the staged CMEMS NetCDF to OpenDrift as a CF-generic reader."""
        if self._dataset is None or not self.dataset_path:
            return None
        try:
            from opendrift.readers.reader_netCDF_CF_generic import Reader as NetCDFReader
            return [NetCDFReader(self.dataset_path)]
        except Exception as exc:
            logger.warning("CMEMS->OpenDrift reader construction failed: %s", exc)
            return None

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "CMEMSCurrentProvider",
            "source": "Copernicus Marine Environment Monitoring Service",
            "product_id": self.product_id,
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
        lon, lat = spill.centroid
        return (lon - padding_deg, lat - padding_deg, lon + padding_deg, lat + padding_deg)
    return (xmin - padding_deg, ymin - padding_deg, xmax + padding_deg, ymax + padding_deg)