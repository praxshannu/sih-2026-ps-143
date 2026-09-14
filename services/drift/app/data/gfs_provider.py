"""
NOAA Global Forecast System (GFS) & ERA5 Atmospheric Wind Forcing Provider for SENTINEL.
Adapted from OceanTrace agent2/adapters/noaa_gfs_adapter.py.

Ingests multi-dimensional gridded wind fields using Xarray and NetCDF4.
Supports standard GFS variables (ugrd10m, vgrd10m) and ECMWF variables (u10, v10).

Live acquisition path: NOAA NOMADS OPeNDAP (https://nomads.ncep.noaa.gov/dods/gfs_0p25)
via xarray's pydap backend. A TDS-referenced .nc subset is cached under
<repo>/data/cache/forcing/gfs/*.nc.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .forcing_base import EnvironmentalForcingProvider
from ..errors import EnvironmentalDataError

logger = logging.getLogger("sentinel.drift.GFS")

# Mirror 1: Unidata THREDDS public GFS 0.25deg TwoD aggregation.
DEFAULT_GFS_OPENDAP = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/TwoD"
# Mirror 2: Unidata THREDDS TwoD (reftime x timeOffset) aggregation.
UNIDATA_GFS_TWO_D = "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/TwoD"


class NOAAGFSWindProvider(EnvironmentalForcingProvider):
    """
    Ingests NOAA GFS / ECMWF ERA5 10m atmospheric wind datasets using Xarray / NetCDF4.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        source_name: str = "NOAA_GFS_0p25",
        opendap_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        wind_u: float = 4.0,
        wind_v: float = 2.0,
    ):
        self.dataset_path = dataset_path
        self.source_name = source_name
        self.opendap_url = opendap_url or os.environ.get("GFS_OPENDAP_URL") or DEFAULT_GFS_OPENDAP
        self.cache_dir = cache_dir or self._default_cache_dir()
        self.wind_u = float(wind_u)
        self.wind_v = float(wind_v)
        self._dataset: Optional[Any] = None
        self._live_state = None

        if self.dataset_path and os.path.exists(self.dataset_path):
            if not np:
                raise EnvironmentalDataError("xarray and netCDF4 are required to read GFS datasets.")
            try:
                import xarray as xr
                import netCDF4
                self._dataset = xr.open_dataset(self.dataset_path)
                self._live_state = {"source": "local_file", "dataset_path": self.dataset_path}
            except Exception as e:
                raise EnvironmentalDataError(f"Failed to open GFS NetCDF file at {self.dataset_path}: {e}")

    # ------------------------------------------------------------------
    # Live acquisition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_cache_dir() -> str:
        base = os.getenv("OCEANTRACE_CACHE_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache")
        return os.path.join(base, "forcing", "gfs")

    @staticmethod
    def _y0(lat_max: float) -> int:
        return int(round((90.0 - lat_max) / 0.25))

    @staticmethod
    def _y1(lat_min: float) -> int:
        return int(round((90.0 - lat_min) / 0.25))

    @staticmethod
    def _x0(lon: float) -> int:
        return int(round((lon - 0.0) / 0.25))

    @staticmethod
    def _x1(lon: float) -> int:
        return int(round((lon - 0.0) / 0.25))

    def live_available(self) -> bool:
        return True

    def require_credentials(self) -> None:
        if not self.live_available():
            raise EnvironmentalDataError(
                "GFS live download unavailable: optional deps missing "
                "(`pip install pydap netCDF4`). Provide a local dataset_path otherwise."
            )

    def stage_winds(
        self,
        bbox: Tuple[float, float, float, float],
        t_start: datetime,
        t_end: datetime,
    ) -> str:
        """
        Pulls the GFS 0.25-deg 10m wind field for the requested spatio-temporal
        window and caches it as a normalized NetCDF (u10/v10, lon/lat/time).
        Mirrors are tried in order: Unidata THREDDS TwoD aggregate, then the
        legacy NOMADS OPeNDAP (publicly retired by SCN 25-81 in 2025, kept for
        on-prem / legacy mirror deployments). If every live mirror fails and the labelled
        synthetic fallback is enabled (OCEANTRACE_GFS_SYNTHETIC_FALLBACK, default
        true), a deterministic gridded wind field is written and its provenance
        is recorded (`data_origin: synthetic_gridded`). It is never reported as
        live data.
        """
        self.require_credentials()
        os.makedirs(self.cache_dir, exist_ok=True)

        lon_min, lat_min, lon_max, lat_max = bbox
        if lon_min < 0:
            # GFS longitudes are 0..360
            lon_min += 360.0
            lon_max += 360.0
        lon_min = max(0.0, min(lon_min, 359.75))
        lon_max = max(0.0, min(lon_max, 359.75))
        fname_date = t_start.strftime("%Y%m%d")
        init_hour = (t_start.hour // 6) * 6

        out_path = os.path.join(self.cache_dir, f"gfs_u10v10_{fname_date}{init_hour:02d}_"
                                f"{lon_min:.0f}_{lat_min:.0f}_{lon_max:.0f}_{lat_max:.0f}.nc")
        if os.path.exists(out_path):
            try:
                import xarray as xr
                self._dataset = xr.open_dataset(out_path)
                self.dataset_path = out_path
                self._live_state = {
                    "source": "cache_hit",
                    "data_origin": str(self._dataset.attrs.get("gfs_source", "unknown")),
                    "cache_path": out_path,
                }
                return out_path
            except Exception:
                os.remove(out_path)

        # Grid coordinates reconstructed from the fixed 0.25-deg GFS index space
        # (the DODS subset does not carry lat/lon values).
        j0 = int(round((90.0 - lat_max) / 0.25))
        j1 = int(round((90.0 - lat_min) / 0.25))
        i0 = int(round((lon_min - 0.0) / 0.25))
        i1 = int(round((lon_max - 0.0) / 0.25))
        lons_vals = 0.25 * np.arange(i0, i1 + 1)
        lats_vals = 90.0 - 0.25 * np.arange(j1, j0 - 1, -1)  # ascending

        # Mirror 1: Unidata THREDDS TwoD aggregation (publicly reachable).
        unidata_base = os.environ.get("GFS_THREDDS_URL") or UNIDATA_GFS_TWO_D
        try:
            unidata_url = self._unidata_subset_url(unidata_base, lon_min, lat_min, lon_max, lat_max, t_start)
            self._pull_and_normalize(
                unidata_url, out_path, "unidata_thredds",
                u_candidates=["ugrd10m", "u10", "10u"],
                v_candidates=["vgrd10m", "v10", "10v"],
                lons=lons_vals, lats=lats_vals,
            )
            return self._finalize(out_path)
        except Exception as e:
            logger.warning("Unidata THREDDS failed: %s", e)

        # Mirror 2: legacy NOMADS OPeNDAP single-file DODS (publicly retired).
        try:
            nomads_url = self._nomads_subset_url(lon_min, lat_min, lon_max, lat_max, t_start, init_hour)
            self._pull_and_normalize(
                nomads_url, out_path, "nomads_opendap",
                u_candidates=["ugrd10m", "u10"],
                v_candidates=["vgrd10m", "v10"],
                lons=lons_vals, lats=lats_vals,
            )
            return self._finalize(out_path)
        except Exception as e:
            logger.warning("NOMADS OPeNDAP failed: %s", e)

        # Labelled synthetic fallback (never mislabeled as live).
        if os.getenv("OCEANTRACE_GFS_SYNTHETIC_FALLBACK", "true").lower() != "false":
            logger.warning(
                "All GFS mirrors failed for %s; writing labelled synthetic gridded "
                "wind field (errors recorded).", out_path,
            )
            self._write_synthetic_grid(out_path, lon_min, lat_min, lon_max, lat_max, t_start)
            self._dataset = xr.open_dataset(out_path)
            self.dataset_path = out_path
            self._live_state = {
                "source": "synthetic_gridded",
                "data_origin": "synthetic_gridded",
                "cache_path": out_path,
            }
            return out_path

        raise EnvironmentalDataError(
            "All GFS mirrors failed and OCEANTRACE_GFS_SYNTHETIC_FALLBACK is disabled. "
            "Provide a local dataset_path or a reachable GFS mirror instead."
        )

    # ------------------------------------------------------------------
    # Subset URL builders + download helpers
    # ------------------------------------------------------------------

    def _nomads_subset_url(
        self, lon_min: float, lat_min: float, lon_max: float, lat_max: float,
        t_start: datetime, init_hour: int,
    ) -> str:
        fname_date = t_start.strftime("%Y%m%d")
        return (f"{self.opendap_url}/gfs{fname_date}/00/atmos/gfs.t00z.pgrb2.0p25.f000_0p25?"
                f"time[0:0:0],ugrd10m[{init_hour // 6}]"
                f"[{self._y0(lat_max)}:1:{self._y1(lat_min)}][{self._x0(lon_min)}:1:{self._x1(lon_max)}],"
                f"vgrd10m[{init_hour // 6}]"
                f"[{self._y0(lat_max)}:1:{self._y1(lat_min)}][{self._x0(lon_min)}:1:{self._x1(lon_max)}]")

    def _unidata_subset_url(
        self, base: str, lon_min: float, lat_min: float, lon_max: float, lat_max: float,
        t_start: datetime,
    ) -> str:
        """Builds a DODS constraint for the Unidata TwoD (reftime x validtime)
        aggregation. Selects the most recent reftime at/before t_start and the
        first forecast offset (the analysis time), plus the 10m wind level."""
        reftime_idx = self._unidata_reftime_index(base, t_start)
        uvar = "u-component_of_wind_height_above_ground"
        vvar = "v-component_of_wind_height_above_ground"
        return (
            f"{base}?reftime[{reftime_idx}:1:{reftime_idx}],"
            f"{uvar}[{reftime_idx}:1:{reftime_idx}][0:1:0][0:1:0]"
            f"[{self._y0(lat_max)}:1:{self._y1(lat_min)}][{self._x0(lon_min)}:1:{self._x1(lon_max)}],"
            f"{vvar}[{reftime_idx}:1:{reftime_idx}][0:1:0][0:1:0]"
            f"[{self._y0(lat_max)}:1:{self._y1(lat_min)}][{self._x0(lon_min)}:1:{self._x1(lon_max)}]"
        )

    def _unidata_reftime_index(self, base: str, t_start: datetime) -> int:
        """Fetches the small reftime axis and returns the index of the latest
        reference time at or before t_start (0 if all are newer)."""
        import numpy as _np
        import xarray as _xr
        refs_url = f"{base}?reftime"
        refs = _np.asarray(
            _xr.open_dataset(refs_url, engine="pydap")["reftime"].values,
            dtype="datetime64[ns]",
        )
        t0 = _np.datetime64(t_start.replace(tzinfo=None))
        valid_i = _np.nonzero(refs <= t0)[0]
        if valid_i.size:
            return int(valid_i[-1])
        return int(_np.argmin(_np.abs(refs - t0)))

    def _pull_and_normalize(
        self, url: str, out_path: str, source_tag: str,
        u_candidates: List[str], v_candidates: List[str],
        lons: np.ndarray, lats: np.ndarray,
    ) -> None:
        """Downloads one 0.25-deg DODS subset (pydap) and normalizes it into a
        small, CF-friendly NetCDF with exactly {time,lat,lon} coords and u10/v10
        grids (lat ascending 90..-90, lon ascending 0..360).

        The subset does NOT carry lat/lon coordinate values, so they are
        reconstructed from the fixed 0.25-deg grid, and the (lat_max..lat_min)
        row order returned by the server is reversed for storage. All input is
        projected out; no mirror-specific naming or attrs leak into the cache.
        """
        logger.info("GFS wind download attempt [%s] %s", source_tag, url)
        import xarray as xr
        ds = xr.open_dataset(url, engine="pydap")

        u_var = next((v for v in u_candidates if v in ds.data_vars), None)
        v_var = next((v for v in v_candidates if v in ds.data_vars), None)
        if not u_var or not v_var:
            raise EnvironmentalDataError(
                f"Wind variables not found in {source_tag} dataset. "
                f"Available: {list(ds.data_vars)[:20]}"
            )

        def _grid(dataarray):
            lat_dim = next((dim for dim in dataarray.dims if "lat" in dim.lower() and "lon" not in dim.lower()), None)
            lon_dim = next((dim for dim in dataarray.dims if "lon" in dim.lower()), None)
            if lat_dim is None or lon_dim is None:
                raise EnvironmentalDataError(f"No lat/lon dims in {source_tag} variable {dataarray.name}")
            selectors = {dim: 0 for dim in dataarray.dims if dim not in (lat_dim, lon_dim)}
            u2d = np.asarray(dataarray.isel(selectors).transpose(lat_dim, lon_dim).values,
                             dtype=np.float64)
            return u2d

        u2d = np.ma.filled(_grid(ds[u_var]), np.nan)
        v2d = np.ma.filled(_grid(ds[v_var]), np.nan)

        if u2d.shape != (lats.size, lons.size):
            raise EnvironmentalDataError(
                f"{source_tag} grid mismatch: got {u2d.shape}, expected "
                f"({lats.size}, {lons.size})"
            )
        # Server order is lat_max..lat_min; flip to ascending for storage.
        u2d = u2d[::-1, :]
        v2d = v2d[::-1, :]

        t_actual = self._resolve_window_time(ds)
        out = xr.Dataset(
            {"u10": (("time", "lat", "lon"), np.nan_to_num(u2d)[None, :, :]),
             "v10": (("time", "lat", "lon"), np.nan_to_num(v2d)[None, :, :])},
            coords={"time": np.array([t_actual], dtype="datetime64[ns]"),
                    "lat": lats, "lon": lons},
        )
        out.attrs["gfs_source"] = source_tag
        out.attrs["title"] = f"NOAA GFS 0.25deg 10m wind subset ({source_tag})"
        out.attrs["conventions"] = "CF-1.8"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_netcdf(out_path, engine="netcdf4")
        ds.close()

    @staticmethod
    def _resolve_window_time(ds: Any) -> np.datetime64:
        """Determines the single representative timestamp of a DODS subset,
        handling both reftime+offset (Unidata TwoD) and plain time (single-file)
        conventions."""
        if "time" in ds.coords:
            t = np.asarray(ds["time"].values, dtype="datetime64[ns]")
            if t.size:
                return t[0]
        reftime = None
        if "reftime" in ds.coords:
            rf = np.asarray(ds["reftime"].values, dtype="datetime64[ns]")
            reftime = rf[0] if rf.size else None
        offset_name = next((c for c in ds.coords if "offset" in c.lower() and ds.sizes.get(c, 0) == 1), None)
        offset_hours = 0.0
        if offset_name is not None and offset_name in ds.coords:
            vals = np.asarray(ds[offset_name].values)
            if vals.size:
                offset_hours = float(vals.reshape(-1)[0])
        if reftime is not None:
            return reftime + np.timedelta64(int(round(offset_hours)), "h")
        return np.datetime64("now", "h")

    def _finalize(self, out_path: str) -> str:
        try:
            import xarray as xr

            self._dataset = xr.open_dataset(out_path)
            self.dataset_path = out_path
            self._live_state = {
                "source": "live",
                "data_origin": self._dataset.attrs.get("gfs_source", "unknown"),
                "cache_path": out_path,
            }
            return out_path
        except Exception as e:
            raise EnvironmentalDataError(f"Failed to open staged GFS NetCDF at {out_path}: {e}")

    def _write_synthetic_grid(
        self, out_path: str, lon_min: float, lat_min: float, lon_max: float, lat_max: float,
        t_start: datetime,
    ) -> None:
        """Deterministic labelled gridded wind field (mock synoptic pattern)
        used only when no live wind mirror is reachable. Provenance is recorded
        in `_live_state` and NetCDF global attributes, never presented as real."""
        import numpy as _np
        import xarray as _xr
        lons = _np.arange(round(lon_min / 0.25) * 0.25, lon_max, 0.25)
        lats = _np.arange(round(lat_max / 0.25) * 0.25, lat_min - 0.24, -0.25)[::-1]
        LON, LAT = _np.meshgrid(lons, lats)
        grad = 1.0 + 0.01 * (LAT - 15.0)
        u10 = self.wind_u * grad + 1.2 * _np.sin(LON * 5.0) * 0.02
        v10 = self.wind_v * _np.ones_like(LAT) + 1.0 * _np.cos(LAT * 5.0) * 0.02
        time = _np.array([t_start.replace(tzinfo=None)], dtype="datetime64[ns]")
        ds = _xr.Dataset(
            {
                "u10": (("time", "lat", "lon"), u10[None, :, :]),
                "v10": (("time", "lat", "lon"), v10[None, :, :]),
            },
            coords={"time": time, "lat": lats, "lon": lons},
        )
        ds.attrs.update({
            "gfs_source": "synthetic_gridded",
            "title": "OceanTrace labelled synthetic wind grid (NO live GFS mirror reachable)",
            "conventions": "CF-1.8",
        })
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ds.to_netcdf(out_path)

    # ------------------------------------------------------------------
    # Prepare region
    # ------------------------------------------------------------------

    def prepare_for_region(self, spill: Any) -> None:
        """Stages GFS 10m winds for the spill's footprint + look-ahead window."""
        if self._dataset is not None:
            return
        try:
            bbox = self._spill_bbox(spill, padding_deg=0.15)
            t0 = spill.observation_timestamp - timedelta(hours=6)
            t1 = spill.observation_timestamp + timedelta(hours=48)
            self.stage_winds(bbox, t0, t1)
        except EnvironmentalDataError:
            raise
        except Exception as e:
            raise EnvironmentalDataError(f"GFS region staging failed: {e}")

    @staticmethod
    def _spill_bbox(spill: Any, padding_deg: float = 0.15) -> Tuple[float, float, float, float]:
        try:
            xmin, ymin, xmax, ymax = spill.geometry.bounds
        except Exception:
            lon, lat = spill.centroid
            return (lon - padding_deg, lat - padding_deg, lon + padding_deg, lat + padding_deg)
        return (xmin - padding_deg, ymin - padding_deg, xmax + padding_deg, ymax + padding_deg)

    # ------------------------------------------------------------------
    # EnvironmentalForcingProvider implementation
    # ------------------------------------------------------------------

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise EnvironmentalDataError("Atmospheric Wind Provider does not provide ocean current vectors.")

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates 10m u/v wind vectors at query coordinates using Xarray."""
        if self._dataset is not None:
            try:
                # Support standard GFS and ERA5 variable naming conventions
                u_var = next((v for v in ["ugrd10m", "u10", "10u", "u_wind"] if v in self._dataset), None)
                v_var = next((v for v in ["vgrd10m", "v10", "10v", "v_wind"] if v in self._dataset), None)

                if not u_var or not v_var:
                    raise EnvironmentalDataError(
                        f"Wind variables not found in GFS dataset. Available: {list(self._dataset.data_vars)}"
                    )

                # xarray sel() with an aware numpy datetime (numpy 2.x yields
                # datetime64[us, UTC]) fails against a 'ns' time coordinate.
                # Normalize to UTC-naive datetime64[ns] to match the index dtype.
                lookup_dt = timestamp
                if getattr(lookup_dt, "tzinfo", None) is not None:
                    lookup_dt = lookup_dt.astimezone(timezone.utc).replace(tzinfo=None)
                # Holds a datetime until the conversion succeeds, then a
                # datetime64; both are valid for Dataset.sel().
                lookup_time: datetime | np.datetime64
                try:
                    lookup_time = np.datetime64(lookup_dt, "ns")
                except (TypeError, ValueError):
                    lookup_time = lookup_dt

                ds_t = self._dataset.sel(time=lookup_time, method="nearest")
                lon_dim = "lon" if "lon" in ds_t.coords else "longitude"
                lat_dim = "lat" if "lat" in ds_t.coords else "latitude"

                u_interp = ds_t[u_var].interp({lon_dim: ("points", lons), lat_dim: ("points", lats)}).values
                v_interp = ds_t[v_var].interp({lon_dim: ("points", lons), lat_dim: ("points", lats)}).values

                u_wind = np.asarray(u_interp, dtype=np.float64)
                v_wind = np.asarray(v_interp, dtype=np.float64)
                # Forecast particles can drift beyond the staged tile bounds;
                # xarray returns NaN there. Fall back to the configured
                # background wind instead of propagating NaN into the RK4
                # integration (labelled: same philosophy as OCEANTRACE_GFS_SYNTHETIC_FALLBACK).
                invalid = ~(np.isfinite(u_wind) & np.isfinite(v_wind))
                if np.any(invalid):
                    u_wind = np.where(invalid, self.wind_u, u_wind)
                    v_wind = np.where(invalid, self.wind_v, v_wind)
                    logger.warning(
                        "GFS wind NaN outside staged tile; applied background wind "
                        "(u=%s, v=%s) to %d particle point(s)", self.wind_u, self.wind_v, int(np.count_nonzero(invalid)),
                    )

                return u_wind, v_wind

            except Exception as e:
                raise EnvironmentalDataError(f"Error interpolating GFS wind vectors: {e}")

        raise EnvironmentalDataError(
            f"NOAA GFS wind forcing is not connected. No dataset path provided. "
            f"In accordance with the Sentinel No-Fabrication Policy, missing wind forcing will not be synthesized."
        )

    def coverage_window(self) -> Tuple[Optional[str], Optional[str]]:
        """(start_utc, end_utc) of the staged wind grid, else (None, None)."""
        from .forcing_selection import coverage_from_dataset

        return coverage_from_dataset(self._dataset)

    def to_opendrift_readers(self) -> Optional[List[Any]]:
        """Exposes the GFS/ERA5 wind NetCDF to OpenDrift as a CF-generic reader."""
        if self._dataset is None or not self.dataset_path:
            return None
        try:
            from opendrift.readers.reader_netCDF_CF_generic import Reader as NetCDFReader
            return [NetCDFReader(self.dataset_path)]
        except Exception as exc:
            logger.warning("GFS->OpenDrift reader construction failed: %s", exc)
            return None

    def metadata(self) -> Dict[str, Any]:
        origin = None
        if self._dataset is not None:
            origin = str(self._dataset.attrs.get("gfs_source", "")) or (
                "synthetic_gridded" if (self._live_state or {}).get("source") == "synthetic_gridded" else None
            )
        return {
            "provider": "NOAAGFSWindProvider",
            "source": self.source_name,
            "dataset_path": self.dataset_path,
            "is_connected": self._dataset is not None,
            "data_origin": origin or ("not_staged" if self._dataset is None else "unknown"),
            "engine": "Xarray/NetCDF4",
            "live_ready": self.live_available(),
            "live_state": self._live_state,
        }