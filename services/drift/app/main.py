"""SENTINEL Drift Service – FastAPI Application.

Lagrangian physics drift engine for backward (origin finding) and
forward (shoreline impact) drift modeling.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from loguru import logger

from .engine.backward_sde import RegimeInputs
from .errors import (
    EnvironmentalDataError,
    MissingCurrentForcingError,
    MissingWindForcingError,
)
from .schemas import (
    BackwardRequest,
    ConfidenceEllipse,
    DriftResult,
    ForwardRequest,
    ForwardResult,
    FullPipelineRequest,
    FullPipelineResult,
    HealthResponse,
    OceanDataRef,
)

# Configure loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/drift_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days", level="DEBUG")

_XGB_MODEL = None


def _get_xgb_model():
    """Lifespan-loaded singleton for the XGBoost residual corrector."""
    global _XGB_MODEL
    if _XGB_MODEL is None:
        try:
            from .engine.xgb_residual import XGBoostPhysicsResidualModel

            _XGB_MODEL = XGBoostPhysicsResidualModel()
            logger.info("XGBoost residual model ready (trained={})", _XGB_MODEL.is_trained)
        except Exception as e:
            logger.warning("XGBoost residual unavailable, empirical fallback: {}", e)
            _XGB_MODEL = None
    return _XGB_MODEL


def _build_fields(spill_lon: float, spill_lat: float, req_ocean, now, allow_synthetic: bool = True):
    """Build (u_field, v_field, K_field, lons, lats, times, provider, provenance).

    Tries local NetCDF loaders first; on any failure falls back to the
    forcing factory (live CMEMS/ERA5 if credentialed, else GFS + labelled
    mock) sampled onto a small grid so the SDE engine always runs
    and the response always labels its source.

    `provenance` is the full per-field selection record: source, real-vs-
    synthetic, coverage window and confidence penalty. Anything synthetic is
    visible here — it is never a silent substitution.
    """
    from .data.forcing_factory import build_forcing_provider

    provider, provenance = build_forcing_provider(
        cmems_path=getattr(
            req_ocean,
            "cmems_base",
            "",
        )
        or None,
        era5_path=getattr(
            req_ocean,
            "era5_base",
            "",
        )
        or None,
        gfs_path=getattr(req_ocean, "gfs_base", "") or None,
        window=(req_ocean.time_start, req_ocean.time_end),
        allow_synthetic=allow_synthetic,
    )

    # Try local NetCDF loaders when paths look real.
    cmems_base = (
        getattr(
            req_ocean,
            "cmems_base",
            "",
        )
        or ""
    )
    era5_base = (
        getattr(
            req_ocean,
            "era5_base",
            "",
        )
        or ""
    )
    bbox = getattr(req_ocean, "bbox", (spill_lon - 1, spill_lat - 1, spill_lon + 1, spill_lat + 1))
    try:
        if cmems_base and Path(cmems_base).exists():
            from .data.cmems_loader import load_cmems_currents

            ocean_ds = load_cmems_currents(
                cmems_base,
                np.datetime64(req_ocean.time_start),
                np.datetime64(req_ocean.time_end),
                bbox,
            )
            u_field = ocean_ds["u"].values if "u" in ocean_ds else ocean_ds["uo"].values
            v_field = ocean_ds["v"].values if "v" in ocean_ds else ocean_ds["vo"].values
            lons = ocean_ds["longitude"].values
            lats = ocean_ds["latitude"].values
            times = _numeric_times(ocean_ds["time"].values)
            K_field = np.zeros((u_field.shape[0], u_field.shape[1], u_field.shape[2], 2, 2))
            # `is_real` must mean the bytes are on disk, not merely that a path
            # was typed into the request.
            era5_staged = bool(era5_base) and Path(era5_base).exists()
            if not era5_staged and not allow_synthetic:
                raise MissingWindForcingError(
                    "Wind forcing is unavailable for this run: no local ERA5 file "
                    "is staged and synthetic wind was refused. Stage an ERA5 (or "
                    "GFS) dataset, or allow synthetic forcing explicitly.",
                    reason="wind_not_staged_synthetic_refused",
                )
            provenance = dict(
                provenance,
                forcing_source="local_files",
                current={
                    "field": "current",
                    "source": "cmems_local_file",
                    "provider": "local_netcdf",
                    "is_real": True,
                    "status": "real_staged",
                    "coverage_start_utc": _iso_time(req_ocean.time_start),
                    "coverage_end_utc": _iso_time(req_ocean.time_end),
                    "reason": f"Opened local CMEMS NetCDF {cmems_base}",
                },
                wind={
                    "field": "wind",
                    "source": "era5_local_file" if era5_staged else "not_staged",
                    "provider": "local_netcdf",
                    "is_real": bool(era5_staged),
                    "status": "real_staged" if era5_staged else "unavailable",
                    "coverage_start_utc": _iso_time(req_ocean.time_start) if era5_staged else None,
                    "coverage_end_utc": _iso_time(req_ocean.time_end) if era5_staged else None,
                    "reason": "Local ERA5 file staged"
                    if era5_staged
                    else "No local ERA5 file staged",
                },
            )
            return (
                u_field,
                v_field,
                K_field,
                lons,
                lats,
                times,
                provider,
                _normalise_provenance(provenance),
            )
    except EnvironmentalDataError:
        # A refused field is a verdict, not a load error — never fall through
        # to the factory grid, which would silently serve the mock anyway.
        raise
    except Exception as e:
        logger.warning("Local NetCDF load failed, using forcing factory: {}", e)

    # Factory-sampled synthetic grid (values from the real provider when live).
    try:
        ts = (
            req_ocean.time_start
            if hasattr(req_ocean.time_start, "tzinfo")
            else req_ocean.time_start
        )
    except Exception:
        ts = now
    lon_grid = np.linspace(spill_lon - 1.0, spill_lon + 1.0, 9)
    lat_grid = np.linspace(spill_lat - 1.0, spill_lat + 1.0, 9)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    try:
        u_s, v_s = provider.get_current_vectors(LON.ravel(), LAT.ravel(), ts)
        u2 = np.asarray(u_s, dtype=np.float64).reshape(9, 9)
        v2 = np.asarray(v_s, dtype=np.float64).reshape(9, 9)
    except Exception as e:
        if not allow_synthetic:
            # A constant analytic substitute IS synthetic data, so serving it
            # here would be exactly the silent fallback the caller refused.
            raise MissingCurrentForcingError(
                f"Current provider {type(provider).__name__} could not be sampled "
                f"onto the model grid and synthetic substitution was refused: {e}",
                reason="current_sampling_failed_synthetic_refused",
            ) from e
        logger.warning("Provider current sampling failed, analytic fallback: {}", e)
        u2 = np.full((9, 9), 0.15)
        v2 = np.full((9, 9), 0.05)
        # A constant analytic substitute is synthetic data. Say so, per field,
        # and let the confidence penalty follow.
        provenance = dict(
            provenance,
            current_origin="analytic_fallback",
            current={
                "field": "current",
                "source": "synthetic_mock",
                "provider": "analytic_constant",
                "is_real": False,
                "status": "synthetic_mock",
                "reason": f"Provider sampling failed ({str(e)[:120]}); constant analytic field substituted",
            },
        )
    u_field = np.stack([u2, u2])
    v_field = np.stack([v2, v2])
    K_field = np.zeros((2, 9, 9, 2, 2))
    K_field[..., 0, 0] = 100.0
    K_field[..., 1, 1] = 100.0
    # Numeric seconds grid: the SDE engine queries interpolators with
    # float seconds-remaining, so datetime64 grids would break argmin.
    times = np.array([0.0, 3600.0])
    return (
        u_field,
        v_field,
        K_field,
        lon_grid,
        lat_grid,
        times,
        provider,
        _normalise_provenance(provenance),
    )


def _iso_time(value: object) -> str | None:
    """ISO-8601 for a datetime (or numpy datetime64), else None."""
    if value is None:
        return None
    try:
        return value.isoformat()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - provenance metadata must never crash a run
        return str(value)


def _as_field(field: str, data: dict) -> dict:
    """Fill a FieldSelection payload with safe defaults."""
    return {
        "field": field,
        "source": str(data.get("source", "not_staged")),
        "provider": str(data.get("provider", "")),
        "is_real": bool(data.get("is_real", False)),
        "status": str(data.get("status", "unavailable")),
        "reason": str(data.get("reason", "")),
        "dataset_id": data.get("dataset_id"),
        "coverage_start_utc": data.get("coverage_start_utc"),
        "coverage_end_utc": data.get("coverage_end_utc"),
    }


def _normalise_provenance(prov: dict, extra_notes: list[str] | None = None) -> dict:
    """Rebuild the provenance record through ForcingSelection.

    Guarantees that `any_synthetic`, `synthetic_fields`, `confidence_penalty`
    and the literal ``SYNTHETIC FORCING`` warning are always consistent with
    the per-field `is_real` flags — including when a code path only knew how
    to patch `current_origin`.
    """
    from .data.forcing_selection import FieldSelection, ForcingSelection

    notes = [n for n in (prov.get("warnings") or []) if not n.startswith("SYNTHETIC FORCING")]
    notes.extend(extra_notes or [])
    selection = ForcingSelection(
        wind=FieldSelection(**_as_field("wind", prov.get("wind") or {})),
        current=FieldSelection(**_as_field("current", prov.get("current") or {})),
        forcing_source=str(prov.get("forcing_source", "synthetic_mock")),
        extra_notes=notes,
    )
    out = selection.to_dict()
    out["forcing_detail"] = dict(prov)
    return out


def _http_forcing_unavailable(exc: EnvironmentalDataError) -> HTTPException:
    """503 carrying the typed error's machine-readable payload.

    A missing forcing field is not a server crash and must not masquerade as
    one: 500 + a prose string forces the UI to guess. 503 + ``to_dict()``
    lets it render an explicit "wind unavailable" state with the reason code.
    """
    return HTTPException(status_code=503, detail=exc.to_dict())


def _numeric_times(times: np.ndarray) -> np.ndarray:
    """Convert datetime64 grids to float seconds offset (engine contract)."""
    try:
        t = np.asarray(times)
        if np.issubdtype(t.dtype, np.datetime64):
            return ((t - t[0]) / np.timedelta64(1, "s")).astype(np.float64)
        return t.astype(np.float64)
    except Exception:
        return np.arange(np.asarray(times).shape[0], dtype=np.float64)


def _apply_xgb_correction(ellipse: ConfidenceEllipse, provider, req, provenance: dict):
    """Post-hoc XGBoost residual correction on the origin ellipse.

    Never skips WMC (that runs inside the SDE); this only shifts the
    already-computed centre by (dx, dy) metres and scales axes by the
    predicted uncertainty_scale.
    """
    try:
        import numpy as _np

        ts = req.ocean_data.time_start
        lon_c, lat_c = ellipse.center_lon, ellipse.center_lat
        try:
            uc, vc = provider.get_current_vectors(_np.array([lon_c]), _np.array([lat_c]), ts)
            uw, vw = provider.get_wind_vectors(_np.array([lon_c]), _np.array([lat_c]), ts)
            u_c, v_c, u_w, v_w = float(uc[0]), float(vc[0]), float(uw[0]), float(vw[0])
        except Exception:
            u_c, v_c, u_w, v_w = 0.15, 0.05, 4.0, 2.0
        xgb = _get_xgb_model()
        if xgb is None:
            return ellipse, False, {}
        corr = xgb.predict_residual(
            u_current=u_c,
            v_current=v_c,
            u_wind=u_w,
            v_wind=v_w,
            windage=0.03,
            sst=28.0,
            duration_hours=float(req.spill_age_hours),
            observed_area_km2=5.0,
            aspect_ratio=2.0,
        )
        dx_m = float(corr["dx_residual_meters"])
        dy_m = float(corr["dy_residual_meters"])
        scale = float(corr["uncertainty_scale"])
        km_lat = 111.0
        km_lon = 111.0 * float(_np.cos(_np.radians(min(max(lat_c, -89), 89))))
        new_lon = lon_c + (dx_m / 1000.0) / max(km_lon, 1e-6)
        new_lat = lat_c + (dy_m / 1000.0) / km_lat
        corrected = ConfidenceEllipse(
            center_lon=float(new_lon),
            center_lat=float(new_lat),
            semi_major_km=float(ellipse.semi_major_km * scale),
            semi_minor_km=float(ellipse.semi_minor_km * scale),
            orientation_deg=float(ellipse.orientation_deg),
        )
        return corrected, True, {"dx": dx_m, "dy": dy_m, "scale": scale}
    except Exception as e:
        logger.warning("XGB correction skipped: {}", e)
        return ellipse, False, {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("SENTINEL Drift Service starting up")
    Path("logs").mkdir(exist_ok=True)
    _get_xgb_model()
    yield
    logger.info("SENTINEL Drift Service shutting down")


app = FastAPI(
    title="SENTINEL Drift Service",
    description="Lagrangian physics drift engine for oil spill origin finding and shoreline impact forecasting",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.post("/drift/backward", response_model=DriftResult)
async def drift_backward(req: BackwardRequest):
    """Find probable origin of a detected spill.

    Given a spill centroid, age, and ocean data references, runs an ensemble
    of N backward SDE integrations to reconstruct the origin distribution.
    Returns a 95% confidence ellipse and particle positions.
    """
    t0 = time.time()
    logger.info(
        "Backward drift request: spill=({:.4f}, {:.4f}), age={}h, N={}",
        req.spill_lon,
        req.spill_lat,
        req.spill_age_hours,
        req.n_particles,
    )

    try:
        from datetime import datetime as _dt

        from .engine.backward_sde import compute_confidence_ellipse, integrate_backward

        # Load ocean data (local NetCDF if present, else forcing factory).
        u_field, v_field, K_field, lons, lats, times, provider, provenance = _build_fields(
            req.spill_lon,
            req.spill_lat,
            req.ocean_data,
            _dt.now(),
            allow_synthetic=req.allow_synthetic,
        )

        from .engine.ensemble import make_interpolators, run_backward_ensemble

        u_interp, v_interp, K_interp = make_interpolators(
            u_field,
            v_field,
            K_field,
            lons,
            lats,
            times,
        )

        # Regime inputs: latitude and the sampled current speed are all this
        # endpoint can honestly supply; bathymetry and coast distance are
        # unknown, so the classifier reports the regime as undetermined
        # rather than being handed a guess.
        try:
            _u0, _v0 = provider.get_current_vectors(
                np.array([req.spill_lon]), np.array([req.spill_lat]), req.ocean_data.time_start
            )
            _speed = np.array([float(np.hypot(float(_u0[0]), float(_v0[0])))])
        except Exception:  # noqa: BLE001 - regime input, not a hard failure
            _speed = None

        regime_inputs = RegimeInputs(
            depth_m=None,
            distance_to_coast_km=None,
            current_speed=_speed,
            lat=np.array([req.spill_lat]),
        )

        # Run the backward ensemble (regime selected first, WMC inside).
        drift = run_backward_ensemble(
            req.spill_lon,
            req.spill_lat,
            req.spill_age_hours,
            u_interp,
            v_interp,
            K_interp,
            n_particles=req.n_particles,
            random_seed=req.random_seed,
            regime_inputs=regime_inputs,
            k_field=K_field,
            grid_lons=np.asarray(lons),
            grid_lats=np.asarray(lats),
            forcing=provenance,
            detection_time=req.ocean_data.time_end,
        )

        # XGBoost physics residual post-hoc correction (never skips WMC).
        ellipse, xgb_applied, xgb_corr = _apply_xgb_correction(
            drift.origin_ellipse, provider, req, provenance
        )
        drift.origin_ellipse = ellipse
        drift.origin_distribution = dict(
            drift.origin_distribution,
            center_lon=ellipse.center_lon,
            center_lat=ellipse.center_lat,
        )

        from .schemas import ForcingReport

        drift.forcing_source = str(provenance.get("forcing_source", "synthetic_mock"))
        drift.forcing_detail = dict(provenance)
        drift.forcing = ForcingReport(
            wind=provenance["wind"],
            current=provenance["current"],
            forcing_source=drift.forcing_source,
            any_synthetic=bool(provenance.get("any_synthetic", True)),
            synthetic_fields=list(provenance.get("synthetic_fields") or []),
            confidence_penalty=float(provenance.get("confidence_penalty", 0.0)),
            warnings=list(provenance.get("warnings") or []),
        )
        drift.xgb_residual_applied = bool(xgb_applied)
        drift.xgb_correction_m = dict(xgb_corr)
        result = drift

        elapsed = time.time() - t0
        logger.info("Backward drift completed in {:.2f}s", elapsed)

        return result

    except EnvironmentalDataError as exc:
        logger.error("Backward drift refused: {} ({})", exc.reason, exc.code)
        raise _http_forcing_unavailable(exc) from exc
    except Exception as e:
        logger.error("Backward drift failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Drift computation failed: {str(e)}")


@app.post("/drift/forward", response_model=ForwardResult)
async def drift_forward(req: ForwardRequest):
    """Forecast 72-hour forward trajectories from an origin.

    Given an origin with uncertainty, runs an ensemble of forward Lagrangian
    simulations and returns trajectory paths with probability quantiles.
    """
    t0 = time.time()
    logger.info(
        "Forward forecast request: origin=({:.4f}, {:.4f}), forecast={}h, N={}",
        req.origin_lon,
        req.origin_lat,
        req.forecast_hours,
        req.n_particles,
    )

    try:
        from datetime import datetime as _dt2

        from .engine.ensemble import make_interpolators
        from .engine.forward_forecast import compute_trajectory_quantiles, integrate_forward

        # Load ocean data (local NetCDF if present, else forcing factory).
        u_field, v_field, K_field, lons, lats, times, provider, provenance = _build_fields(
            req.origin_lon,
            req.origin_lat,
            req.ocean_data,
            _dt2.now(),
            allow_synthetic=req.allow_synthetic,
        )

        u_interp, v_interp, K_interp = make_interpolators(
            u_field,
            v_field,
            K_field,
            lons,
            lats,
            times,
        )

        # Spawn particles from origin distribution
        rng = np.random.default_rng(req.random_seed)
        origin_lons = rng.normal(req.origin_lon, req.origin_lon_sigma, req.n_particles)
        origin_lats = rng.normal(req.origin_lat, req.origin_lat_sigma, req.n_particles)

        # Forward integration
        fwd_result = integrate_forward(
            origin_lons,
            origin_lats,
            req.forecast_hours,
            u_interp,
            v_interp,
            K_interp,
            random_seed=req.random_seed,
        )

        # Compute quantiles
        traj = compute_trajectory_quantiles(
            fwd_result["lons"], fwd_result["lats"], fwd_result["times"]
        )

        from .schemas import TrajectoryEnsemble, TrajectoryPoint

        trajectories = TrajectoryEnsemble(
            mean_path=[TrajectoryPoint(**p) for p in traj["mean_path"]],
            quantile_5_path=[TrajectoryPoint(**p) for p in traj["quantile_5_path"]],
            quantile_95_path=[TrajectoryPoint(**p) for p in traj["quantile_95_path"]],
            all_paths_lon=traj["all_paths_lon"],
            all_paths_lat=traj["all_paths_lat"],
        )

        result = ForwardResult(
            trajectories=trajectories,
            regime="markov1",
            forecast_hours=req.forecast_hours,
        )

        elapsed = time.time() - t0
        logger.info("Forward forecast completed in {:.2f}s", elapsed)

        return result

    except EnvironmentalDataError as exc:
        logger.error("Forward forecast refused: {} ({})", exc.reason, exc.code)
        raise _http_forcing_unavailable(exc) from exc
    except Exception as e:
        logger.error("Forward forecast failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Forecast computation failed: {str(e)}")


@app.post("/drift/full", response_model=FullPipelineResult)
async def drift_full_pipeline(req: FullPipelineRequest):
    """Complete pipeline: backward + forward + shoreline risk.

    Runs the full SENTINEL drift analysis:
        1. Backward SDE to find origin distribution
        2. Forward forecast from origin ellipse
        3. Shoreline impact risk assessment
    """
    t0 = time.time()
    logger.info(
        "Full pipeline request: spill=({:.4f}, {:.4f}), age={}h, forecast={}h, N={}",
        req.spill_lon,
        req.spill_lat,
        req.spill_age_hours,
        req.forecast_hours,
        req.n_particles,
    )

    try:
        from datetime import datetime as _dt3

        from .engine.ensemble import (
            make_interpolators,
            run_backward_ensemble,
            run_forward_ensemble,
        )
        from .engine.forward_forecast import compute_shoreline_risk
        from .schemas import ShorelineRisk

        # Load data (local NetCDF if present, else forcing factory).
        u_field, v_field, K_field, lons, lats, times, provider, provenance = _build_fields(
            req.spill_lon,
            req.spill_lat,
            req.ocean_data,
            _dt3.now(),
            allow_synthetic=req.allow_synthetic,
        )

        u_interp, v_interp, K_interp = make_interpolators(
            u_field,
            v_field,
            K_field,
            lons,
            lats,
            times,
        )

        # Step 1: Backward
        logger.info("Step 1/3: Backward drift")
        backward_result = run_backward_ensemble(
            req.spill_lon,
            req.spill_lat,
            req.spill_age_hours,
            u_interp,
            v_interp,
            K_interp,
            n_particles=req.n_particles,
            random_seed=req.random_seed,
        )
        # Label provenance + apply XGB correction (same as /drift/backward).
        try:
            _corr_ellipse, _xgb_ok, _xgb_corr = _apply_xgb_correction(
                backward_result.origin_ellipse, provider, req, provenance
            )
            backward_result.origin_ellipse = _corr_ellipse
            backward_result.forcing_source = str(provenance.get("forcing_source", "synthetic_mock"))
            backward_result.forcing_detail = dict(provenance)
            backward_result.xgb_residual_applied = bool(_xgb_ok)
            backward_result.xgb_correction_m = dict(_xgb_corr)
        except Exception as e:
            logger.warning("Full-pipeline XGB/provenance labelling skipped: {}", e)

        # Step 2: Forward
        logger.info("Step 2/3: Forward forecast")
        origin_lons = np.array(backward_result.origin_points_lon)
        origin_lats = np.array(backward_result.origin_points_lat)

        forward_result = run_forward_ensemble(
            origin_lons,
            origin_lats,
            req.forecast_hours,
            u_interp,
            v_interp,
            K_interp,
            n_particles=req.n_particles,
            random_seed=req.random_seed,
        )

        # Step 3: Shoreline risk
        logger.info("Step 3/3: Shoreline risk assessment")
        final_lons = np.array(forward_result.trajectories.all_paths_lon)
        final_lats = np.array(forward_result.trajectories.all_paths_lat)

        risk_data = compute_shoreline_risk(final_lons, final_lats, None, None)

        shoreline = ShorelineRisk(
            risk_index=risk_data["risk_index"],
            nearest_impact_km=risk_data["nearest_impact_km"],
            impact_probability=risk_data["impact_probability"],
            coastline_segments_at_risk=risk_data["coastline_segments_at_risk"],
            trajectory_landfall_points_lon=risk_data["impact_points_lon"],
            trajectory_landfall_points_lat=risk_data["impact_points_lat"],
        )

        result = FullPipelineResult(
            backward=backward_result,
            forward=forward_result,
            shoreline_risk=shoreline,
        )

        elapsed = time.time() - t0
        logger.info(
            "Full pipeline completed in {:.2f}s: origin=({:.4f}, {:.4f}), risk={:.3f}",
            elapsed,
            backward_result.origin_ellipse.center_lon,
            backward_result.origin_ellipse.center_lat,
            shoreline.risk_index,
        )

        return result

    except EnvironmentalDataError as exc:
        logger.error("Full pipeline refused: {} ({})", exc.reason, exc.code)
        raise _http_forcing_unavailable(exc) from exc
    except Exception as e:
        logger.error("Full pipeline failed: {}", str(e))
        raise HTTPException(status_code=500, detail=f"Pipeline computation failed: {str(e)}")


def start_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the drift service."""
    import uvicorn

    logger.info("Starting SENTINEL Drift Service on {}:{}", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
