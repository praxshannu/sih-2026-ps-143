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

from .schemas import (
    BackwardRequest,
    ConfidenceEllipse,
    DriftResult,
    ForwardRequest,
    ForwardResult,
    FullPipelineRequest,
    FullPipelineResult,
    HealthResponse,
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


def _build_fields(spill_lon: float, spill_lat: float, req_ocean, now):
    """Build (u_field, v_field, K_field, lons, lats, times, provider, provenance).

    Tries local NetCDF loaders first; on any failure falls back to the
    forcing factory (live CMEMS/ERA5 if credentialed, else GFS + labelled
    mock) sampled onto a small synthetic grid so the SDE engine always runs
    and the response always labels its source.
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
            provenance = {
                "forcing_source": "local_files",
                "current_origin": "local_cmems_file",
                "wind_origin": "local_era5_file" if era5_base else "not_staged",
            }
            return u_field, v_field, K_field, lons, lats, times, provider, provenance
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
        logger.warning("Provider current sampling failed, analytic fallback: {}", e)
        u2 = np.full((9, 9), 0.15)
        v2 = np.full((9, 9), 0.05)
        provenance = dict(provenance, current_origin="analytic_fallback")
    u_field = np.stack([u2, u2])
    v_field = np.stack([v2, v2])
    K_field = np.zeros((2, 9, 9, 2, 2))
    K_field[..., 0, 0] = 100.0
    K_field[..., 1, 1] = 100.0
    # Numeric seconds grid: the SDE engine queries interpolators with
    # float seconds-remaining, so datetime64 grids would break argmin.
    times = np.array([0.0, 3600.0])
    return u_field, v_field, K_field, lon_grid, lat_grid, times, provider, provenance


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
            req.spill_lon, req.spill_lat, req.ocean_data, _dt.now()
        )

        from .engine.ensemble import make_interpolators

        u_interp, v_interp, K_interp = make_interpolators(
            u_field,
            v_field,
            K_field,
            lons,
            lats,
            times,
        )

        # Run backward integration
        origin_lons, origin_lats, regime = integrate_backward(
            req.spill_lon,
            req.spill_lat,
            req.spill_age_hours,
            u_interp,
            v_interp,
            K_interp,
            n_particles=req.n_particles,
            random_seed=req.random_seed,
        )

        # Compute confidence ellipse
        ellipse = compute_confidence_ellipse(origin_lons, origin_lats, confidence=0.95)

        # XGBoost physics residual post-hoc correction (never skips WMC).
        ellipse, xgb_applied, xgb_corr = _apply_xgb_correction(ellipse, provider, req, provenance)

        # Mean dispersion
        km_per_deg = 111.0 * np.cos(np.radians(np.clip(req.spill_lat, -89, 89)))
        dispersion_km = float(
            np.sqrt(
                ((origin_lons - origin_lons.mean()) * km_per_deg) ** 2
                + ((origin_lats - origin_lats.mean()) * 111.0) ** 2
            ).mean()
        )

        result = DriftResult(
            origin_ellipse=ellipse,
            particle_count=req.n_particles,
            regime=regime.value,
            origin_points_lon=origin_lons.tolist(),
            origin_points_lat=origin_lats.tolist(),
            mean_dispersion_km=dispersion_km,
            forcing_source=str(provenance.get("forcing_source", "synthetic_mock")),
            forcing_detail=dict(provenance),
            xgb_residual_applied=bool(xgb_applied),
            xgb_correction_m=dict(xgb_corr),
        )

        elapsed = time.time() - t0
        logger.info("Backward drift completed in {:.2f}s", elapsed)

        return result

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
            req.origin_lon, req.origin_lat, req.ocean_data, _dt2.now()
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
            req.spill_lon, req.spill_lat, req.ocean_data, _dt3.now()
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
