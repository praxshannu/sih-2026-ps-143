"""Ensemble runner: spawns N particles, runs backward/forward, computes statistics.

Handles the full lifecycle:
    1. Spawn particles at spill centroid
    2. Backward integration to find origin
    3. Forward integration from origin ellipse
    4. Compute ensemble statistics (mean, 95% CI, shoreline risk)
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ..schemas import (
    ConfidenceEllipse,
    DriftResult,
    ForwardResult,
    FullPipelineResult,
    ShorelineRisk,
    TrajectoryEnsemble,
    TrajectoryPoint,
)
from .backward_sde import compute_confidence_ellipse, integrate_backward
from .forward_forecast import (
    compute_shoreline_risk,
    compute_trajectory_quantiles,
    integrate_forward,
)
from .regime_classifier import OceanRegime


def make_interpolators(
    u_field: np.ndarray,
    v_field: np.ndarray,
    K_field: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
):
    """Create simple nearest-neighbour interpolators for velocity and K fields.

    In production, use scipy.interpolate.RegularGridInterpolator.
    Here we use a fast vectorised lookup.
    """
    # Precompute grid spacings
    lon_grid = np.sort(np.unique(lons))
    lat_grid = np.sort(np.unique(lats))
    time_grid = np.sort(np.unique(times))

    def u_interp(t, x, y):
        ti = np.argmin(np.abs(time_grid - t))
        xi = np.clip(np.searchsorted(lon_grid, x), 0, len(lon_grid) - 1)
        yi = np.clip(np.searchsorted(lat_grid, y), 0, len(lat_grid) - 1)
        return u_field[ti, yi, xi]

    def v_interp(t, x, y):
        ti = np.argmin(np.abs(time_grid - t))
        xi = np.clip(np.searchsorted(lon_grid, x), 0, len(lon_grid) - 1)
        yi = np.clip(np.searchsorted(lat_grid, y), 0, len(lat_grid) - 1)
        return v_field[ti, yi, xi]

    def K_interp(t, x, y):
        n = len(x)
        ti = np.argmin(np.abs(time_grid - t))
        xi = np.clip(np.searchsorted(lon_grid, x), 0, len(lon_grid) - 1)
        yi = np.clip(np.searchsorted(lat_grid, y), 0, len(lat_grid) - 1)
        return K_field[ti, yi, xi]

    return u_interp, v_interp, K_interp


def run_backward_ensemble(
    spill_lon: float,
    spill_lat: float,
    spill_age_hours: float,
    u_interp,
    v_interp,
    K_interp,
    *,
    n_particles: int = 1000,
    random_seed: int | None = None,
) -> DriftResult:
    """Run backward ensemble to find origin.

    Returns DriftResult with origin ellipse and particle distribution.
    """
    logger.info(
        "Running backward ensemble: {} particles from ({:.4f}, {:.4f}), age={}h",
        n_particles, spill_lon, spill_lat, spill_age_hours,
    )

    origin_lons, origin_lats, regime = integrate_backward(
        spill_lon, spill_lat, spill_age_hours,
        u_interp, v_interp, K_interp,
        n_particles=n_particles,
        random_seed=random_seed,
    )

    # Compute 95% confidence ellipse
    ellipse = compute_confidence_ellipse(origin_lons, origin_lats, confidence=0.95)

    # Mean dispersion distance from centroid
    km_per_deg = 111.0 * np.cos(np.radians(np.clip(spill_lat, -89, 89)))
    dispersion_km = float(
        np.sqrt(
            ((origin_lons - origin_lons.mean()) * km_per_deg) ** 2
            + ((origin_lats - origin_lats.mean()) * 111.0) ** 2
        ).mean()
    )

    return DriftResult(
        origin_ellipse=ellipse,
        particle_count=n_particles,
        regime=regime.value,
        origin_points_lon=origin_lons.tolist(),
        origin_points_lat=origin_lats.tolist(),
        mean_dispersion_km=dispersion_km,
    )


def run_forward_ensemble(
    origin_lons: np.ndarray,
    origin_lats: np.ndarray,
    forecast_hours: float,
    u_interp,
    v_interp,
    K_interp,
    *,
    n_particles: int = 1000,
    random_seed: int | None = None,
) -> ForwardResult:
    """Run forward ensemble from origin distribution.

    Returns ForwardResult with trajectory paths and quantiles.
    """
    logger.info(
        "Running forward ensemble: {} particles, {}hr forecast",
        n_particles, forecast_hours,
    )

    result = integrate_forward(
        origin_lons, origin_lats, forecast_hours,
        u_interp, v_interp, K_interp,
        random_seed=random_seed,
    )

    # Compute quantiles
    traj = compute_trajectory_quantiles(
        result["lons"], result["lats"], result["times"]
    )

    # Build trajectory ensemble
    trajectories = TrajectoryEnsemble(
        mean_path=[TrajectoryPoint(**p) for p in traj["mean_path"]],
        quantile_5_path=[TrajectoryPoint(**p) for p in traj["quantile_5_path"]],
        quantile_95_path=[TrajectoryPoint(**p) for p in traj["quantile_95_path"]],
        all_paths_lon=traj["all_paths_lon"],
        all_paths_lat=traj["all_paths_lat"],
    )

    return ForwardResult(
        trajectories=trajectories,
        regime=OceanRegime.MARKOV1.value,
        forecast_hours=forecast_hours,
    )


def run_full_pipeline(
    spill_lon: float,
    spill_lat: float,
    spill_age_hours: float,
    forecast_hours: float,
    u_interp,
    v_interp,
    K_interp,
    *,
    n_particles: int = 1000,
    random_seed: int | None = None,
    coastline_lons: np.ndarray | None = None,
    coastline_lats: np.ndarray | None = None,
) -> FullPipelineResult:
    """Run complete pipeline: backward + forward + shoreline risk.

    Parameters
    ----------
    spill_lon, spill_lat : detected spill centroid
    spill_age_hours : hours since spill began
    forecast_hours : forward forecast duration
    u_interp, v_interp, K_interp : field interpolators
    n_particles : ensemble size
    random_seed : reproducibility seed
    coastline_lons, coastline_lats : optional coastline geometry

    Returns
    -------
    FullPipelineResult with backward, forward, and shoreline risk
    """
    logger.info(
        "Full pipeline: spill=({:.4f}, {:.4f}), age={}h, forecast={}h, N={}",
        spill_lon, spill_lat, spill_age_hours, forecast_hours, n_particles,
    )

    # --- Backward ---
    drift = run_backward_ensemble(
        spill_lon, spill_lat, spill_age_hours,
        u_interp, v_interp, K_interp,
        n_particles=n_particles,
        random_seed=random_seed,
    )

    # --- Forward ---
    origin_lons = np.array(drift.origin_points_lon)
    origin_lats = np.array(drift.origin_points_lat)

    fwd = run_forward_ensemble(
        origin_lons, origin_lats, forecast_hours,
        u_interp, v_interp, K_interp,
        n_particles=n_particles,
        random_seed=random_seed,
    )

    # --- Shoreline risk ---
    final_lons = np.array(fwd.trajectories.all_paths_lon)
    final_lats = np.array(fwd.trajectories.all_paths_lat)

    risk_data = compute_shoreline_risk(
        final_lons, final_lats,
        coastline_lons, coastline_lats,
    )

    shoreline = ShorelineRisk(
        risk_index=risk_data["risk_index"],
        nearest_impact_km=risk_data["nearest_impact_km"],
        impact_probability=risk_data["impact_probability"],
        coastline_segments_at_risk=risk_data["coastline_segments_at_risk"],
        trajectory_landfall_points_lon=risk_data["impact_points_lon"],
        trajectory_landfall_points_lat=risk_data["impact_points_lat"],
    )

    logger.info(
        "Full pipeline complete: origin=({:.4f}, {:.4f}), shoreline_risk={:.3f}",
        drift.origin_ellipse.center_lon,
        drift.origin_ellipse.center_lat,
        shoreline.risk_index,
    )

    return FullPipelineResult(
        backward=drift,
        forward=fwd,
        shoreline_risk=shoreline,
    )
