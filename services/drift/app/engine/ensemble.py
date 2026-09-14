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

from ..data.forcing_selection import apply_confidence_penalty
from ..schemas import (
    DriftResult,
    ForecastConeStep,
    ForwardResult,
    FullPipelineResult,
    OriginTimeInterval,
    ShorelineRisk,
    TrajectoryEnsemble,
    TrajectoryPoint,
    UncertaintyMetrics,
)
from .backward_sde import (
    RegimeInputs,
    compute_confidence_ellipse,
    integrate_backward,
    origin_distribution,
)
from .forward_forecast import (
    compute_forecast_spread,
    compute_shoreline_risk,
    compute_trajectory_quantiles,
    integrate_forward,
)


def base_confidence_for_ensemble(n_particles: int) -> float:
    """Confidence before any forcing penalty, from ensemble size alone.

    An ellipse fitted to 3 particles is not an ellipse, it is an artefact —
    so the score has to say so. Thresholds follow the UI's weak-ensemble
    warning (<16 members).
    """
    if n_particles >= 64:
        return 0.85
    if n_particles >= 16:
        return 0.70
    return 0.50


def build_origin_time_interval(
    detection_time: Any,
    backtrack_hours: float,
    *,
    uncertainty_hours: float | None = None,
) -> OriginTimeInterval:
    """Origin TIME INTERVAL — a spill starts over a window, not at an instant.

    SAR fixes the *acquisition* time exactly; it says nothing about when the
    oil left the ship. The most likely release is `detection - backtrack`, and
    the half-width defaults to 10 % of the backtrack (clamped to 0.5–12 h),
    which is the honest size of the age uncertainty for a Fay-style estimate.
    """
    from datetime import timedelta

    if uncertainty_hours is None:
        uncertainty_hours = min(max(0.10 * float(backtrack_hours), 0.5), 12.0)
    mid = detection_time - timedelta(hours=float(backtrack_hours))
    half = timedelta(hours=float(uncertainty_hours))
    return OriginTimeInterval(
        earliest_utc=(mid - half).isoformat(),
        most_likely_utc=mid.isoformat(),
        latest_utc=(mid + half).isoformat(),
        uncertainty_hours=round(float(uncertainty_hours), 3),
        basis=(
            f"detection_time minus {backtrack_hours:g} h backtrack, widened by "
            f"{uncertainty_hours:g} h (±{10.0:g}% of the backtrack, clamped to "
            "0.5–12 h). SAR gives an acquisition instant, not a release instant."
        ),
    )


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
    regime_inputs: RegimeInputs | None = None,
    k_field: np.ndarray | None = None,
    grid_lons: np.ndarray | None = None,
    grid_lats: np.ndarray | None = None,
    forcing: dict | None = None,
    detection_time: Any = None,
    origin_uncertainty_hours: float | None = None,
) -> DriftResult:
    """Run backward ensemble to find origin.

    Returns DriftResult carrying the origin DISTRIBUTION (ellipse + percentile
    radii), the selected K_ij regime (or None), the WMC bookkeeping, the
    origin TIME INTERVAL and uncertainty metrics.
    """
    logger.info(
        "Running backward ensemble: {} particles from ({:.4f}, {:.4f}), age={}h",
        n_particles,
        spill_lon,
        spill_lat,
        spill_age_hours,
    )

    backward = integrate_backward(
        spill_lon,
        spill_lat,
        spill_age_hours,
        u_interp,
        v_interp,
        K_interp,
        n_particles=n_particles,
        random_seed=random_seed,
        regime_inputs=regime_inputs,
        k_field=k_field,
        grid_lons=grid_lons,
        grid_lats=grid_lats,
    )
    origin_lons, origin_lats = backward.origin_lons, backward.origin_lats

    # 95% confidence ellipse + non-parametric percentile radii.
    ellipse = compute_confidence_ellipse(origin_lons, origin_lats, confidence=0.95)
    distribution = origin_distribution(origin_lons, origin_lats, confidence=0.95)

    # Mean dispersion distance from centroid
    km_per_deg = 111.0 * np.cos(np.radians(np.clip(spill_lat, -89, 89)))
    dispersion_km = float(
        np.sqrt(
            ((origin_lons - origin_lons.mean()) * km_per_deg) ** 2
            + ((origin_lats - origin_lats.mean()) * 111.0) ** 2
        ).mean()
    )

    n_kept = int(np.isfinite(origin_lons).sum())
    confidence, basis = apply_confidence_penalty(
        base_confidence_for_ensemble(n_particles), forcing or {}
    )
    warnings: list[str] = list((forcing or {}).get("warnings") or [])
    if n_particles < 16:
        warnings.append(
            f"Weak ensemble: {n_particles} particles — the ellipse is indicative only."
        )
    if backward.regime is None:
        warnings.append(
            "K_ij regime undetermined: " + str(backward.regime_selection.get("reason"))
        )

    interval = (
        build_origin_time_interval(
            detection_time, spill_age_hours, uncertainty_hours=origin_uncertainty_hours
        )
        if detection_time is not None
        else None
    )

    return DriftResult(
        origin_ellipse=ellipse,
        particle_count=n_particles,
        regime=backward.regime_label,
        origin_points_lon=origin_lons.tolist(),
        origin_points_lat=origin_lats.tolist(),
        mean_dispersion_km=dispersion_km,
        origin_distribution=distribution,
        origin_time_interval=interval,
        uncertainty=UncertaintyMetrics(
            ensemble_size=int(n_particles),
            surviving_particles=n_kept,
            survival_fraction=round(n_kept / max(n_particles, 1), 4),
            origin_p50_radius_km=distribution["percentile_radii_km"]["p50"],
            origin_p95_radius_km=distribution["percentile_radii_km"]["p95"],
            ellipse_area_km2=distribution["covariance_ellipse"]["area_km2"],
            mean_dispersion_km=dispersion_km,
            confidence=confidence,
            confidence_basis=basis,
        ),
        regime_detail=backward.regime_selection,
        wmc=backward.wmc,
        warnings=warnings,
        confidence=confidence,
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
    forcing: dict | None = None,
) -> ForwardResult:
    """Run forward ensemble from origin distribution.

    Returns ForwardResult with trajectory paths, per-quantile envelopes, and
    the p10/p50/p95 spread cone.
    """
    logger.info(
        "Running forward ensemble: {} particles, {}hr forecast",
        n_particles,
        forecast_hours,
    )

    result = integrate_forward(
        origin_lons,
        origin_lats,
        forecast_hours,
        u_interp,
        v_interp,
        K_interp,
        random_seed=random_seed,
    )

    # Compute quantiles
    traj = compute_trajectory_quantiles(result["lons"], result["lats"], result["times"])

    # Build trajectory ensemble
    trajectories = TrajectoryEnsemble(
        mean_path=[TrajectoryPoint(**p) for p in traj["mean_path"]],
        quantile_5_path=[TrajectoryPoint(**p) for p in traj["quantile_5_path"]],
        quantile_95_path=[TrajectoryPoint(**p) for p in traj["quantile_95_path"]],
        all_paths_lon=traj["all_paths_lon"],
        all_paths_lat=traj["all_paths_lat"],
    )

    spread = compute_forecast_spread(result["lons"], result["lats"], result["times"])
    confidence, _basis = apply_confidence_penalty(
        base_confidence_for_ensemble(int(np.asarray(origin_lons).size)), forcing or {}
    )
    warnings = list((forcing or {}).get("warnings") or [])

    return ForwardResult(
        trajectories=trajectories,
        # The forward path does not classify a regime — it inherits whatever
        # K field the caller supplies. None, not a guess.
        regime=None,
        forecast_hours=forecast_hours,
        spread=[ForecastConeStep(**s) for s in spread],
        wmc=dict(result.get("wmc") or {}),
        warnings=warnings,
        confidence=confidence,
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
        spill_lon,
        spill_lat,
        spill_age_hours,
        forecast_hours,
        n_particles,
    )

    # --- Backward ---
    drift = run_backward_ensemble(
        spill_lon,
        spill_lat,
        spill_age_hours,
        u_interp,
        v_interp,
        K_interp,
        n_particles=n_particles,
        random_seed=random_seed,
    )

    # --- Forward ---
    origin_lons = np.array(drift.origin_points_lon)
    origin_lats = np.array(drift.origin_points_lat)

    fwd = run_forward_ensemble(
        origin_lons,
        origin_lats,
        forecast_hours,
        u_interp,
        v_interp,
        K_interp,
        n_particles=n_particles,
        random_seed=random_seed,
    )

    # --- Shoreline risk ---
    final_lons = np.array(fwd.trajectories.all_paths_lon)
    final_lats = np.array(fwd.trajectories.all_paths_lat)

    risk_data = compute_shoreline_risk(
        final_lons,
        final_lats,
        coastline_lons,
        coastline_lats,
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
