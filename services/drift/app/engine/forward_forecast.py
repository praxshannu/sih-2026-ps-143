"""Forward Lagrangian forecast: spread/cone (p10/p50/p95) over time.

This module is the FORWARD half of the drift engine and shares no output
plumbing with `backward_sde` (that one answers "where did it come from",
this one answers "where is it going"). They are exercised by separate tests
so a regression in one can never be masked by the other.

Both call `wmc_correction.wmc_divergence` for the ∇·K term, so the two
directions can never disagree about what the Well-Mixed Criterion did.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from .wmc_correction import wmc_divergence


def rk4_step_forward(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    K: np.ndarray,
    div_K: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single 4th-order Runge-Kutta step for forward integration.

    Forward SDE drift: a(x) = u - ∇·K

    Parameters
    ----------
    x, y : (N,) current positions
    u, v : (N,) velocity at current position
    K : (N, 2, 2) diffusion tensor
    div_K : (N, 2) divergence of K
    dt : time step (positive)

    Returns
    -------
    x_new, y_new : (N,) positions after one RK4 step
    """

    def drift(px, py):
        return u - div_K[:, 0], v - div_K[:, 1]

    dx1, dy1 = drift(x, y)
    dx2, dy2 = drift(x + 0.5 * dt * dx1, y + 0.5 * dt * dy1)
    dx3, dy3 = drift(x + 0.5 * dt * dx2, y + 0.5 * dt * dy2)
    dx4, dy4 = drift(x + dt * dx3, y + dt * dy3)

    x_new = x + (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
    y_new = y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)

    return x_new, y_new


def integrate_forward(
    origin_lons: np.ndarray,
    origin_lats: np.ndarray,
    forecast_hours: float,
    u_interp,
    v_interp,
    K_interp,
    *,
    dt_seconds: float = 1800.0,
    random_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Run forward Lagrangian simulation from origin distribution.

    Parameters
    ----------
    origin_lons, origin_lats : (N,) initial particle positions
    forecast_hours : forecast duration in hours
    u_interp, v_interp : callable(t, lon, lat) -> (u, v)
    K_interp : callable(t, lon, lat) -> (N, 2, 2) diffusion tensor
    dt_seconds : time step
    random_seed : reproducibility seed

    Returns
    -------
    dict with:
        'lons': (N, n_timesteps+1) longitude history
        'lats': (N, n_timesteps+1) latitude history
        'times': (n_timesteps+1,) time in hours from origin
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    total_seconds = forecast_hours * 3600.0
    n_steps = int(np.ceil(total_seconds / dt_seconds))
    dt = total_seconds / n_steps

    n_particles = len(origin_lons)
    wmc_steps: list[dict[str, Any]] = []

    logger.info(
        "Forward integration: {} particles, {} steps of {:.0f}s, {}hr forecast",
        n_particles,
        n_steps,
        dt,
        forecast_hours,
    )

    # Storage
    lons = np.zeros((n_particles, n_steps + 1), dtype=np.float64)
    lats = np.zeros((n_particles, n_steps + 1), dtype=np.float64)
    times = np.linspace(0, forecast_hours, n_steps + 1)

    # Initialize
    x = origin_lons.copy()
    y = origin_lats.copy()
    lons[:, 0] = x
    lats[:, 0] = y

    for step in range(n_steps):
        t_hours = step * dt / 3600.0
        t_sec = step * dt  # interpolators use the seconds grid (see _build_fields)

        # Get velocity and K
        u = u_interp(t_sec, x, y)
        v = v_interp(t_sec, x, y)
        K = K_interp(t_sec, x, y)

        # WMC ∇·K — same ladder as the backward integrator (grid -> constant-K
        # no-op -> scattered fit -> explicitly unavailable). The old version
        # differentiated along the particle *index*, which is not a direction.
        div_K, wmc_step = wmc_divergence(K, x, y)
        wmc_steps.append(wmc_step)

        # Units: u/v are m/s but x/y are degrees — convert to deg/s first.
        cos_lat = np.cos(np.radians(np.clip(y, -89.0, 89.0)))
        cos_lat = np.maximum(cos_lat, 0.1)
        u_deg = (u - div_K[:, 0]) / (111320.0 * cos_lat)
        v_deg = (v - div_K[:, 1]) / 110540.0

        # RK4 step (divergence already folded into u_deg/v_deg)
        x, y = rk4_step_forward(x, y, u_deg, v_deg, K, div_K, dt)

        # Diffusion noise
        from .backward_sde import wiener_increment

        dW_x, dW_y = wiener_increment(K, dt, rng)
        x += dW_x / 111000.0 / np.cos(np.radians(np.clip(y, -89, 89)))
        y += dW_y / 111000.0

        lons[:, step + 1] = x
        lats[:, step + 1] = y

        if step % max(n_steps // 10, 1) == 0:
            logger.debug(
                "Step {}/{} (t={:.1f}h): mean=({:.4f}, {:.4f})",
                step,
                n_steps,
                t_hours + dt / 3600.0,
                x.mean(),
                y.mean(),
            )

    logger.info(
        "Forward complete: {} trajectories, final mean=({:.4f}, {:.4f})",
        n_particles,
        lons[:, -1].mean(),
        lats[:, -1].mean(),
    )

    return {
        "lons": lons,
        "lats": lats,
        "times": times,
        "wmc": wmc_steps[-1] if wmc_steps else {},
    }


def compute_forecast_spread(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
    *,
    center_lon: float | None = None,
    center_lat: float | None = None,
) -> list[dict[str, Any]]:
    """Per-timestep spread cone: p10 / p50 / p95 radius around the centroid.

    A trajectory bundle is not a forecast unless the spread is quantified, so
    each step reports three radial percentiles (km) plus how many particles
    were still in play. Steps where every particle has been deactivated keep
    the previous finite centre and report zero radii rather than being
    dropped — a collapsed cone must not masquerade as a short run.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    n_particles, n_steps = lons.shape

    km_per_deg_lat = 111.0
    last_cx = float(center_lon if center_lon is not None else lons[:, 0].mean())
    last_cy = float(center_lat if center_lat is not None else lats[:, 0].mean())

    steps: list[dict[str, Any]] = []
    for i in range(n_steps):
        L, A = lons[:, i], lats[:, i]
        ok = np.isfinite(L) & np.isfinite(A)
        n_active = int(ok.sum())
        if n_active:
            cx = float(np.nanmean(L[ok])) if center_lon is None else float(center_lon)
            cy = float(np.nanmean(A[ok])) if center_lat is None else float(center_lat)
            last_cx, last_cy = cx, cy
            km_per_deg_lon = km_per_deg_lat * float(
                np.cos(np.radians(np.clip(cy, -89.0, 89.0)))
            )
            dx = (L[ok] - cx) * km_per_deg_lon
            dy = (A[ok] - cy) * km_per_deg_lat
            radii = np.sqrt(dx * dx + dy * dy)
            p10, p50, p95 = (
                float(np.percentile(radii, 10)),
                float(np.percentile(radii, 50)),
                float(np.percentile(radii, 95)),
            )
        else:
            cx, cy = last_cx, last_cy
            p10 = p50 = p95 = 0.0

        steps.append(
            {
                "hours_ahead": round(float(times[i]), 3),
                "center_lon": round(cx, 6),
                "center_lat": round(cy, 6),
                "p10_radius_km": round(p10, 3),
                "p50_radius_km": round(p50, 3),
                "p95_radius_km": round(p95, 3),
                "n_active": n_active,
                "n_particles": int(n_particles),
            }
        )
    return steps


def compute_trajectory_quantiles(
    lons: np.ndarray,
    lats: np.ndarray,
    times: np.ndarray,
) -> dict:
    """Compute mean path and probability quantiles from ensemble.

    Parameters
    ----------
    lons, lats : (N, T) particle trajectories
    times : (T,) time array in hours

    Returns
    -------
    dict with mean_path, quantile_5_path, quantile_95_path as lists of (lon, lat)
    """
    n_particles, n_steps = lons.shape

    mean_lon = np.mean(lons, axis=0)
    mean_lat = np.mean(lats, axis=0)
    q5_lon = np.percentile(lons, 5, axis=0)
    q5_lat = np.percentile(lats, 5, axis=0)
    q95_lon = np.percentile(lons, 95, axis=0)
    q95_lat = np.percentile(lats, 95, axis=0)

    result = {
        "mean_path": [
            {"lon": float(mean_lon[i]), "lat": float(mean_lat[i]), "time_hours": float(times[i])}
            for i in range(n_steps)
        ],
        "quantile_5_path": [
            {"lon": float(q5_lon[i]), "lat": float(q5_lat[i]), "time_hours": float(times[i])}
            for i in range(n_steps)
        ],
        "quantile_95_path": [
            {"lon": float(q95_lon[i]), "lat": float(q95_lat[i]), "time_hours": float(times[i])}
            for i in range(n_steps)
        ],
        "all_paths_lon": lons.tolist(),
        "all_paths_lat": lats.tolist(),
    }

    return result


def compute_shoreline_risk(
    lons: np.ndarray,
    lats: np.ndarray,
    coastline_lons: np.ndarray | None,
    coastline_lats: np.ndarray | None,
    *,
    impact_distance_km: float = 10.0,
) -> dict:
    """Assess shoreline impact risk from trajectory ensemble.

    Parameters
    ----------
    lons, lats : (N, T) final positions of particles
    coastline_lons, coastline_lats : coastline geometry (None = no coastline data)
    impact_distance_km : threshold for "impact" detection

    Returns
    -------
    dict with risk_index, nearest_impact_km, impact_probability,
          coastline_segments_at_risk, impact points
    """
    final_lons = lons[:, -1]
    final_lats = lats[:, -1]
    n = len(final_lons)

    if coastline_lons is None or coastline_lats is None:
        logger.warning("No coastline data provided, estimating risk from open-ocean dispersion")
        # Estimate risk from particle spread
        spread_km = np.sqrt(
            np.var(final_lons * 111.0 * np.cos(np.radians(np.mean(final_lats)))) * 111.0**2
            + np.var(final_lats * 111.0) ** 2
        )
        risk = min(spread_km / 100.0, 1.0)
        return {
            "risk_index": float(risk),
            "nearest_impact_km": float(spread_km),
            "impact_probability": float(min(spread_km / 50.0, 1.0)),
            "coastline_segments_at_risk": 0,
            "impact_points_lon": [],
            "impact_points_lat": [],
        }

    # Compute distance from each particle to nearest coastline point
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(np.clip(np.mean(final_lats), -89, 89)))

    min_distances = np.full(n, np.inf)
    for i in range(n):
        dx = (final_lons[i] - coastline_lons) * km_per_deg_lon
        dy = (final_lats[i] - coastline_lats) * km_per_deg_lat
        dist = np.sqrt(dx**2 + dy**2)
        min_distances[i] = dist.min()

    # Risk metrics
    impacts = min_distances < impact_distance_km
    risk_index = float(impacts.sum() / n)
    nearest_impact = float(min_distances.min())

    # Find which coastline segments are at risk
    segments_at_risk = set()
    impact_lons = []
    impact_lats = []
    for i in range(n):
        if min_distances[i] < impact_distance_km:
            nearest_idx = np.argmin(
                np.sqrt(
                    ((final_lons[i] - coastline_lons) * km_per_deg_lon) ** 2
                    + ((final_lats[i] - coastline_lats) * km_per_deg_lat) ** 2
                )
            )
            segments_at_risk.add(int(nearest_idx))
            impact_lons.append(float(final_lons[i]))
            impact_lats.append(float(final_lats[i]))

    logger.info(
        "Shoreline risk: index={:.3f}, nearest={:.1f}km, segments_at_risk={}",
        risk_index,
        nearest_impact,
        len(segments_at_risk),
    )

    return {
        "risk_index": risk_index,
        "nearest_impact_km": nearest_impact,
        "impact_probability": float(np.exp(-nearest_impact / 20.0)),
        "coastline_segments_at_risk": len(segments_at_risk),
        "impact_points_lon": impact_lons,
        "impact_points_lat": impact_lats,
    }
