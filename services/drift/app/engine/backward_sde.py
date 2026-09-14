"""Backward SDE integrator: 4th-order Runge-Kutta for origin finding.

    dX = (u - ∇·K)dt - sqrt(2K) dW

Integrates backward from spill centroid to find probable origin.
Returns the origin DISTRIBUTION (covariance ellipse + percentile radii), the
selected K_ij regime, and the WMC ∇·K bookkeeping.

Order of operations (AGENTS.md hard rule)
-----------------------------------------
1. **Select the K_ij regime FIRST** — `select_regime()` runs before any
   integration and before the WMC correction. If the classifier cannot
   decide, the run reports `regime=None` with the reason; it never falls back
   to MARKOV1 in silence.
2. **Then apply the WMC ∇·K correction** to the drift. It is never skipped.
   Where K is spatially constant, ∇·K ≡ 0 and the correction is a genuine
   no-op — VERIFICATION §9.5 requires stating that precondition rather than
   printing a number derived from particle ordering.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from ..schemas import ConfidenceEllipse
from .kij_tensor import k_eigenvalues
from .regime_classifier import (
    REGIME_LABELS,
    OceanRegime,
    RegimeLabel,
    RegimeSelection,
    regime_k_params,
    select_regime,
)
from .wmc_correction import (
    WMC_APPLIED,
    WMC_NO_OP_CONSTANT_K,
    WMC_UNRESOLVED,
    wmc_divergence,
)


@dataclass
class RegimeInputs:
    """Whatever the caller knows about the water column at the seed points.

    All fields optional; anything left None makes the regime undetermined,
    which is reported instead of guessed.
    """

    depth_m: np.ndarray | None = None
    distance_to_coast_km: np.ndarray | None = None
    current_speed: np.ndarray | None = None
    lat: np.ndarray | None = None


@dataclass
class BackwardResult:
    """Output of one backward ensemble — a distribution, never a single line."""

    origin_lons: np.ndarray
    origin_lats: np.ndarray
    regime: OceanRegime | None
    regime_selection: dict[str, Any]
    wmc: dict[str, Any] = field(default_factory=dict)
    k_diagnostics: dict[str, Any] = field(default_factory=dict)
    n_particles: int = 0
    n_steps: int = 0

    @property
    def regime_label(self) -> RegimeLabel | None:
        return REGIME_LABELS[self.regime] if self.regime is not None else None


def rk4_step_backward(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    K: np.ndarray,
    div_K: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single 4th-order Runge-Kutta step for backward integration.

    The backward SDE drift term is:
        a(x) = -(u - ∇·K)   [negative because we go backward in time]

    Parameters
    ----------
    x, y : (N,) current positions
    u, v : (N,) velocity at current position
    K : (N, 2, 2) diffusion tensor
    div_K : (N, 2) divergence of K
    dt : time step (positive, integration goes backward)

    Returns
    -------
    x_new, y_new : (N,) positions after one RK4 step
    """

    def drift(px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Backward drift: a = -(u - div_K)"""
        return -(u - div_K[:, 0]), -(v - div_K[:, 1])

    # k1
    dx1, dy1 = drift(x, y)

    # k2
    dx2, dy2 = drift(x + 0.5 * dt * dx1, y + 0.5 * dt * dy1)

    # k3
    dx3, dy3 = drift(x + 0.5 * dt * dx2, y + 0.5 * dt * dy2)

    # k4
    dx4, dy4 = drift(x + dt * dx3, y + dt * dy3)

    x_new = x + (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
    y_new = y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)

    return x_new, y_new


def wiener_increment(
    K: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated Wiener process increments sqrt(2K*dt) * N(0,1).

    For a 2x2 symmetric tensor K, the noise is correlated:
        dW_x = sqrt(2 * lambda_1) * N1
        dW_y = sqrt(2 * lambda_2) * N2
    rotated by the eigenvectors of K.

    Parameters
    ----------
    K : (N, 2, 2) diffusion tensor
    dt : time step (positive)
    rng : numpy random generator

    Returns
    -------
    dW_x, dW_y : (N,) correlated noise increments
    """
    n = K.shape[0]

    # White noise
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)

    # Eigendecomposition of each 2x2 tensor
    a = K[:, 0, 0]
    b = K[:, 0, 1]
    d = K[:, 1, 1]

    trace = a + d
    det = a * d - b * b
    discriminant = np.maximum(trace**2 - 4.0 * det, 0.0)
    lam1 = 0.5 * (trace + np.sqrt(discriminant))
    lam2 = 0.5 * (trace - np.sqrt(discriminant))

    # Ensure non-negative eigenvalues
    lam1 = np.maximum(lam1, 0.0)
    lam2 = np.maximum(lam2, 0.0)

    # Eigenvector angle (for 2x2 symmetric)
    theta = 0.5 * np.arctan2(2.0 * b, a - d + 1e-30)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Scale noise by eigenvalues
    s1 = np.sqrt(2.0 * lam1 * dt)
    s2 = np.sqrt(2.0 * lam2 * dt)

    # Rotate back to physical coordinates
    dW_x = cos_t * s1 * z1 - sin_t * s2 * z2
    dW_y = sin_t * s1 * z1 + cos_t * s2 * z2

    return dW_x, dW_y


def integrate_backward(
    spill_lon: float,
    spill_lat: float,
    spill_age_hours: float,
    u_interp: Callable[..., np.ndarray],
    v_interp: Callable[..., np.ndarray],
    K_interp: Callable[..., np.ndarray],
    *,
    n_particles: int = 1000,
    dt_seconds: float = 3600.0,
    random_seed: int | None = None,
    regime_inputs: RegimeInputs | None = None,
    k_field: np.ndarray | None = None,
    grid_lons: np.ndarray | None = None,
    grid_lats: np.ndarray | None = None,
) -> BackwardResult:
    """Run backward SDE integration to find the probable origin.

    Parameters
    ----------
    spill_lon, spill_lat : centroid of detected spill
    spill_age_hours : hours since spill began
    u_interp, v_interp : callable(t, lon, lat) -> (u, v) velocity
    K_interp : callable(t, lon, lat) -> (N, 2, 2) diffusion tensor
    n_particles : ensemble size
    dt_seconds : time step
    random_seed : reproducibility seed
    regime_inputs : bathymetry / coast distance / current speed / latitude
        samples used to select the K_ij regime BEFORE the run
    k_field, grid_lons, grid_lats : optional gridded K field enabling an exact
        ∇·K instead of the scattered-fit fallback

    Returns
    -------
    BackwardResult with the particle cloud, the selected regime (or None) and
    the WMC bookkeeping.
    """
    rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()

    total_seconds = spill_age_hours * 3600.0
    n_steps = max(int(np.ceil(total_seconds / dt_seconds)), 1)
    dt = total_seconds / n_steps

    # ── STEP 1: select the K_ij regime, before anything else ──────────────
    if regime_inputs is None:
        selection: RegimeSelection = select_regime()
    else:
        selection = select_regime(
            depth_m=regime_inputs.depth_m,
            distance_to_coast_km=regime_inputs.distance_to_coast_km,
            current_speed=regime_inputs.current_speed,
            lat=regime_inputs.lat,
            n=n_particles,
        )
    params = regime_k_params(selection.regime) if selection.regime else None
    logger.info(
        "Backward SDE: regime={} ({})",
        selection.regime.value if selection.regime else "UNDETERMINED",
        selection.reason,
    )
    if selection.regime is None:
        logger.warning(
            "K_ij regime undetermined — recorded as None, not defaulted: {}",
            selection.reason,
        )

    logger.info(
        "Backward integration: {} steps of {:.1f}s from ({:.4f}, {:.4f})",
        n_steps,
        dt,
        spill_lon,
        spill_lat,
    )

    # Initialize particles at spill centroid with small Gaussian spread
    spread_deg = 0.01  # ~1 km initial spread
    x = spill_lon + rng.normal(0, spread_deg, n_particles)
    y = spill_lat + rng.normal(0, spread_deg, n_particles)

    wmc_states: list[dict[str, Any]] = []
    wmc_max: float | None = None
    k_principal: list[float] = []

    for step in range(n_steps):
        t_remaining = (n_steps - step) * dt

        # Get velocity and K at current positions
        u = np.asarray(u_interp(t_remaining, x, y), dtype=np.float64)
        v = np.asarray(v_interp(t_remaining, x, y), dtype=np.float64)
        K = np.asarray(K_interp(t_remaining, x, y), dtype=np.float64)

        # ── STEP 2: WMC ∇·K correction (never skipped) ───────────────────
        div_K, wmc_step = wmc_divergence(
            K, x, y, k_field=k_field, grid_lons=grid_lons, grid_lats=grid_lats
        )
        wmc_states.append(wmc_step)
        if wmc_step.get("divergence_max") is not None:
            wmc_max = (
                max(wmc_max or 0.0, float(wmc_step["divergence_max"]))
                if wmc_max is not None
                else float(wmc_step["divergence_max"])
            )
        if step == 0:
            eig_max, eig_min = k_eigenvalues(K)
            k_principal = [float(np.mean(eig_max)), float(np.mean(eig_min))]

        # Units: u/v/div_K are m/s but x/y are degrees. Convert the
        # combined advective velocity to deg/s BEFORE the RK4 step —
        # otherwise each step jumps hundreds of degrees and the origin
        # diverges to garbage coordinates.
        cos_lat = np.cos(np.radians(np.clip(y, -89.0, 89.0)))
        cos_lat = np.maximum(cos_lat, 0.1)
        u_deg = (u - div_K[:, 0]) / (111320.0 * cos_lat)
        v_deg = (v - div_K[:, 1]) / 110540.0

        # RK4 step (divergence already folded into u_deg/v_deg)
        x, y = rk4_step_backward(x, y, u_deg, v_deg, K, div_K, dt)

        # Diffusion (stochastic term)
        dW_x, dW_y = wiener_increment(K, dt, rng)
        x += dW_x / 111000.0 / np.cos(np.radians(np.clip(y, -89, 89)))  # convert m to deg
        y += dW_y / 111000.0  # convert m to deg

        if step % max(n_steps // 10, 1) == 0:
            logger.debug(
                "Step {}/{}: mean pos = ({:.4f}, {:.4f}), std = ({:.4f}, {:.4f})",
                step,
                n_steps,
                x.mean(),
                y.mean(),
                x.std(),
                y.std(),
            )

    statuses = {s["status"] for s in wmc_states}
    if WMC_UNRESOLVED in statuses and len(statuses) == 1:
        wmc_summary: dict[str, Any] = {
            "applied": False,
            "status": WMC_UNRESOLVED,
            "divergence_max": None,
            "reason": wmc_states[-1].get("reason"),
        }
    elif statuses <= {WMC_NO_OP_CONSTANT_K}:
        wmc_summary = {
            "applied": True,
            "status": WMC_NO_OP_CONSTANT_K,
            "divergence_max": 0.0,
            "precondition": wmc_states[-1].get("precondition"),
            "method": "analytic_zero_constant_K",
        }
    else:
        wmc_summary = {
            "applied": True,
            "status": WMC_APPLIED,
            "divergence_max": wmc_max,
            "method": wmc_states[-1].get("method"),
        }

    logger.info(
        "Backward complete: mean origin = ({:.4f}, {:.4f}), spread = ({:.4f}, {:.4f})",
        x.mean(),
        y.mean(),
        x.std(),
        y.std(),
    )

    return BackwardResult(
        origin_lons=x,
        origin_lats=y,
        regime=selection.regime,
        regime_selection=selection.to_dict(),
        wmc=wmc_summary,
        k_diagnostics={
            "regime_params": params.to_dict() if params else None,
            "principal_diffusivity_max_m2s": k_principal[0] if k_principal else None,
            "principal_diffusivity_min_m2s": k_principal[1] if k_principal else None,
        },
        n_particles=int(n_particles),
        n_steps=int(n_steps),
    )


def origin_distribution(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Origin DISTRIBUTION — covariance ellipse plus percentile radii.

    A single ellipse is not an uncertainty statement; the percentile radii
    (p50/p90/p95 of distance-from-centroid in km) let the UI say
    "half the ensemble is within X km" without assuming Gaussianity.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    finite = np.isfinite(lons) & np.isfinite(lats)
    lons, lats = lons[finite], lats[finite]
    ellipse = compute_confidence_ellipse(lons, lats, confidence=confidence)

    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * float(np.cos(np.radians(np.clip(lats.mean(), -89, 89))))
    dx = (lons - lons.mean()) * km_per_deg_lon
    dy = (lats - lats.mean()) * km_per_deg_lat
    radii = np.sqrt(dx * dx + dy * dy)

    return {
        "center_lon": ellipse.center_lon,
        "center_lat": ellipse.center_lat,
        "covariance_ellipse": {
            "confidence": confidence,
            "semi_major_km": ellipse.semi_major_km,
            "semi_minor_km": ellipse.semi_minor_km,
            "orientation_deg": ellipse.orientation_deg,
            "area_km2": float(
                np.pi * max(ellipse.semi_major_km, 0.0) * max(ellipse.semi_minor_km, 0.0)
            ),
        },
        "percentile_radii_km": {
            "p50": float(np.percentile(radii, 50)) if radii.size else 0.0,
            "p75": float(np.percentile(radii, 75)) if radii.size else 0.0,
            "p90": float(np.percentile(radii, 90)) if radii.size else 0.0,
            "p95": float(np.percentile(radii, 95)) if radii.size else 0.0,
        },
        "n_particles": int(lons.size),
        "mean_dispersion_km": float(np.mean(radii)) if radii.size else 0.0,
    }


def compute_confidence_ellipse(
    lons: np.ndarray,
    lats: np.ndarray,
    confidence: float = 0.95,
) -> ConfidenceEllipse:
    """Compute confidence ellipse from particle distribution.

    Uses eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    lons, lats : (N,) particle positions
    confidence : confidence level (default 0.95)

    Returns
    -------
    ConfidenceEllipse with centre, semi-axes, orientation and percentile radii
    """
    # Convert to km for meaningful ellipse dimensions
    mean_lon = float(np.mean(lons))
    mean_lat = float(np.mean(lats))

    # Approximate conversion to km
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(np.clip(mean_lat, -89, 89)))

    x_km = (lons - mean_lon) * km_per_deg_lon
    y_km = (lats - mean_lat) * km_per_deg_lat

    # Covariance matrix
    cov = np.cov(np.stack([x_km, y_km], axis=0))

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Chi-squared scaling for 2D Gaussian
    chi2_scale = np.sqrt(-2.0 * np.log(1.0 - confidence))

    semi_major = chi2_scale * np.sqrt(max(eigvals[0], 0.0))
    semi_minor = chi2_scale * np.sqrt(max(eigvals[1], 0.0))

    # Orientation angle (from x-axis / East)
    orientation = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    radii = np.sqrt(x_km**2 + y_km**2)

    logger.info(
        "Confidence ellipse ({}%): centre=({:.4f}, {:.4f}), a={:.2f}km, b={:.2f}km, θ={:.1f}°",
        int(confidence * 100),
        mean_lon,
        mean_lat,
        semi_major,
        semi_minor,
        orientation,
    )

    return ConfidenceEllipse(
        center_lon=mean_lon,
        center_lat=mean_lat,
        semi_major_km=float(semi_major),
        semi_minor_km=float(semi_minor),
        orientation_deg=float(orientation),
        confidence=float(confidence),
        p50_radius_km=float(np.percentile(radii, 50)) if radii.size else 0.0,
        p95_radius_km=float(np.percentile(radii, 95)) if radii.size else 0.0,
        n_particles=int(np.asarray(lons).size),
    )
