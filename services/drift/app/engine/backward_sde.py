"""Backward SDE integrator: 4th-order Runge-Kutta for origin finding.

    dX = (u - ∇·K)dt - sqrt(2K) dW

Integrates backward from spill centroid to find probable origin.
Returns 95% confidence ellipse from particle distribution.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ..schemas import ConfidenceEllipse
from .regime_classifier import OceanRegime


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
    n = len(x)

    def drift(px, py):
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
    u_interp,
    v_interp,
    K_interp,
    *,
    n_particles: int = 1000,
    dt_seconds: float = 3600.0,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, OceanRegime]:
    """Run backward SDE integration to find probable origin.

    Parameters
    ----------
    spill_lon, spill_lat : centroid of detected spill
    spill_age_hours : hours since spill began
    u_interp, v_interp : callable(t, lon, lat) -> (u, v) velocity
    K_interp : callable(t, lon, lat) -> (N, 2, 2) diffusion tensor
    n_particles : ensemble size
    dt_seconds : time step
    random_seed : reproducibility seed

    Returns
    -------
    origin_lons, origin_lats : (N,) particle positions at t=0
    regime : OceanRegime used
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    total_seconds = spill_age_hours * 3600.0
    n_steps = int(np.ceil(total_seconds / dt_seconds))
    dt = total_seconds / n_steps

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

    # Determine regime (approximate – use midpoint)
    regime = OceanRegime.MARKOV1

    for step in range(n_steps):
        t_remaining = (n_steps - step) * dt

        # Get velocity and K at current positions
        u = u_interp(t_remaining, x, y)
        v = v_interp(t_remaining, x, y)
        K = K_interp(t_remaining, x, y)

        # WMC correction
        div_K_x = np.zeros(n_particles)
        div_K_y = np.zeros(n_particles)
        # Approximate divergence from K gradient
        if n_particles > 1:
            dk_dx = (
                np.gradient(K[:, 0, 0], axis=0) / max(np.abs(x.max() - x.min()), 1e-6) * 111000.0
            )
            dk_dy = (
                np.gradient(K[:, 1, 1], axis=0) / max(np.abs(y.max() - y.min()), 1e-6) * 111000.0
            )
            div_K_x = dk_dx
            div_K_y = dk_dy

        # Units: u/v/div_K are m/s but x/y are degrees. Convert the
        # combined advective velocity to deg/s BEFORE the RK4 step —
        # otherwise each step jumps hundreds of degrees and the origin
        # diverges to garbage coordinates.
        cos_lat = np.cos(np.radians(np.clip(y, -89.0, 89.0)))
        cos_lat = np.maximum(cos_lat, 0.1)
        u_deg = (u - div_K_x) / (111320.0 * cos_lat)
        v_deg = (v - div_K_y) / 110540.0

        # RK4 step (zero div: already folded into u_deg/v_deg)
        x, y = rk4_step_backward(x, y, u_deg, v_deg, K, np.zeros((n_particles, 2)), dt)

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

    logger.info(
        "Backward complete: mean origin = ({:.4f}, {:.4f}), spread = ({:.4f}, {:.4f})",
        x.mean(),
        y.mean(),
        x.std(),
        y.std(),
    )

    return x, y, regime


def compute_confidence_ellipse(
    lons: np.ndarray,
    lats: np.ndarray,
    confidence: float = 0.95,
) -> ConfidenceEllipse:
    """Compute 95% confidence ellipse from particle distribution.

    Uses eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    lons, lats : (N,) particle positions
    confidence : confidence level (default 0.95)

    Returns
    -------
    ConfidenceEllipse with centre, semi-axes, and orientation
    """
    # Convert to km for meaningful ellipse dimensions
    mean_lon = lons.mean()
    mean_lat = lats.mean()

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
        center_lon=float(mean_lon),
        center_lat=float(mean_lat),
        semi_major_km=float(semi_major),
        semi_minor_km=float(semi_minor),
        orientation_deg=float(orientation),
    )
