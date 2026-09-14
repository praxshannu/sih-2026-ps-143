"""Full anisotropic K_ij diffusion tensor assembly.

    K = K_molecular + K_eddy(Smagorinsky) + K_redi + K_shear

Returns 2x2 tensor at each grid point as (N, 2, 2) array.
"""

from __future__ import annotations

import numpy as np
from loguru import logger


# Physical defaults
K_MOLECULAR = 1e-6          # m^2/s (molecular diffusion of heat/dye)
CS_SMAG = 0.15               # Smagorinsky coefficient
KR_H = 100.0                # Redi horizontal diffusivity m^2/s
KR_KV_RATIO = 1000.0        # Redi horizontal/vertical aspect ratio
K_SHEAR_COEFF = 0.1          # Shear-induced mixing coefficient
DELTA_GRID = 1000.0          # Nominal grid spacing (m) – overridden if dx/dy available


def strain_rate_magnitude(
    du_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dx: np.ndarray,
    dv_dy: np.ndarray,
) -> np.ndarray:
    """Compute |S| = sqrt(2 * S_ij * S_ij) for 2D horizontal strain.

    Parameters
    ----------
    du_dx, du_dy, dv_dx, dv_dy : (N,) velocity gradients

    Returns
    -------
    |S| : (N,) strain rate magnitude
    """
    s11 = du_dx
    s22 = dv_dy
    s12 = 0.5 * (du_dy + dv_dx)
    return np.sqrt(2.0 * (s11**2 + s22**2 + 2.0 * s12**2))


def k_tensor_assembly(
    du_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dx: np.ndarray,
    dv_dy: np.ndarray,
    depth_m: np.ndarray,
    lat: np.ndarray,
    *,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
    cs: float = CS_SMAG,
    kr_h: float = KR_H,
    kr_kv_ratio: float = KR_KV_RATIO,
    k_shear_coeff: float = K_SHEAR_COEFF,
    k_molecular: float = K_MOLECULAR,
) -> np.ndarray:
    """Assemble full anisotropic K_ij tensor at particle positions.

    Parameters
    ----------
    du_dx, du_dy, dv_dx, dv_dy : (N,) velocity gradients (s^-1)
    depth_m : (N,) bathymetry in metres (positive = depth)
    lat : (N,) latitude in degrees
    dx, dy : (N,) optional local grid spacing (m). Defaults to DELTA_GRID.
    cs : Smagorinsky coefficient
    kr_h : Redi horizontal diffusivity
    kr_kv_ratio : Redi horizontal/vertical anisotropy ratio
    k_shear_coeff : shear mixing coefficient
    k_molecular : molecular diffusivity

    Returns
    -------
    K : (N, 2, 2) diffusion tensor in m^2/s
    """
    n = len(du_dx)

    if dx is None:
        dx = np.full(n, DELTA_GRID)
    if dy is None:
        dy = np.full(n, DELTA_GRID)

    # --- 1. Molecular (isotropic, diagonal) ---
    K_mol = np.zeros((n, 2, 2), dtype=np.float64)
    K_mol[:, 0, 0] = k_molecular
    K_mol[:, 1, 1] = k_molecular

    # --- 2. Smagorinsky eddy diffusivity ---
    #    K_S = Cs * Delta^2 * |S|  (isotropic, diagonal)
    S_mag = strain_rate_magnitude(du_dx, du_dy, dv_dx, dv_dy)
    delta2 = dx * dy  # area-based grid spacing
    K_smag = cs * delta2 * S_mag  # (N,)

    K_eddy = np.zeros((n, 2, 2), dtype=np.float64)
    K_eddy[:, 0, 0] = K_smag
    K_eddy[:, 1, 1] = K_smag

    # --- 3. Redi mixing (anisotropic) ---
    #    K_R = Kr_h * (1, 1/aspect) where aspect = H/L (depth/shelf width)
    #    Off-diagonal terms from isopycnal slope coupling (simplified here)
    aspect = np.maximum(depth_m, 1.0) / 1000.0  # rough scale
    kr_v = kr_h / np.maximum(kr_kv_ratio, 1.0)

    K_redi = np.zeros((n, 2, 2), dtype=np.float64)
    K_redi[:, 0, 0] = kr_h
    K_redi[:, 1, 1] = kr_h * np.minimum(aspect, 1.0)
    # Off-diagonal: Coriolis-induced cross-diffusion (proportional to f)
    omega_earth = 7.2921e-5
    f = 2.0 * omega_earth * np.sin(np.radians(np.clip(lat, -89.9, 89.9)))
    K_redi[:, 0, 1] = k_shear_coeff * np.abs(f) * depth_m
    K_redi[:, 1, 0] = K_redi[:, 0, 1]

    # --- 4. Shear-induced mixing (vertical shear → horizontal dispersion) ---
    #    K_Shear = k_shear * du_dz * H (simplified 2D projection)
    K_shear = np.zeros((n, 2, 2), dtype=np.float64)
    shear_diag = k_shear_coeff * S_mag * depth_m
    K_shear[:, 0, 0] = shear_diag
    K_shear[:, 1, 1] = shear_diag

    # --- Total ---
    K_total = K_mol + K_eddy + K_redi + K_shear

    # Ensure positive-definite by clamping eigenvalues
    _enforce_positive_definite(K_total)

    logger.debug(
        "K_ij assembled: mean(Kxx)={:.3e}, mean(Kyy)={:.3e}, mean(Kxy)={:.3e}",
        K_total[:, 0, 0].mean(),
        K_total[:, 1, 1].mean(),
        K_total[:, 0, 1].mean(),
    )

    return K_total


def _enforce_positive_definite(K: np.ndarray) -> None:
    """Clamp K tensor eigenvalues to be >= k_molecular.

    Operates in-place on (N, 2, 2) tensor.
    """
    n = K.shape[0]

    # For each 2x2 symmetric tensor, enforce eigenvalues > 0
    a = K[:, 0, 0]
    b = K[:, 0, 1]
    d = K[:, 1, 1]

    # Eigenvalues of [[a, b], [b, d]]
    trace = a + d
    det = a * d - b * b
    discriminant = np.maximum(trace**2 - 4.0 * det, 0.0)
    eig_min = 0.5 * (trace - np.sqrt(discriminant))
    eig_max = 0.5 * (trace + np.sqrt(discriminant))

    # Clamp minimum eigenvalue
    min_eig = K_MOLECULAR
    needs_fix = eig_min < min_eig
    if np.any(needs_fix):
        shift = np.where(needs_fix, min_eig - eig_min, 0.0)
        K[:, 0, 0] += shift
        K[:, 1, 1] += shift


def k_eigenvalues(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract principal diffusivities (eigenvalues) from K tensor.

    Returns
    -------
    eig_max, eig_min : (N,) arrays
    """
    a = K[:, 0, 0]
    b = K[:, 0, 1]
    d = K[:, 1, 1]
    trace = a + d
    det = a * d - b * b
    discriminant = np.maximum(trace**2 - 4.0 * det, 0.0)
    eig_max = 0.5 * (trace + np.sqrt(discriminant))
    eig_min = 0.5 * (trace - np.sqrt(discriminant))
    return eig_max, eig_min
