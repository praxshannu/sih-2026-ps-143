"""Regime classifier: Markov-1 / Redi / Smagorinsky based on bathymetry and flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger


class OceanRegime(str, Enum):
    MARKOV1 = "markov1"
    REDI = "redi"
    SMAGORINSKY = "smagorinsky"


@dataclass
class RegimeSelection:
    """Result of regime selection, including the honest 'undetermined' case.

    AGENTS.md: select the K_ij regime first, and never silently default. So
    `regime` is ``None`` when the classifier cannot decide, and `reason` says
    exactly which input was missing.
    """

    regime: OceanRegime | None
    reason: str
    votes: dict[str, int]
    inputs_available: dict[str, bool]

    @property
    def determined(self) -> bool:
        return self.regime is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value if self.regime else None,
            "determined": self.determined,
            "reason": self.reason,
            "votes": self.votes,
            "inputs_available": self.inputs_available,
        }


def select_regime(
    *,
    depth_m: np.ndarray | None = None,
    distance_to_coast_km: np.ndarray | None = None,
    current_speed: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    n: int = 1,
) -> RegimeSelection:
    """Select the K_ij diffusion regime, or admit that it cannot be selected.

    All four inputs are required by the classifier. When any is missing the
    regime is reported as undetermined rather than defaulted to MARKOV1 —
    a silently wrong K_ij is worse than an explicitly unknown one.
    """
    available = {
        "depth_m": depth_m is not None,
        "distance_to_coast_km": distance_to_coast_km is not None,
        "current_speed": current_speed is not None,
        "lat": lat is not None,
    }
    missing = [k for k, ok in available.items() if not ok]
    if missing:
        return RegimeSelection(
            regime=None,
            reason="regime undetermined: missing input(s) " + ", ".join(sorted(missing)),
            votes={},
            inputs_available=available,
        )

    depth = np.atleast_1d(np.asarray(depth_m, dtype=np.float64))  # type: ignore[arg-type]
    coast = np.atleast_1d(np.asarray(distance_to_coast_km, dtype=np.float64))  # type: ignore[arg-type]
    speed = np.atleast_1d(np.asarray(current_speed, dtype=np.float64))  # type: ignore[arg-type]
    lat_a = np.atleast_1d(np.asarray(lat, dtype=np.float64))  # type: ignore[arg-type]
    size = max(depth.size, coast.size, speed.size, lat_a.size, int(n))
    if size > 1:

        def _fill(a: np.ndarray) -> np.ndarray:
            return np.full(size, float(a[0])) if a.size == 1 else np.resize(a, size)

        depth, coast, speed, lat_a = _fill(depth), _fill(coast), _fill(speed), _fill(lat_a)

    regime = classify_regime(depth, coast, speed, lat_a)
    votes = {
        OceanRegime.MARKOV1.value: int(((depth > 200.0) & (coast > 50.0)).sum()),
        OceanRegime.REDI.value: int(
            (((depth > 50.0) & (depth <= 200.0)) | (coast <= 50.0)).sum()
        ),
        OceanRegime.SMAGORINSKY.value: int(
            ((depth <= 50.0) | (speed > 0.5) | (np.abs(lat_a) < 60.0)).sum()
        ),
    }
    return RegimeSelection(
        regime=regime,
        reason=f"classified from {size} sample(s): " + ", ".join(f"{k}={v}" for k, v in votes.items()),
        votes=votes,
        inputs_available=available,
    )


@dataclass
class RegimeKParams:
    """Per-regime constants handed to the K_ij tensor assembly."""

    cs: float = 0.15  # Smagorinsky coefficient
    kr_h: float = 100.0  # Redi horizontal diffusivity (m^2/s)
    kr_kv_ratio: float = 1000.0
    k_shear_coeff: float = 0.1
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def regime_k_params(regime: OceanRegime) -> RegimeKParams:
    """Diffusivity parameters for the selected regime.

    These are the documented defaults of the three closures; the point is that
    the *choice* is recorded, so a reader can see which closure produced K.
    """
    if regime is OceanRegime.SMAGORINSKY:
        return RegimeKParams(
            cs=0.20,
            kr_h=20.0,
            kr_kv_ratio=100.0,
            k_shear_coeff=0.20,
            note="tidal/nearshore: Smagorinsky eddy term dominates",
        )
    if regime is OceanRegime.REDI:
        return RegimeKParams(
            cs=0.10,
            kr_h=300.0,
            kr_kv_ratio=1000.0,
            k_shear_coeff=0.10,
            note="shelf/coastal: Redi isopycnal mixing dominates",
        )
    return RegimeKParams(
        cs=0.15,
        kr_h=100.0,
        kr_kv_ratio=1000.0,
        k_shear_coeff=0.10,
        note="deep ocean: Markov-1 isotropic eddy diffusivity",
    )


def classify_regime(
    depth_m: np.ndarray,
    distance_to_coast_km: np.ndarray,
    current_speed: np.ndarray,
    lat: np.ndarray,
) -> OceanRegime:
    """Classify oceanic diffusion regime at particle positions.

    Classification rules (depth-averaged):
        1. Markov-1 (deep ocean):
           - depth > 200 m  AND  distance_to_coast > 50 km
        2. Redi (shelf / coastal):
           - 50 m < depth <= 200 m  OR  distance_to_coast <= 50 km
        3. Smagorinsky (tidal / estuarine):
           - depth <= 50 m  OR  current_speed > 0.5 m/s  OR  lat < 60 deg (nearshore)

    Parameters
    ----------
    depth_m : (N,) bathymetry in metres (positive = depth)
    distance_to_coast_km : (N,) distance from nearest coast
    current_speed : (N,) surface current speed |u| in m/s
    lat : (N,) latitude in degrees

    Returns
    -------
    OceanRegime : the dominant regime (majority vote over particles)
    """
    n = len(depth_m)
    votes = np.zeros(3, dtype=np.int64)

    # Conditions
    deep = depth_m > 200.0
    far_from_coast = distance_to_coast_km > 50.0
    shelf = (depth_m > 50.0) & (depth_m <= 200.0)
    coastal = distance_to_coast_km <= 50.0
    shallow = depth_m <= 50.0
    strong_current = current_speed > 0.5
    high_lat = np.abs(lat) < 60.0

    # Markov-1: deep + far from coast
    markov1_mask = deep & far_from_coast
    votes[0] = int(markov1_mask.sum())

    # Redi: shelf or coastal (but not shallow estuarine)
    redi_mask = (shelf | coastal) & ~shallow
    votes[1] = int(redi_mask.sum())

    # Smagorinsky: shallow, strong currents, or nearshore
    smag_mask = shallow | strong_current | high_lat
    votes[2] = int(smag_mask.sum())

    # Resolve ties / fallback: prefer deeper regime (more physical)
    if votes[2] > votes[1] and votes[2] > votes[0]:
        regime = OceanRegime.SMAGORINSKY
    elif votes[1] > votes[0]:
        regime = OceanRegime.REDI
    else:
        regime = OceanRegime.MARKOV1

    logger.info(
        "Regime classification: markov1={}, redi={}, smag={} => {}",
        votes[0],
        votes[1],
        votes[2],
        regime.value,
    )
    return regime


def get_regime_depth_thresholds() -> dict[str, float]:
    """Return depth thresholds used for classification."""
    return {
        "deep_threshold_m": 200.0,
        "shelf_threshold_m": 50.0,
        "coast_distance_km": 50.0,
        "strong_current_ms": 0.5,
        "high_lat_deg": 60.0,
    }
