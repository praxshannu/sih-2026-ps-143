"""Regime classifier: Markov-1 / Redi / Smagorinsky based on bathymetry and flow."""

from __future__ import annotations

from enum import Enum

import numpy as np
from loguru import logger


class OceanRegime(str, Enum):
    MARKOV1 = "markov1"
    REDI = "redi"
    SMAGORINSKY = "smagorinsky"


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
