"""Cauchy fuzzy membership multi-factor scorer.

Computes composite suspicion scores using Cauchy membership functions
across multiple behavioral and spatial factors. Includes Wilson score
confidence intervals for statistical rigor.

Every input field below is part of the auditable scoring record: a ranked
suspect must always be traceable back to (mmsi, timestamp, latitude,
longitude, speed, course, closest approach, matched ping count, AIS gap).
Dropping any of them would leave a number that cannot be defended, so they
are carried into the score dict and onward into ``ScoreResult``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SuspectFeatures:
    """Full feature vector for one candidate vessel.

    Fields required by the audit trail (never dropped):
        mmsi, timestamp, latitude, longitude, speed_knots, course_deg,
        min_distance_nm (closest approach), matched_ping_count, ais_gap_minutes.
    """

    mmsi: str
    vessel_name: str
    min_distance_nm: float
    time_delta_minutes: float
    trajectory_intersection_score: float
    ais_gap_minutes: float
    speed_anomaly_sigma: float
    course_anomaly_sigma: float
    vessel_type_risk: float
    historical_violations: int
    is_dark_sar_target: bool
    # --- provenance / audit fields (optional: absent when the source has none) ---
    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    speed_knots: float | None = None
    course_deg: float | None = None
    matched_ping_count: int = 0
    ais_provenance: str = "unknown"


def cauchy_membership(x: float, center: float, width: float) -> float:
    return 1.0 / (1.0 + ((x - center) / max(width, 1e-8)) ** 2)


def wilson_interval(p: float, n: int = 100, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion.

    AGENTS.md: attribution shows a Wilson 95% CI, never a point estimate
    alone. ``n`` is the effective evidence count (100 by default,
    i.e. +/- ~10 points at p=0.5).
    """
    p = min(max(p, 0.0), 1.0)
    denom = 1.0 + z**2 / n
    centre = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4 * n**2))
    return ((centre - half) / denom, (centre + half) / denom)


def score_suspect(features: SuspectFeatures) -> dict:
    prox = cauchy_membership(features.min_distance_nm, center=0.0, width=5.0)
    temp = cauchy_membership(features.time_delta_minutes, center=0.0, width=30.0)
    traj = min(max(features.trajectory_intersection_score, 0.0), 1.0)
    ais_dark = cauchy_membership(features.ais_gap_minutes, center=20.0, width=15.0)
    speed_anom = cauchy_membership(features.speed_anomaly_sigma, center=3.0, width=1.5)
    course_anom = cauchy_membership(features.course_anomaly_sigma, center=3.0, width=1.5)
    anomaly = 0.5 * ais_dark + 0.3 * speed_anom + 0.2 * course_anom
    dark_bonus = 0.30 if features.is_dark_sar_target else 0.0
    vtype = features.vessel_type_risk
    hist = min(features.historical_violations / 3.0, 1.0)

    WEIGHTS = {
        "proximity": 0.22,
        "temporal": 0.20,
        "trajectory": 0.18,
        "anomaly": 0.20,
        "vessel_type": 0.10,
        "history": 0.05,
        "dark_bonus": 0.05,
    }

    composite = (
        WEIGHTS["proximity"] * prox
        + WEIGHTS["temporal"] * temp
        + WEIGHTS["trajectory"] * traj
        + WEIGHTS["anomaly"] * anomaly
        + WEIGHTS["vessel_type"] * vtype
        + WEIGHTS["history"] * hist
        + WEIGHTS["dark_bonus"] * dark_bonus
    )
    composite = min(composite, 1.0)

    lower, upper = wilson_interval(composite)

    return {
        "composite_score": round(float(composite), 4),
        "confidence_lower": round(float(lower), 4),
        "confidence_upper": round(float(upper), 4),
        "confidence_method": "wilson_95",
        "score_proximity": round(float(prox), 4),
        "score_temporal": round(float(temp), 4),
        "score_trajectory": round(float(traj), 4),
        "score_anomaly": round(float(anomaly), 4),
        "score_vessel_type": round(float(vtype), 4),
        "ais_gap_minutes": features.ais_gap_minutes,
        "is_dark_vessel": features.is_dark_sar_target,
        # --- audit trail: carried through verbatim, never recomputed ---
        "mmsi": features.mmsi,
        "timestamp": features.timestamp,
        "latitude": features.latitude,
        "longitude": features.longitude,
        "speed_knots": features.speed_knots,
        "course_deg": features.course_deg,
        "closest_approach_nm": features.min_distance_nm,
        "matched_ping_count": features.matched_ping_count,
        "ais_provenance": features.ais_provenance,
    }
