"""Cauchy fuzzy membership multi-factor scorer.

Computes composite suspicion scores using Cauchy membership functions
across multiple behavioral and spatial factors. Includes Wilson score
confidence intervals for statistical rigor.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SuspectFeatures:
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


def cauchy_membership(x: float, center: float, width: float) -> float:
    return 1.0 / (1.0 + ((x - center) / max(width, 1e-8)) ** 2)

def score_suspect(features: SuspectFeatures) -> dict:
    prox = cauchy_membership(features.min_distance_nm, center=0.0, width=5.0)
    temp = cauchy_membership(features.time_delta_minutes, center=0.0, width=30.0)
    traj = np.clip(features.trajectory_intersection_score, 0.0, 1.0)
    ais_dark = cauchy_membership(features.ais_gap_minutes, center=20.0, width=15.0)
    speed_anom = cauchy_membership(features.speed_anomaly_sigma, center=3.0, width=1.5)
    course_anom = cauchy_membership(features.course_anomaly_sigma, center=3.0, width=1.5)
    anomaly = 0.5 * ais_dark + 0.3 * speed_anom + 0.2 * course_anom
    dark_bonus = 0.30 if features.is_dark_sar_target else 0.0
    vtype = features.vessel_type_risk
    hist = min(features.historical_violations / 3.0, 1.0)

    WEIGHTS = {
        "proximity": 0.22, "temporal": 0.20, "trajectory": 0.18,
        "anomaly": 0.20, "vessel_type": 0.10, "history": 0.05, "dark_bonus": 0.05,
    }

    composite = (
        WEIGHTS["proximity"] * prox + WEIGHTS["temporal"] * temp +
        WEIGHTS["trajectory"] * traj + WEIGHTS["anomaly"] * anomaly +
        WEIGHTS["vessel_type"] * vtype + WEIGHTS["history"] * hist +
        WEIGHTS["dark_bonus"] * dark_bonus
    )
    composite = min(composite, 1.0)

    n = 100
    z = 1.96
    lower = (composite + z**2/(2*n) - z*np.sqrt(composite*(1-composite)/n + z**2/(4*n**2))) / (1 + z**2/n)
    upper = (composite + z**2/(2*n) + z*np.sqrt(composite*(1-composite)/n + z**2/(4*n**2))) / (1 + z**2/n)

    return {
        "composite_score": round(float(composite), 4),
        "confidence_lower": round(float(lower), 4),
        "confidence_upper": round(float(upper), 4),
        "score_proximity": round(float(prox), 4),
        "score_temporal": round(float(temp), 4),
        "score_trajectory": round(float(traj), 4),
        "score_anomaly": round(float(anomaly), 4),
        "score_vessel_type": round(float(vtype), 4),
        "ais_gap_minutes": features.ais_gap_minutes,
        "is_dark_vessel": features.is_dark_sar_target,
    }
