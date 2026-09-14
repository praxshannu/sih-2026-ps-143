"""Weighted composite suspect ranker.

Aggregates all suspect scores from the fuzzy scorer, sorts by composite
score, assigns ranks, and computes Wilson score confidence intervals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from app.engine.fuzzy_scorer import SuspectFeatures, score_suspect

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RankedSuspect:
    """A suspect vessel with rank and confidence intervals.

    The provenance fields (timestamp, latitude, longitude, speed, course,
    closest approach, matched ping count) are part of the audit trail and are
    carried from ``SuspectFeatures`` without modification.
    """

    rank: int
    mmsi: str
    vessel_name: str = "UNKNOWN"
    composite_score: float = 0.0
    confidence_lower: float = 0.0
    confidence_upper: float = 0.0
    score_proximity: float = 0.0
    score_temporal: float = 0.0
    score_trajectory: float = 0.0
    score_anomaly: float = 0.0
    score_vessel_type: float = 0.0
    score_history: float = 0.0
    ais_gap_minutes: float = 0.0
    is_dark_vessel: bool = False
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    speed_knots: float | None = None
    course_deg: float | None = None
    closest_approach_nm: float | None = None
    matched_ping_count: int = 0
    ais_provenance: str = "unknown"


@dataclass
class RankResult:
    """Result of the ranking pipeline."""

    suspects: list[RankedSuspect] = field(default_factory=list)
    total: int = 0
    dark_vessel_count: int = 0
    query_ms: float = 0.0


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


class SuspectRanker:
    """Weighted composite ranker.

    Takes scored suspects, sorts by composite_score descending,
    assigns 1-based ranks, and returns the ranked list.
    """

    def __init__(self) -> None:
        self._top_n: int = 20  # max suspects to return

    def rank_from_features(
        self,
        suspect_features: list[SuspectFeatures],
        top_n: int = 20,
    ) -> RankResult:
        """Score and rank suspects from feature vectors.

        Args:
            suspect_features: List of SuspectFeatures for each candidate.
            top_n: Maximum number of suspects to return.

        Returns:
            RankResult with ranked suspects.
        """
        t0 = time.monotonic()
        self._top_n = top_n
        ranked: list[RankedSuspect] = []

        for feat in suspect_features:
            scores = score_suspect(feat)
            ranked.append(
                RankedSuspect(
                    rank=0,  # assigned below
                    mmsi=feat.mmsi,
                    vessel_name=feat.vessel_name,
                    composite_score=scores["composite_score"],
                    confidence_lower=scores["confidence_lower"],
                    confidence_upper=scores["confidence_upper"],
                    score_proximity=scores["score_proximity"],
                    score_temporal=scores["score_temporal"],
                    score_trajectory=scores["score_trajectory"],
                    score_anomaly=scores["score_anomaly"],
                    score_vessel_type=scores["score_vessel_type"],
                    score_history=min(feat.historical_violations / 3.0, 1.0),
                    ais_gap_minutes=scores["ais_gap_minutes"],
                    is_dark_vessel=scores["is_dark_vessel"],
                    timestamp=scores["timestamp"],
                    latitude=scores["latitude"],
                    longitude=scores["longitude"],
                    speed_knots=scores["speed_knots"],
                    course_deg=scores["course_deg"],
                    closest_approach_nm=scores["closest_approach_nm"],
                    matched_ping_count=scores["matched_ping_count"],
                    ais_provenance=scores["ais_provenance"],
                )
            )

        # Sort descending by composite_score
        ranked.sort(key=lambda s: s.composite_score, reverse=True)

        # Assign ranks
        for i, s in enumerate(ranked[:top_n], start=1):
            s.rank = i

        dark_count = sum(1 for s in ranked if s.is_dark_vessel)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = RankResult(
            suspects=ranked[:top_n],
            total=len(ranked),
            dark_vessel_count=dark_count,
            query_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "Suspect ranker: {} suspects ranked, {} dark, top score={:.4f} ({:.1f}ms)",
            result.total,
            dark_count,
            ranked[0].composite_score if ranked else 0.0,
            result.query_ms,
        )

        return result

    def rank_from_dicts(
        self,
        suspects: list[dict[str, Any]],
        top_n: int = 20,
    ) -> RankResult:
        """Rank pre-scored suspects from dictionaries.

        Expects each dict to have at minimum: mmsi, composite_score.
        Re-computes the composite using the fuzzy scorer for consistency.
        """
        features_list: list[SuspectFeatures] = []
        for s in suspects:
            feat = SuspectFeatures(
                mmsi=s.get("mmsi", "000000000"),
                vessel_name=s.get("vessel_name", "UNKNOWN"),
                min_distance_nm=s.get("min_distance_nm", 50.0),
                time_delta_minutes=s.get("time_delta_minutes", 180.0),
                trajectory_intersection_score=s.get("trajectory_intersection_score", 0.0),
                ais_gap_minutes=s.get("ais_gap_minutes", 0.0),
                speed_anomaly_sigma=s.get("speed_anomaly_sigma", 0.0),
                course_anomaly_sigma=s.get("course_anomaly_sigma", 0.0),
                vessel_type_risk=s.get("vessel_type_risk", 0.0),
                historical_violations=s.get("historical_violations", 0),
                is_dark_sar_target=s.get("is_dark_vessel", False),
                timestamp=s.get("timestamp"),
                latitude=s.get("latitude"),
                longitude=s.get("longitude"),
                speed_knots=s.get("speed_knots"),
                course_deg=s.get("course_deg"),
                matched_ping_count=int(s.get("matched_ping_count", 0) or 0),
                ais_provenance=str(s.get("ais_provenance", "unknown")),
            )
            features_list.append(feat)

        return self.rank_from_features(features_list, top_n=top_n)

    def aggregate_scores(
        self,
        ais_scores: dict[str, dict],
        anomaly_scores: dict[str, dict],
        dark_mmsis: set[str],
        vessel_type_risk: dict[str, float],
        historical_violations: dict[str, int],
    ) -> list[SuspectFeatures]:
        """Aggregate per-suspect data into feature vectors for scoring.

        Merges AIS slicer results, anomaly scores, dark vessel flags,
        vessel type risk, and historical violation counts into a unified
        SuspectFeatures per candidate MMSI.

        Args:
            ais_scores: mmsi -> {min_distance_nm, time_delta_minutes, ...}
            anomaly_scores: mmsi -> {speed_anomaly_sigma, course_anomaly_sigma, ...}
            dark_mmsis: Set of MMSIs confirmed as dark vessels.
            vessel_type_risk: mmsi -> risk score (0-1).
            historical_violations: mmsi -> count.

        Returns:
            List of SuspectFeatures, one per unique MMSI.
        """
        all_mmsi = set(ais_scores.keys()) | set(anomaly_scores.keys())
        features: list[SuspectFeatures] = []

        for mmsi in all_mmsi:
            ais = ais_scores.get(mmsi, {})
            anom = anomaly_scores.get(mmsi, {})

            features.append(
                SuspectFeatures(
                    mmsi=mmsi,
                    vessel_name=ais.get("vessel_name", "UNKNOWN"),
                    min_distance_nm=ais.get("min_distance_nm", 50.0),
                    time_delta_minutes=ais.get("time_delta_minutes", 180.0),
                    trajectory_intersection_score=ais.get("trajectory_intersection_score", 0.0),
                    ais_gap_minutes=ais.get("ais_gap_minutes", 0.0),
                    speed_anomaly_sigma=anom.get("speed_anomaly_sigma", 0.0),
                    course_anomaly_sigma=anom.get("course_anomaly_sigma", 0.0),
                    vessel_type_risk=vessel_type_risk.get(mmsi, 0.0),
                    historical_violations=historical_violations.get(mmsi, 0),
                    is_dark_sar_target=mmsi in dark_mmsis,
                    timestamp=ais.get("timestamp"),
                    latitude=ais.get("latitude"),
                    longitude=ais.get("longitude"),
                    speed_knots=ais.get("speed_knots"),
                    course_deg=ais.get("course_deg"),
                    matched_ping_count=int(ais.get("matched_ping_count", 0) or 0),
                    ais_provenance=str(ais.get("ais_provenance", "unknown")),
                )
            )

        return features
