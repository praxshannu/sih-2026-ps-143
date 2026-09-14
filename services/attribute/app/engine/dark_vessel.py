"""SAR non-AIS target matcher for dark vessel detection.

When a vessel is detected in SAR (non-AIS return) at the origin location
and time but has no matching AIS broadcast, it qualifies as a dark vessel
candidate. This module cross-references SAR ship signatures with the
origin ellipse and identifies MMSI gaps.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

import asyncpg
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NM_TO_DEG_LAT = 1.0 / 60.0  # 1 nautical mile ~ 1/60 degree latitude
NM_TO_METERS = 1852.0


def _as_datetime(value: Any) -> Any:
    """Coerce ISO strings to datetime for asyncpg timestamptz params."""
    if isinstance(value, str):
        from datetime import datetime

        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


@dataclass
class DarkVesselMatch:
    """A dark vessel candidate identified from SAR."""

    sar_lon: float
    sar_lat: float
    sar_timestamp: Any
    sar_confidence: float
    nearest_ais_mmsi: Optional[str]
    nearest_ais_distance_nm: float
    nearest_ais_time_delta_min: float
    is_dark: bool  # True if no AIS match within threshold
    matching_mmsi: Optional[str] = None
    match_distance_nm: float = 0.0
    match_time_delta_min: float = 0.0


@dataclass
class DarkVesselResult:
    """Result of dark vessel analysis."""

    matches: list[DarkVesselMatch]
    dark_vessel_count: int
    sar_targets_count: int
    query_ms: float = 0.0


class DarkVesselDetector:
    """SAR non-AIS target matcher.

    Cross-references SAR ship detections with AIS positions to find
    vessels transmitting no AIS (dark vessels) at the spill origin.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self._ais_match_radius_nm: float = 2.0
        self._ais_match_window_min: float = 30.0

    async def detect_dark_vessels(
        self,
        sar_targets: list[dict[str, Any]],
        search_radius_nm: float = 50.0,
    ) -> DarkVesselResult:
        """Analyze SAR targets for dark vessel signatures.

        For each SAR detection, query PostGIS for the nearest AIS position
        within a tight spatial-temporal window. If no AIS match is found,
        the SAR target is classified as a dark vessel candidate.

        Args:
            sar_targets: List of SAR detection dicts with keys:
                lon, lat, timestamp, confidence, radar_cross_section_db.
            search_radius_nm: Max radius to search for AIS matches.

        Returns:
            DarkVesselResult with match details per SAR target.
        """
        t0 = time.monotonic()
        matches: list[DarkVesselMatch] = []
        radius_m = self._ais_match_radius_nm * NM_TO_METERS

        # Step 1: Find AIS positions near each SAR target
        sar_ais_map: dict[int, list[dict]] = {}

        for idx, sar in enumerate(sar_targets):
            sar_lon = sar["lon"]
            sar_lat = sar["lat"]
            sar_ts = sar["timestamp"]

            sql = """
                SELECT
                    mmsi, vessel_name, lon, lat, sog, cog, timestamp,
                    ST_Distance(
                        geom::geography,
                        ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography
                    ) / $3 AS distance_nm,
                    ABS(EXTRACT(EPOCH FROM (timestamp - $4::timestamptz)) / 60.0)
                        AS time_delta_min
                FROM ais_positions
                WHERE ST_DWithin(
                    geom::geography,
                    ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
                    $5
                )
                AND timestamp BETWEEN ($4::timestamptz - interval '30 minutes')
                                  AND ($4::timestamptz + interval '30 minutes')
                ORDER BY
                    ST_Distance(
                        geom::geography,
                        ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography
                    ) ASC,
                    ABS(EXTRACT(EPOCH FROM (timestamp - $4::timestamptz))) ASC
                LIMIT 1
            """

            rows = await self._pool.fetch(
                sql,
                sar_lon,
                sar_lat,
                NM_TO_METERS,
                _as_datetime(sar_ts),
                radius_m,
            )
            sar_ais_map[idx] = [dict(r) for r in rows]

        # Step 2: Classify each SAR target
        for idx, sar in enumerate(sar_targets):
            ais_candidates = sar_ais_map.get(idx, [])

            if not ais_candidates:
                # No AIS positions near SAR target -> dark vessel
                match = DarkVesselMatch(
                    sar_lon=sar["lon"],
                    sar_lat=sar["lat"],
                    sar_timestamp=sar["timestamp"],
                    sar_confidence=sar.get("confidence", 0.0),
                    nearest_ais_mmsi=None,
                    nearest_ais_distance_nm=float("inf"),
                    nearest_ais_time_delta_min=float("inf"),
                    is_dark=True,
                )
                matches.append(match)
                continue

            best = ais_candidates[0]
            dist_nm = float(best["distance_nm"])
            td_min = float(best["time_delta_min"])

            is_dark = (
                dist_nm > self._ais_match_radius_nm
                or td_min > self._ais_match_window_min
            )

            match = DarkVesselMatch(
                sar_lon=sar["lon"],
                sar_lat=sar["lat"],
                sar_timestamp=sar["timestamp"],
                sar_confidence=sar.get("confidence", 0.0),
                nearest_ais_mmsi=best["mmsi"] if not is_dark else None,
                nearest_ais_distance_nm=dist_nm,
                nearest_ais_time_delta_min=td_min,
                is_dark=is_dark,
                matching_mmsi=best["mmsi"] if not is_dark else None,
                match_distance_nm=dist_nm,
                match_time_delta_min=td_min,
            )
            matches.append(match)

        dark_count = sum(1 for m in matches if m.is_dark)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = DarkVesselResult(
            matches=matches,
            dark_vessel_count=dark_count,
            sar_targets_count=len(sar_targets),
            query_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "Dark vessel detector: {}/{} dark vessels identified in {:.1f}ms",
            dark_count,
            len(sar_targets),
            result.query_ms,
        )

        return result

    def get_dark_mmsis(self, result: DarkVesselResult) -> set[str]:
        """Extract MMSIs of vessels flagged as NOT dark (i.e., they have AIS).

        Inverted logic: if a SAR target has a nearby AIS vessel, that MMSI
        becomes a suspect. Dark vessels have no MMSI to attribute.
        """
        return {
            m.matching_mmsi
            for m in result.matches
            if m.matching_mmsi is not None
        }

    def get_dark_sar_mmsi_set(
        self, sar_targets: list[dict[str, Any]], dark_result: DarkVesselResult
    ) -> set[str]:
        """Return set of MMSIs that were near dark SAR targets.

        For attribution, we look for AIS vessels that were within the broader
        search radius of a dark SAR detection - these are strong suspects.
        """
        dark_sars = [
            m for m in dark_result.matches if m.is_dark
        ]
        if not dark_sars:
            return set()
        # In practice, this queries broader AIS context; for now return empty
        # as dark vessels have no AIS by definition.
        return set()
