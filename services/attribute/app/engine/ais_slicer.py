"""PostGIS spatiotemporal query engine for AIS vessel proximity search.

Finds all vessels within N nautical miles of an origin ellipse during a
T-hour time window using ST_DWithin with geographic coordinates and
temporal filtering against the ais_positions hypertable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

import asyncpg
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NM_TO_METERS = 1852.0
DEG_TO_RAD = math.pi / 180.0


def _as_datetime(value: Any) -> Any:
    """Coerce ISO strings to datetime; pass datetimes through.

    asyncpg rejects str for timestamptz params, so all SQL time arguments
    must be real datetime objects (preserves parameterized-SQL policy).
    """
    if isinstance(value, str):
        from datetime import datetime

        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value


@dataclass
class VesselTrack:
    """A single AIS position record with derived fields.

    ``source`` is the provenance label carried by the row (e.g.
    ``live_terrestrial`` or ``synthetic_mock``). It is what the coverage gate
    uses to decide whether this vessel may be scored at all — see
    ``app/engine/ais_coverage.py``.
    """

    mmsi: str
    vessel_name: str
    vessel_type: int
    lon: float
    lat: float
    sog: float
    cog: float
    heading: int | None
    nav_status: int | None
    imo_number: str | None
    flag_state: str | None
    timestamp: Any
    min_distance_nm: float = 0.0
    time_delta_minutes: float = 0.0
    source: str = "unknown"
    matched_ping_count: int = 0


@dataclass
class SliceResult:
    """Result of an AIS slice query."""

    vessels: list[VesselTrack] = field(default_factory=list)
    distinct_mmsi: int = 0
    query_ms: float = 0.0
    total_positions: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AisSlicer:
    """PostGIS spatiotemporal query engine.

    Uses ST_DWithin on geography casts for accurate nautical-mile proximity
    queries and temporal range filtering on the ais_positions hypertable.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def find_vessels(
        self,
        center_lon: float,
        center_lat: float,
        search_radius_nm: float,
        spill_time: Any,
        time_window_hours: float,
    ) -> SliceResult:
        """Find all vessels within radius of center during the time window.

        Args:
            center_lon: Origin ellipse center longitude.
            center_lat: Origin ellipse center latitude.
            search_radius_nm: Search radius in nautical miles.
            spill_time: Estimated spill origin time (datetime or ISO string).
            time_window_hours: Half-window in hours around spill_time.

        Returns:
            SliceResult with all matching vessel positions.
        """
        t0 = time.monotonic()
        radius_m = search_radius_nm * NM_TO_METERS
        spill_time = _as_datetime(spill_time)

        sql = """
            SELECT DISTINCT ON (mmsi)
                mmsi,
                vessel_name,
                vessel_type,
                lon,
                lat,
                sog,
                cog,
                heading,
                nav_status,
                imo_number,
                flag_state,
                timestamp,
                COALESCE(source, 'unknown') AS source,
                COUNT(*) OVER (PARTITION BY mmsi) AS matched_ping_count,
                ST_Distance(
                    geom::geography,
                    ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography
                ) / $3 AS min_distance_nm
            FROM ais_positions
            WHERE ST_DWithin(
                geom::geography,
                ST_SetSRID(ST_MakePoint($1, $2), 4326)::geography,
                $3
            )
            AND timestamp BETWEEN ($4::timestamptz - ($5 || ' hours')::interval)
                              AND ($4::timestamptz + ($5 || ' hours')::interval)
            ORDER BY mmsi, timestamp DESC
        """

        logger.debug(
            "AIS slicer query: center=({}, {}) radius={}nm window=±{}h",
            center_lon,
            center_lat,
            search_radius_nm,
            time_window_hours,
        )

        rows = await self._pool.fetch(
            sql,
            center_lon,
            center_lat,
            radius_m,
            spill_time,
            str(time_window_hours),
        )

        vessels: list[VesselTrack] = []
        seen_mmsi: set[str] = set()
        origin_ts = spill_time

        for row in rows:
            mmsi = row["mmsi"]
            seen_mmsi.add(mmsi)

            # Compute time delta from origin
            pos_ts = row["timestamp"]
            td = abs((pos_ts - origin_ts).total_seconds() / 60.0) if origin_ts and pos_ts else 0.0

            track = VesselTrack(
                mmsi=mmsi,
                vessel_name=row["vessel_name"] or "UNKNOWN",
                vessel_type=row["vessel_type"] or 0,
                lon=row["lon"],
                lat=row["lat"],
                sog=row["sog"] or 0.0,
                cog=row["cog"] or 0.0,
                heading=row["heading"],
                nav_status=row["nav_status"],
                imo_number=row["imo_number"],
                flag_state=row["flag_state"],
                timestamp=pos_ts,
                min_distance_nm=float(row["min_distance_nm"] or 0.0),
                time_delta_minutes=td,
                source=row["source"] or "unknown",
                matched_ping_count=int(row["matched_ping_count"] or 0),
            )
            vessels.append(track)

        elapsed_ms = (time.monotonic() - t0) * 1000.0

        result = SliceResult(
            vessels=vessels,
            distinct_mmsi=len(seen_mmsi),
            query_ms=round(elapsed_ms, 2),
            total_positions=len(rows),
        )

        logger.info(
            "AIS slicer: {} vessels ({} positions) in {:.1f}ms",
            result.distinct_mmsi,
            result.total_positions,
            result.query_ms,
        )

        return result

    async def get_vessel_history(
        self,
        mmsi: str,
        hours: float = 24.0,
        ref_time: Any = None,
    ) -> list[VesselTrack]:
        """Retrieve recent AIS track for a single vessel.

        Used by the anomaly detector to build behavioral profiles.
        """
        if ref_time is None:
            from datetime import datetime

            ref_time = datetime.now(UTC)
        sql = """
            SELECT
                mmsi, vessel_name, vessel_type, lon, lat,
                sog, cog, heading, nav_status, imo_number,
                flag_state, timestamp,
                COALESCE(source, 'unknown') AS source
            FROM ais_positions
            WHERE mmsi = $1
            AND timestamp BETWEEN ($2::timestamptz - ($3 || ' hours')::interval)
                              AND ($2::timestamptz)
            ORDER BY timestamp ASC
        """

        rows = await self._pool.fetch(sql, mmsi, _as_datetime(ref_time), str(hours))

        return [
            VesselTrack(
                mmsi=row["mmsi"],
                vessel_name=row["vessel_name"] or "UNKNOWN",
                vessel_type=row["vessel_type"] or 0,
                lon=row["lon"],
                lat=row["lat"],
                sog=row["sog"] or 0.0,
                cog=row["cog"] or 0.0,
                heading=row["heading"],
                nav_status=row["nav_status"],
                imo_number=row["imo_number"],
                flag_state=row["flag_state"],
                timestamp=row["timestamp"],
                source=row["source"] or "unknown",
            )
            for row in rows
        ]

    async def get_ais_gap(
        self,
        mmsi: str,
        ref_time: Any,
        lookback_hours: float = 6.0,
    ) -> float:
        """Compute the AIS gap duration (minutes) ending at ref_time.

        A gap is the longest continuous period without AIS transmissions
        within the lookback window.
        """
        sql = """
            WITH positions AS (
                SELECT timestamp
                FROM ais_positions
                WHERE mmsi = $1
                AND timestamp BETWEEN ($2::timestamptz - ($3 || ' hours')::interval)
                                  AND ($2::timestamptz)
                ORDER BY timestamp DESC
            ),
            gaps AS (
                SELECT EXTRACT(EPOCH FROM (
                    LAG(timestamp) OVER (ORDER BY timestamp DESC) - timestamp
                )) / 60.0 AS gap_minutes
                FROM positions
            )
            SELECT COALESCE(MAX(gap_minutes), 0.0) AS max_gap_minutes
            FROM gaps
        """

        row = await self._pool.fetchrow(sql, mmsi, _as_datetime(ref_time), str(lookback_hours))
        return float(row["max_gap_minutes"]) if row else 0.0
