"""AIS data normalizer for PostGIS ingestion.

Handles MMSI deduplication, position interpolation for short gaps,
PostGIS geometry creation, and bulk upsert with ON CONFLICT.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import asyncpg
from loguru import logger


class AisNormalizer:
    """Normalizes and cleans AIS position data for PostGIS storage.

    Features:
      - MMSI-based deduplication (keeps latest position per timestamp)
      - Position interpolation for gaps < 5 minutes
      - PostGIS geometry creation
      - Bulk upsert with ON CONFLICT handling
    """

    def __init__(self, pool: asyncpg.Pool | None = None) -> None:
        self.pool = pool
        self.interpolation_threshold_seconds = 300  # 5 minutes

    async def normalize_and_upsert(
        self,
        positions: list[dict],
        interpolate: bool = True,
    ) -> tuple[int, int, int]:
        """Full normalization pipeline: dedup -> interpolate -> upsert.

        Parameters
        ----------
        positions : List of raw AIS position dicts from fetcher.
        interpolate : Whether to interpolate gaps < 5 min.

        Returns
        -------
        (upserted_count, unique_vessels, interpolated_gaps)
        """
        if not positions:
            return 0, 0, 0

        # Step 1: Parse and validate
        parsed = [self._parse_position(p) for p in positions]
        parsed = [p for p in parsed if p is not None]

        # Step 2: Deduplicate by (mmsi, timestamp)
        deduped = self._deduplicate(parsed)
        logger.info("Dedup: {} -> {} positions", len(parsed), len(deduped))

        # Step 3: Interpolate short gaps
        interpolated_count = 0
        if interpolate:
            deduped, interpolated_count = self._interpolate_gaps(deduped)
            logger.info("Interpolated {} gap positions", interpolated_count)

        # Step 4: Upsert to database
        upserted = 0
        unique_vessels: set[str] = set()

        if self.pool and deduped:
            upserted = await self._bulk_upsert(deduped)
            unique_vessels = {p["mmsi"] for p in deduped}

        return upserted, len(unique_vessels), interpolated_count

    def _parse_position(self, raw: dict) -> dict | None:
        """Parse and validate a single AIS position record."""
        try:
            mmsi = str(raw.get("MMSI") or raw.get("mmsi", "")).strip()
            if not mmsi or len(mmsi) != 9 or not mmsi.isdigit():
                return None

            lon = float(raw.get("LONGITUDE") or raw.get("lon") or raw.get("longitude", 0))
            lat = float(raw.get("LATITUDE") or raw.get("lat") or raw.get("latitude", 0))

            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                return None

            ts_raw = raw.get("TIMESTAMP") or raw.get("timestamp") or raw.get("time")
            ts = self._parse_timestamp(ts_raw)
            if ts is None:
                return None

            return {
                "mmsi": mmsi,
                "timestamp": ts,
                "lon": lon,
                "lat": lat,
                "sog": self._safe_float(raw.get("SPEED") or raw.get("sog")),
                "cog": self._safe_float(raw.get("COURSE") or raw.get("cog")),
                "heading": self._safe_int(raw.get("HEADING") or raw.get("heading")),
                "nav_status": self._safe_int(raw.get("NAV_STATUS") or raw.get("nav_status")),
                "vessel_name": self._safe_str(raw.get("NAME") or raw.get("vessel_name")),
                "vessel_type": self._safe_int(raw.get("TYPE") or raw.get("vessel_type")),
                "imo_number": self._safe_str(raw.get("IMO") or raw.get("imo_number")),
                "flag_state": self._safe_str(raw.get("FLAG") or raw.get("flag_state")),
                "source": raw.get("source", "marinecadastre"),
                "is_interpolated": False,
            }
        except Exception:
            return None

    def _deduplicate(self, positions: list[dict]) -> list[dict]:
        """Deduplicate positions by (mmsi, timestamp), keeping latest."""
        seen: dict[tuple[str, datetime], dict] = {}

        for p in positions:
            key = (p["mmsi"], p["timestamp"])
            if key not in seen or p["timestamp"] >= seen[key]["timestamp"]:
                seen[key] = p

        return sorted(seen.values(), key=lambda x: (x["mmsi"], x["timestamp"]))

    def _interpolate_gaps(
        self, positions: list[dict]
    ) -> tuple[list[dict], int]:
        """Interpolate positions for gaps shorter than the threshold.

        For each vessel (MMSI), if consecutive positions are separated by
        less than `interpolation_threshold_seconds`, linearly interpolate
        intermediate positions at 1-minute intervals.
        """
        if not positions:
            return [], 0

        # Group by MMSI
        by_mmsi: dict[str, list[dict]] = {}
        for p in positions:
            by_mmsi.setdefault(p["mmsi"], []).append(p)

        result: list[dict] = []
        interpolated_count = 0

        for mmsi, vessel_positions in by_mmsi.items():
            vessel_positions.sort(key=lambda x: x["timestamp"])

            result.extend(vessel_positions[:1])

            for i in range(1, len(vessel_positions)):
                prev = vessel_positions[i - 1]
                curr = vessel_positions[i]

                gap = (curr["timestamp"] - prev["timestamp"]).total_seconds()

                if 0 < gap <= self.interpolation_threshold_seconds:
                    # Interpolate at 1-minute intervals
                    n_points = int(gap / 60)
                    for j in range(1, n_points):
                        frac = j / n_points
                        interp = self._interpolate_position(prev, curr, frac)
                        interp["is_interpolated"] = True
                        result.append(interp)
                        interpolated_count += 1

                result.append(curr)

        return result, interpolated_count

    @staticmethod
    def _interpolate_position(
        prev: dict, curr: dict, fraction: float
    ) -> dict:
        """Linearly interpolate between two AIS positions."""
        lon = prev["lon"] + fraction * (curr["lon"] - prev["lon"])
        lat = prev["lat"] + fraction * (curr["lat"] - prev["lat"])

        ts = prev["timestamp"] + timedelta(
            seconds=fraction * (curr["timestamp"] - prev["timestamp"]).total_seconds()
        )

        sog = None
        if prev.get("sog") is not None and curr.get("sog") is not None:
            sog = prev["sog"] + fraction * (curr["sog"] - prev["sog"])

        cog = None
        if prev.get("cog") is not None and curr.get("cog") is not None:
            cog = prev["cog"] + fraction * (curr["cog"] - prev["cog"])

        return {
            "mmsi": prev["mmsi"],
            "timestamp": ts,
            "lon": lon,
            "lat": lat,
            "sog": sog,
            "cog": cog,
            "heading": prev.get("heading"),
            "nav_status": prev.get("nav_status"),
            "vessel_name": prev.get("vessel_name"),
            "vessel_type": prev.get("vessel_type"),
            "imo_number": prev.get("imo_number"),
            "flag_state": prev.get("flag_state"),
            "source": prev.get("source", "marinecadastre"),
            "is_interpolated": True,
        }

    async def _bulk_upsert(self, positions: list[dict]) -> int:
        """Bulk upsert positions into PostGIS with ON CONFLICT."""
        if not self.pool or not positions:
            return 0

        async with self.pool.acquire() as conn:
            rows = [
                (
                    p["mmsi"],
                    p.get("vessel_name"),
                    p.get("vessel_type"),
                    p["lon"],
                    p["lat"],
                    p.get("sog"),
                    p.get("cog"),
                    p.get("heading"),
                    p.get("nav_status"),
                    p.get("imo_number"),
                    p.get("flag_state"),
                    p["timestamp"],
                    f"SRID=4326;POINT({p['lon']} {p['lat']})",
                    p.get("source", "marinecadastre"),
                    p.get("is_interpolated", False),
                )
                for p in positions
            ]

            await conn.executemany(
                """
                INSERT INTO ais_positions (
                    mmsi, vessel_name, vessel_type, lon, lat, sog, cog,
                    heading, nav_status, imo_number, flag_state, timestamp,
                    geom, source, is_interpolated
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                    ST_GeomFromEWKT($13), $14, $15)
                ON CONFLICT (mmsi, timestamp)
                DO UPDATE SET
                    vessel_name = COALESCE(EXCLUDED.vessel_name, ais_positions.vessel_name),
                    vessel_type = COALESCE(EXCLUDED.vessel_type, ais_positions.vessel_type),
                    lon = EXCLUDED.lon,
                    lat = EXCLUDED.lat,
                    sog = EXCLUDED.sog,
                    cog = EXCLUDED.cog,
                    heading = EXCLUDED.heading,
                    nav_status = EXCLUDED.nav_status,
                    geom = EXCLUDED.geom,
                    source = EXCLUDED.source,
                    is_interpolated = EXCLUDED.is_interpolated
                """,
                rows,
            )

            return len(rows)

    @staticmethod
    def _parse_timestamp(val: object) -> datetime | None:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val, tz=timezone.utc)
        if isinstance(val, datetime):
            return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
        if isinstance(val, str):
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
            ]:
                try:
                    return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _safe_float(val: object) -> float | None:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(val: object) -> int | None:
        if val is None:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_str(val: object) -> str | None:
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None
