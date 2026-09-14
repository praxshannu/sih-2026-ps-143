"""MarineCadastre / AISHub AIS data ingestion.

Fetches recent vessel position data, normalizes fields,
and inserts into PostGIS database.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import asyncpg
import httpx
from loguru import logger

from ..models.schemas import AisIngestResult, AisPosition

# Indian EEZ bounding box
INDIA_EEZ_BBOX = (68.0, 5.0, 88.0, 25.0)

MARINECADASTRE_URL = "https://data.aishub.net/ws.php"
AISHUB_API_URL = "https://data.aishub.net/api/positions"


class AisFetcher:
    """Fetches AIS position data from MarineCadastre/AISHub."""

    def __init__(
        self,
        api_key: str | None = None,
        pool: asyncpg.Pool | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("AISHUB_API_KEY", "")
        self.pool = pool

    async def fetch(
        self,
        bbox: tuple[float, float, float, float] = INDIA_EEZ_BBOX,
        lookback_hours: int = 6,
    ) -> AisIngestResult:
        """Fetch recent AIS data and upsert into PostGIS.

        Parameters
        ----------
        bbox : (lon_min, lat_min, lon_max, lat_max)
        lookback_hours : Hours of historical data to fetch

        Returns
        -------
        AisIngestResult with ingestion metadata.
        """
        t0 = time.time()
        lon_min, lat_min, lon_max, lat_max = bbox

        logger.info(
            "Fetching AIS data: bbox={}, lookback={}h",
            bbox,
            lookback_hours,
        )

        # Fetch from MarineCadastre
        raw_positions = await self._fetch_marinecadastre(bbox, lookback_hours)

        # Normalize
        normalized = [self._normalize_position(p) for p in raw_positions]
        normalized = [p for p in normalized if p is not None]

        # Upsert to database
        upserted = 0
        unique_vessels: set[str] = set()

        if self.pool and normalized:
            upserted = await self._bulk_upsert(normalized)
            unique_vessels = {p.mmsi for p in normalized}

        elapsed = time.time() - t0

        logger.info(
            "AIS fetch complete: {} raw, {} normalized, {} upserted, {} vessels in {:.1f}s",
            len(raw_positions),
            len(normalized),
            upserted,
            len(unique_vessels),
            elapsed,
        )

        return AisIngestResult(
            records_fetched=len(raw_positions),
            records_upserted=upserted,
            unique_vessels=len(unique_vessels),
            interpolated_gaps=0,
            fetch_duration_seconds=round(elapsed, 2),
        )

    async def _fetch_marinecadastre(
        self,
        bbox: tuple[float, float, float, float],
        lookback_hours: int,
    ) -> list[dict]:
        """Fetch AIS data from MarineCadastre API."""
        lon_min, lat_min, lon_max, lat_max = bbox

        params = {
            "username": self.api_key,
            "format": 1,  # JSON
            "output": "json",
            "compress": 0,
            "latmin": lat_min,
            "latmax": lat_max,
            "lonmin": lon_min,
            "lonmax": lon_max,
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(MARINECADASTRE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                elif isinstance(data, list):
                    return data
                return []

        except Exception as e:
            logger.warning("MarineCadastre fetch failed, trying AISHub: {}", str(e))
            return await self._fetch_aishub(bbox, lookback_hours)

    async def _fetch_aishub(
        self,
        bbox: tuple[float, float, float, float],
        lookback_hours: int,
    ) -> list[dict]:
        """Fallback: fetch from AISHub API."""
        lon_min, lat_min, lon_max, lat_max = bbox

        params = {
            "api_key": self.api_key,
            "format": "json",
            "output": "json",
            "latmin": lat_min,
            "latmax": lat_max,
            "lonmin": lon_min,
            "lonmax": lon_max,
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(AISHUB_API_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, dict) and "data" in data:
                    return data["data"]
                elif isinstance(data, list):
                    return data
                return []
        except Exception as e:
            logger.error("AISHub fetch also failed: {}", str(e))
            return []

    def _normalize_position(self, raw: dict) -> AisPosition | None:
        """Normalize a raw AIS position report.

        Handles different field naming conventions from various AIS providers.
        """
        try:
            mmsi = str(raw.get("MMSI") or raw.get("mmsi", "")).strip()
            if not mmsi or len(mmsi) != 9:
                return None

            # Parse timestamp
            ts_raw = raw.get("TIMESTAMP") or raw.get("timestamp") or raw.get("time", "")
            if isinstance(ts_raw, (int, float)):
                ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
            elif isinstance(ts_raw, str):
                for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]:
                    try:
                        ts = datetime.strptime(ts_raw, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                else:
                    return None
            else:
                return None

            lon = float(raw.get("LONGITUDE") or raw.get("lon") or raw.get("longitude", 0))
            lat = float(raw.get("LATITUDE") or raw.get("lat") or raw.get("latitude", 0))

            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                return None

            return AisPosition(
                mmsi=mmsi,
                timestamp=ts,
                lon=lon,
                lat=lat,
                sog=self._safe_float(raw.get("SPEED") or raw.get("sog")),
                cog=self._safe_float(raw.get("COURSE") or raw.get("cog")),
                heading=self._safe_int(raw.get("HEADING") or raw.get("heading")),
                nav_status=self._safe_int(raw.get("NAV_STATUS") or raw.get("nav_status")),
                vessel_name=self._safe_str(raw.get("NAME") or raw.get("vessel_name")),
                vessel_type=self._safe_int(raw.get("TYPE") or raw.get("vessel_type")),
                imo_number=self._safe_str(raw.get("IMO") or raw.get("imo_number")),
                flag_state=self._safe_str(raw.get("FLAG") or raw.get("flag_state")),
                source="marinecadastre",
            )
        except Exception as e:
            logger.debug("Failed to normalize AIS record: {}", str(e))
            return None

    async def _bulk_upsert(self, positions: list[AisPosition]) -> int:
        """Bulk upsert AIS positions into PostGIS with ON CONFLICT."""
        if not self.pool or not positions:
            return 0

        async with self.pool.acquire() as conn:
            rows = [
                (
                    p.mmsi,
                    p.vessel_name,
                    p.vessel_type,
                    p.lon,
                    p.lat,
                    p.sog,
                    p.cog,
                    p.heading,
                    p.nav_status,
                    p.imo_number,
                    p.flag_state,
                    p.timestamp,
                    f"SRID=4326;POINT({p.lon} {p.lat})",
                    p.source,
                    p.is_interpolated,
                )
                for p in positions
            ]

            result = await conn.executemany(
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
                    source = EXCLUDED.source
                """,
                rows,
            )

            return len(rows)

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
