"""MarineCadastre / AISHub AIS data ingestion.

Fetches recent vessel position data, normalizes fields,
and inserts into PostGIS database.

FAIL-CLOSED CONTRACT
--------------------
There is no free AIS source for the open Indian Ocean (VERIFICATION §2.7, §6),
and SENTINEL does not fill that hole with invented ships. When no real feed is
reachable this module returns an explicit ``no_real_coverage`` state with zero
vessels — never a fabricated track.

Synthetic tracks are produced *only* when all four conditions hold:

1. ``SENTINEL_DEMO_MODE=true``
2. ``ALLOW_SYNTHETIC_AIS=true``
3. the AOI overlaps ``NO_REAL_COVERAGE_BOXES``
4. the AOI does not overlap ``KNOWN_COASTAL_COVERAGE_BOXES``

Conditions 3 and 4 are decided by ``ais_synthetic.should_use_synthetic()`` —
reused, never re-implemented — and 1 and 2 by ``provenance.decide_synthetic()``,
the same gate CMEMS and ERA5 use.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import httpx
from loguru import logger

from ..models.schemas import AisIngestResult, AisPosition
from ..provenance import (
    AIS_GATES,
    PROVENANCE_NO_COVERAGE,
    PROVENANCE_SYNTHETIC,
    bbox_overlap_fraction,
    coverage_report,
    decide_synthetic,
    provenance_envelope,
)
from .ais_synthetic import (
    KNOWN_COASTAL_COVERAGE_BOXES,
    NO_REAL_COVERAGE_BOXES,
    generate_synthetic_ais,
    should_use_synthetic,
    synthetic_disclaimer,
)

# Indian EEZ bounding box
INDIA_EEZ_BBOX = (68.0, 5.0, 88.0, 25.0)

MARINECADASTRE_URL = "https://data.aishub.net/ws.php"
AISHUB_API_URL = "https://data.aishub.net/api/positions"

PROVENANCE_LIVE = "live_terrestrial"


def ais_coverage_block(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """Geographic AIS coverage for an AOI, honestly graded.

    ``full`` means receivers are expected everywhere. ``none`` means the whole
    AOI sits in the no-coverage region. ``partial`` means the AOI **straddles a
    source boundary** — that case is never resolved by blending: the caller
    returns real data for the covered part and names the rest.
    """
    open_report = coverage_report(bbox, NO_REAL_COVERAGE_BOXES[0], "no_real_ais_coverage")
    open_fraction = float(open_report["covered_fraction"])
    coastal = [
        coverage_report(bbox, box, "known_coastal_receivers")
        for box in KNOWN_COASTAL_COVERAGE_BOXES
        if bbox_overlap_fraction(bbox, box) > 0.0
    ]
    if coastal:
        status = "partial" if open_fraction > 0.0 else "full"
    elif open_fraction <= 0.0:
        status = "full"
    elif open_fraction >= 1.0:
        status = "none"
    else:
        status = "partial"
    return {
        "status": status,
        "open_ocean_fraction": round(open_fraction, 6),
        "coastal_carveouts": coastal,
        "no_coverage_box": open_report,
        "partial_note": (
            "AOI straddles an AIS coverage boundary. Real vessels are returned "
            "for the covered part only; the remainder is reported as "
            "no-coverage and is never filled with synthetic tracks."
            if status == "partial"
            else None
        ),
    }


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

        try:
            # Fetch from MarineCadastre (which falls back to AISHub).
            raw_positions = await self._fetch_marinecadastre(bbox, lookback_hours)
        except Exception as exc:
            # No real feed: report the gap. SENTINEL never fabricates a track.
            elapsed = time.time() - t0
            logger.error(
                "AIS unavailable for bbox={} — reporting no_real_coverage, no vessels invented",
                bbox,
            )
            return AisIngestResult(
                records_fetched=0,
                records_upserted=0,
                unique_vessels=0,
                interpolated_gaps=0,
                fetch_duration_seconds=round(elapsed, 2),
                provenance=PROVENANCE_NO_COVERAGE,
                is_synthetic=False,
                synthetic_reason="real_ais_unavailable",
                warning=(
                    "No real AIS coverage for this AOI: "
                    f"{type(exc).__name__}: {exc}. "
                    "SENTINEL does not fabricate vessel tracks — zero vessels returned."
                ),
                coverage=ais_coverage_block(bbox),
            )

        # Normalize
        normalized = [p for p in (self._normalize_position(r) for r in raw_positions) if p]

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
            provenance=PROVENANCE_LIVE,
            is_synthetic=False,
            coverage=ais_coverage_block(bbox),
        )

    async def fetch_envelope(
        self,
        bbox: tuple[float, float, float, float] = INDIA_EEZ_BBOX,
        start: datetime | None = None,
        end: datetime | None = None,
        n_vessels: int = 6,
        seed: int = 20200725,
    ) -> dict[str, Any]:
        """Vessel data for an AOI with an explicit, machine-readable provenance.

        This is the contract the archive browser serves. It never mixes real and
        synthetic rows in one list: the payload is either entirely real
        (``live_terrestrial``), entirely synthetic (``synthetic_mock``, only
        behind every gate), or an explicit ``no_real_coverage`` state with zero
        vessels.
        """
        end_dt = end or datetime.now(UTC)
        start_dt = start or (end_dt - timedelta(hours=6))
        coverage = ais_coverage_block(bbox)

        if should_use_synthetic(bbox):
            # No receiver network exists here at all, so a live call would be
            # pointless. Either the operator has explicitly opted into synthetic
            # demo data, or they get an honest empty state.
            decision = decide_synthetic(
                "ais",
                required_env=AIS_GATES,
                geo_allowed=coverage["status"] == "none",
                geo_reason=(
                    "aoi_partially_covered"
                    if coverage["status"] == "partial"
                    else "aoi_outside_no_coverage_box"
                ),
            )
            if not decision.allowed:
                logger.warning(
                    "AIS: no real coverage for bbox={} and synthetic AIS is NOT enabled ({}) — "
                    "returning zero vessels",
                    bbox,
                    decision.reason,
                )
                return provenance_envelope(
                    "ais",
                    PROVENANCE_NO_COVERAGE,
                    error={
                        "reason": decision.reason,
                        "required_env": list(AIS_GATES),
                        "detail": (
                            "No free AIS source covers this AOI, and synthetic tracks are "
                            "disabled. Set SENTINEL_DEMO_MODE and ALLOW_SYNTHETIC_AIS for an "
                            "explicitly labelled demo."
                        ),
                    },
                    coverage=coverage,
                    extra={
                        "bbox": list(bbox),
                        "window": [
                            start_dt.isoformat().replace("+00:00", "Z"),
                            end_dt.isoformat().replace("+00:00", "Z"),
                        ],
                        "count": 0,
                        "vessels": [],
                        "notice": None,
                        "disclaimer": None,
                    },
                )

            payload = generate_synthetic_ais(
                bbox, start_dt, end_dt, n_vessels=n_vessels, seed=seed
            )
            payload["source"] = "ais"
            payload["coverage"] = coverage
            payload["disclaimer"] = synthetic_disclaimer()
            logger.warning(
                "Serving SYNTHETIC AIS for bbox={} window={}..{} — no real coverage exists",
                bbox,
                start_dt.isoformat(),
                end_dt.isoformat(),
            )
            return payload

        # Real coverage exists for at least part of the AOI: live only.
        lookback_hours = max(1, int((end_dt - start_dt).total_seconds() // 3600) or 1)
        try:
            raw = await self._fetch_marinecadastre(bbox, lookback_hours)
        except Exception as exc:
            logger.error(
                "AIS live feed failed for bbox={} ({}); returning no_real_coverage, "
                "not synthetic tracks",
                bbox,
                exc,
            )
            return provenance_envelope(
                "ais",
                PROVENANCE_NO_COVERAGE,
                error={
                    "reason": "real_ais_unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
                coverage=coverage,
                extra={
                    "bbox": list(bbox),
                    "window": [
                        start_dt.isoformat().replace("+00:00", "Z"),
                        end_dt.isoformat().replace("+00:00", "Z"),
                    ],
                    "count": 0,
                    "vessels": [],
                },
            )

        vessels = self._group_vessels(
            [p for p in (self._normalize_position(r) for r in raw) if p]
        )
        return provenance_envelope(
            "ais",
            PROVENANCE_LIVE,
            coverage=coverage,
            extra={
                "bbox": list(bbox),
                "window": [
                    start_dt.isoformat().replace("+00:00", "Z"),
                    end_dt.isoformat().replace("+00:00", "Z"),
                ],
                "count": len(vessels),
                "vessels": vessels,
                "reason": (
                    "Live terrestrial AIS. Coverage reported per AOI; a partial AOI returns "
                    "only the covered part."
                    if coverage["status"] != "partial"
                    else coverage["partial_note"]
                ),
            },
        )

    @staticmethod
    def _group_vessels(positions: list[AisPosition]) -> list[dict[str, Any]]:
        """Group normalized fixes into one entry per vessel.

        The shape matches ``ais_synthetic.generate_synthetic_ais`` so the UI can
        render either without branching — but the two are never merged into the
        same list.
        """
        by_mmsi: dict[str, list[AisPosition]] = {}
        for p in positions:
            by_mmsi.setdefault(p.mmsi, []).append(p)

        vessels: list[dict[str, Any]] = []
        for mmsi, fixes in by_mmsi.items():
            ordered = sorted(fixes, key=lambda p: p.timestamp)
            head = ordered[0]
            vessels.append(
                {
                    "mmsi": mmsi,
                    "name": head.vessel_name,
                    "vessel_type": head.vessel_type,
                    "flag": head.flag_state,
                    "imo": head.imo_number,
                    "provenance": PROVENANCE_LIVE,
                    "track": [
                        {
                            "timestamp": p.timestamp.isoformat(),
                            "longitude": p.lon,
                            "latitude": p.lat,
                            "sog": p.sog,
                            "cog": p.cog,
                            "heading": p.heading,
                            "nav_status": p.nav_status,
                            "source": p.source,
                            "provenance": PROVENANCE_LIVE,
                        }
                        for p in ordered
                    ],
                    "gaps": [],
                }
            )
        return vessels

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
        """Fallback: fetch from AISHub API.

        Raises on failure. Returning ``[]`` here is what made a dead feed look
        like "no vessels in the area" — the caller must be able to tell those
        two states apart, so the error propagates.
        """
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
        except Exception as e:
            logger.error("AISHub fetch also failed: {}", str(e))
            raise

        if isinstance(data, dict) and "data" in data:
            rows = data["data"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"unrecognised AIS payload from AISHub: {type(data).__name__}")
        if not isinstance(rows, list):
            raise ValueError(f"AISHub 'data' is {type(rows).__name__}, expected a list")
        return rows

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
                ts = datetime.fromtimestamp(ts_raw, tz=UTC)
            elif isinstance(ts_raw, str):
                for fmt in [
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                ]:
                    try:
                        ts = datetime.strptime(ts_raw, fmt).replace(tzinfo=UTC)
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
                    source = EXCLUDED.source
                """,
                rows,
            )

            return len(rows)

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(val: Any) -> int | None:
        if val is None:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_str(val: Any) -> str | None:
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None
