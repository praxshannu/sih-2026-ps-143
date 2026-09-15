"""Seed AIS rows for the demo: real MarineCadastre first, synthetic only if forced.

History
-------
This script used to download a MarineCadastre sample and, when that failed,
silently generate 55 fabricated vessel tracks — 8 MMSIs x exactly 200 rows at a
perfect 600 s cadence in the shipped CSV, which is how `data/ais_demo.csv` was
shown to be synthetic (VERIFICATION §4). Fabricated rows in the same table as
real ones are indistinguishable downstream, so the fallback is now gated.

Rules
-----
1. Real data wins. The MarineCadastre download is attempted first and used if
   it works.
2. If it fails, the script **exits non-zero** unless all of these hold:
   * ``SENTINEL_DEMO_MODE=true``
   * ``ALLOW_SYNTHETIC_AIS=true``
   * the demo AOI sits in ``NO_REAL_COVERAGE_BOXES`` and outside
     ``KNOWN_COASTAL_COVERAGE_BOXES`` (decided by
     ``ais_synthetic.should_use_synthetic`` — reused, never re-implemented)
3. Every synthetic row is stamped ``source='synthetic_mock'`` so it is labelled
   in the database itself, not only in the API response.

Usage:
    python scripts/seed_ais_demo.py
"""

from __future__ import annotations

import asyncio
import csv
import io
import math
import random
import sys
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import asyncpg
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.ingest.app.provenance import (  # noqa: E402
    AIS_GATES,
    PROVENANCE_SYNTHETIC,
    env_flag,
)
from services.ingest.app.sources.ais_synthetic import (  # noqa: E402
    should_use_synthetic,
)

DB_URL = "postgresql://sentinel:sentinel_secret@db:5432/sentinel"

MARINECADASTRE_SAMPLE_URL = "https://coast.noaa.gov/htdata/CSV/AIS/2020/AIS_2020_07_07.zip"

# The AOI synthetic tracks are generated for. It must be a region with no real
# receiver coverage; `should_use_synthetic` is the authority on that.
DEMO_BBOX = (57.6, -21.0, 58.2, -20.4)  # Wakashio zone, SE of Mauritius

VESSEL_PROFILES: list[dict[str, Any]] = [
    {"type": 70, "name": "Bulk Carrier", "speed_range": (10.0, 15.0), "weight": 8},
    {"type": 80, "name": "Container Ship", "speed_range": (12.0, 22.0), "weight": 6},
    {"type": 82, "name": "Container Ship", "speed_range": (14.0, 24.0), "weight": 5},
    {"type": 30, "name": "Fishing", "speed_range": (3.0, 8.0), "weight": 12},
    {"type": 31, "name": "Towing", "speed_range": (2.0, 6.0), "weight": 3},
    {"type": 32, "name": "Towing", "speed_range": (2.0, 5.0), "weight": 2},
    {"type": 60, "name": "Passenger", "speed_range": (14.0, 20.0), "weight": 2},
    {"type": 83, "name": "Container Ship", "speed_range": (16.0, 25.0), "weight": 3},
    {"type": 90, "name": "Tanker", "speed_range": (11.0, 16.0), "weight": 7},
    {"type": 92, "name": "Chemical Tanker", "speed_range": (12.0, 17.0), "weight": 4},
]

FLAGS: list[str] = [
    "HKG",
    "PAN",
    "LBR",
    "MAR",
    "SGP",
    "NOR",
    "GBR",
    "JPN",
    "KOR",
    "IND",
    "PA",
    "MHL",
    "GRC",
    "CHN",
    "RUS",
]

Indian_Ocean_ROUTES: list[dict[str, Any]] = [
    {"name": "Strait of Malacca Exit", "center": (2.0, 100.0), "heading": 270.0, "spread": 5.0},
    {"name": "Mozambique Channel N", "center": (-12.0, 44.0), "heading": 190.0, "spread": 3.0},
    {"name": "Mozambique Channel S", "center": (-20.0, 38.0), "heading": 200.0, "spread": 4.0},
    {"name": "Suez to Mumbai", "center": (12.0, 62.0), "heading": 120.0, "spread": 8.0},
    {"name": "Cape Route", "center": (-33.0, 30.0), "heading": 60.0, "spread": 10.0},
    {"name": "Bay of Bengal", "center": (12.0, 88.0), "heading": 150.0, "spread": 6.0},
    {"name": "Arabian Sea", "center": (15.0, 65.0), "heading": 200.0, "spread": 7.0},
    {"name": "Mauritius EEZ", "center": (-20.4, 57.7), "heading": 180.0, "spread": 4.0},
]


def generate_vessel(mmsi_base: int, idx: int) -> dict[str, Any]:
    profile = random.choices(VESSEL_PROFILES, weights=[p["weight"] for p in VESSEL_PROFILES])[0]
    route = random.choice(Indian_Ocean_ROUTES)
    flag = random.choice(FLAGS)
    name_parts = [
        random.choice(
            [
                "STAR",
                "OCEAN",
                "SEA",
                "WAVE",
                "SPIRIT",
                "LUCKY",
                "GOLDEN",
                "SILVER",
                "NORTH",
                "SOUTH",
            ]
        ),
        random.choice(
            [
                "BREEZE",
                "DRAGON",
                "HAWK",
                "WOLF",
                "TIGER",
                "EAGLE",
                "PEARL",
                "QUEEN",
                "PRINCE",
                "KING",
            ]
        ),
    ]
    return {
        "mmsi": str(mmsi_base + idx),
        "vessel_name": f"{' '.join(random.sample(name_parts, 2))}",
        "vessel_type": profile["type"],
        "vessel_type_name": profile["name"],
        "flag_state": flag,
        "imo_number": f"9{random.randint(1000000, 9999999)}",
        "speed_range": profile["speed_range"],
        "route_center": route["center"],
        "route_heading": route["heading"] + random.uniform(-route["spread"], route["spread"]),
        "route_spread": route["spread"],
    }


def generate_track(
    vessel: dict,
    base_time: datetime,
    n_positions: int = 100,
    include_gap: bool = False,
) -> list[dict]:
    positions = []
    sog = random.uniform(*vessel["speed_range"])
    cog = vessel["route_heading"]
    center_lat, center_lon = vessel["route_center"]

    lat = center_lat + random.uniform(-2.0, 2.0)
    lon = center_lon + random.uniform(-3.0, 3.0)

    gap_start = random.randint(n_positions // 4, 3 * n_positions // 4) if include_gap else -1
    gap_length = random.randint(3, 8) if include_gap else 0

    for i in range(n_positions):
        if gap_start <= i < gap_start + gap_length:
            continue

        t = base_time - timedelta(hours=n_positions - i)

        cog_var = cog + random.uniform(-5.0, 5.0)
        cog_rad = math.radians(cog_var)

        speed = sog + random.uniform(-1.0, 1.0)
        speed = max(0.5, speed)

        d_lat = speed * 0.008333 * math.cos(cog_rad)
        d_lon = speed * 0.008333 * math.sin(cog_rad) / max(0.01, math.cos(math.radians(lat)))

        lat += d_lat * 0.05 + random.uniform(-0.002, 0.002)
        lon += d_lon * 0.05 + random.uniform(-0.002, 0.002)

        nav_status = random.choices([0, 1, 5, 8, 15], weights=[70, 5, 5, 5, 15])[0]

        positions.append(
            {
                "mmsi": vessel["mmsi"],
                "vessel_name": vessel["vessel_name"],
                "vessel_type": vessel["vessel_type"],
                "lon": round(lon, 6),
                "lat": round(lat, 6),
                "sog": round(speed, 1),
                "cog": round(cog_var % 360, 1),
                "heading": int(cog_var) % 360,
                "nav_status": nav_status,
                "imo_number": vessel["imo_number"],
                "flag_state": vessel["flag_state"],
                "timestamp": t,
                # Stamped in the row itself: a synthetic track stays labelled
                # even after it has been copied into another table or CSV.
                "source": PROVENANCE_SYNTHETIC,
            }
        )
    return positions


async def try_download_sample() -> str | None:
    logger.info("Attempting MarineCadastre download: {}", MARINECADASTRE_SAMPLE_URL)
    try:
        req = urllib.request.Request(
            MARINECADASTRE_SAMPLE_URL,
            headers={"User-Agent": "SENTINEL/1.0 (research)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            logger.info("Downloaded {} bytes", len(data))
            return data.decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("MarineCadastre download failed: {}", e)
        return None


async def insert_positions(
    conn: asyncpg.Connection, positions: list[dict], batch_size: int = 500
) -> int:
    total = 0
    for i in range(0, len(positions), batch_size):
        batch = positions[i : i + batch_size]
        records = []
        for p in batch:
            records.append(
                (
                    p["mmsi"],
                    p["vessel_name"],
                    p["vessel_type"],
                    p["lon"],
                    p["lat"],
                    p["sog"],
                    p["cog"],
                    p["heading"],
                    p["nav_status"],
                    p["imo_number"],
                    p["flag_state"],
                    p["timestamp"],
                    p["source"],
                )
            )
        await conn.executemany(
            """INSERT INTO ais_positions (
                mmsi, vessel_name, vessel_type, lon, lat, sog, cog, heading,
                nav_status, imo_number, flag_state, timestamp, geom, source
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                ST_SetSRID(ST_MakePoint($4, $5), 4326), $13
            )""",
            records,
        )
        total += len(batch)
        logger.info("Inserted {}/{} positions", total, len(positions))
    return total


def _assert_synthetic_permitted() -> None:
    """Refuse to fabricate vessel tracks unless every gate is open."""
    if not should_use_synthetic(DEMO_BBOX):
        raise SystemExit(
            f"refusing synthetic AIS: DEMO_BBOX={DEMO_BBOX} is not a no-coverage region "
            "(it overlaps a coastal carve-out or lies outside NO_REAL_COVERAGE_BOXES)"
        )
    missing = [name for name in AIS_GATES if not env_flag(name)]
    if missing:
        raise SystemExit(
            f"refusing synthetic AIS: MarineCadastre is unavailable and "
            f"{', '.join(missing)} are not set. SENTINEL does not fabricate vessel "
            f"tracks silently — set {' and '.join(AIS_GATES)} for a labelled demo."
        )


async def seed_synthetic(conn: asyncpg.Connection) -> None:
    _assert_synthetic_permitted()
    logger.warning(
        "SYNTHETIC AIS: generating fabricated tracks ({}). Not real data — "
        "must not be used as evidence.",
        ", ".join(AIS_GATES),
    )
    base_time = datetime(2020, 7, 25, 10, 0, tzinfo=UTC)
    n_vessels = 55
    mmsi_base = 200000000

    all_positions = []
    vessels = [generate_vessel(mmsi_base, i) for i in range(n_vessels)]

    for vessel in vessels:
        include_gap = random.random() < 0.15
        n_pos = random.randint(60, 150)
        track = generate_track(vessel, base_time, n_positions=n_pos, include_gap=include_gap)
        all_positions.extend(track)

    random.shuffle(all_positions)
    await insert_positions(conn, all_positions)


async def seed_from_csv(conn: asyncpg.Connection, csv_text: str) -> None:
    reader = csv.DictReader(io.StringIO(csv_text))
    positions = []
    for row in reader:
        try:
            mmsi = row.get("MMSI", "")
            if not mmsi:
                continue
            positions.append(
                {
                    "mmsi": mmsi,
                    "vessel_name": row.get("VesselName", "").strip()[:100] or f"VESSEL_{mmsi}",
                    "vessel_type": int(float(row.get("VesselType", 0) or 0)),
                    "lon": float(row.get("LON", 0)),
                    "lat": float(row.get("LAT", 0)),
                    "sog": float(row.get("SOG", 0) or 0),
                    "cog": float(row.get("COG", 0) or 0),
                    "heading": int(float(row.get("Heading", 0) or 0)),
                    "nav_status": int(float(row.get("Status", 0) or 0)),
                    "imo_number": row.get("IMO", ""),
                    "flag_state": row.get("Flag", ""),
                    "timestamp": datetime.fromisoformat(
                        row.get("BaseDateTime", "2020-07-07T00:00:00").replace("Z", "+00:00")
                    ),
                    "source": "marinecadastre",
                }
            )
        except (ValueError, KeyError):
            continue

    if positions:
        await insert_positions(conn, positions)


async def main() -> None:
    conn = await asyncpg.connect(DB_URL)
    try:
        csv_text = await try_download_sample()
        if csv_text and len(csv_text) > 1000:
            await seed_from_csv(conn, csv_text)
        else:
            await seed_synthetic(conn)
        count = await conn.fetchval("SELECT COUNT(*) FROM ais_positions")
        logger.info("Total AIS positions in database: {}", count)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
