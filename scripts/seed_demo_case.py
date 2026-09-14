"""Seed the MV Wakashio (Mauritius, July 2020) demo case.

Inserts a complete investigation case with:
- Spill detection record (20.4S, 57.7E)
- Drift result with origin ellipse
- 5 suspect vessels (primary: MMSI 477218700, 91.4% confidence)
- Investigation case with narrative
- ~200 synthetic AIS positions around the incident

Usage:
    python scripts/seed_demo_case.py
    docker compose exec sentinel-api python scripts/seed_demo_case.py
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import uuid
from datetime import UTC, datetime, timedelta

import asyncpg

DB_URL = "postgresql://sentinel:sentinel_secret@db:5432/sentinel"

SPILL_CENTER_LAT = -20.4
SPILL_CENTER_LON = 57.7
SPILL_DETECTED_AT = datetime(2020, 7, 25, 10, 30, tzinfo=UTC)
SPILL_ID = uuid.uuid4()
DRIFT_ID = uuid.uuid4()
CASE_ID = uuid.uuid4()

SUSPECTS = [
    {
        "mmsi": "477218700",
        "vessel_name": "MV WAKASHIO",
        "flag_state": "HKG",
        "vessel_type": 70,
        "imo_number": "9811000",
        "rank": 1,
        "composite_score": 0.914,
        "score_proximity": 0.95,
        "score_temporal": 0.88,
        "score_trajectory": 0.92,
        "score_anomaly": 0.94,
        "score_vessel_type": 0.87,
        "ais_gap_minutes": 23.0,
        "min_distance_nm": 0.8,
        "is_dark_vessel": False,
        "confidence_lower": 0.87,
        "confidence_upper": 0.95,
    },
    {
        "mmsi": "636019234",
        "vessel_name": "PACIFIC EXPLORER",
        "flag_state": "PA",
        "vessel_type": 80,
        "imo_number": "9765432",
        "rank": 2,
        "composite_score": 0.312,
        "score_proximity": 0.45,
        "score_temporal": 0.28,
        "score_trajectory": 0.33,
        "score_anomaly": 0.22,
        "score_vessel_type": 0.27,
        "ais_gap_minutes": 2.0,
        "min_distance_nm": 8.4,
        "is_dark_vessel": False,
        "confidence_lower": 0.21,
        "confidence_upper": 0.41,
    },
    {
        "mmsi": "538006789",
        "vessel_name": "NAKAMA STAR",
        "flag_state": "MH",
        "vessel_type": 70,
        "imo_number": "9432187",
        "rank": 3,
        "composite_score": 0.198,
        "score_proximity": 0.32,
        "score_temporal": 0.18,
        "score_trajectory": 0.21,
        "score_anomaly": 0.15,
        "score_vessel_type": 0.13,
        "ais_gap_minutes": 0.0,
        "min_distance_nm": 15.2,
        "is_dark_vessel": False,
        "confidence_lower": 0.11,
        "confidence_upper": 0.29,
    },
    {
        "mmsi": "219014852",
        "vessel_name": "RED STAR 7",
        "flag_state": "DK",
        "vessel_type": 82,
        "imo_number": "9187654",
        "rank": 4,
        "composite_score": 0.087,
        "score_proximity": 0.12,
        "score_temporal": 0.09,
        "score_trajectory": 0.07,
        "score_anomaly": 0.06,
        "score_vessel_type": 0.10,
        "ais_gap_minutes": 1.0,
        "min_distance_nm": 42.7,
        "is_dark_vessel": False,
        "confidence_lower": 0.04,
        "confidence_upper": 0.13,
    },
    {
        "mmsi": "503045600",
        "vessel_name": "SOUTHERN LADY",
        "flag_state": "AUS",
        "vessel_type": 30,
        "imo_number": "8765432",
        "rank": 5,
        "composite_score": 0.041,
        "score_proximity": 0.05,
        "score_temporal": 0.04,
        "score_trajectory": 0.03,
        "score_anomaly": 0.02,
        "score_vessel_type": 0.06,
        "ais_gap_minutes": 0.0,
        "min_distance_nm": 78.1,
        "is_dark_vessel": False,
        "confidence_lower": 0.01,
        "confidence_upper": 0.07,
    },
]


def _make_spill_polygon(center_lat: float, center_lon: float) -> str:
    coords = []
    for i in range(24):
        angle = 2 * math.pi * i / 24
        r_lat = 0.015 * math.cos(angle)
        r_lon = 0.006 * math.sin(angle)
        noise = random.uniform(0.92, 1.08)
        coords.append(f"{center_lon + r_lon * noise:.6f} {center_lat + r_lat * noise:.6f}")
    coords.append(coords[0])
    return f"SRID=4326;POLYGON(({', '.join(coords)}))"


def _make_origin_ellipse(lat: float, lon: float) -> str:
    coords = []
    for i in range(32):
        angle = 2 * math.pi * i / 32
        r_lat = 0.08 * math.cos(angle)
        r_lon = 0.12 * math.sin(angle)
        coords.append(f"{lon + r_lon:.6f} {lat + r_lat:.6f}")
    coords.append(coords[0])
    return f"SRID=4326;POLYGON(({', '.join(coords)}))"


def _make_track_segment(
    center_lat: float, center_lon: float, heading_deg: float, length_nm: float = 5.0
) -> str:
    heading_rad = math.radians(heading_deg)
    coords = []
    for i in range(8):
        t = i / 7.0
        d_lat = t * length_nm * 0.008333 * math.cos(heading_rad)
        d_lon = (
            t * length_nm * 0.008333 * math.sin(heading_rad) / math.cos(math.radians(center_lat))
        )
        coords.append(f"{center_lon + d_lon:.6f} {center_lat + d_lat:.6f}")
    return f"SRID=4326;LINESTRING({', '.join(coords)})"


def _make_ais_positions(
    mmsi: str,
    vessel_name: str,
    vessel_type: int,
    flag_state: str,
    imo_number: str,
    center_lat: float,
    center_lon: float,
    base_time: datetime,
    n_positions: int = 40,
    include_gap: bool = False,
    gap_minutes: int = 23,
) -> list[dict]:
    positions = []
    sog = random.uniform(8.0, 14.0)
    cog = random.uniform(170.0, 190.0)

    for i in range(n_positions):
        t = base_time - timedelta(hours=n_positions - i)
        if include_gap and n_positions // 2 <= i < n_positions // 2 + gap_minutes // 5:
            continue
        progress = i / n_positions
        cog_var = cog + random.uniform(-3.0, 3.0)
        cog_rad = math.radians(cog_var)

        d_lat = sog * 0.008333 * math.cos(cog_rad)
        d_lon = sog * 0.008333 * math.sin(cog_rad) / math.cos(math.radians(center_lat))

        lat = center_lat + d_lat * progress + random.uniform(-0.001, 0.001)
        lon = center_lon + d_lon * progress + random.uniform(-0.001, 0.001)

        positions.append(
            {
                "mmsi": mmsi,
                "vessel_name": vessel_name,
                "vessel_type": vessel_type,
                "lon": lon,
                "lat": lat,
                "sog": round(sog + random.uniform(-0.5, 0.5), 1),
                "cog": round(cog_var, 1),
                "heading": int(cog_var) % 360,
                "nav_status": 0,
                "imo_number": imo_number,
                "flag_state": flag_state,
                "timestamp": t,
                "source": "synthetic_demo",
            }
        )
    return positions


async def seed(conn: asyncpg.Connection) -> None:
    print(f"[seed] Spill detection {SPILL_ID}...")

    await conn.execute(
        """INSERT INTO spill_detections (
            id, detected_at, satellite_pass, polygon, area_km2, perimeter_km,
            centroid, orientation_deg, age_hours_est, confidence, lookalike_prob,
            wind_speed_ms, wind_dir_deg, current_speed, current_dir_deg,
            sar_image_path, metadata
        ) VALUES (
            $1, $2, $3, ST_GeomFromEWKT($4), $5, $6,
            ST_SetSRID(ST_MakePoint($7, $8), 4326), $9, $10, $11, $12,
            $13, $14, $15, $16, $17, $18
        )""",
        SPILL_ID,
        SPILL_DETECTED_AT,
        "S1A_IW_GRDH_1SDV_20200725T103025",
        _make_spill_polygon(SPILL_CENTER_LAT, SPILL_CENTER_LON),
        27.0,
        18.4,
        SPILL_CENTER_LON,
        SPILL_CENTER_LAT,
        45.0,
        12.0,
        0.94,
        0.02,
        6.2,
        135.0,
        0.35,
        220.0,
        "/data/sar/s1a_20200725_mauritius.tif",
        json.dumps(
            {
                "satellite": "Sentinel-1A",
                "mode": "IW GRDH",
                "polarization": "VV+VH",
                "incidence_angle": 38.2,
                "region": "Indian Ocean - Mauritius",
            }
        ),
    )

    print("[seed] Drift result...")
    await conn.execute(
        """INSERT INTO drift_results (
            id, spill_id, origin_lon, origin_lat, origin_ellipse,
            t_origin_est, t_origin_minus, t_origin_plus,
            confidence_level, particle_count, k_regime, k_tensor_trace,
            forecast_paths, shoreline_risk
        ) VALUES (
            $1, $2, $3, $4, ST_GeomFromEWKT($5),
            $6, $7, $8, $9, $10, $11, $12, $13, $14
        )""",
        DRIFT_ID,
        SPILL_ID,
        SPILL_CENTER_LON,
        SPILL_CENTER_LAT,
        _make_origin_ellipse(SPILL_CENTER_LAT, SPILL_CENTER_LON),
        SPILL_DETECTED_AT - timedelta(hours=12),
        timedelta(minutes=18),
        timedelta(minutes=18),
        0.95,
        1000,
        "tidal",
        0.34,
        json.dumps(
            {
                "paths": [
                    {"hours": 24, "lat": -20.35, "lon": 57.75, "spread_km": 2.1},
                    {"hours": 48, "lat": -20.28, "lon": 57.82, "spread_km": 4.8},
                    {"hours": 72, "lat": -20.21, "lon": 57.91, "spread_km": 8.3},
                ]
            }
        ),
        json.dumps(
            {"rodrigues": 0.12, "reunion": 0.03, "mauritius_east": 0.87, "mauritius_south": 0.64}
        ),
    )

    print("[seed] Suspect vessels...")
    suspect_ids = []
    for suspect in SUSPECTS:
        sid = uuid.uuid4()
        suspect_ids.append(sid)
        track_wkt = _make_track_segment(
            SPILL_CENTER_LAT + random.uniform(-0.3, 0.3),
            SPILL_CENTER_LON + random.uniform(-0.5, 0.5),
            heading_deg=random.uniform(170.0, 190.0),
            length_nm=5.0,
        )
        anomalies = []
        if suspect["ais_gap_minutes"] > 10:
            anomalies.append({"type": "ais_gap", "duration_min": suspect["ais_gap_minutes"]})
        if suspect["score_anomaly"] > 0.8:
            anomalies.append({"type": "speed_anomaly", "detail": "speed drop near origin"})

        await conn.execute(
            """INSERT INTO suspect_vessels (
                id, case_id, mmsi, vessel_name, flag_state, vessel_type,
                imo_number, rank, composite_score, score_proximity, score_temporal,
                score_trajectory, score_anomaly, score_vessel_type,
                ais_gap_minutes, min_distance_nm, track_segment,
                anomalies, is_dark_vessel, confidence_lower, confidence_upper
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, ST_GeomFromEWKT($17), $18, $19, $20, $21
            )""",
            sid,
            CASE_ID,
            suspect["mmsi"],
            suspect["vessel_name"],
            suspect["flag_state"],
            suspect["vessel_type"],
            suspect["imo_number"],
            suspect["rank"],
            suspect["composite_score"],
            suspect["score_proximity"],
            suspect["score_temporal"],
            suspect["score_trajectory"],
            suspect["score_anomaly"],
            suspect["score_vessel_type"],
            suspect["ais_gap_minutes"],
            suspect["min_distance_nm"],
            track_wkt,
            json.dumps(anomalies),
            suspect["is_dark_vessel"],
            suspect["confidence_lower"],
            suspect["confidence_upper"],
        )

    primary_suspect_id = suspect_ids[0]

    print("[seed] Investigation case...")
    narrative = (
        "CASE 2020-MU-001: MV WAKASHIO GROUNDING & OIL SPILL\n\n"
        "On 25 July 2020 at approximately 10:30 UTC, Sentinel-1A SAR imagery "
        "revealed a significant oil slick southeast of Mauritius, centered at "
        "20.4S 57.7E. The slick covered approximately 27 km2 with an elongated "
        "morphology consistent with a point-source release under tidal forcing.\n\n"
        "BACKWARD DRIFT ANALYSIS\n"
        "Lagrangian particle tracking with CMEMS ocean currents and ERA5 winds "
        "retraced the slick origin to a 95% confidence ellipse encompassing the "
        "region 20.3-20.5S, 57.6-57.8E, with estimated release time 12+/-3 hours "
        "before satellite acquisition. The K_ij diffusion tensor trace of 0.34 m2/s "
        "indicates a tidal-dominant dispersion regime.\n\n"
        "VESSEL ATTRIBUTION\n"
        "AIS analysis identified 5 vessels within the 50 NM attribution radius during "
        "the temporal search window. MV WAKASHIO (MMSI 477218700, IMO 9811000), a "
        "Hong Kong-flagged bulk carrier, achieved a composite K_ij score of 91.4%, "
        "significantly exceeding all other candidates.\n\n"
        "Key attribution factors:\n"
        "  - Proximity score: 0.95 (track passed 0.8 NM from spill centroid)\n"
        "  - Temporal alignment: 0.88 (present during estimated release window)\n"
        "  - Trajectory match: 0.92 (course/heading consistent with slick orientation)\n"
        "  - Behavioral anomaly: 0.94 (23-minute AIS gap coinciding with origin)\n"
        "  - Vessel-type risk: 0.87 (bulk carrier with potential cargo residues)\n\n"
        "The 23-minute AIS gap (10:17-10:40 UTC) aligns precisely with the backward-"
        "projected origin time and location. This gap, combined with a simultaneous "
        "speed reduction from 11.2 to 3.4 knots, constitutes strong circumstantial "
        "evidence of a deliberate or accidental discharge.\n\n"
        "RECOMMENDATION\n"
        "Forward to IMO MEPC investigation. Coordinate with Mauritius Maritime "
        "Authorities and Hong Kong Marine Department for flag-state inquiry."
    )

    await conn.execute(
        """INSERT INTO investigation_cases (
            id, case_number, status, spill_id, drift_id, primary_suspect,
            narrative, evidence_hash, case_file_path, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
        CASE_ID,
        "2020-MU-001",
        "OPEN",
        SPILL_ID,
        DRIFT_ID,
        primary_suspect_id,
        narrative,
        "sha256:a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
        "/data/case_files/2020-MU-001.json",
        json.dumps({"priority": "HIGH", "region": "Indian Ocean", "flag_state": "MUR"}),
    )

    print("[seed] AIS positions (~200)...")
    all_positions = []
    base_time = SPILL_DETECTED_AT

    primary_positions = _make_ais_positions(
        "477218700",
        "MV WAKASHIO",
        70,
        "HKG",
        "9811000",
        SPILL_CENTER_LAT - 0.05,
        SPILL_CENTER_LON - 0.1,
        base_time,
        n_positions=50,
        include_gap=True,
        gap_minutes=23,
    )
    all_positions.extend(primary_positions)

    other_vessels = [
        ("636019234", "PACIFIC EXPLORER", 80, "PA", "9765432", -20.35, 57.55, 35),
        ("538006789", "NAKAMA STAR", 70, "MH", "9432187", -20.55, 57.65, 40),
        ("219014852", "RED STAR 7", 82, "DK", "9187654", -20.28, 57.82, 30),
        ("503045600", "SOUTHERN LADY", 30, "AUS", "8765432", -20.62, 57.90, 25),
    ]
    for mmsi, name, vtype, flag, imo, lat, lon, n in other_vessels:
        all_positions.extend(
            _make_ais_positions(mmsi, name, vtype, flag, imo, lat, lon, base_time, n_positions=n)
        )

    for pos in all_positions:
        await conn.execute(
            """INSERT INTO ais_positions (
                mmsi, vessel_name, vessel_type, lon, lat, sog, cog, heading,
                nav_status, imo_number, flag_state, timestamp, geom, source
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                ST_SetSRID(ST_MakePoint($4, $5), 4326), $13
            )""",
            pos["mmsi"],
            pos["vessel_name"],
            pos["vessel_type"],
            pos["lon"],
            pos["lat"],
            pos["sog"],
            pos["cog"],
            pos["heading"],
            pos["nav_status"],
            pos["imo_number"],
            pos["flag_state"],
            pos["timestamp"],
            pos["source"],
        )

    print(f"[seed] Done: {len(all_positions)} AIS positions, 5 suspects, case 2020-MU-001")


async def main() -> None:
    print("[seed] Connecting to database...")
    conn = await asyncpg.connect(DB_URL)
    try:
        await seed(conn)
    finally:
        await conn.close()
    print("[seed] Connection closed.")


if __name__ == "__main__":
    asyncio.run(main())
