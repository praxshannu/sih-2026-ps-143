"""SENTINEL Attribute Service - Vessel Attribution & Scoring Engine.

FastAPI application exposing the full attribution pipeline:
  AIS slicer -> dark vessel check -> anomaly detection -> fuzzy scoring -> ranked suspects
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

import asyncpg
from fastapi import FastAPI, HTTPException
from loguru import logger

from app.engine.ais_coverage import (
    REAL_AIS_PROVENANCE,
    SYNTHETIC_AIS_PROVENANCE,
    AoiError,
    CoverageVerdict,
    coverage_verdict,
    provenance_verdict,
    resolve_aoi,
)
from app.engine.ais_slicer import AisSlicer, VesselTrack
from app.engine.anomaly_detector import AnomalyDetector
from app.engine.dark_vessel import DarkVesselDetector
from app.engine.fuzzy_scorer import SuspectFeatures, score_suspect
from app.engine.suspect_ranker import SuspectRanker
from app.schemas import (
    AttribRequest,
    AttribResponse,
    HealthResponse,
    RankRequest,
    RankResponse,
    ScoreRequest,
    ScoreResult,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://sentinel:sentinel_secret@db:5432/sentinel",
)
# asyncpg uses plain postgres://, SQLAlchemy driver prefix is for sqlalchemy
ASYNC_DSN = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

ATTRIBUTION_SEARCH_RADIUS_NM = float(os.getenv("ATTRIBUTION_SEARCH_RADIUS_NM", "50"))
ATTRIBUTION_TIME_WINDOW_HOURS = float(os.getenv("ATTRIBUTION_TIME_WINDOW_HOURS", "6"))

ATTRIBUTION_FUZZY_WEIGHT = float(os.getenv("ATTRIBUTION_FUZZY_WEIGHT", "0.6"))
ATTRIBUTION_XGB_WEIGHT = float(os.getenv("ATTRIBUTION_XGB_WEIGHT", "0.4"))
AIS_PARQUET_PATH = os.getenv("AIS_PARQUET_PATH", "data/ais_demo.parquet")


def _blend_scores(fuzzy: float, xgb: float | None) -> tuple[float, str]:
    """Weighted composite of fuzzy + XGB; labels the method used."""
    wf, wx = ATTRIBUTION_FUZZY_WEIGHT, ATTRIBUTION_XGB_WEIGHT
    total = wf + wx
    if total <= 0:
        wf, wx, total = 0.6, 0.4, 1.0
    if xgb is None:
        return round(float(max(0.0, min(1.0, fuzzy))), 4), "fuzzy_only"
    blended = (wf * fuzzy + wx * xgb) / total
    return round(float(max(0.0, min(1.0, blended))), 4), "fuzzy_xgb_blend"


VESSEL_TYPE_RISK: dict[int, float] = {
    0: 0.5,  # Unknown
    30: 0.1,  # Fishing
    31: 0.1,  # Towing
    32: 0.1,  # Towing (large)
    33: 0.15,  # Dredger
    34: 0.15,  # Diving ops
    35: 0.1,  # Military
    36: 0.1,  # Sailing
    37: 0.1,  # Pleasure craft
    40: 0.6,  # High-speed craft
    50: 0.7,  # Pilot vessel
    51: 0.8,  # Search & rescue
    52: 0.3,  # Tug
    53: 0.3,  # Port tender
    54: 0.3,  # Anti-pollution
    55: 0.3,  # Law enforcement
    60: 0.4,  # Passenger (inland)
    61: 0.4,  # Passenger (ocean)
    62: 0.4,  # Passenger (cargo)
    63: 0.4,  # Passenger (ferry)
    64: 0.4,  # Passenger (ferry)
    65: 0.4,  # Passenger (ferry)
    66: 0.4,  # Passenger (ferry)
    67: 0.4,  # Passenger (ferry)
    70: 0.9,  # Cargo (inland)
    71: 0.9,  # Cargo (hazmat A)
    72: 0.9,  # Cargo (hazmat B)
    73: 0.9,  # Cargo (hazmat C)
    74: 0.9,  # Cargo (hazmat D)
    79: 0.9,  # Cargo (special)
    80: 0.85,  # Tanker (inland)
    81: 0.95,  # Tanker (hazmat A)
    82: 0.95,  # Tanker (hazmat B)
    83: 0.95,  # Tanker (hazmat C)
    84: 0.95,  # Tanker (hazmat D)
    89: 0.9,  # Tanker (special)
}


# ---------------------------------------------------------------------------
# Lifespan / DB pool
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None


async def _init_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=ASYNC_DSN,
        min_size=2,
        max_size=10,
        timeout=30,
        command_timeout=60,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    logger.info("Attribute service starting...")
    _pool = await _init_pool()
    logger.info("Database pool connected")
    yield
    if _pool:
        await _pool.close()
    logger.info("Attribute service stopped")


app = FastAPI(
    title="SENTINEL Attribute Service",
    description="Vessel attribution and scoring engine for oil spill suspect identification",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise HTTPException(status_code=503, detail="Database pool not initialized")
    return _pool


def _vessel_type_risk(vessel_type: int) -> float:
    """Map AIS vessel type code to risk score."""
    return VESSEL_TYPE_RISK.get(vessel_type, 0.5)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    pool = _get_pool()
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception as e:
        logger.warning("Health check DB error: {}", e)
        db_ok = False

    return HealthResponse(
        status="ok" if db_ok else "degraded",
    )


@app.post("/attribute/score", response_model=ScoreResult)
async def score_single(req: ScoreRequest):
    """Score a single suspect vessel with given features."""
    feat = SuspectFeatures(
        mmsi=req.mmsi,
        vessel_name=req.vessel_name,
        min_distance_nm=req.min_distance_nm,
        time_delta_minutes=req.time_delta_minutes,
        trajectory_intersection_score=req.trajectory_intersection_score,
        ais_gap_minutes=req.ais_gap_minutes,
        speed_anomaly_sigma=req.speed_anomaly_sigma,
        course_anomaly_sigma=req.course_anomaly_sigma,
        vessel_type_risk=req.vessel_type_risk,
        historical_violations=req.historical_violations,
        is_dark_sar_target=req.is_dark_sar_target,
        timestamp=req.timestamp,
        latitude=req.latitude,
        longitude=req.longitude,
        speed_knots=req.speed_knots,
        course_deg=req.course_deg,
        matched_ping_count=req.matched_ping_count,
        ais_provenance=req.ais_provenance,
    )
    scores = score_suspect(feat)
    composite, method = _blend_scores(scores["composite_score"], None)

    return ScoreResult(
        mmsi=req.mmsi,
        vessel_name=req.vessel_name,
        composite_score=composite,
        confidence_lower=scores["confidence_lower"],
        confidence_upper=scores["confidence_upper"],
        score_proximity=scores["score_proximity"],
        score_temporal=scores["score_temporal"],
        score_trajectory=scores["score_trajectory"],
        score_anomaly=scores["score_anomaly"],
        score_vessel_type=scores["score_vessel_type"],
        score_history=min(req.historical_violations / 3.0, 1.0),
        fuzzy_score=scores["composite_score"],
        xgb_score=None,
        xgb_method="origin_required_use_/attribute",
        shap_breakdown={},
        ais_gap_minutes=scores["ais_gap_minutes"],
        is_dark_vessel=scores["is_dark_vessel"],
        rank=1,
        timestamp=req.timestamp,
        latitude=req.latitude,
        longitude=req.longitude,
        speed_knots=req.speed_knots,
        course_deg=req.course_deg,
        closest_approach_nm=req.min_distance_nm,
        matched_ping_count=req.matched_ping_count,
        ais_provenance=req.ais_provenance,
    )


@app.post("/attribute/rank", response_model=RankResponse)
async def rank_suspects(req: RankRequest):
    """Rank pre-scored suspects by composite score."""
    ranker = SuspectRanker()
    features: list[SuspectFeatures] = []
    for s in req.suspects:
        features.append(
            SuspectFeatures(
                mmsi=s.mmsi,
                vessel_name=s.vessel_name,
                min_distance_nm=0.0,  # not used for re-scoring
                time_delta_minutes=0.0,
                trajectory_intersection_score=s.score_trajectory,
                ais_gap_minutes=s.ais_gap_minutes,
                speed_anomaly_sigma=0.0,
                course_anomaly_sigma=0.0,
                vessel_type_risk=s.score_vessel_type,
                historical_violations=0,
                is_dark_sar_target=s.is_dark_vessel,
                timestamp=s.timestamp,
                latitude=s.latitude,
                longitude=s.longitude,
                speed_knots=s.speed_knots,
                course_deg=s.course_deg,
                matched_ping_count=s.matched_ping_count,
                ais_provenance=s.ais_provenance,
            )
        )

    result = ranker.rank_from_features(features)

    return RankResponse(
        suspects=[
            ScoreResult(
                mmsi=s.mmsi,
                vessel_name=s.vessel_name,
                composite_score=s.composite_score,
                confidence_lower=s.confidence_lower,
                confidence_upper=s.confidence_upper,
                score_proximity=s.score_proximity,
                score_temporal=s.score_temporal,
                score_trajectory=s.score_trajectory,
                score_anomaly=s.score_anomaly,
                score_vessel_type=s.score_vessel_type,
                score_history=s.score_history,
                ais_gap_minutes=s.ais_gap_minutes,
                is_dark_vessel=s.is_dark_vessel,
                rank=s.rank,
                timestamp=s.timestamp,
                latitude=s.latitude,
                longitude=s.longitude,
                speed_knots=s.speed_knots,
                course_deg=s.course_deg,
                closest_approach_nm=s.closest_approach_nm,
                matched_ping_count=s.matched_ping_count,
                ais_provenance=s.ais_provenance,
            )
            for s in result.suspects
        ],
        total=result.total,
    )


def _decide_coverage(
    req: AttribRequest,
    vessels: list[VesselTrack],
    aoi: tuple[float, float, float, float] | None,
) -> CoverageVerdict:
    """Decide whether the AIS behind this request may be ranked at all.

    Evidence order (strongest first): the caller's declared provenance, then
    the provenance carried by the matched rows, then the AOI policy. An AOI
    verdict of "no coverage" always wins over a *terrestrial* row label —
    a terrestrial feed over the open Indian Ocean contradicts the measured
    result (0 messages in 40 s) and must not be trusted. Satellite AIS is the
    one exception: it genuinely covers open ocean.
    """
    real_rows = [v for v in vessels if (v.source or "unknown").lower() in REAL_AIS_PROVENANCE]
    synthetic_rows = [
        v for v in vessels if (v.source or "unknown").lower() in SYNTHETIC_AIS_PROVENANCE
    ]

    if req.ais_provenance:
        verdict = provenance_verdict(req.ais_provenance, matched=len(real_rows) or len(vessels))
    elif real_rows:
        verdict = provenance_verdict(real_rows[0].source, matched=len(real_rows))
    elif synthetic_rows:
        verdict = provenance_verdict(synthetic_rows[0].source, matched=0)
    elif vessels:
        # Rows exist but carry a provenance we do not recognise.
        verdict = provenance_verdict(vessels[0].source, matched=len(vessels))
    else:
        verdict = coverage_verdict(aoi)

    if (
        aoi is not None
        and not coverage_verdict(aoi).rankable
        and verdict.provenance != ("live_satellite")
    ):
        return coverage_verdict(aoi)
    return verdict


def _empty_attrib_response(
    req: AttribRequest, verdict: CoverageVerdict, elapsed_ms: float
) -> AttribResponse:
    """The only legal response when AIS coverage is not real.

    Empty suspects + a machine-readable reason. Never a ranking, never a
    placeholder score, never silently omitted fields.
    """
    logger.warning(
        "[ATTR] No suspect ranking produced — coverage={} reason={}",
        verdict.coverage,
        verdict.reason,
    )
    return AttribResponse(
        case_id=req.case_id,
        spill_id=req.spill_id,
        suspects=[],
        total_candidates=0,
        dark_vessels_found=0,
        xgb_candidates=0,
        scoring_weights={
            "fuzzy": ATTRIBUTION_FUZZY_WEIGHT,
            "xgb": ATTRIBUTION_XGB_WEIGHT,
        },
        pipeline_ms=round(elapsed_ms, 2),
        coverage=verdict.coverage,  # type: ignore[arg-type]
        coverage_reason=verdict.reason,
        coverage_message=verdict.message,
        ais_provenance=verdict.provenance,
        rankable=False,
    )


@app.post("/attribute", response_model=AttribResponse)
async def full_attribution(req: AttribRequest):
    """Full attribution pipeline.

    Pipeline stages:
      0. AIS coverage gate - refuse to rank without real coverage
      1. AIS slicer - PostGIS spatiotemporal proximity query
      2. Dark vessel check - SAR non-AIS target matching
      3. Anomaly detection - LSTM autoencoder behavioral scoring
      4. Fuzzy scoring - Cauchy membership multi-factor composite
      5. Ranked suspects - sorted by composite score with Wilson 95% CI
    """
    t0 = time.monotonic()

    # Stage 0: resolve the AOI using named axes only. A partial set is a 422 —
    # a guessed axis silently relocates the AOI and flips the verdict.
    try:
        aoi = resolve_aoi(req.min_lon, req.min_lat, req.max_lon, req.max_lat, req.bbox)
    except AoiError as exc:
        logger.warning("[ATTR] AOI rejected: {} {}", exc.code, exc.message)
        raise HTTPException(
            status_code=422, detail={"reason": exc.code, "message": exc.message}
        ) from exc

    pool = _get_pool()

    # Stage 1: AIS Slicer
    logger.info(
        "[ATTR] Stage 1: AIS slicer - center=({}, {}) radius={}nm window=±{}h",
        req.origin.center_lon,
        req.origin.center_lat,
        req.search_radius_nm,
        req.time_window_hours,
    )
    slicer = AisSlicer(pool)
    slice_result = await slicer.find_vessels(
        center_lon=req.origin.center_lon,
        center_lat=req.origin.center_lat,
        search_radius_nm=req.search_radius_nm,
        spill_time=req.spill_time,
        time_window_hours=req.time_window_hours,
    )
    logger.info("[ATTR] AIS slicer: {} distinct vessels", slice_result.distinct_mmsi)

    # Stage 0b: the coverage gate. Nothing below this line may produce a
    # ranking when the AIS is synthetic, unverified, or absent.
    verdict = _decide_coverage(req, slice_result.vessels, aoi)
    if not verdict.rankable:
        return _empty_attrib_response(req, verdict, (time.monotonic() - t0) * 1000.0)

    # Only vessels whose rows carry a real provenance are scored.
    real_vessels = [
        v for v in slice_result.vessels if (v.source or "unknown").lower() in REAL_AIS_PROVENANCE
    ]

    # Stage 2: Dark Vessel Detection
    dark_result = None
    dark_mmsis: set[str] = set()
    sar_dicts = [s.model_dump() for s in req.sar_targets] if req.sar_targets else []
    if sar_dicts:
        logger.info("[ATTR] Stage 2: Dark vessel check - {} SAR targets", len(sar_dicts))
        dark_detector = DarkVesselDetector(pool)
        dark_result = await dark_detector.detect_dark_vessels(
            sar_targets=sar_dicts,
            search_radius_nm=req.search_radius_nm,
        )
        # MMSIs with AIS near a SAR target (not dark)
        dark_mmsis = dark_detector.get_dark_mmsis(dark_result)
        logger.info(
            "[ATTR] Dark vessels: {}/{} targets",
            dark_result.dark_vessel_count,
            dark_result.sar_targets_count,
        )

    # Stage 3: Anomaly Detection
    logger.info("[ATTR] Stage 3: Anomaly detection for {} vessels", slice_result.distinct_mmsi)
    anomaly_detector = AnomalyDetector()
    anomaly_scores: dict[str, dict] = {}

    # Collect per-vessel tracks for anomaly detection
    vessel_tracks: dict[str, list[dict]] = {}
    for v in real_vessels:
        vessel_tracks.setdefault(v.mmsi, []).append(
            {
                "sog": v.sog,
                "cog": v.cog,
                "lon": v.lon,
                "lat": v.lat,
                "timestamp": v.timestamp,
            }
        )

    for mmsi, tracks in vessel_tracks.items():
        if len(tracks) >= 3:
            result = anomaly_detector.detect(mmsi, tracks)
            anomaly_scores[mmsi] = {
                "speed_anomaly_sigma": result.speed_anomaly_sigma,
                "course_anomaly_sigma": result.course_anomaly_sigma,
                "is_anomalous": result.is_anomalous,
                "anomaly_score": result.anomaly_score,
            }

    logger.info("[ATTR] Anomaly detection complete for {} vessels", len(anomaly_scores))

    # Stage 4: Build feature vectors and score
    logger.info("[ATTR] Stage 4: Fuzzy scoring")
    ranker = SuspectRanker()

    # Build AIS score dicts. Every audit field the scorer needs is carried
    # here: timestamp, latitude, longitude, speed, course, closest approach,
    # matched ping count, AIS gap. Losing any of them makes the score
    # unfalsifiable, so they are attached at the point of extraction.
    ais_scores: dict[str, dict] = {}
    for v in real_vessels:
        mmsi = v.mmsi
        if mmsi not in ais_scores:
            ais_scores[mmsi] = {
                "min_distance_nm": v.min_distance_nm,
                "time_delta_minutes": v.time_delta_minutes,
                "vessel_name": v.vessel_name,
                "ais_gap_minutes": await slicer.get_ais_gap(
                    mmsi, req.spill_time, lookback_hours=req.time_window_hours
                ),
                "trajectory_intersection_score": 0.0,
                "timestamp": v.timestamp,
                "latitude": v.lat,
                "longitude": v.lon,
                "speed_knots": v.sog,
                "course_deg": v.cog,
                "matched_ping_count": v.matched_ping_count,
                "ais_provenance": v.source,
            }

    # Dark SAR targets become high-priority suspects: a vessel whose AIS
    # disappears next to a SAR return gets its trajectory score lifted.
    if sar_dicts and dark_mmsis:
        for mmsi, ais in ais_scores.items():
            if mmsi in dark_mmsis:
                ais["trajectory_intersection_score"] = max(
                    ais.get("trajectory_intersection_score", 0.0), 0.8
                )

    # Build vessel type risk map
    vtr: dict[str, float] = {}
    for v in real_vessels:
        vtr[v.mmsi] = _vessel_type_risk(v.vessel_type)

    # Aggregate and rank
    suspect_features = ranker.aggregate_scores(
        ais_scores=ais_scores,
        anomaly_scores=anomaly_scores,
        dark_mmsis=dark_mmsis,
        vessel_type_risk=vtr,
        historical_violations=req.historical_violations,
    )

    rank_result = ranker.rank_from_features(suspect_features, top_n=20)

    # Stage 4b: DuckDB/XGBoost parallel scorer (origin-centred, SHAP).
    xgb_by_mmsi: dict[str, dict] = {}
    try:
        from app.engine.duckdb_scorer import DuckDBScorer

        duck = DuckDBScorer(source_path=AIS_PARQUET_PATH)
        for item in duck.score_candidates(
            origin_lon=req.origin.center_lon,
            origin_lat=req.origin.center_lat,
            origin_time=req.spill_time,
            window_hours=max(1, int(req.time_window_hours)),
        ):
            xgb_by_mmsi[str(item["mmsi"])] = item
        logger.info("[ATTR] DuckDB/XGB scorer: {} candidates", len(xgb_by_mmsi))
    except Exception as e:
        logger.warning("[ATTR] DuckDB scorer unavailable, fuzzy-only: {}", e)

    # Stage 5: Blend fuzzy + XGB, re-rank by composite, build response.
    merged: list[ScoreResult] = []
    for s in rank_result.suspects:
        xgb_item = xgb_by_mmsi.get(str(s.mmsi))
        xgb_score = float(xgb_item["xgb_score"]) if xgb_item else None
        composite, method = _blend_scores(float(s.composite_score), xgb_score)
        merged.append(
            ScoreResult(
                mmsi=s.mmsi,
                vessel_name=s.vessel_name,
                composite_score=composite,
                confidence_lower=s.confidence_lower,
                confidence_upper=s.confidence_upper,
                score_proximity=s.score_proximity,
                score_temporal=s.score_temporal,
                score_trajectory=s.score_trajectory,
                score_anomaly=s.score_anomaly,
                score_vessel_type=s.score_vessel_type,
                score_history=s.score_history,
                fuzzy_score=float(s.composite_score),
                xgb_score=xgb_score,
                xgb_method=str(xgb_item.get("method", "unavailable"))
                if xgb_item
                else "unavailable",
                shap_breakdown=dict(xgb_item.get("shap_breakdown", {})) if xgb_item else {},
                ais_gap_minutes=s.ais_gap_minutes,
                is_dark_vessel=s.is_dark_vessel,
                rank=s.rank,
                timestamp=s.timestamp,
                latitude=s.latitude,
                longitude=s.longitude,
                speed_knots=s.speed_knots,
                course_deg=s.course_deg,
                closest_approach_nm=s.closest_approach_nm,
                matched_ping_count=s.matched_ping_count,
                ais_provenance=s.ais_provenance,
            )
        )
    merged.sort(key=lambda r: r.composite_score, reverse=True)
    for i, r in enumerate(merged, start=1):
        r.rank = i

    elapsed_ms = (time.monotonic() - t0) * 1000.0

    logger.info(
        "[ATTR] Pipeline complete: {} suspects ({} with XGB) in {:.1f}ms",
        len(merged),
        len(xgb_by_mmsi),
        elapsed_ms,
    )

    return AttribResponse(
        case_id=req.case_id,
        spill_id=req.spill_id,
        suspects=merged,
        total_candidates=rank_result.total,
        dark_vessels_found=rank_result.dark_vessel_count,
        xgb_candidates=len(xgb_by_mmsi),
        scoring_weights={
            "fuzzy": ATTRIBUTION_FUZZY_WEIGHT,
            "xgb": ATTRIBUTION_XGB_WEIGHT,
        },
        pipeline_ms=round(elapsed_ms, 2),
        timestamp=datetime.utcnow(),
        coverage=verdict.coverage,  # type: ignore[arg-type]
        coverage_reason=verdict.reason or None,
        coverage_message=verdict.message,
        ais_provenance=verdict.provenance,
        rankable=True,
    )
