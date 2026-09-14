"""SENTINEL pipeline orchestration (Celery).

Calls the REAL service endpoints with valid payloads, publishes the five
pipeline WS events (SPILL_DETECTED, DRIFT_COMPLETE, SUSPECT_RANKED,
NARRATIVE_READY, CASE_FILE_READY) plus PIPELINE_PROGRESS per stage, and
writes every stage result into the shared Redis result cache so the gateway
GET routes (/cases/:id/suspects, /drift/forecast/:id) serve live data.

Payload passthrough: trigger accepts an optional `payload` dict; missing
keys fall back to the Wakashio-demo defaults below.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from loguru import logger

from app.celery_app import celery_app

DETECT_SERVICE_URL = os.getenv("DETECT_SERVICE_URL", "http://sentinel-detect:8002")
DRIFT_SERVICE_URL = os.getenv("DRIFT_SERVICE_URL", "http://sentinel-drift:8003")
ATTRIBUTE_SERVICE_URL = os.getenv("ATTRIBUTE_SERVICE_URL", "http://sentinel-attribute:8004")
INTEL_SERVICE_URL = os.getenv("INTEL_SERVICE_URL", "http://sentinel-intel:8005")
REDIS_URL = os.getenv("REDIS_URL", os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0"))

# Wakashio (Mauritius, 25 Jul 2020) demo defaults — used when the trigger
# payload omits a key. Explicit, labelled, never fabricated live data.
DEFAULTS: dict[str, Any] = {
    "case_id": "wakashio-demo",
    "spill_id": "spill-wakashio-001",
    "spill_lon": 57.7,
    "spill_lat": -20.4,
    "spill_age_hours": 48.0,
    "forecast_hours": 72.0,
    "n_particles": 200,
    "spill_time": "2020-07-25T12:00:00Z",
    "image_path": "/app/data/sample_sentinel1.png",
    "search_radius_nm": 50.0,
    "time_window_hours": 6.0,
    "origin": {
        "center_lon": 57.55,
        "center_lat": -20.45,
        "semi_major_km": 8.0,
        "semi_minor_km": 4.0,
        "orientation_deg": 45.0,
    },
}


def _merge(payload: Optional[dict] = None) -> dict[str, Any]:
    merged = dict(DEFAULTS)
    if payload:
        merged.update({k: v for k, v in payload.items() if v is not None})
    return merged


def _publish(case_id: str, event_type: str, payload: dict) -> None:
    """Publish one WS event to Redis (API replicas forward to sockets)."""
    try:
        import redis as sync_redis

        client = sync_redis.Redis.from_url(REDIS_URL, decode_responses=True)
        try:
            body = {"type": event_type, "payload": {"case_id": case_id, **payload}}
            client.publish(f"sentinel:ws:{case_id}", json.dumps(body, default=str))
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception as e:
        logger.warning("WS publish failed for {}: {}", case_id, e)


def _save_stage(case_id: str, stage: str, data: Any) -> None:
    """Write one stage result into the shared Redis result cache."""
    try:
        import redis as sync_redis

        client = sync_redis.Redis.from_url(REDIS_URL, decode_responses=True)
        try:
            key = f"sentinel:result:{case_id}"
            raw = client.get(key)
            all_data = json.loads(raw) if raw else {}
            all_data[stage] = data
            client.setex(key, 86400, json.dumps(all_data, default=str))
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception as e:
        logger.warning("Result cache write failed for {}: {}", case_id, e)


def _post(url: str, body: dict, timeout: float = 120.0) -> dict:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=body)
            if resp.status_code == 200:
                return {"ok": True, "data": resp.json()}
            return {"ok": False, "code": resp.status_code, "detail": resp.text[:500]}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)[:500]}


def _progress(case_id: str, stage: str, pct: float) -> None:
    _publish(case_id, "PIPELINE_PROGRESS", {"stage": stage, "pct_complete": pct})


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_detect(case_id: str, p: dict) -> dict:
    _progress(case_id, "detect", 5.0)
    res = _post(
        f"{DETECT_SERVICE_URL}/detect/analyze",
        {"image_path": p["image_path"], "confidence_threshold": 0.5},
        timeout=300.0,
    )
    if res.get("ok"):
        data = res["data"]
        _save_stage(case_id, "detect", data)
        spills = data.get("spills", [])
        _publish(case_id, "SPILL_DETECTED", {"spills": spills, "total": len(spills)})
        return {"status": "ok", "spills": len(spills), "data": data}
    _publish(case_id, "PIPELINE_PROGRESS", {"stage": "detect", "pct_complete": 0.0, "error": res.get("detail")})
    return {"status": "error", **{k: v for k, v in res.items() if k != "ok"}}


def stage_drift(case_id: str, p: dict) -> dict:
    _progress(case_id, "drift", 30.0)
    t_end = datetime.now(timezone.utc)
    t_start = t_end - timedelta(hours=float(p["spill_age_hours"]) + 24)
    body = {
        "spill_lon": float(p["spill_lon"]),
        "spill_lat": float(p["spill_lat"]),
        "spill_age_hours": float(p["spill_age_hours"]),
        "forecast_hours": float(p["forecast_hours"]),
        "n_particles": int(p["n_particles"]),
        "ocean_data": {
            "cmems_base": "",
            "era5_base": "",
            "time_start": t_start.isoformat(),
            "time_end": t_end.isoformat(),
            "bbox": [
                float(p["spill_lon"]) - 2.0,
                float(p["spill_lat"]) - 2.0,
                float(p["spill_lon"]) + 2.0,
                float(p["spill_lat"]) + 2.0,
            ],
        },
    }
    res = _post(f"{DRIFT_SERVICE_URL}/drift/full", body, timeout=600.0)
    if res.get("ok"):
        data = res["data"]
        _save_stage(case_id, "drift", data)
        origin = (data.get("backward") or {}).get("origin_ellipse", {})
        _publish(
            case_id,
            "DRIFT_COMPLETE",
            {"origin_ellipse": origin, "forcing": (data.get("backward") or {}).get("forcing_source")},
        )
        return {"status": "ok", "data": data}
    return {"status": "error", **{k: v for k, v in res.items() if k != "ok"}}


def stage_attribute(case_id: str, p: dict, drift_data: Optional[dict] = None) -> dict:
    _progress(case_id, "attribute", 60.0)
    origin = dict(p["origin"])
    try:
        backward = (drift_data or {}).get("backward", drift_data or {})
        ell = backward.get("origin_ellipse") or {}
        if ell.get("center_lon"):
            origin = {
                "center_lon": float(ell["center_lon"]),
                "center_lat": float(ell["center_lat"]),
                "semi_major_km": float(ell.get("semi_major_km", 8.0)),
                "semi_minor_km": float(ell.get("semi_minor_km", 4.0)),
                "orientation_deg": float(ell.get("orientation_deg", 0.0)),
            }
    except Exception:
        pass
    body = {
        "case_id": case_id,
        "spill_id": str(p["spill_id"]),
        "origin": origin,
        "spill_time": str(p["spill_time"]),
        "search_radius_nm": float(p["search_radius_nm"]),
        "time_window_hours": float(p["time_window_hours"]),
    }
    res = _post(f"{ATTRIBUTE_SERVICE_URL}/attribute", body, timeout=300.0)
    if res.get("ok"):
        data = res["data"]
        _save_stage(case_id, "attribute", data)
        suspects = data.get("suspects", [])
        _publish(case_id, "SUSPECT_RANKED", {"suspects": suspects[:5], "total": len(suspects)})
        return {"status": "ok", "suspects": len(suspects), "data": data}
    return {"status": "error", **{k: v for k, v in res.items() if k != "ok"}}


def _intel_detection(p: dict, case_id: str, detect_data: Optional[dict] = None) -> dict:
    spills = (detect_data or {}).get("spills", []) if detect_data else []
    first = spills[0] if spills else {}
    lon = float(first.get("centroid_lon", p["spill_lon"]))
    lat = float(first.get("centroid_lat", p["spill_lat"]))
    if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
        # Non-georeferenced input image yields pixel coords; fall back to the
        # trigger payload location (labelled) instead of failing validation.
        logger.warning(
            "Detect centroid ({}, {}) out of range — using trigger location",
            lon,
            lat,
        )
        lon, lat = float(p["spill_lon"]), float(p["spill_lat"])
    return {
        "spill_id": str(p["spill_id"]),
        "case_id": case_id,
        "centroid_lon": lon,
        "centroid_lat": lat,
        "area_m2": float(first.get("area_m2", 5000.0)),
        "confidence": float(first.get("confidence", 0.8)),
        "age_hours": float(p["spill_age_hours"]),
        "image_path": str(p["image_path"]),
        "detected_at": str(p["spill_time"]),
    }


def _intel_drift(drift_data: Optional[dict]) -> Optional[dict]:
    try:
        backward = (drift_data or {}).get("backward", drift_data or {})
        ell = backward.get("origin_ellipse") or {}
        if not ell:
            return None
        return {
            "origin_lon": float(ell.get("center_lon", 0.0)),
            "origin_lat": float(ell.get("center_lat", 0.0)),
            "semi_major_km": float(ell.get("semi_major_km", 0.0)),
            "semi_minor_km": float(ell.get("semi_minor_km", 0.0)),
            "orientation_deg": float(ell.get("orientation_deg", 0.0)),
            "regime": str(backward.get("regime", "markov1")),
        }
    except Exception:
        return None


def _intel_suspects(attr_data: Optional[dict]) -> list[dict]:
    out = []
    for i, s in enumerate((attr_data or {}).get("suspects", [])[:5], start=1):
        out.append(
            {
                "mmsi": str(s.get("mmsi", "")),
                "vessel_name": str(s.get("vessel_name", "UNKNOWN")),
                "composite_score": float(s.get("composite_score", 0.0)),
                "score_proximity": float(s.get("score_proximity", 0.0)),
                "score_temporal": float(s.get("score_temporal", 0.0)),
                "score_trajectory": float(s.get("score_trajectory", 0.0)),
                "score_anomaly": float(s.get("score_anomaly", 0.0)),
                "score_vessel_type": float(s.get("score_vessel_type", 0.0)),
                "ais_gap_minutes": float(s.get("ais_gap_minutes", 0.0)),
                "is_dark_vessel": bool(s.get("is_dark_vessel", False)),
                "rank": int(s.get("rank", i)),
            }
        )
    return out


def stage_intel(
    case_id: str,
    p: dict,
    detect_data: Optional[dict] = None,
    drift_data: Optional[dict] = None,
    attr_data: Optional[dict] = None,
) -> dict:
    _progress(case_id, "intel", 80.0)
    body = {
        "detection": _intel_detection(p, case_id, detect_data),
        "drift": _intel_drift(drift_data),
        "suspects": _intel_suspects(attr_data),
        "evidence_hash": "",
    }
    res = _post(f"{INTEL_SERVICE_URL}/intel/narrative", body, timeout=180.0)
    if res.get("ok"):
        data = res["data"]
        _save_stage(case_id, "intel", data)
        _publish(
            case_id,
            "NARRATIVE_READY",
            {"summary": data.get("summary", "")[:500], "priority": data.get("alert_priority")},
        )
        return {"status": "ok", "data": data}
    return {"status": "error", **{k: v for k, v in res.items() if k != "ok"}}


def stage_case_file(
    case_id: str,
    p: dict,
    detect_data: Optional[dict] = None,
    drift_data: Optional[dict] = None,
    attr_data: Optional[dict] = None,
) -> dict:
    _progress(case_id, "case_file", 95.0)
    body = {
        "case_id": case_id,
        "detection": _intel_detection(p, case_id, detect_data),
        "drift": _intel_drift(drift_data),
        "suspects": _intel_suspects(attr_data),
    }
    res = _post(f"{INTEL_SERVICE_URL}/intel/case-file", body, timeout=300.0)
    if res.get("ok"):
        data = res["data"]
        _save_stage(case_id, "case_file", data)
        _publish(
            case_id,
            "CASE_FILE_READY",
            {"pdf_path": data.get("pdf_path"), "evidence_hash": data.get("evidence_hash")},
        )
        _progress(case_id, "done", 100.0)
        return {"status": "ok", "data": data}
    return {"status": "error", **{k: v for k, v in res.items() if k != "ok"}}


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------

@celery_app.task(name="app.tasks.run_detect_pipeline", bind=True)
def run_detect_pipeline(self, case_id: str, payload: dict | None = None):
    logger.info("Running detection pipeline for case: {}", case_id)
    return stage_detect(case_id, _merge(payload))


@celery_app.task(name="app.tasks.run_drift_pipeline", bind=True)
def run_drift_pipeline(self, case_id: str, payload: dict | None = None):
    logger.info("Running drift pipeline for case: {}", case_id)
    return stage_drift(case_id, _merge(payload))


@celery_app.task(name="app.tasks.run_attribute_pipeline", bind=True)
def run_attribute_pipeline(self, case_id: str, payload: dict | None = None):
    logger.info("Running attribute pipeline for case: {}", case_id)
    return stage_attribute(case_id, _merge(payload))


@celery_app.task(name="app.tasks.run_intel_pipeline", bind=True)
def run_intel_pipeline(self, case_id: str, payload: dict | None = None):
    logger.info("Running intel pipeline for case: {}", case_id)
    p = _merge(payload)
    return stage_intel(case_id, p)


@celery_app.task(name="app.tasks.generate_case_file", bind=True)
def generate_case_file(self, case_id: str, payload: dict | None = None):
    logger.info("Generating case file for case: {}", case_id)
    p = _merge(payload)
    return stage_case_file(case_id, p)


@celery_app.task(name="app.tasks.run_full_pipeline", bind=True)
def run_full_pipeline(self, case_id: str, stages: list[str] | None = None, payload: dict | None = None):
    logger.info("Running full pipeline for case: {}, stages: {}", case_id, stages)
    p = _merge(payload)
    target_stages = stages or ["detect", "drift", "attribute", "intel"]
    results: dict[str, Any] = {}
    detect_data = drift_data = attr_data = None

    if "detect" in target_stages:
        r = stage_detect(case_id, p)
        results["detect"] = {k: v for k, v in r.items() if k != "data"}
        if r.get("status") == "ok":
            detect_data = r.get("data")
    if "drift" in target_stages:
        r = stage_drift(case_id, p)
        results["drift"] = {k: v for k, v in r.items() if k != "data"}
        if r.get("status") == "ok":
            drift_data = r.get("data")
    if "attribute" in target_stages:
        r = stage_attribute(case_id, p, drift_data)
        results["attribute"] = {k: v for k, v in r.items() if k != "data"}
        if r.get("status") == "ok":
            attr_data = r.get("data")
    if "intel" in target_stages:
        r = stage_intel(case_id, p, detect_data, drift_data, attr_data)
        results["intel"] = {k: v for k, v in r.items() if k != "data"}
    if "case_file" in target_stages or "intel" in target_stages:
        r = stage_case_file(case_id, p, detect_data, drift_data, attr_data)
        results["case_file"] = {k: v for k, v in r.items() if k != "data"}
    return results
