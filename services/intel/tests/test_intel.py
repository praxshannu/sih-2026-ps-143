"""Intel tests — hasher stability/canonical coverage + offline narrative."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_hash = _load("sentinel_hasher", "services/intel/app/hasher.py")
_nar = _load("sentinel_narrative", "services/intel/app/narrative.py")


def test_evidence_hash_stable():
    a = _hash.compute_object_hash({"x": 1})
    b = _hash.compute_object_hash({"x": 1})
    assert a == b and len(a) == 64


def test_canonical_package_covers_required_fields():
    pkg = _hash.canonical_evidence_package(
        case_id="wakashio-demo",
        spill_id="spill-1",
        detection_data={
            "case_id": "wakashio-demo",
            "spill_id": "spill-1",
            "centroid_lat": -20.4,
            "centroid_lon": 57.7,
            "confidence": 0.88,
            "area_m2": 5000.0,
            "detected_at": "2020-07-25T12:00:00Z",
        },
        drift_data={
            "origin_ellipse": {
                "center_lon": 57.48,
                "center_lat": -20.47,
                "semi_major_km": 24.0,
                "semi_minor_km": 23.0,
            },
            "forcing_source": "composite_gfs_mock",
            "xgb_residual_applied": True,
            "xgb_correction_m": {"dx": 1.0, "dy": 2.0, "scale": 1.1},
        },
        suspects=[
            {
                "mmsi": "477218700",
                "composite_score": 0.4,
                "shap_breakdown": {"distance_to_origin": 0.3},
            }
        ],
    )
    assert pkg["detection"]["confidence"] == 0.88
    assert pkg["drift"]["origin_lon"] == 57.48
    assert pkg["suspects_top3"][0]["shap_breakdown"] == {"distance_to_origin": 0.3}
    # Canonical: sorted keys reproduce the identical hash.
    assert _hash.compute_object_hash(pkg) == _hash.compute_evidence_hash(
        case_id="wakashio-demo",
        spill_id="spill-1",
        detection_data={
            "case_id": "wakashio-demo",
            "spill_id": "spill-1",
            "centroid_lat": -20.4,
            "centroid_lon": 57.7,
            "confidence": 0.88,
            "area_m2": 5000.0,
            "detected_at": "2020-07-25T12:00:00Z",
        },
        drift_data={
            "origin_ellipse": {
                "center_lon": 57.48,
                "center_lat": -20.47,
                "semi_major_km": 24.0,
                "semi_minor_km": 23.0,
            },
            "forcing_source": "composite_gfs_mock",
            "xgb_residual_applied": True,
            "xgb_correction_m": {"dx": 1.0, "dy": 2.0, "scale": 1.1},
        },
        suspects=[
            {
                "mmsi": "477218700",
                "composite_score": 0.4,
                "shap_breakdown": {"distance_to_origin": 0.3},
            }
        ],
    )


def test_offline_narrative_deterministic():
    out = _nar.offline_narrative(
        {
            "spill_id": "spill-1",
            "case_id": "wakashio-demo",
            "centroid_lat": -20.4,
            "centroid_lon": 57.7,
            "confidence": 0.88,
        },
        None,
        [],
        "abc123",
    )
    assert "57.7000" in out["summary"]
    assert out["model_used"] == "offline_template"
    assert "abc123" in out["summary"]


def test_narrative_validation_rejects_bad_priority():
    import pytest

    with pytest.raises(ValueError):
        _nar._validate_narrative(
            {
                "summary": "s",
                "key_finding": "k",
                "evidentiary_gaps": [],
                "legal_basis": "l",
                "alert_priority": "UNKNOWN",
            }
        )


def test_llm_status_shape():
    status = asyncio.run(_nar.llm_status())
    assert set(status) == {"provider", "model", "available", "latency_ms"}
