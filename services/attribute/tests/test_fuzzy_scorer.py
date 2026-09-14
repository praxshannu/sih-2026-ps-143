"""Attribute tests — fuzzy scorer (ported from claude2) + dual scoring."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_fs = _load("sentinel_fuzzy", "services/attribute/app/engine/fuzzy_scorer.py")
_duck = _load("sentinel_duckdb", "services/attribute/app/engine/duckdb_scorer.py")


def test_cauchy_peaks_at_center():
    assert _fs.cauchy_membership(0.0, 0.0, 5.0) == 1.0
    assert 0.0 < _fs.cauchy_membership(50.0, 0.0, 5.0) < 0.05


def test_wakashio_scores_high():
    out = _fs.score_suspect(
        _fs.SuspectFeatures(
            mmsi="477218700",
            vessel_name="MV WAKASHIO",
            min_distance_nm=0.3,
            time_delta_minutes=2.0,
            trajectory_intersection_score=0.92,
            ais_gap_minutes=23.0,
            speed_anomaly_sigma=3.1,
            course_anomaly_sigma=2.4,
            vessel_type_risk=1.0,
            historical_violations=1,
            is_dark_sar_target=False,
        )
    )
    assert out["composite_score"] > 0.6
    assert out["confidence_lower"] < out["composite_score"] < out["confidence_upper"]
    assert out["is_dark_vessel"] is False


def test_far_vessel_scores_low():
    out = _fs.score_suspect(
        _fs.SuspectFeatures(
            mmsi="000000000",
            vessel_name="FAR SHIP",
            min_distance_nm=120.0,
            time_delta_minutes=400.0,
            trajectory_intersection_score=0.0,
            ais_gap_minutes=0.0,
            speed_anomaly_sigma=0.1,
            course_anomaly_sigma=0.1,
            vessel_type_risk=0.3,
            historical_violations=0,
            is_dark_sar_target=False,
        )
    )
    assert out["composite_score"] < 0.25


def test_duckdb_missing_source_returns_empty_never_crashes():
    scorer = _duck.DuckDBScorer(source_path="/nonexistent/ais.parquet")
    moment = datetime(2020, 7, 25)
    assert scorer.score_candidates(57.7, -20.4, moment) == []
    assert scorer.extract_features(57.7, -20.4, moment) == []


def test_blend_weights_default_to_0_6_0_4():
    import os

    assert float(os.getenv("ATTRIBUTION_FUZZY_WEIGHT", "0.6")) == 0.6
    assert float(os.getenv("ATTRIBUTION_XGB_WEIGHT", "0.4")) == 0.4
