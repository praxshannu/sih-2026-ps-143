"""DuckDB AIS feature scorer + XGBoost/SHAP parallel path for SENTINEL attribute.

Adapted from OceanTrace agent3/attribution_engine.py — DuckDB spatial query
kept, imports decoupled, all heavy deps (duckdb, xgboost, shap) optional so
the service never hard-crashes when they are absent. When the parquet source
(AIS_PARQUET_PATH) or a dep is missing, returns [] and the PostGIS + fuzzy
path remains primary.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from loguru import logger


def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except Exception:
        return False


class DuckDBScorer:
    """Parallel AIS scorer over a local parquet/CSV snapshot."""

    FEATURES = [
        "distance_to_origin",
        "min_speed_in_window",
        "ais_gap_duration",
        "vessel_type_weight",
    ]

    def __init__(
        self,
        source_path: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self.source_path = source_path or os.getenv("AIS_PARQUET_PATH", "data/ais_demo.parquet")
        self.model_path = model_path or os.getenv(
            "ATTRIBUTION_XGB_PATH", "data/xgb_attribution.json"
        )
        self._con: Any = None
        self._model: Any = None
        self._explainer: Any = None
        self._base_value: float = 0.0

    # ------------------------------------------------------------------
    def _connect(self) -> Any:
        import duckdb

        if self._con is None:
            self._con = duckdb.connect(":memory:")
            try:
                # Container user has no $HOME; point DuckDB at a writable dir
                # so INSTALL/LOAD spatial can persist the extension.
                import os as _os

                _os.makedirs("/tmp/duckdb_home", exist_ok=True)
                self._con.execute("SET home_directory='/tmp/duckdb_home';")
                self._con.execute("INSTALL spatial;")
                self._con.execute("LOAD spatial;")
            except Exception as e:
                logger.warning("DuckDB spatial extension unavailable: {}", e)
        return self._con

    def _ensure_view(self) -> bool:
        if not _duckdb_available() or not self.source_path or not os.path.exists(self.source_path):
            return False
        try:
            con = self._connect()
            src = self.source_path.replace("'", "''")
            if src.lower().endswith(".parquet"):
                reader = f"read_parquet('{src}')"
            else:
                reader = f"read_csv_auto('{src}')"
            con.execute(
                f"""
                CREATE OR REPLACE VIEW ais_data AS
                SELECT mmsi, timestamp, lon, lat, sog, cog, vessel_type,
                       ST_Point(lon, lat) AS geom
                FROM {reader}
                """
            )
            return True
        except Exception as e:
            logger.warning("DuckDB AIS view failed ({}): {}", self.source_path, e)
            return False

    def _ensure_model(self) -> str:
        """Returns 'xgb' | 'shap' | 'fallback' depending on availability."""
        try:
            import xgboost as xgb  # noqa: F401

            has_xgb = True
        except Exception:
            has_xgb = False
        try:
            import shap  # noqa: F401

            has_shap = True
        except Exception:
            has_shap = False

        if has_xgb and self.model_path and os.path.exists(self.model_path):
            try:
                import xgboost as xgb

                self._model = xgb.XGBClassifier()
                self._model.load_model(self.model_path)
                if has_shap:
                    import shap

                    self._explainer = shap.TreeExplainer(self._model)
                    self._base_value = float(self._explainer.expected_value)
                    return "shap"
                return "xgb"
            except Exception as e:
                logger.warning("XGB model load failed, fallback scoring: {}", e)
        return "fallback"

    # ------------------------------------------------------------------
    def extract_features(
        self,
        origin_lon: float,
        origin_lat: float,
        origin_time: datetime,
        window_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """DuckDB spatial aggregation around the origin ellipse centre."""
        if not self._ensure_view():
            return []
        try:
            con = self._con
            t = origin_time.strftime("%Y-%m-%d %H:%M:%S")
            rows = con.execute(
                f"""
                WITH windowed_ais AS (
                    SELECT mmsi, timestamp, sog, vessel_type,
                           ST_Distance(geom, ST_Point({float(origin_lon)}, {float(origin_lat)})) AS dist_deg,
                           LAG(timestamp) OVER (PARTITION BY mmsi ORDER BY timestamp) AS prev_time
                    FROM ais_data
                    WHERE timestamp BETWEEN CAST('{t}' AS TIMESTAMP) - INTERVAL {int(window_hours)} HOUR
                                        AND CAST('{t}' AS TIMESTAMP) + INTERVAL {int(window_hours)} HOUR
                ),
                vessel_aggregates AS (
                    SELECT mmsi,
                           MIN(dist_deg) * 111000 AS distance_to_origin,
                           MIN(sog) AS min_speed_in_window,
                           MAX(EXTRACT(EPOCH FROM (timestamp - prev_time))) AS ais_gap_duration,
                           ANY_VALUE(vessel_type) AS vessel_type
                    FROM windowed_ais
                    GROUP BY mmsi
                )
                SELECT mmsi, distance_to_origin, min_speed_in_window,
                       COALESCE(ais_gap_duration, 0) AS ais_gap_duration, vessel_type
                FROM vessel_aggregates
                WHERE distance_to_origin < 50000
                """
            ).fetchall()
            out: list[dict[str, Any]] = []
            for mmsi, dist_m, min_sog, gap_s, vtype in rows:
                vt = str(vtype or "")
                w = 1.0 if vt.lower() in ("tanker", "cargo", "81", "82", "70", "71", "72") else 0.2
                out.append(
                    {
                        "mmsi": str(mmsi),
                        "distance_to_origin": float(dist_m or 0.0),
                        "min_speed_in_window": float(min_sog or 0.0),
                        "ais_gap_duration": float(gap_s or 0.0),
                        "vessel_type_weight": float(w),
                    }
                )
            return out
        except Exception as e:
            logger.warning("DuckDB feature extraction failed: {}", e)
            return []

    # ------------------------------------------------------------------
    def score_candidates(
        self,
        origin_lon: float,
        origin_lat: float,
        origin_time: datetime,
        window_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Score candidates; each item has xgb_score + shap_breakdown + method."""
        feats = self.extract_features(origin_lon, origin_lat, origin_time, window_hours)
        if not feats:
            return []
        mode = self._ensure_model()
        results: list[dict[str, Any]] = []

        if mode in ("xgb", "shap") and self._model is not None:
            try:
                import numpy as _np

                X = _np.array([[f[k] for k in self.FEATURES] for f in feats])
                scores = self._model.predict_proba(X)[:, 1]
                if mode == "shap" and self._explainer is not None:
                    shap_values = self._explainer.shap_values(X)
                    for i, f in enumerate(feats):
                        breakdown = {
                            k: float(shap_values[i, j]) for j, k in enumerate(self.FEATURES)
                        }
                        results.append(
                            {
                                "mmsi": f["mmsi"],
                                "xgb_score": float(scores[i]),
                                "features": {k: float(f[k]) for k in self.FEATURES},
                                "shap_breakdown": breakdown,
                                "base_value": float(self._base_value),
                                "method": "xgb_shap",
                            }
                        )
                else:
                    for i, f in enumerate(feats):
                        results.append(
                            {
                                "mmsi": f["mmsi"],
                                "xgb_score": float(scores[i]),
                                "features": {k: float(f[k]) for k in self.FEATURES},
                                "shap_breakdown": {},
                                "base_value": 0.0,
                                "method": "xgb",
                            }
                        )
                return results
            except Exception as e:
                logger.warning("XGB/SHAP scoring failed, linear fallback: {}", e)

        # Deterministic linear fallback (no heavy deps, labelled as such).
        for f in feats:
            dist_score = max(0.0, 1.0 - f["distance_to_origin"] / 50000.0)
            gap_score = min(1.0, f["ais_gap_duration"] / 3600.0 / 4.0)
            slow_score = max(0.0, 1.0 - f["min_speed_in_window"] / 10.0)
            score = (
                0.45 * dist_score
                + 0.25 * gap_score
                + 0.15 * slow_score
                + 0.15 * f["vessel_type_weight"]
            )
            results.append(
                {
                    "mmsi": f["mmsi"],
                    "xgb_score": round(float(max(0.0, min(1.0, score))), 4),
                    "features": {k: float(f[k]) for k in self.FEATURES},
                    "shap_breakdown": {
                        "distance_to_origin": round(0.45 * dist_score, 4),
                        "ais_gap_duration": round(0.25 * gap_score, 4),
                        "min_speed_in_window": round(0.15 * slow_score, 4),
                        "vessel_type_weight": round(0.15 * f["vessel_type_weight"], 4),
                    },
                    "base_value": 0.0,
                    "method": "linear_fallback",
                }
            )
        return results

    def to_json(self, *args: Any, **kwargs: Any) -> str:
        return json.dumps(self.score_candidates(*args, **kwargs), indent=2)
