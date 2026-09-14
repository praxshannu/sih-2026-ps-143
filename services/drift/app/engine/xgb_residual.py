"""XGBoost physics residual + uncertainty correction for SENTINEL drift.

Ported from OceanTrace agent2/ml/xgboost_residual.py — interface preserved
(predict_residual / build_features / save / load), imports adapted, and the
OceanTrace OilProperties dependency replaced with plain floats so the drift
service stays lightweight. Falls back to the empirical literature formula
when xgboost is not installed, so the service never hard-crashes on import.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

try:
    import xgboost as xgb

    _XGB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    xgb = None  # type: ignore[assignment]
    _XGB_AVAILABLE = False


class XGBoostPhysicsResidualModel:
    """Gradient-boosted correction: residual = observed − physics displacement.

    Predicts (dx_m, dy_m) post-hoc correction plus an uncertainty scale factor
    applied to the hindcast ellipse semi-axes.
    """

    FEATURE_NAMES = [
        "u_current",
        "v_current",
        "u_wind",
        "v_wind",
        "windage",
        "sst",
        "duration_hours",
        "observed_area_km2",
        "aspect_ratio",
        "oil_density",
        "oil_viscosity",
    ]

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.is_trained = False
        self.model_dx: Any = None
        self.model_dy: Any = None
        self.model_uncertainty: Any = None

        if _XGB_AVAILABLE:
            self.model_dx = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                objective="reg:squarederror",
                random_state=42,
            )
            self.model_dy = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                objective="reg:squarederror",
                random_state=42,
            )
            self.model_uncertainty = xgb.XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                objective="reg:squarederror",
                random_state=42,
            )

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            default_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "models",
                    "xgb_residual",
                )
            )
            if os.path.exists(os.path.join(default_dir, "xgb_dx_residual.json")):
                self.load(default_dir)
            else:
                self._fit_default_baseline()

    # ------------------------------------------------------------------
    def _fit_default_baseline(self) -> None:
        """Baseline fit on synthetic physics calibration bounds.

        Uses the empirical Stokes-drift residual distribution from the
        literature when no trained checkpoint exists. If xgboost is missing,
        just marks the empirical fallback as trained.
        """
        if not _XGB_AVAILABLE:
            self.is_trained = True
            return
        rng = np.random.RandomState(42)
        n_samples = 500

        u_curr = rng.uniform(-0.8, 0.8, n_samples)
        v_curr = rng.uniform(-0.8, 0.8, n_samples)
        u_wind = rng.uniform(-15.0, 15.0, n_samples)
        v_wind = rng.uniform(-15.0, 15.0, n_samples)
        windage = rng.uniform(0.025, 0.044, n_samples)
        sst = rng.uniform(15.0, 32.0, n_samples)
        dur = rng.uniform(2.0, 72.0, n_samples)
        area = rng.uniform(0.5, 50.0, n_samples)
        ar = rng.uniform(1.0, 5.0, n_samples)
        density = rng.uniform(820.0, 950.0, n_samples)
        visc = rng.uniform(5.0, 150.0, n_samples)

        X = np.column_stack(
            [u_curr, v_curr, u_wind, v_wind, windage, sst, dur, area, ar, density, visc]
        )
        dx_res = 0.005 * u_wind * dur * 3600.0 * (1.0 + 0.1 * rng.randn(n_samples))
        dy_res = 0.005 * v_wind * dur * 3600.0 * (1.0 + 0.1 * rng.randn(n_samples))
        unc_scale = 1.0 + 0.05 * np.sqrt(u_wind**2 + v_wind**2) * (dur / 24.0)

        assert self.model_dx is not None
        self.model_dx.fit(X, dx_res)
        assert self.model_dy is not None
        self.model_dy.fit(X, dy_res)
        assert self.model_uncertainty is not None
        self.model_uncertainty.fit(X, unc_scale)
        self.is_trained = True

    # ------------------------------------------------------------------
    def build_features(
        self,
        u_current: float,
        v_current: float,
        u_wind: float,
        v_wind: float,
        windage: float,
        sst: float,
        duration_hours: float,
        observed_area_km2: float,
        aspect_ratio: float,
        oil_density: float = 880.0,
        oil_viscosity: float = 50.0,
    ) -> np.ndarray:
        """Assemble the 11-dim feature vector for inference."""
        return np.array(
            [
                u_current,
                v_current,
                u_wind,
                v_wind,
                windage,
                sst,
                duration_hours,
                observed_area_km2,
                aspect_ratio,
                oil_density,
                oil_viscosity,
            ],
            dtype=np.float32,
        ).reshape(1, -1)

    def predict_residual(
        self,
        u_current: float,
        v_current: float,
        u_wind: float,
        v_wind: float,
        windage: float,
        sst: float,
        duration_hours: float,
        observed_area_km2: float,
        aspect_ratio: float = 1.0,
        oil_density: float = 880.0,
        oil_viscosity: float = 50.0,
    ) -> Dict[str, float]:
        """Predict displacement correction (dx_m, dy_m) + uncertainty scale."""
        if not _XGB_AVAILABLE or not self.is_trained or self.model_dx is None:
            dx_m = 0.005 * u_wind * duration_hours * 3600.0
            dy_m = 0.005 * v_wind * duration_hours * 3600.0
            unc = 1.0 + 0.05 * float(np.hypot(u_wind, v_wind)) * (duration_hours / 24.0)
            return {
                "dx_residual_meters": round(float(dx_m), 2),
                "dy_residual_meters": round(float(dy_m), 2),
                "uncertainty_scale": round(max(float(unc), 0.5), 3),
            }
        X = self.build_features(
            u_current=u_current,
            v_current=v_current,
            u_wind=u_wind,
            v_wind=v_wind,
            windage=windage,
            sst=sst,
            duration_hours=duration_hours,
            observed_area_km2=observed_area_km2,
            aspect_ratio=aspect_ratio,
            oil_density=oil_density,
            oil_viscosity=oil_viscosity,
        )
        dx_m = float(self.model_dx.predict(X)[0])
        dy_m = float(self.model_dy.predict(X)[0])
        unc_scale = float(self.model_uncertainty.predict(X)[0])
        return {
            "dx_residual_meters": round(dx_m, 2),
            "dy_residual_meters": round(dy_m, 2),
            "uncertainty_scale": round(max(unc_scale, 0.5), 3),
        }

    # ------------------------------------------------------------------
    def save(self, directory: str) -> None:
        """Serialize trained XGBoost models to directory."""
        if not _XGB_AVAILABLE:
            raise RuntimeError("xgboost is not installed; cannot save models")
        os.makedirs(directory, exist_ok=True)
        assert self.model_dx is not None
        self.model_dx.save_model(os.path.join(directory, "xgb_dx_residual.json"))
        assert self.model_dy is not None
        self.model_dy.save_model(os.path.join(directory, "xgb_dy_residual.json"))
        assert self.model_uncertainty is not None
        self.model_uncertainty.save_model(
            os.path.join(directory, "xgb_uncertainty.json")
        )

    def load(self, directory: str) -> None:
        """Load serialized XGBoost models."""
        if not _XGB_AVAILABLE:
            self.is_trained = True
            return
        assert self.model_dx is not None
        self.model_dx.load_model(os.path.join(directory, "xgb_dx_residual.json"))
        assert self.model_dy is not None
        self.model_dy.load_model(os.path.join(directory, "xgb_dy_residual.json"))
        assert self.model_uncertainty is not None
        self.model_uncertainty.load_model(
            os.path.join(directory, "xgb_uncertainty.json")
        )
        self.is_trained = True
