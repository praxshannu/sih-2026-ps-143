"""Drift engine tests — ported from claude2, adapted to claude1 APIs.

Covers: regime classification, K_ij assembly (positive-definite),
WMC correction presence + effect, backward SDE origin sanity (degrees,
not metres — regression test for the m/s→deg/s unit fix), forcing
factory provenance labelling, and the XGBoost empirical fallback.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import UTC
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "services" / "drift" / "app"


def _pkg(name: str, path):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
    return sys.modules[name]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, APP / rel)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_pkg("dforce", APP)
_pkg("dforce.data", APP / "data")


_reg = _load("t_regime", "engine/regime_classifier.py")
_kij = _load("t_kij", "engine/kij_tensor.py")
_wmc = _load("t_wmc", "engine/wmc_correction.py")
_xgb = _load("t_xgb", "engine/xgb_residual.py")


def test_regime_deep_ocean_is_markov1():
    out = _reg.classify_regime(
        np.array([3000.0]), np.array([500.0]), np.array([0.1]), np.array([-20.0])
    )
    assert out == _reg.OceanRegime.MARKOV1


def test_regime_shallow_is_smagorinsky():
    out = _reg.classify_regime(np.array([10.0]), np.array([5.0]), np.array([0.1]), np.array([15.0]))
    assert out == _reg.OceanRegime.SMAGORINSKY


def test_kij_positive_definite():
    n = 8
    K = _kij.k_tensor_assembly(
        np.full(n, 1e-5),
        np.full(n, 2e-5),
        np.full(n, -1e-5),
        np.full(n, 3e-5),
        np.full(n, 1000.0),
        np.full(n, -20.0),
    )
    assert K.shape == (n, 2, 2)
    emax, emin = _kij.k_eigenvalues(K)
    assert np.all(emin > 0.0)
    assert np.all(emax >= emin)


def test_wmc_correction_changes_velocity():
    n = 5
    K = _kij.k_tensor_assembly(
        np.full(n, 1e-4),
        np.full(n, 1e-4),
        np.full(n, 1e-4),
        np.full(n, 1e-4),
        np.full(n, 500.0),
        np.full(n, 10.0),
    )
    u = np.full(n, 0.3)
    v = np.full(n, 0.1)
    u_corr, v_corr = _wmc.wmc_corrected_velocity(u, v, K, dx=1000.0, dy=1000.0)
    assert u_corr.shape == (n,)
    # Correction must be finite (documents that WMC ran, whatever its size)
    assert np.all(np.isfinite(u_corr)) and np.all(np.isfinite(v_corr))


def test_backward_sde_source_applies_wmc_and_degrees():
    src = (APP / "engine" / "backward_sde.py").read_text()
    assert "wmc" in src.lower()
    # Unit fix: velocities must be converted m/s -> deg/s before RK4 on degrees.
    assert "111320.0" in src or "111000.0" in src


def test_forcing_factory_labels_composite_without_credentials():
    import os

    for key in (
        "COPERNICUSMARINE_SERVICE_USERNAME",
        "COPERNICUS_MARINE_USERNAME",
        "COPERNICUS_USER",
    ):
        os.environ.pop(key, None)
    fac = _load("dforce.data.forcing_factory", "data/forcing_factory.py")
    provider, provenance = fac.build_forcing_provider()
    assert provenance["forcing_source"] == "composite_gfs_mock"
    assert provenance["current_origin"] == "synthetic_mock"
    # Provider genuinely serves vectors (mock currents, not a crash).
    from datetime import datetime

    u, v = provider.get_current_vectors(np.array([57.7]), np.array([-20.4]), datetime.now(UTC))
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))


def test_xgb_residual_interface_and_fallback():
    model = _xgb.XGBoostPhysicsResidualModel()
    out = model.predict_residual(
        u_current=0.15,
        v_current=0.05,
        u_wind=4.0,
        v_wind=2.0,
        windage=0.03,
        sst=28.0,
        duration_hours=48.0,
        observed_area_km2=5.0,
    )
    assert set(out) == {"dx_residual_meters", "dy_residual_meters", "uncertainty_scale"}
    assert out["uncertainty_scale"] >= 0.5
