"""Detect inference test — runs on synthetic SAR inside containers/CI.

Skips cleanly where torch is unavailable (host without GPU deps).
Spec: 256x256 synthetic SAR -> assert output shape and value range.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_unet = _load("sentinel_unetpp", "services/detect/app/models/unetpp_scse.py")
_tiled = _load("sentinel_tiled", "services/detect/app/models/inference.py")


def _synthetic_sar(size: int = 256, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    img = rng.normal(0.6, 0.1, size=(6, size, size)).astype("float32")
    # Dark rectangular slick in VV channel.
    img[0, 90:170, 90:170] -= 0.4
    return np.clip(img, 0.0, 1.0)


def test_tiled_inference_shape_and_range():
    image = _synthetic_sar()
    model = _unet.UNetPlusPlusSCSE(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=1,
    )
    model.eval()
    engine = _tiled.TiledInference(model, tile_size=128, overlap=16, device="cpu")
    logits = engine.predict(image, batch_size=2)
    assert logits.shape == (1, 256, 256)
    assert bool((logits > -50).all() and (logits < 50).all())


def test_channel_adapter_never_crashes():
    import numpy as np

    model = _unet.UNetPlusPlusSCSE(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=1,
    )
    model.eval()
    engine = _tiled.TiledInference(model, tile_size=128, overlap=16, device="cpu")
    # 3-channel input against a 6-channel model must pad, not crash.
    out = engine.predict(np.zeros((3, 256, 256), dtype="float32"), batch_size=2)
    assert out.shape == (1, 256, 256)
