"""Fail-closed provenance: every source must refuse to invent data.

Background
----------
``docs/VERIFICATION.md`` §9.2 records the failure this file exists to prevent:
an ERA5 pull silently degraded to synthetic wind and *the run succeeded and
returned invented numbers*. The rule now is that a missing upstream is an error
unless an operator explicitly opens a gate, and that anything synthetic says so
in the payload itself.

Every test here runs **offline**: the fetch functions are monkeypatched, so no
call reaches CDSE, CMEMS, CDS or AISHub. Anything that would need the network
is a bug in this file, not a flake.

Note on the environment: the gates are read from ``os.environ`` at call time, so
tests mutate the environment through the ``clean_env`` fixture rather than
setting module globals.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
INGEST_DIR = ROOT / "services" / "ingest"

# The archive router creates DATA_DIR/sar at import time; keep it off the tree.
os.environ.setdefault("SENTINEL_DATA_DIR", str(ROOT / "services" / "ingest" / "tests" / ".tmp"))


def _purge_app_namespace() -> None:
    """Every service ships a top-level `app` package, so they collide in
    sys.modules when the whole suite runs in one process.
    """
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]


_purge_app_namespace()
sys.path.insert(0, str(INGEST_DIR))

from app.provenance import (  # noqa: E402
    AIS_GATES,
    FORCING_GATES,
    PROVENANCE_NO_COVERAGE,
    PROVENANCE_SYNTHETIC,
    ProvenanceError,
    decide_synthetic,
    env_flag,
)
from app.sources.ais import AisFetcher  # noqa: E402
from app.sources.cmems import CmemsFetcher  # noqa: E402
from app.sources.era5 import Era5Fetcher  # noqa: E402

WAKASHIO = (57.6, -21.0, 58.2, -20.4)  # open Indian Ocean — no real receivers
MUMBAI = (72.6, 18.8, 73.1, 19.3)  # coastal carve-out — real receivers exist
NORTH_SEA = (3.0, 52.0, 5.0, 54.0)  # outside the no-coverage box entirely
# Straddles the no-coverage boundary: half open ocean, half coastal.
STRADDLE = (56.4, 26.2, 57.4, 27.2)

# Small enough that the synthetic NetCDF fallback is written in milliseconds.
TINY_BBOX = (57.6, -21.0, 58.2, -20.4)
TINY_DEPTH = (0.0, 4.0)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Strip every synthetic gate so 'unset' really means unset."""
    for name in (
        "ALLOW_SYNTHETIC_FORCING",
        "ALLOW_SYNTHETIC_AIS",
        "SENTINEL_DEMO_MODE",
        "ALLOW_SYNTHETIC_TRAINING",
    ):
        monkeypatch.delenv(name, raising=False)
    return {}  # tests that need an explicit env pass it to decide_synthetic


@pytest.fixture
def enable_forcing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_SYNTHETIC_FORCING", "true")


@pytest.fixture
def enable_demo_ais(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_DEMO_MODE", "true")
    monkeypatch.setenv("ALLOW_SYNTHETIC_AIS", "true")


def _boom(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("upstream 503 — simulated outage")


# ── the gate itself ───────────────────────────────────────────────────────


def test_env_flag_unset_is_false(clean_env):  # noqa: ARG001
    assert env_flag("ALLOW_SYNTHETIC_FORCING") is False
    assert env_flag("ALLOW_SYNTHETIC_FORCING", {}) is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_env_flag_truthy_forms(value: str):
    assert env_flag("X", {"X": value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", "troo"])
def test_env_flag_unrecognised_is_false(value: str):
    """A typo must never open the gate."""
    assert env_flag("X", {"X": value}) is False


def test_forcing_gate_closed_by_default(clean_env):  # noqa: ARG001
    decision = decide_synthetic("cmems")
    assert decision.allowed is False
    assert decision.provenance != PROVENANCE_SYNTHETIC
    assert decision.reason == "ALLOW_SYNTHETIC_FORCING_not_enabled"
    assert decision.failed_gates == FORCING_GATES


def test_ais_gate_needs_both_flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SENTINEL_DEMO_MODE", "true")
    monkeypatch.delenv("ALLOW_SYNTHETIC_AIS", raising=False)
    decision = decide_synthetic("ais", required_env=AIS_GATES)
    assert decision.allowed is False
    assert decision.reason == "ALLOW_SYNTHETIC_AIS_not_enabled"

    monkeypatch.setenv("ALLOW_SYNTHETIC_AIS", "true")
    assert decide_synthetic("ais", required_env=AIS_GATES).allowed is True


def test_geographic_gate_is_a_gate(monkeypatch: pytest.MonkeyPatch):
    """Both env flags set is still not enough outside a permitted box."""
    monkeypatch.setenv("SENTINEL_DEMO_MODE", "true")
    monkeypatch.setenv("ALLOW_SYNTHETIC_AIS", "true")
    decision = decide_synthetic("ais", required_env=AIS_GATES, geo_allowed=False)
    assert decision.allowed is False
    assert decision.reason == "outside_permitted_box"
    assert decision.provenance == PROVENANCE_NO_COVERAGE


# ── CMEMS ─────────────────────────────────────────────────────────────────


async def test_cmems_failure_raises_when_gate_closed(
    clean_env,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(CmemsFetcher, "_download_via_cds", _boom)
    fetcher = CmemsFetcher(api_key="x", storage_path=str(tmp_path))

    with pytest.raises(ProvenanceError) as exc:
        await fetcher.fetch(bbox=TINY_BBOX, depth_range=TINY_DEPTH, lookback_days=2)

    err = exc.value
    assert err.source == "cmems"
    assert err.reason == "ALLOW_SYNTHETIC_FORCING_not_enabled"
    assert "cmems" in str(err)
    assert err.required_env == FORCING_GATES
    # Machine-readable envelope, not just a sentence.
    assert err.to_dict()["provenance"] == "unavailable"
    assert err.to_dict()["error"]["reason"] == "ALLOW_SYNTHETIC_FORCING_not_enabled"


async def test_cmems_failure_under_flag_returns_labelled_synthetic(
    clean_env,  # noqa: ARG001
    enable_forcing,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(CmemsFetcher, "_download_via_cds", _boom)
    fetcher = CmemsFetcher(api_key="x", storage_path=str(tmp_path))

    result = await fetcher.fetch(bbox=TINY_BBOX, depth_range=TINY_DEPTH, lookback_days=2)

    assert result.provenance == PROVENANCE_SYNTHETIC
    assert result.is_synthetic is True
    assert result.warning, "synthetic forcing must carry a visible warning"
    assert "SYNTHETIC" in result.warning.upper()
    assert result.total_bytes > 0


async def test_cmems_success_is_stamped_real(
    clean_env,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """A real download must never be labelled synthetic."""

    async def _fake_download(self: CmemsFetcher, _payload: dict, out_file: str) -> int:
        _write_currents(out_file)
        return os.path.getsize(out_file)

    monkeypatch.setattr(CmemsFetcher, "_download_via_cds", _fake_download)
    fetcher = CmemsFetcher(api_key="x", storage_path=str(tmp_path))

    result = await fetcher.fetch(bbox=TINY_BBOX, depth_range=TINY_DEPTH, lookback_days=1)
    assert result.provenance == "cmems"
    assert result.is_synthetic is False
    assert result.warning is None


# ── ERA5 ──────────────────────────────────────────────────────────────────


async def test_era5_failure_raises_when_gate_closed(
    clean_env,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(Era5Fetcher, "_download_via_cds", _boom)
    fetcher = Era5Fetcher(api_key="x", storage_path=str(tmp_path))

    with pytest.raises(ProvenanceError) as exc:
        await fetcher.fetch(bbox=TINY_BBOX, lookback_days=2)

    err = exc.value
    assert err.source == "era5"
    assert err.reason == "ALLOW_SYNTHETIC_FORCING_not_enabled"
    assert "era5" in str(err).lower()


async def test_era5_failure_under_flag_returns_labelled_synthetic(
    clean_env,  # noqa: ARG001
    enable_forcing,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(Era5Fetcher, "_download_via_cds", _boom)
    fetcher = Era5Fetcher(api_key="x", storage_path=str(tmp_path))

    result = await fetcher.fetch(bbox=TINY_BBOX, lookback_days=2)

    assert result.provenance == PROVENANCE_SYNTHETIC
    assert result.is_synthetic is True
    assert result.warning and "SYNTHETIC" in result.warning.upper()


async def test_era5_success_is_stamped_real(
    clean_env,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    async def _fake_download(self: Era5Fetcher, _payload: dict, out_file: str) -> int:
        _write_winds(out_file)
        return os.path.getsize(out_file)

    monkeypatch.setattr(Era5Fetcher, "_download_via_cds", _fake_download)
    fetcher = Era5Fetcher(api_key="x", storage_path=str(tmp_path))

    result = await fetcher.fetch(bbox=TINY_BBOX, lookback_days=1)
    assert result.provenance == "era5"
    assert result.is_synthetic is False


# ── AIS ───────────────────────────────────────────────────────────────────


async def test_ais_no_coverage_returns_empty_state_not_synthetic(
    clean_env,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
):
    """No receivers and no gate -> zero vessels, and provenance is NOT synthetic."""
    monkeypatch.setattr(AisFetcher, "_fetch_marinecadastre", _boom)
    payload = await AisFetcher(api_key="x").fetch_envelope(
        WAKASHIO, _utc(2020, 8, 9), _utc(2020, 8, 10)
    )

    assert payload["provenance"] == PROVENANCE_NO_COVERAGE
    assert payload["is_synthetic"] is False
    assert payload["provenance"] != PROVENANCE_SYNTHETIC
    assert payload["count"] == 0
    assert payload["vessels"] == []
    assert payload["error"]["reason"] == "+".join(
        f"{name}_not_enabled" for name in AIS_GATES
    )
    assert set(payload["error"]["required_env"]) == set(AIS_GATES)


async def test_ais_synthetic_only_with_both_flags_and_permitted_box(
    clean_env,  # noqa: ARG001
    enable_demo_ais,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(AisFetcher, "_fetch_marinecadastre", _boom)
    payload = await AisFetcher(api_key="x").fetch_envelope(
        WAKASHIO, _utc(2020, 8, 9), _utc(2020, 8, 10), n_vessels=3
    )

    assert payload["provenance"] == PROVENANCE_SYNTHETIC
    assert payload["is_synthetic"] is True
    assert payload["count"] == 3
    # Every record — not just the envelope — carries the label.
    for vessel in payload["vessels"]:
        assert vessel["provenance"] == PROVENANCE_SYNTHETIC
        for fix in vessel["track"]:
            assert fix["provenance"] == PROVENANCE_SYNTHETIC
    assert payload["disclaimer"]["provenance"] == PROVENANCE_SYNTHETIC


async def test_ais_synthetic_rejected_in_coastal_carveout(
    clean_env,  # noqa: ARG001
    enable_demo_ais,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
):
    """Mumbai has real receivers: synthetic must be refused there even with flags."""
    monkeypatch.setattr(AisFetcher, "_fetch_marinecadastre", _boom)
    payload = await AisFetcher(api_key="x").fetch_envelope(
        MUMBAI, _utc(2020, 8, 9), _utc(2020, 8, 10)
    )

    assert payload["provenance"] == PROVENANCE_NO_COVERAGE
    assert payload["is_synthetic"] is False
    assert payload["count"] == 0


async def test_ais_synthetic_rejected_outside_no_coverage_box(
    clean_env,  # noqa: ARG001
    enable_demo_ais,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(AisFetcher, "_fetch_marinecadastre", _boom)
    payload = await AisFetcher(api_key="x").fetch_envelope(
        NORTH_SEA, _utc(2020, 8, 9), _utc(2020, 8, 10)
    )
    assert payload["provenance"] == PROVENANCE_NO_COVERAGE
    assert payload["count"] == 0


async def test_ais_demo_mode_alone_is_not_enough(
    clean_env,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SENTINEL_DEMO_MODE", "true")
    monkeypatch.setattr(AisFetcher, "_fetch_marinecadastre", _boom)
    payload = await AisFetcher(api_key="x").fetch_envelope(
        WAKASHIO, _utc(2020, 8, 9), _utc(2020, 8, 10)
    )
    assert payload["provenance"] == PROVENANCE_NO_COVERAGE
    assert payload["error"]["required_env"] == list(AIS_GATES)


async def test_ais_fetch_returns_no_coverage_state(clean_env, monkeypatch):  # noqa: ARG001
    """The DB-facing fetch path fails closed the same way."""
    monkeypatch.setattr(AisFetcher, "_fetch_marinecadastre", _boom)
    result = await AisFetcher(api_key="x").fetch(bbox=WAKASHIO, lookback_hours=6)

    assert result.provenance == PROVENANCE_NO_COVERAGE
    assert result.is_synthetic is False
    assert result.unique_vessels == 0
    assert result.records_fetched == 0
    assert result.synthetic_reason == "real_ais_unavailable"


async def test_ais_real_feed_is_labelled_live(clean_env, monkeypatch):  # noqa: ARG001
    async def _fake_live(
        self: AisFetcher, bbox: tuple[float, float, float, float], hours: int
    ) -> list[dict]:
        return [
            {
                "MMSI": "477218700",
                "TIMESTAMP": "2020-08-09T01:00:00Z",
                "LATITUDE": 19.0,
                "LONGITUDE": 72.9,
                "SPEED": 11.2,
                "COURSE": 180.0,
                "NAME": "REAL VESSEL",
            },
            {
                "MMSI": "477218700",
                "TIMESTAMP": "2020-08-09T02:00:00Z",
                "LATITUDE": 19.1,
                "LONGITUDE": 72.9,
                "SPEED": 11.4,
                "COURSE": 181.0,
                "NAME": "REAL VESSEL",
            },
            {"MMSI": "123", "TIMESTAMP": "2020-08-09T01:00:00Z"},  # dropped: bad MMSI
        ]

    monkeypatch.setattr(AisFetcher, "_fetch_marinecadastre", _fake_live)
    payload = await AisFetcher(api_key="x").fetch_envelope(
        MUMBAI, _utc(2020, 8, 9), _utc(2020, 8, 10)
    )

    assert payload["provenance"] == "live_terrestrial"
    assert payload["is_synthetic"] is False
    assert payload["count"] == 1
    assert payload["vessels"][0]["provenance"] == "live_terrestrial"
    assert len(payload["vessels"][0]["track"]) == 2


# ── partial coverage is reported, never blended ────────────────────────────


def test_partial_coverage_is_reported_not_blended():
    from app.sources.ais import ais_coverage_block

    block = ais_coverage_block(STRADDLE)
    # Half the AOI has real receivers (Hormuz carve-out), half is open ocean.
    assert block["status"] == "partial"
    assert block["coastal_carveouts"], "the covered part must be named"
    assert block["partial_note"], "the operator must be told how it was resolved"

    assert ais_coverage_block(WAKASHIO)["status"] == "none"
    assert ais_coverage_block(NORTH_SEA)["status"] == "full"


async def test_partial_aoi_never_returns_synthetic_rows(
    clean_env,  # noqa: ARG001
    enable_demo_ais,  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
):
    """Even with both flags set, a straddling AOI gets real data or nothing."""
    monkeypatch.setattr(AisFetcher, "_fetch_marinecadastre", _boom)
    payload = await AisFetcher(api_key="x").fetch_envelope(
        STRADDLE, _utc(2020, 8, 9), _utc(2020, 8, 10)
    )
    assert payload["provenance"] == PROVENANCE_NO_COVERAGE
    assert payload["vessels"] == []
    assert payload["coverage"]["status"] == "partial"


# ── SAR ───────────────────────────────────────────────────────────────────


def test_missing_sar_dataset_raises_naming_the_path(tmp_path: Path):
    from app.processors.sar_preprocessor import SarPreprocessor

    missing = tmp_path / "no_such_scene.tif"
    processor = SarPreprocessor(output_dir=str(tmp_path / "out"))

    with pytest.raises(ProvenanceError) as exc:
        processor.preprocess(str(missing))

    err = exc.value
    assert err.reason == "sar_dataset_missing"
    assert str(missing.resolve()) in err.detail
    assert str(missing) in str(err)
    assert err.provenance == "unavailable"


# ── the named-bbox contract still holds ───────────────────────────────────


def test_named_bbox_contract_still_rejects_partial_set():
    """Pinned by an existing test; repeated here so a provenance refactor
    cannot quietly regress it."""
    from fastapi import HTTPException

    from app.routers.archive import _resolve_bbox

    with pytest.raises(HTTPException) as exc:
        _resolve_bbox(57.6, -21.0, None, None, None)
    assert exc.value.status_code == 422
    assert "together" in str(exc.value.detail)


# ── synthetic training gate ───────────────────────────────────────────────


def test_synthetic_train_refuses_without_the_flag(clean_env, tmp_path):  # noqa: ARG001
    from scripts import synthetic_train

    with pytest.raises(RuntimeError) as exc:
        synthetic_train.generate_benchmark_dataset(base_dir=str(tmp_path / "syn"), n_train=1)
    assert "ALLOW_SYNTHETIC_TRAINING" in str(exc.value)
    assert not (tmp_path / "syn").exists()


def test_synthetic_train_stamps_every_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import synthetic_train

    monkeypatch.setenv("ALLOW_SYNTHETIC_TRAINING", "true")
    base = tmp_path / "syn"
    manifest = synthetic_train.generate_benchmark_dataset(
        base_dir=str(base), n_train=2, n_val=1, n_test=1
    )

    assert manifest["provenance"] == PROVENANCE_SYNTHETIC
    assert manifest["is_synthetic"] is True
    assert manifest["count"] == len(manifest["samples"]) == 5
    for sample in manifest["samples"]:
        assert sample["provenance"] == PROVENANCE_SYNTHETIC
        # The stamp is inside the file, not only in the manifest.
        meta = _png_text(sample["image"])
        assert meta["provenance"] == PROVENANCE_SYNTHETIC
        assert meta["provenance"] in ("synthetic_mock",)
    assert (base / "provenance.json").exists()


# ── helpers ───────────────────────────────────────────────────────────────


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _write_currents(out_file: str) -> None:
    # Naive on purpose: numpy datetime64 carries no timezone and warns if given one.
    now = datetime.now(UTC).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    times = np.array([now - timedelta(hours=6), now, now + timedelta(hours=6)], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            "uo": (["time", "depth", "latitude", "longitude"], np.zeros((3, 1, 2, 2), "f4")),
            "vo": (["time", "depth", "latitude", "longitude"], np.zeros((3, 1, 2, 2), "f4")),
        },
        coords={
            "time": times,
            "depth": np.array([0.5], "f4"),
            "latitude": np.array([-21.0, -20.4], "f4"),
            "longitude": np.array([57.6, 58.2], "f4"),
        },
    )
    ds.to_netcdf(out_file)


def _write_winds(out_file: str) -> None:
    # Naive on purpose: numpy datetime64 carries no timezone and warns if given one.
    now = datetime.now(UTC).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    times = np.array([now - timedelta(hours=3), now, now + timedelta(hours=3)], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {
            "u10": (["time", "latitude", "longitude"], np.zeros((3, 2, 2), "f4")),
            "v10": (["time", "latitude", "longitude"], np.zeros((3, 2, 2), "f4")),
        },
        coords={
            "time": times,
            "latitude": np.array([-21.0, -20.4], "f4"),
            "longitude": np.array([57.6, 58.2], "f4"),
        },
    )
    ds.to_netcdf(out_file)


def _png_text(path: str) -> dict[str, str]:
    from PIL import Image

    with Image.open(path) as img:
        return {k: v for k, v in img.text.items()}  # type: ignore[union-attr]
