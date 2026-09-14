"""Missing forcing fields must fail closed, with a reason the API can surface.

The No-Fabrication Policy is only worth anything if it is enforced where the
data is actually fetched. A missing wind or current field therefore has to
raise a *typed* error carrying a machine-readable ``code``/``field``/``reason``
triple — never a mock field with a polite log line, and never a bare 500 whose
``detail`` is a sentence the UI would have to parse to understand.

Every provider here is monkeypatched. No test touches the network, CDS,
Copernicus Marine or NOMADS, and none needs a database.

Two holes were open when this file was written and are now closed in
production code:

1. ``runner.run_backward_attribution`` / ``run_forward_forecast`` substituted
   constant synthetic wind/current whenever a reader was ``None``, with no way
   for a caller to refuse. They now take ``allow_synthetic``.
2. ``main._build_fields`` silently served a constant analytic current when the
   provider could not be sampled, and reported ``wind=not_staged`` while
   letting the run continue when a local CMEMS file was staged without ERA5.
   Both now raise when ``allow_synthetic=False``.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
APP = ROOT / "services" / "drift" / "app"


def _pkg(name: str, path) -> types.ModuleType:
    """Register a namespace package so relative imports inside the app resolve."""
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
    return sys.modules[name]


for _name, _path in (
    ("dfail", APP),
    ("dfail.data", APP / "data"),
    ("dfail.engine", APP / "engine"),
    ("dfail.sources", APP / "sources"),
):
    _pkg(_name, _path)

errors = importlib.import_module("dfail.errors")
fac = importlib.import_module("dfail.data.forcing_factory")
runner = importlib.import_module("dfail.runner")


# ---------------------------------------------------------------------------
# Fakes — these replace providers so nothing reaches the network.
# ---------------------------------------------------------------------------


class _FakeCurrent:
    """Stands in for CMEMSCurrentProvider; only `live_available` is consulted."""

    product_id = "GLOBAL_ANALYSISFORECAST_PHY_001_024"

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def live_available(self) -> bool:
        return self._available

    def coverage_window(self):
        return "2020-08-09T00:00:00", "2020-08-10T00:00:00"


class _FakeWind:
    """Stands in for ERA5WindProvider."""

    def __init__(self, available: bool = True) -> None:
        self._available = available

    def live_available(self) -> bool:
        return self._available

    def coverage_window(self):
        return "2020-08-09T00:00:00", "2020-08-10T00:00:00"


#: A 2020 window: outside NOMADS retention, so GFS is legitimately refused
#: without any network call.
WINDOW = (
    datetime(2020, 8, 10, 1, 37, tzinfo=UTC),
    datetime(2020, 8, 11, 1, 37, tzinfo=UTC),
)


@pytest.fixture()
def no_real_current(monkeypatch):
    monkeypatch.setattr(fac, "CMEMSCurrentProvider", lambda *a, **k: _FakeCurrent(False))


@pytest.fixture()
def no_real_wind(monkeypatch):
    monkeypatch.setattr(fac, "ERA5WindProvider", lambda *a, **k: _FakeWind(False))


@pytest.fixture()
def real_current(monkeypatch):
    monkeypatch.setattr(fac, "CMEMSCurrentProvider", lambda *a, **k: _FakeCurrent(True))


@pytest.fixture()
def real_wind(monkeypatch):
    monkeypatch.setattr(fac, "ERA5WindProvider", lambda *a, **k: _FakeWind(True))


# ---------------------------------------------------------------------------
# 1. Missing WIND
# ---------------------------------------------------------------------------


def test_missing_wind_raises_typed_error(real_current, no_real_wind):
    with pytest.raises(errors.MissingWindForcingError) as caught:
        fac.build_forcing_provider(window=WINDOW, allow_synthetic=False)

    exc = caught.value
    assert exc.code == "missing_wind_forcing"
    assert exc.field == "wind"
    assert exc.reason == "wind_unavailable_synthetic_refused"


def test_missing_wind_is_not_masked_by_a_current_failure(real_current, no_real_wind):
    """A current field that IS available must not be blamed for missing wind."""
    with pytest.raises(errors.MissingWindForcingError) as caught:
        fac.build_forcing_provider(window=WINDOW, allow_synthetic=False)
    assert not isinstance(caught.value, errors.MissingCurrentForcingError)


def test_missing_wind_never_returns_a_mock_provider(real_current, no_real_wind):
    """The whole point: no provider object is produced, so nothing can serve it."""
    provider = None
    with contextlib.suppress(errors.MissingWindForcingError):
        provider, _provenance = fac.build_forcing_provider(window=WINDOW, allow_synthetic=False)
    assert provider is None


def test_missing_wind_becomes_mock_only_when_explicitly_allowed(real_current, no_real_wind):
    """allow_synthetic=True is the opt-in that makes the difference visible."""
    _provider, provenance = fac.build_forcing_provider(window=WINDOW, allow_synthetic=True)
    assert provenance["wind_origin"] == "synthetic_mock"
    assert provenance["wind"]["is_real"] is False
    assert "wind" in provenance["synthetic_fields"]


# ---------------------------------------------------------------------------
# 2. Missing CURRENT
# ---------------------------------------------------------------------------


def test_missing_current_raises_typed_error(no_real_current, real_wind):
    with pytest.raises(errors.MissingCurrentForcingError) as caught:
        fac.build_forcing_provider(window=WINDOW, allow_synthetic=False)

    exc = caught.value
    assert exc.code == "missing_current_forcing"
    assert exc.field == "current"
    assert exc.reason == "cmems_unavailable_synthetic_refused"


def test_missing_current_is_not_masked_by_a_wind_failure(no_real_current, real_wind):
    with pytest.raises(errors.MissingCurrentForcingError) as caught:
        fac.build_forcing_provider(window=WINDOW, allow_synthetic=False)
    assert not isinstance(caught.value, errors.MissingWindForcingError)


def test_missing_current_from_selection_api(no_real_current, real_wind):
    """build_forcing_selection (metadata-only entry point) fails identically."""
    with pytest.raises(errors.MissingCurrentForcingError):
        fac.build_forcing_selection(window=WINDOW, allow_synthetic=False)


# ---------------------------------------------------------------------------
# 3. Machine-readable payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("missing", "expected_code", "expected_field"),
    [
        ("wind", "missing_wind_forcing", "wind"),
        ("current", "missing_current_forcing", "current"),
    ],
)
def test_error_payload_is_machine_readable(missing, expected_code, expected_field):
    exc = (
        errors.MissingWindForcingError("no wind", reason="wind_unavailable_synthetic_refused")
        if missing == "wind"
        else errors.MissingCurrentForcingError(
            "no current", reason="cmems_unavailable_synthetic_refused"
        )
    )
    payload = exc.to_dict()
    assert set(payload) == {"error", "code", "field", "reason", "message"}
    assert payload["code"] == expected_code
    assert payload["field"] == expected_field
    assert payload["reason"] and payload["reason"] != payload["message"]
    assert all(isinstance(v, str) for v in payload.values())


def test_every_forcing_error_shares_one_contract():
    """A UI switch on `code` is only safe if the vocabulary is closed and typed."""
    for cls in (
        errors.MissingWindForcingError,
        errors.MissingCurrentForcingError,
    ):
        exc = cls("boom", reason="unit_test")
        assert isinstance(exc, errors.EnvironmentalDataError)
        assert isinstance(exc, errors.ForcingUnavailableError)
        assert exc.to_dict()["error"] == cls.__name__


# ---------------------------------------------------------------------------
# 4. The OpenDrift runner must refuse too, not just the factory
# ---------------------------------------------------------------------------


class _Reached(Exception):
    """Raised by the stubbed OpenDrift import to prove the gate was passed."""


@pytest.fixture()
def stub_opendrift(monkeypatch):
    monkeypatch.setattr(runner, "_opendrift", lambda: (_ for _ in ()).throw(_Reached()))


def _sentinel_reader():
    return object()


def test_hindcast_refuses_synthetic_wind(stub_opendrift):
    with pytest.raises(errors.MissingWindForcingError) as caught:
        runner.run_backward_attribution(
            detection_time=WINDOW[1],
            detection_lon=57.7,
            detection_lat=-20.4,
            detection_area_km2=5.0,
            wind_reader=None,
            current_reader=_sentinel_reader(),
            allow_synthetic=False,
        )
    assert caught.value.reason == "wind_reader_absent_synthetic_refused"


def test_hindcast_refuses_synthetic_current(stub_opendrift):
    with pytest.raises(errors.MissingCurrentForcingError) as caught:
        runner.run_backward_attribution(
            detection_time=WINDOW[1],
            detection_lon=57.7,
            detection_lat=-20.4,
            detection_area_km2=5.0,
            wind_reader=_sentinel_reader(),
            current_reader=None,
            allow_synthetic=False,
        )
    assert caught.value.reason == "current_reader_absent_synthetic_refused"


def test_forecast_refuses_synthetic_wind(stub_opendrift):
    with pytest.raises(errors.MissingWindForcingError):
        runner.run_forward_forecast(
            origin_time=WINDOW[0],
            origin_lon=57.7,
            origin_lat=-20.4,
            seed_radius_km=1.0,
            duration_h=2.0,
            wind_reader=None,
            current_reader=_sentinel_reader(),
            allow_synthetic=False,
        )


def test_forecast_refuses_synthetic_current(stub_opendrift):
    with pytest.raises(errors.MissingCurrentForcingError):
        runner.run_forward_forecast(
            origin_time=WINDOW[0],
            origin_lon=57.7,
            origin_lat=-20.4,
            seed_radius_km=1.0,
            duration_h=2.0,
            wind_reader=_sentinel_reader(),
            current_reader=None,
            allow_synthetic=False,
        )


def test_runner_gate_passes_when_both_fields_are_real(stub_opendrift):
    """With both readers supplied the gate must not fire — it reached OpenDrift."""
    with pytest.raises(_Reached):
        runner.run_backward_attribution(
            detection_time=WINDOW[1],
            detection_lon=57.7,
            detection_lat=-20.4,
            detection_area_km2=5.0,
            wind_reader=_sentinel_reader(),
            current_reader=_sentinel_reader(),
            allow_synthetic=False,
        )


def test_runner_still_substitutes_when_synthetic_allowed(monkeypatch):
    """Default behaviour is unchanged: labelled synthetic, not a hard failure."""
    monkeypatch.setattr(runner, "_opendrift", lambda: (_ for _ in ()).throw(_Reached()))
    with pytest.raises(_Reached):
        runner.run_backward_attribution(
            detection_time=WINDOW[1],
            detection_lon=57.7,
            detection_lat=-20.4,
            detection_area_km2=5.0,
            wind_reader=None,
            current_reader=None,
        )


# ---------------------------------------------------------------------------
# 5. The HTTP layer has to keep the reason code
# ---------------------------------------------------------------------------


@pytest.fixture()
def drift_main(tmp_path, monkeypatch):
    """Import the FastAPI module with CWD redirected (it opens ./logs at import)."""
    monkeypatch.chdir(tmp_path)
    return importlib.import_module("dfail.main")


def _request(drift_main, **overrides):
    ocean = drift_main.OceanDataRef(
        time_start=WINDOW[0],
        time_end=WINDOW[1],
        bbox=(56.7, -21.4, 58.7, -19.4),
    )
    kwargs = {
        "spill_lon": 57.7,
        "spill_lat": -20.4,
        "spill_age_hours": 24.0,
        "ocean_data": ocean,
        "n_particles": 10,
        "allow_synthetic": False,
    }
    kwargs.update(overrides)
    return drift_main.BackwardRequest(**kwargs)


async def test_backward_endpoint_reports_503_with_the_reason_code(drift_main, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise errors.MissingWindForcingError("no wind", reason="wind_unavailable_synthetic_refused")

    monkeypatch.setattr(drift_main, "_build_fields", _boom)

    with pytest.raises(HTTPException) as caught:
        await drift_main.drift_backward(_request(drift_main))

    assert caught.value.status_code == 503
    detail = caught.value.detail
    assert detail["code"] == "missing_wind_forcing"
    assert detail["field"] == "wind"
    assert detail["reason"] == "wind_unavailable_synthetic_refused"


async def test_current_failure_is_also_surfaced_not_swallowed(drift_main, monkeypatch):
    def _boom(*_args, **_kwargs):
        raise errors.MissingCurrentForcingError(
            "no current", reason="cmems_unavailable_synthetic_refused"
        )

    monkeypatch.setattr(drift_main, "_build_fields", _boom)

    with pytest.raises(HTTPException) as caught:
        await drift_main.drift_backward(_request(drift_main))

    assert caught.value.status_code == 503
    assert caught.value.detail["code"] == "missing_current_forcing"


async def test_unrelated_failures_stay_500(drift_main, monkeypatch):
    """Only a refusal is 503; a genuine crash must not be relabelled."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(drift_main, "_build_fields", _boom)

    with pytest.raises(HTTPException) as caught:
        await drift_main.drift_backward(_request(drift_main))

    assert caught.value.status_code == 500


# ---------------------------------------------------------------------------
# 6. A staged local CMEMS file is not a licence to invent wind
# ---------------------------------------------------------------------------


class _Arr:
    def __init__(self, values) -> None:
        self.values = values


def _fake_ocean_dataset() -> dict:
    return {
        "u": _Arr(np.zeros((2, 9, 9), dtype=np.float64)),
        "v": _Arr(np.zeros((2, 9, 9), dtype=np.float64)),
        "longitude": _Arr(np.linspace(56.7, 58.7, 9)),
        "latitude": _Arr(np.linspace(-21.4, -19.4, 9)),
        "time": _Arr(np.array([0.0, 3600.0])),
    }


def _local_file_request(drift_main, tmp_path):
    cmems = tmp_path / "cmems_local.nc"
    cmems.write_bytes(b"")
    return SimpleNamespace(
        cmems_base=str(cmems),
        era5_base="",
        gfs_base="",
        time_start=WINDOW[0],
        time_end=WINDOW[1],
        bbox=(56.7, -21.4, 58.7, -19.4),
    )


@pytest.fixture()
def staged_local_currents(monkeypatch):
    loader = importlib.import_module("dfail.data.cmems_loader")
    monkeypatch.setattr(loader, "load_cmems_currents", lambda *_a, **_k: _fake_ocean_dataset())
    monkeypatch.setattr(
        fac,
        "build_forcing_provider",
        lambda *_a, **_k: (
            None,
            {
                "forcing_source": "local_files",
                "wind": {"is_real": False, "source": "synthetic_mock"},
                "current": {"is_real": True, "source": "cmems_local_file"},
            },
        ),
    )


def test_local_currents_without_wind_refuses_when_synthetic_denied(
    drift_main, tmp_path, staged_local_currents
):
    req = _local_file_request(drift_main, tmp_path)
    with pytest.raises(errors.MissingWindForcingError) as caught:
        drift_main._build_fields(57.7, -20.4, req, datetime.now(UTC), allow_synthetic=False)
    assert caught.value.reason == "wind_not_staged_synthetic_refused"


def test_local_currents_without_wind_is_labelled_when_synthetic_allowed(
    drift_main, tmp_path, staged_local_currents
):
    """With synthetic allowed the run proceeds, but wind is honestly 'not_staged'."""
    req = _local_file_request(drift_main, tmp_path)
    _u, _v, _k, _lons, _lats, _times, _provider, provenance = drift_main._build_fields(
        57.7, -20.4, req, datetime.now(UTC), allow_synthetic=True
    )
    assert provenance["wind"]["is_real"] is False
    assert provenance["wind"]["status"] == "unavailable"
    assert provenance["current"]["is_real"] is True


def test_unsamplable_currents_refuse_instead_of_using_an_analytic_constant(
    drift_main, monkeypatch, tmp_path
):
    """The analytic 0.15/0.05 grid is synthetic; serving it silently was a hole."""

    class _Broken:
        def get_current_vectors(self, *_a, **_k):
            raise RuntimeError("provider exploded")

        def get_wind_vectors(self, *_a, **_k):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(
        fac,
        "build_forcing_provider",
        lambda *_a, **_k: (_Broken(), {"forcing_source": "live_cmems_era5"}),
    )
    req = SimpleNamespace(
        cmems_base="",
        era5_base="",
        gfs_base="",
        time_start=WINDOW[0],
        time_end=WINDOW[1],
        bbox=(56.7, -21.4, 58.7, -19.4),
    )

    with pytest.raises(errors.MissingCurrentForcingError) as caught:
        drift_main._build_fields(57.7, -20.4, req, datetime.now(UTC), allow_synthetic=False)
    assert caught.value.reason == "current_sampling_failed_synthetic_refused"
    # Nothing was written to disk as if it were a prepared dataset.
    assert not (tmp_path / "manifest.jsonl").exists()
