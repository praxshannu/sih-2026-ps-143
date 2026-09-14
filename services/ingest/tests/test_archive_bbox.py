"""Regression tests for the AOI axis-order contract on the archive endpoints.

Background
----------
An AOI used to travel as a positional ``west,south,east,north`` string. A
lat-first caller sends four numbers that are all legal in either slot, so no
validator can tell the two apart: the box is silently relocated and, for the
AIS endpoints, the real/synthetic verdict flips with it. Verified 2026-09-14 —
``-21.0,57.6,-20.4,58.2`` (Mauritius, written lat-first) was accepted as
lon -21..-20.4 / lat 57.6..58.2, i.e. the North Atlantic, and reported
``live_terrestrial`` for an Indian Ocean AOI.

The contract is now: named axes win, partial sets are rejected, and the string
survives only as a deprecated alias. These tests pin that behaviour.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
INGEST_DIR = ROOT / "services" / "ingest"

# The router creates DATA_DIR/sar at import time; keep it off the real tree.
os.environ.setdefault("SENTINEL_DATA_DIR", str(ROOT / "services" / "ingest" / "tests" / ".tmp"))


def _purge_app_namespace() -> None:
    """Every service ships a top-level `app` package, so they collide in
    sys.modules when the whole suite runs in one process. Drop any `app.*`
    already cached (e.g. by the api service's tests) before importing ours.
    """
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]


_purge_app_namespace()
sys.path.insert(0, str(INGEST_DIR))

from app.routers.archive import _resolve_bbox, router  # noqa: E402

WAKASHIO = (57.6, -21.0, 58.2, -20.4)          # open Indian Ocean → synthetic
MUMBAI = (72.6, 18.8, 73.1, 19.3)              # coastal carve-out → live
ATLANTIC = (-21.0, 57.6, -20.4, 58.2)          # what a lat-first Mauritius parses to


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A bare app mounting only the archive router.

    Deliberately not ``app.main:app`` — that pulls in the DB pool and the
    APScheduler daemon, which have nothing to do with the bbox contract.
    """
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _named(box: tuple[float, float, float, float]) -> dict[str, float]:
    return {
        "min_lon": box[0],
        "min_lat": box[1],
        "max_lon": box[2],
        "max_lat": box[3],
    }


# ── resolver unit tests ───────────────────────────────────────────────────


def test_named_axes_are_returned_in_declared_order():
    assert _resolve_bbox(*WAKASHIO, None) == WAKASHIO


def test_named_axes_preserve_a_lat_first_caller_intent():
    """The original bug: a lat-first box used to be reinterpreted.

    With named axes there is nothing to reinterpret — declaring lon -21..-20.4
    means the Atlantic, and the Atlantic is what you get.
    """
    assert _resolve_bbox(*ATLANTIC, None) == ATLANTIC


@pytest.mark.parametrize(
    "partial",
    [
        (57.6, None, None, None),
        (57.6, -21.0, None, None),
        (57.6, -21.0, 58.2, None),
        (None, None, None, -20.4),
    ],
)
def test_partial_axis_sets_are_rejected(partial):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _resolve_bbox(*partial, None)
    assert exc.value.status_code == 422
    assert "together" in str(exc.value.detail)


def test_named_axes_still_range_checked():
    from fastapi import HTTPException

    # north/south flipped
    with pytest.raises(HTTPException) as exc:
        _resolve_bbox(57.6, -20.4, 58.2, -21.0, None)
    assert "latitudes" in str(exc.value.detail)

    # west/east flipped
    with pytest.raises(HTTPException) as exc:
        _resolve_bbox(58.2, -21.0, 57.6, -20.4, None)
    assert "longitudes" in str(exc.value.detail)


def test_legacy_string_still_parses():
    """Deprecated, but callers in the wild still send it."""
    assert _resolve_bbox(None, None, None, None, "57.6,-21.0,58.2,-20.4") == WAKASHIO


def test_nothing_supplied_is_rejected_with_guidance():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _resolve_bbox(None, None, None, None, None)
    assert "min_lon" in str(exc.value.detail)


# ── endpoint contract ─────────────────────────────────────────────────────


def test_ais_coverage_named_axes_wakashio_is_synthetic(client):
    r = client.get("/archive/ais/coverage", params=_named(WAKASHIO))
    assert r.status_code == 200
    body = r.json()
    assert body["use_synthetic"] is True
    assert body["mode"] == "synthetic_only"
    assert body["provenance"] == "synthetic_mock"
    assert body["bbox"] == list(WAKASHIO)


def test_ais_coverage_named_axes_mumbai_is_live(client):
    r = client.get("/archive/ais/coverage", params=_named(MUMBAI))
    assert r.status_code == 200
    body = r.json()
    assert body["use_synthetic"] is False
    assert body["mode"] == "live_terrestrial"
    assert body["coastal_carveouts"]


def test_ais_coverage_rejects_partial_axes(client):
    r = client.get(
        "/archive/ais/coverage", params={"min_lon": 57.6, "min_lat": -21.0}
    )
    assert r.status_code == 422
    assert "together" in r.json()["detail"]


def test_ais_coverage_rejects_missing_aoi(client):
    r = client.get("/archive/ais/coverage")
    assert r.status_code == 422
    assert "min_lon" in r.json()["detail"]


def test_ais_coverage_legacy_string_still_works(client):
    r = client.get(
        "/archive/ais/coverage", params={"bbox": "57.6,-21.0,58.2,-20.4"}
    )
    assert r.status_code == 200
    assert r.json()["use_synthetic"] is True


def test_ais_endpoint_honours_named_axes(client):
    r = client.get(
        "/archive/ais",
        params={
            **_named(WAKASHIO),
            "start": "2020-08-09T00:00:00Z",
            "end": "2020-08-10T00:00:00Z",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["bbox"] == list(WAKASHIO)
    assert body["provenance"] == "synthetic_mock"
    assert body["count"] > 0
