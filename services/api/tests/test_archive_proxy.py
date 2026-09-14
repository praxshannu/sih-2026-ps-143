"""Gateway proxy tests — AOI forwarding and None-stripping.

Two real bugs live in this layer:

1. ``INGEST_SERVICE_URL`` defaults to the Docker hostname
   ``http://sentinel-ingest:8001``, which does not resolve on a native run.
   The proxy then gets an HTML error body and ``resp.json()`` explodes.
2. **httpx serialises ``None`` as a bare ``key=``** (verified:
   ``QueryParams({'a': 1, 'b': None})`` → ``'a=1&b='``). The gateway was
   forwarding ``max_lon=&max_lat=``, which upstream fails to parse as a float
   — a confusing ``float_parsing`` 422 instead of a clear "supply all four
   axes".
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "services" / "api"


def _purge_app_namespace() -> None:
    """Every service ships a top-level `app` package, so they collide in
    sys.modules when the whole suite runs in one process. Drop any `app.*`
    already cached (e.g. by the ingest service's tests) before importing ours.
    """
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]


_purge_app_namespace()
sys.path.insert(0, str(API_DIR))

import httpx  # noqa: E402
from app.routers.archive import _bbox_params, _proxy_get  # noqa: E402

WAKASHIO = (57.6, -21.0, 58.2, -20.4)


class _FakeResponse:
    status_code = 200
    content = b'{"ok":true}'
    text = '{"ok":true}'
    headers = {"content-type": "application/json"}

    def json(self):
        return {"ok": True}


class _FakeAsyncClient:
    """Records the outbound request so we can assert on the query string."""

    captured: dict[str, object] = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params=None):
        _FakeAsyncClient.captured = {"url": url, "params": params}
        return _FakeResponse()


def _run_proxy(params: dict[str, object]) -> dict[str, object]:
    original = httpx.AsyncClient
    httpx.AsyncClient = _FakeAsyncClient  # type: ignore[misc, assignment]
    try:
        asyncio.run(_proxy_get("ais/coverage", params))
    finally:
        httpx.AsyncClient = original  # type: ignore[misc, assignment]
    return _FakeAsyncClient.captured  # type: ignore[return-value]


def test_named_axes_are_forwarded_named():
    out = _bbox_params(*WAKASHIO, None)
    assert out == {
        "min_lon": 57.6,
        "min_lat": -21.0,
        "max_lon": 58.2,
        "max_lat": -20.4,
    }


def test_partial_named_axes_stay_named():
    """Upstream owns the 'all four together' rule — forward what we were given."""
    out = _bbox_params(57.6, -21.0, None, None, None)
    assert out["min_lon"] == 57.6
    assert out["min_lat"] == -21.0
    assert out["max_lon"] is None


def test_legacy_string_falls_through():
    assert _bbox_params(None, None, None, None, "57.6,-21.0,58.2,-20.4") == {
        "bbox": "57.6,-21.0,58.2,-20.4"
    }


def test_none_params_are_stripped_before_proxying():
    captured = _run_proxy(
        {
            "min_lon": 57.6,
            "min_lat": -21.0,
            "max_lon": None,
            "max_lat": None,
            "bbox": None,
        }
    )
    params = captured["params"]
    assert isinstance(params, dict)
    # No key may carry a None value — httpx would render it as `key=`.
    assert not any(v is None for v in params.values())
    assert params == {"min_lon": 57.6, "min_lat": -21.0}


def test_absent_optional_filters_are_dropped_too():
    """Also stops `platform`/`mode` being sent upstream as empty strings."""
    captured = _run_proxy({"bbox": "57.6,-21.0,58.2,-20.4", "platform": None})
    assert captured["params"] == {"bbox": "57.6,-21.0,58.2,-20.4"}


def test_proxy_targets_the_archive_path():
    captured = _run_proxy({"bbox": "57.6,-21.0,58.2,-20.4"})
    assert str(captured["url"]).endswith("/archive/ais/coverage")
