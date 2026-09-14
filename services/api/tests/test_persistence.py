"""A processed case must survive a service restart.

Before this, cases lived in a module-level ``dict`` in ``routers/cases.py``.
Restarting the gateway discarded every investigation, including the provenance
that made them auditable. "It is in memory" is not persistence, and a result you
cannot reopen is not evidence.

The test that matters is :func:`test_case_survives_a_restart`: it writes through
one store instance, then reads through a **completely fresh** instance over the
same backing path. That is the closest thing to a restart available in-process —
it proves nothing is held in the object graph or in a module global.

The database is NOT running on this machine, and these tests must not require it.
They assert the durable file backend works and that the backend actually used is
reported, so a caller can never mistake a JSON file for Postgres.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "services" / "api"


def _purge_app_namespace() -> None:
    """Every service ships a top-level `app` package; purge before importing."""
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]


_purge_app_namespace()
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.case_store import (  # noqa: E402
    BACKEND_FILE,
    CaseRecord,
    CaseStore,
)


def _sample_case(case_id: str = "case-000001") -> CaseRecord:
    """A record carrying every field the brief requires to be persisted."""
    return CaseRecord(
        case_id=case_id,
        scene_id="wakashio_20200810_peak",
        title="MV Wakashio — 10 Aug 2020",
        scene={
            "label": "wakashio_20200810_peak",
            "platform": "S1A",
            "product_id": "S1A_IW_GRDH_1SDV_20200810T013730",
            "acquisition_time": "2020-08-10T01:37:30.042000Z",
            "source": "cdse_sentinel_hub_process",
            "checksum_sha256": "b" * 64,
            "crs": "EPSG:4326",
            "width": 2048,
            "height": 2048,
            "bands": 3,
        },
        detection={
            "state": "ok",
            "state_reason": None,
            "flags": ["missing_wind_forcing"],
            "detector": "deterministic-tier-a-lee-solberg-v1",
            "model_status": "untrained",
            "count": 11,
            "area_km2": 3.34711940424636,
            "centroid": [57.72945073545187, -20.416901124019486],
            "length_km": 4.823489232238755,
            "width_km": 1.571098460828948,
            "orientation_deg": 153.7512649893496,
            "polygon": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [57.7154, -20.4369],
                        [57.7441, -20.4006],
                        [57.7294, -20.4169],
                        [57.7154, -20.4369],
                    ]
                ],
            },
            "confidence": {
                "value": 0.934,
                "low": 0.902,
                "high": 0.956,
                "method": "wilson_95",
                "n_effective": 334,
            },
        },
        drift={
            "origin_lon": 57.8186,
            "origin_lat": -20.3943,
            "origin_time": "2020-08-09T01:37:30Z",
            "semi_major_km": 1.71,
            "semi_minor_km": 0.77,
            "p50_radius_km": 0.61,
            "p95_radius_km": 1.11,
            "n_particles": 64,
            "wmc_divergence_max": 0.0,
            "stranded_fraction": 0.969,
        },
        forcing_provenance={
            "wind_source": "era5",
            "current_source": "cmems",
            "synthetic": False,
        },
        ais_provenance={
            "provenance": "no_real_coverage",
            "reason": "SENTINEL_DEMO_MODE_not_enabled+ALLOW_SYNTHETIC_AIS_not_enabled",
        },
        coverage="none",
        suspects=[
            {
                "mmsi": "477218700",
                "name": "MV WAKASHIO",
                "latitude": -20.46,
                "longitude": 57.74,
                "distance_to_origin_km": 6.73,
                "confidence_low": 0.71,
                "confidence_high": 0.93,
            }
        ],
        model_version="deterministic-tier-a-lee-solberg-v1",
        warnings=["missing_wind_forcing", "wind_viability=LOW_CONFIDENCE_NO_WIND"],
    )


@pytest.fixture
def store(tmp_path: Path) -> CaseStore:
    """A store forced onto the durable file backend.

    `backend="file"` is not a shortcut — the DB is genuinely down on this
    machine, and a test that needs a live Postgres is a test that fails for
    everyone without one.
    """
    return CaseStore(directory=tmp_path / "cases", backend="file", database_url="")


async def test_case_survives_a_restart(store: CaseStore, tmp_path: Path):
    """The headline requirement: write, then read from a brand-new instance."""
    record = await store.save(_sample_case())

    assert record.persistence_backend == BACKEND_FILE
    assert record.created_at and record.updated_at

    # A fresh instance over the same path. Nothing is shared with `store`
    # except the directory — no cache, no module global, no object identity.
    reopened = CaseStore(directory=tmp_path / "cases", backend="file", database_url="")
    loaded = await reopened.load("case-000001")

    assert loaded is not None, "case did not survive a fresh store instance"

    # Every field the brief requires to be persisted, checked individually.
    assert loaded.case_id == "case-000001"
    assert loaded.scene_id == "wakashio_20200810_peak"

    assert loaded.scene.checksum_sha256 == "b" * 64
    assert loaded.scene.source == "cdse_sentinel_hub_process"
    assert loaded.scene.acquisition_time == "2020-08-10T01:37:30.042000Z"
    assert loaded.scene.platform == "S1A"

    assert loaded.detection.state == "ok"
    assert loaded.detection.count == 11
    assert loaded.detection.area_km2 == pytest.approx(3.34711940424636)
    assert loaded.detection.polygon is not None
    assert loaded.detection.polygon["type"] == "Polygon"

    # The interval must round-trip, not just the point estimate (AGENTS.md).
    assert loaded.detection.confidence.value == pytest.approx(0.934)
    assert loaded.detection.confidence.low == pytest.approx(0.902)
    assert loaded.detection.confidence.high == pytest.approx(0.956)
    assert loaded.detection.confidence.method == "wilson_95"

    assert loaded.drift.origin_lon == pytest.approx(57.8186)
    assert loaded.drift.n_particles == 64

    # Provenance is the reason this store exists — check it survived.
    assert loaded.forcing_provenance["wind_source"] == "era5"
    assert loaded.forcing_provenance["current_source"] == "cmems"
    assert loaded.forcing_provenance["synthetic"] is False
    assert loaded.ais_provenance["provenance"] == "no_real_coverage"
    assert loaded.coverage == "none"

    assert loaded.model_version == "deterministic-tier-a-lee-solberg-v1"
    assert "missing_wind_forcing" in loaded.warnings
    assert len(loaded.suspects) == 1
    assert loaded.suspects[0]["mmsi"] == "477218700"


async def test_backend_is_reported_not_assumed(store: CaseStore):
    """A caller must be able to tell a JSON file from a database."""
    record = await store.save(_sample_case("case-000002"))
    assert record.persistence_backend == BACKEND_FILE
    assert await store.active_backend() == BACKEND_FILE


async def test_save_is_an_upsert_not_an_append(store: CaseStore):
    """Re-processing a scene must update its case, not accumulate duplicates."""
    await store.save(_sample_case("case-000003"))

    updated = _sample_case("case-000003")
    updated.detection.count = 3
    updated.detection.confidence.value = 0.5
    await store.save(updated)

    records = await store.list()
    assert len(records) == 1
    assert records[0].detection.count == 3
    assert records[0].detection.confidence.value == pytest.approx(0.5)


async def test_list_and_delete(store: CaseStore):
    await store.save(_sample_case("case-000004"))
    await store.save(_sample_case("case-000005"))

    listed = await store.list()
    assert {r.case_id for r in listed} == {"case-000004", "case-000005"}

    assert await store.delete("case-000004") is True
    assert await store.load("case-000004") is None
    assert await store.delete("case-000004") is False
    assert {r.case_id for r in await store.list()} == {"case-000005"}


async def test_missing_case_returns_none_not_an_error(store: CaseStore):
    assert await store.load("never-existed") is None


async def test_case_id_cannot_escape_the_store_directory(store: CaseStore, tmp_path: Path):
    """A crafted case id must not write outside the store."""
    await store.save(_sample_case("../../etc/passwd"))

    written = list((tmp_path / "cases").glob("*.json"))
    assert len(written) == 1
    assert written[0].parent == tmp_path / "cases"
    assert ".." not in written[0].name

    loaded = await store.load("../../etc/passwd")
    assert loaded is not None
    assert loaded.case_id == "../../etc/passwd"


async def test_corrupt_record_does_not_break_listing(store: CaseStore, tmp_path: Path):
    """One bad file must not make the whole case list unreadable."""
    await store.save(_sample_case("case-000006"))
    (tmp_path / "cases" / "broken.json").write_text("{not json", encoding="utf-8")

    listed = await store.list()
    assert [r.case_id for r in listed] == ["case-000006"]


async def test_atomic_write_leaves_no_partial_file(store: CaseStore, tmp_path: Path):
    """Writes go through a temp file, so a crash cannot truncate a record."""
    await store.save(_sample_case("case-000007"))
    leftovers = list((tmp_path / "cases").glob("*.tmp"))
    assert leftovers == [], f"temp files left behind: {leftovers}"
