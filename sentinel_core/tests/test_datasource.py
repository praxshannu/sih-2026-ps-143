"""Data-source switching tests.

These exercise the contract the trainer depends on:

* ``real`` reads the archive where it lives and never copies it;
* ``synthetic`` is labelled as such and can never support a scientific claim;
* the switch happens at runtime, on a live registry;
* a missing real archive is an **error**, not a silent substitution, unless
  the operator has explicitly allowed one — and then the substitution is
  recorded on the spec rather than being invisible.

Rasters are written as real GeoTIFFs (small ones) rather than mocked, because
the thing under test is partly the pairing and probing of actual files.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sentinel_core.config import DataSourceSettings, PathSettings, Settings, TrainingSettings
from sentinel_core.datasource import (
    MANIFEST_NAME,
    SOURCE_META_NAME,
    DataSourceRegistry,
    resolve_source,
)
from sentinel_core.errors import DataSourceMismatchError, DataSourceUnavailableError
from sentinel_core.provenance import DataProvenance

SIZE = 32
PIXEL_DEG = 0.001

#: Edge length of the tiles the helpers below write: 32 px at 0.001 deg.
TILE_DEG = SIZE * PIXEL_DEG


def _write_raster(
    path: Path,
    *,
    bands: int = 2,
    value: float = -20.0,
    lon: float = 4.0,
    lat: float = 55.0,
    dtype: str = "float32",
    crs: str | None = "EPSG:4326",
) -> None:
    """A tiny georeferenced GeoTIFF, standing in for one Sentinel-1 scene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((bands, SIZE, SIZE), value, dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=from_origin(lon, lat, PIXEL_DEG, PIXEL_DEG),
    ) as dst:
        dst.write(data)


def _write_tile(
    path: Path, *, left: float, bottom: float, bands: int = 2, crs: str | None = "EPSG:4326"
) -> None:
    """A tile occupying exactly ``[left, left+TILE_DEG] x [bottom, bottom+TILE_DEG]``.

    :func:`_write_raster` takes the *north-west* corner. Footprints are compared
    by their lower-left, so these helpers take the south-west corner instead and
    let a test place a tile on a grid boundary to the degree.
    """
    _write_raster(path, bands=bands, lon=left, lat=bottom + TILE_DEG, crs=crs)


def _make_footprint_archive(root: Path, boxes: Sequence[tuple[float, float]]) -> tuple[Path, Path]:
    """One image/mask pair per ``(left, bottom)``, named by position."""
    images, masks = root / "images", root / "masks"
    for i, (left, bottom) in enumerate(boxes):
        _write_tile(images / f"{i:05d}.tif", left=left, bottom=bottom)
        _write_tile(masks / f"{i:05d}.tif", left=left, bottom=bottom, bands=1)
    return images, masks


def _manifest_rows(spec: Any) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (spec.prepared_root / MANIFEST_NAME).read_text().splitlines()
        if line.strip()
    ]


def _scene_ids(spec: Any) -> dict[str, str]:
    """``{file stem: scene id}`` as written into the manifest."""
    return {Path(row["image"]).stem: row["scene_id"] for row in _manifest_rows(spec)}


def _make_real_archive(
    root: Path, n: int = 10, *, images_bands: int = 2, unpaired: bool = False
) -> tuple[Path, Path]:
    images, masks = root / "images", root / "masks"
    for i in range(n):
        # Spread the tiles so the spatial grid assigns several distinct scenes.
        _write_raster(images / f"{i:05d}.tif", bands=images_bands, lon=4.0 + i, lat=55.0)
        if unpaired and i == n - 1:
            continue
        _write_raster(masks / f"{i:05d}.tif", bands=1, value=0.0, lon=4.0 + i, lat=55.0)
    return images, masks


def _settings(
    root: Path,
    *,
    training: TrainingSettings | None = None,
    **datasource_overrides: Any,
) -> Settings:
    defaults: dict[str, Any] = {
        "real_images_dir": root / "images",
        "real_masks_dir": root / "masks",
        "synthetic_root": root / "synthetic",
    }
    defaults.update(datasource_overrides)
    extra: dict[str, Any] = {} if training is None else {"training": training}
    return Settings(
        environment="ci",
        data_source="real",
        paths=PathSettings(data_dir=root / "data", checkpoint_dir=root / "ckpt"),
        datasources=DataSourceSettings(**defaults),
        **extra,
    )


# ---------------------------------------------------------------------------
# Real source
# ---------------------------------------------------------------------------


def test_real_source_is_read_in_place(tmp_path: Path) -> None:
    images, masks = _make_real_archive(tmp_path)
    spec = resolve_source("real", settings=_settings(tmp_path))

    assert spec.kind == "real"
    assert spec.in_place is True
    assert spec.provenance is DataProvenance.REAL
    assert spec.scientifically_valid is True
    assert spec.is_synthetic is False
    assert spec.n_pairs == 10
    assert spec.bands == 2

    # The index points at the originals; it does not hold a copy of them.
    rows = [
        json.loads(line)
        for line in (spec.prepared_root / MANIFEST_NAME).read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 10
    for row in rows:
        assert Path(row["image"]).is_absolute()
        assert Path(row["image"]).parent == images.resolve()
        assert Path(row["mask"]).parent == masks.resolve()
        assert row["provenance"] == "real"
        assert row["split"] in {"train", "val", "test"}


def test_real_index_records_its_provenance_and_split_strategy(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    spec = resolve_source("real", settings=_settings(tmp_path))
    meta = json.loads((spec.prepared_root / SOURCE_META_NAME).read_text())
    assert meta["provenance"] == "real"
    assert meta["in_place"] is True
    assert meta["bands"] == 2
    # Headers are not read by default, so the split is per file.
    assert meta["split_strategy"] == "per_file_hash"
    assert meta["scene_grid_deg"] is None
    assert "not scanned" in meta["band_validation"]


def test_default_split_warns_that_it_is_not_scene_aware(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    spec = resolve_source("real", settings=_settings(tmp_path))
    assert spec.split_strategy == "per_file_hash"
    assert any("per file, not per scene" in warning for warning in spec.warnings)


def test_spatial_grouping_is_opt_in_and_groups_by_grid(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    spec = resolve_source("real", settings=_settings(tmp_path, spatial_scene_grouping=True))

    assert spec.split_strategy.startswith("spatial_grid")
    assert spec.bands == 2  # this path does read and check every header
    meta = json.loads((spec.prepared_root / SOURCE_META_NAME).read_text())
    assert meta["band_validation"].startswith("all ")
    assert meta["scene_grid_deg"] == 1.0
    # The fixture places tiles one degree apart, so they must not all collide.
    assert meta["n_scenes"] >= 2
    rows = [
        json.loads(line)
        for line in (spec.prepared_root / MANIFEST_NAME).read_text().splitlines()
        if line.strip()
    ]
    scene_ids = {row["scene_id"] for row in rows}
    assert all(scene_id.startswith("g") for scene_id in scene_ids)
    assert len(scene_ids) == meta["n_scenes"]


def test_spatial_grouping_keeps_a_scene_inside_one_split(tmp_path: Path) -> None:
    """The invariant the whole feature exists for: no scene straddles splits."""
    _make_real_archive(tmp_path)
    spec = resolve_source("real", settings=_settings(tmp_path, spatial_scene_grouping=True))
    rows = [
        json.loads(line)
        for line in (spec.prepared_root / MANIFEST_NAME).read_text().splitlines()
        if line.strip()
    ]
    by_scene: dict[str, set[str]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_scene.values()), by_scene


# ---------------------------------------------------------------------------
# Footprint grouping
#
# The grid asks "are these two tiles in the same 1 degree cell?", which is a
# question about the coordinate system. The footprint asks "do these two tiles
# show the same patch of sea?", which is the question the split needs answered.
# These tests pin the difference at a cell boundary, in both directions.
# ---------------------------------------------------------------------------

#: A whole-numbered degree, so ``round(lon)`` flips between the two tiles that
#: straddle it and the grid puts them in different cells.
GRID_EDGE = 3.5


def test_footprint_grouping_joins_tiles_a_grid_boundary_separates(tmp_path: Path) -> None:
    """Two touching tiles, one acquisition, split by the grid for no reason."""
    _make_footprint_archive(tmp_path, ((GRID_EDGE - TILE_DEG, 55.0), (GRID_EDGE, 55.0)))
    grid = resolve_source("real", settings=_settings(tmp_path, scene_grouping="grid"), rebuild=True)
    assert len(set(_scene_ids(grid).values())) == 2  # the boundary runs between them

    footprint = resolve_source(
        "real", settings=_settings(tmp_path, scene_grouping="footprint"), rebuild=True
    )
    assert len(set(_scene_ids(footprint).values())) == 1  # they share an edge


def test_footprint_grouping_separates_tiles_a_grid_cell_merges(tmp_path: Path) -> None:
    """Two tiles 30 km apart, one cell, two acquisitions."""
    _make_footprint_archive(tmp_path, ((3.60, 55.0), (3.90, 55.0)))
    grid = resolve_source("real", settings=_settings(tmp_path, scene_grouping="grid"), rebuild=True)
    assert len(set(_scene_ids(grid).values())) == 1

    footprint = resolve_source(
        "real", settings=_settings(tmp_path, scene_grouping="footprint"), rebuild=True
    )
    assert len(set(_scene_ids(footprint).values())) == 2


def test_footprint_grouping_is_transitive(tmp_path: Path) -> None:
    """A touches B and B touches C, but A does not touch C — still one group.

    A pairwise implementation that compares each tile against its group's first
    box gets this wrong and shatters a continuous acquisition into pieces, which
    is the leakage the grouping exists to prevent.
    """
    _make_footprint_archive(tmp_path, tuple((3.0 + i * TILE_DEG, 55.0) for i in range(3)))
    spec = resolve_source(
        "real", settings=_settings(tmp_path, scene_grouping="footprint"), rebuild=True
    )
    # One group, labelled by the lowest stem in it — so the label is a stable
    # name for the acquisition rather than a function of traversal order.
    assert set(_scene_ids(spec).values()) == {"00000"}


def test_footprint_grouping_keeps_an_acquisition_on_one_side_of_the_split(
    tmp_path: Path,
) -> None:
    """Twenty tiles in a row, plus five unrelated ones, must not interleave."""
    run = tuple((3.0 + i * TILE_DEG, 55.0) for i in range(20))
    far = tuple((40.0 + i, -10.0) for i in range(5))
    _make_footprint_archive(tmp_path, run + far)
    spec = resolve_source(
        "real", settings=_settings(tmp_path, scene_grouping="footprint"), rebuild=True
    )

    rows = _manifest_rows(spec)
    assert len({row["scene_id"] for row in rows}) == 6  # the run is one acquisition
    run_splits = {row["split"] for row in rows if Path(row["image"]).stem < f"{len(run):05d}"}
    assert len(run_splits) == 1, run_splits


def test_footprint_grouping_reports_ungeoreferenced_tiles(tmp_path: Path) -> None:
    """A tile with no CRS cannot be placed, so it is named rather than guessed at."""
    images, masks = tmp_path / "images", tmp_path / "masks"
    for i, crs in enumerate(("EPSG:4326", None, "EPSG:4326")):
        _write_tile(images / f"{i:05d}.tif", left=3.0, bottom=55.0, crs=crs)
        _write_tile(masks / f"{i:05d}.tif", left=3.0, bottom=55.0, bands=1)

    spec = resolve_source(
        "real", settings=_settings(tmp_path, scene_grouping="footprint"), rebuild=True
    )
    ids = _scene_ids(spec)
    # The two located tiles group together; the blind one stands alone.
    assert ids["00000"] == ids["00002"]
    assert ids["00001"] == "00001"
    assert any("no georeferencing" in warning for warning in spec.warnings)


def test_footprint_grouping_refuses_a_mixed_crs_archive(tmp_path: Path) -> None:
    """Comparing metres against degrees returns a confident, meaningless answer."""
    images, masks = tmp_path / "images", tmp_path / "masks"
    for i, crs in enumerate(("EPSG:4326", "EPSG:3857")):
        _write_tile(images / f"{i:05d}.tif", left=3.0, bottom=55.0, crs=crs)
        _write_tile(masks / f"{i:05d}.tif", left=3.0, bottom=55.0, bands=1)

    with pytest.raises(DataSourceMismatchError) as caught:
        resolve_source(
            "real", settings=_settings(tmp_path, scene_grouping="footprint"), rebuild=True
        )
    assert caught.value.reason == "inconsistent_crs"
    assert caught.value.context["crs_seen"] == ["EPSG:3857", "EPSG:4326"]


def test_footprint_index_records_the_mode_and_its_tolerance(tmp_path: Path) -> None:
    _make_footprint_archive(tmp_path, ((3.0, 55.0),))
    spec = resolve_source(
        "real", settings=_settings(tmp_path, scene_grouping="footprint"), rebuild=True
    )
    assert spec.split_strategy == "footprint_connected"
    meta = json.loads((spec.prepared_root / SOURCE_META_NAME).read_text())
    assert meta["scene_grouping"] == "footprint"
    assert meta["scene_grid_deg"] is None
    assert meta["footprint_tol_deg"] == 0.005
    assert meta["band_validation"].startswith("all ")


def test_the_legacy_spatial_flag_still_means_the_grid(tmp_path: Path) -> None:
    """``spatial_scene_grouping`` predates ``scene_grouping``; it keeps working."""
    _make_real_archive(tmp_path)
    spec = resolve_source(
        "real", settings=_settings(tmp_path, spatial_scene_grouping=True), rebuild=True
    )
    assert spec.split_strategy.startswith("spatial_grid")
    meta = json.loads((spec.prepared_root / SOURCE_META_NAME).read_text())
    assert meta["scene_grouping"] == "grid"


def test_an_explicit_mode_beats_the_legacy_flag(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    spec = resolve_source(
        "real",
        settings=_settings(tmp_path, spatial_scene_grouping=True, scene_grouping="footprint"),
        rebuild=True,
    )
    assert spec.split_strategy == "footprint_connected"


def test_changing_the_grouping_mode_invalidates_the_cached_index(tmp_path: Path) -> None:
    """A cached split answers the question it was built for, and no other.

    Reusing a per-file index for a footprint-grouped run would silently hand the
    experiment the other grouping's validation set.
    """
    _make_real_archive(tmp_path)
    settings = _settings(tmp_path)
    per_file = resolve_source("real", settings=settings, rebuild=True)
    assert per_file.split_strategy == "per_file_hash"

    footprint = resolve_source("real", settings=_settings(tmp_path, scene_grouping="footprint"))
    assert footprint.split_strategy == "footprint_connected"
    assert "cached" not in footprint.note
    meta = json.loads((footprint.prepared_root / SOURCE_META_NAME).read_text())
    assert meta["scene_grouping"] == "footprint"


def test_the_cached_index_is_reused_when_the_mode_matches(tmp_path: Path) -> None:
    """The header scan costs minutes on removable media; it must not repeat."""
    _make_real_archive(tmp_path)
    settings = _settings(tmp_path, scene_grouping="footprint")
    first = resolve_source("real", settings=settings, rebuild=True)
    again = resolve_source("real", settings=settings)
    assert "cached" in again.note
    assert again.split_strategy == first.split_strategy
    assert again.n_pairs == first.n_pairs


def test_the_index_records_the_split_parameters_that_produced_it(tmp_path: Path) -> None:
    """The split is baked into the manifest, so it belongs in the cache key."""
    _make_real_archive(tmp_path)
    spec = resolve_source(
        "real",
        settings=_settings(
            tmp_path, training=TrainingSettings(val_fraction=0.2, test_fraction=0.1, seed=7)
        ),
        rebuild=True,
    )
    meta = json.loads((spec.prepared_root / SOURCE_META_NAME).read_text())
    assert meta["split_params"] == {"val_fraction": 0.2, "test_fraction": 0.1, "seed": 7}


def test_raising_test_fraction_invalidates_the_cached_index(tmp_path: Path) -> None:
    """Otherwise a held-out run silently gets no held-out ground.

    The manifest carries its split. Reusing one built with ``test_fraction=0``
    for a run that asks for a test set hands back a manifest with no test rows,
    and the trainer then reports zero test pairs — which reads as "the archive
    has no held-out data" rather than "the cache answered a different question".
    """
    _make_real_archive(tmp_path)
    two_way = resolve_source("real", settings=_settings(tmp_path), rebuild=True)
    assert "cached" not in two_way.note

    three_way = resolve_source(
        "real",
        settings=_settings(
            tmp_path, training=TrainingSettings(val_fraction=0.2, test_fraction=0.2, seed=42)
        ),
    )
    assert "cached" not in three_way.note


def test_changing_the_seed_invalidates_the_cached_index(tmp_path: Path) -> None:
    """The seed picks the split, so a new seed is a different dataset."""
    _make_real_archive(tmp_path)
    resolve_source(
        "real", settings=_settings(tmp_path, training=TrainingSettings(seed=1)), rebuild=True
    )
    reseeded = resolve_source(
        "real", settings=_settings(tmp_path, training=TrainingSettings(seed=2))
    )
    assert "cached" not in reseeded.note


def test_an_index_without_split_params_is_rebuilt_rather_than_trusted(tmp_path: Path) -> None:
    """An index that cannot say how it was split cannot be checked against it.

    Indexes written before ``split_params`` existed are the case that matters:
    there is no way to confirm the cached split answers the current question, so
    the honest answer is to rebuild once rather than assume that it does.
    """
    _make_real_archive(tmp_path)
    settings = _settings(tmp_path, scene_grouping="footprint")
    first = resolve_source("real", settings=settings, rebuild=True)

    meta_path = first.prepared_root / SOURCE_META_NAME
    meta = json.loads(meta_path.read_text())
    meta.pop("split_params")
    meta_path.write_text(json.dumps(meta))

    again = resolve_source("real", settings=settings)
    assert "cached" not in again.note


def test_the_cache_is_still_reused_when_every_split_parameter_matches(tmp_path: Path) -> None:
    """The guard must not fire on the happy path — a rebuild costs a header scan."""
    _make_real_archive(tmp_path)
    settings = _settings(
        tmp_path, training=TrainingSettings(val_fraction=0.2, test_fraction=0.1, seed=7)
    )
    first = resolve_source("real", settings=settings, rebuild=True)
    again = resolve_source("real", settings=settings)
    assert "cached" in again.note
    assert again.n_pairs == first.n_pairs


def test_real_split_is_deterministic(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    settings = _settings(tmp_path)
    first = resolve_source("real", settings=settings, rebuild=True)
    second = resolve_source("real", settings=settings, rebuild=True)
    read = lambda spec: [  # noqa: E731 - terse on purpose
        json.loads(line) for line in (spec.prepared_root / MANIFEST_NAME).read_text().splitlines()
    ]
    assert read(first) == read(second)


def test_real_index_is_cached_between_resolutions(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    settings = _settings(tmp_path)
    first = resolve_source("real", settings=settings, rebuild=True)
    cached = resolve_source("real", settings=settings)
    assert cached.n_pairs == first.n_pairs
    assert "cached" in cached.note


def test_appledouble_sidecars_are_not_indexed(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    # macOS writes one of these next to every file on a non-native filesystem.
    (tmp_path / "images" / "._00000.tif").write_bytes(b"\x00" * 16)
    spec = resolve_source("real", settings=_settings(tmp_path))
    assert spec.n_pairs == 10


# ---------------------------------------------------------------------------
# Synthetic source
# ---------------------------------------------------------------------------


def _make_synthetic(root: Path, per_split: int = 4) -> Path:
    for split in ("train", "val", "test"):
        for i in range(per_split):
            _write_raster(root / split / "images" / f"{split}_{i:03d}.tif", lon=4.0 + i)
            _write_raster(
                root / split / "masks" / f"{split}_{i:03d}.tif", bands=1, value=0.0, lon=4.0 + i
            )
    return root


def test_synthetic_source_is_labelled_and_not_evidence(tmp_path: Path) -> None:
    _make_synthetic(tmp_path / "synthetic")
    spec = resolve_source("synthetic", settings=_settings(tmp_path))

    assert spec.kind == "synthetic"
    assert spec.provenance is DataProvenance.SYNTHETIC
    assert spec.is_synthetic is True
    assert spec.scientifically_valid is False
    assert spec.in_place is False
    assert spec.n_pairs == 12
    assert "SYNTHETIC" in spec.note


def test_synthetic_manifest_rows_are_labelled_synthetic(tmp_path: Path) -> None:
    _make_synthetic(tmp_path / "synthetic")
    spec = resolve_source("synthetic", settings=_settings(tmp_path))
    rows = [
        json.loads(line)
        for line in (spec.prepared_root / MANIFEST_NAME).read_text().splitlines()
        if line.strip()
    ]
    assert {row["provenance"] for row in rows} == {"synthetic_mock"}
    assert {row["split"] for row in rows} == {"train", "val", "test"}


# ---------------------------------------------------------------------------
# Failure modes: never a silent substitution
# ---------------------------------------------------------------------------


def test_missing_real_archive_raises_rather_than_substituting(tmp_path: Path) -> None:
    settings = _settings(tmp_path)  # nothing was written
    with pytest.raises(DataSourceUnavailableError) as caught:
        resolve_source("real", settings=settings)
    assert caught.value.reason == "real_images_missing"
    assert caught.value.context["exists"] is False


def test_missing_masks_names_the_fix(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    (tmp_path / "masks" / "00000.tif").unlink()
    settings = _settings(tmp_path)
    # Wipe the masks dir entirely so the "no masks at all" branch is taken.
    for leftover in (tmp_path / "masks").glob("*.tif"):
        leftover.unlink()
    with pytest.raises(DataSourceUnavailableError) as caught:
        resolve_source("real", settings=settings)
    assert caught.value.reason == "real_masks_missing"
    assert "prepare_real_masks" in str(caught.value)


def test_fallback_is_off_by_default_and_on_only_when_asked(tmp_path: Path) -> None:
    _make_synthetic(tmp_path / "synthetic")

    strict = _settings(tmp_path)
    with pytest.raises(DataSourceUnavailableError):
        resolve_source("real", settings=strict)

    lenient = _settings(tmp_path, allow_synthetic_fallback=True)
    spec = resolve_source("real", settings=lenient)
    assert spec.kind == "synthetic"
    assert spec.provenance is DataProvenance.SYNTHETIC
    assert spec.scientifically_valid is False
    assert any(w.startswith("substituted_for_real") for w in spec.warnings)
    assert "SUBSTITUTED" in spec.note


def test_band_count_mismatch_is_caught_when_headers_are_read(tmp_path: Path) -> None:
    _make_real_archive(tmp_path, images_bands=3)
    with pytest.raises(DataSourceMismatchError) as caught:
        resolve_source("real", settings=_settings(tmp_path, spatial_scene_grouping=True))
    assert caught.value.reason == "band_count_mismatch"
    assert caught.value.context == {"found": 3, "required": 2}


def test_band_count_mismatch_is_deferred_to_the_loader_by_default(tmp_path: Path) -> None:
    """Without a header scan the index is a catalogue; the loader is the check.

    This is the trade-off the module documents. It is only acceptable because
    the deferred check genuinely exists, which is what this test pins down.
    """
    from ml.training.dataset import SAROilSpillDataset

    _make_real_archive(tmp_path, images_bands=3)
    spec = resolve_source("real", settings=_settings(tmp_path))  # no raise here

    with pytest.raises(ValueError) as caught:
        SAROilSpillDataset(spec.prepared_root, expected_bands=2)
    assert "3 bands found, 2 expected" in str(caught.value)


def test_unpaired_image_is_refused_not_dropped(tmp_path: Path) -> None:
    _make_real_archive(tmp_path, unpaired=True)
    with pytest.raises(DataSourceUnavailableError) as caught:
        resolve_source("real", settings=_settings(tmp_path))
    assert caught.value.reason == "unpaired_images"
    assert caught.value.context["n_missing"] == 1


def test_missing_synthetic_root_names_the_generator(tmp_path: Path) -> None:
    with pytest.raises(DataSourceUnavailableError) as caught:
        resolve_source("synthetic", settings=_settings(tmp_path))
    assert caught.value.reason == "synthetic_missing"
    assert "generate_synthetic" in str(caught.value)


# ---------------------------------------------------------------------------
# Runtime switching
# ---------------------------------------------------------------------------


def test_registry_switches_source_at_runtime(tmp_path: Path) -> None:
    _make_real_archive(tmp_path)
    _make_synthetic(tmp_path / "synthetic")
    registry = DataSourceRegistry(_settings(tmp_path))

    assert registry.active().kind == "real"
    switched = registry.set_active("synthetic")
    assert switched.kind == "synthetic"
    # The switch is real: the next read returns the other spec, not a cached one.
    assert registry.active().kind == "synthetic"
    assert registry.set_active("real").kind == "real"
    assert registry.active().provenance is DataProvenance.REAL


def test_registry_probe_reports_without_raising(tmp_path: Path) -> None:
    _make_synthetic(tmp_path / "synthetic")
    registry = DataSourceRegistry(_settings(tmp_path))

    real_probe = registry.probe("real")
    assert real_probe["available"] is False
    assert real_probe["reason"] == "real_images_missing"

    synthetic_probe = registry.probe("synthetic")
    assert synthetic_probe["n_pairs"] == 12
    assert synthetic_probe["provenance"] == "synthetic_mock"


def test_registry_report_never_raises_even_with_nothing_configured(tmp_path: Path) -> None:
    report = DataSourceRegistry(_settings(tmp_path)).report()
    assert report["configured"] == "real"
    assert report["active"]["available"] is False
    assert set(report["available"]) == {"real", "synthetic"}
