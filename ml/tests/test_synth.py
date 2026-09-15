"""Tests for the synthetic-data profiler and generator.

The generator's whole claim is that a synthetic scene is *statistically
substitutable* for a real one. These tests build a small fixture whose
statistics are known by construction, then check that the profiler measures
them and the generator reproduces them — including the two properties that were
silently wrong before: no-data blocks must not enter the statistics, and the oil
budget must be met rather than emerge from a Poisson draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from ml.synth.generate import (
    GenerationConfig,
    _apply_extremes,
    _blob_patch,
    _extreme_canvas,
    _nodata_canvas,
    _pick_extreme_record,
    generate_dataset,
    synthesize_scene,
)
from ml.synth.profile import (
    NODATA_DB,
    DistributionProfile,
    _extremes,
    compare_profiles,
    profile_scenes,
)
from sentinel_core.config import DataSourceSettings, Settings
from sentinel_core.datasource import _build_synthetic

SCENE = 256


# ---------------------------------------------------------------------------
# Fixture: a tiny "real" dataset with known statistics
# ---------------------------------------------------------------------------


def _write_pair(image_path: Path, mask_path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    import rasterio
    from rasterio.transform import from_origin

    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    bands, height, width = image.shape
    # Georeferenced like the archive, so nothing in the profiler is exercised
    # under a different assumption than production.
    transform = from_origin(4.0, 55.0, 1e-4, 1e-4)
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(image)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(mask[np.newaxis, ...])


def _fixture_scene(
    rng: np.random.Generator, size: int = SCENE, nodata: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Two bands around a known sea level, with one rectangular slick."""
    image = np.stack(
        [
            rng.normal(-30.0, 1.0, (size, size)),
            rng.normal(-18.0, 1.0, (size, size)),
        ]
    ).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4 : size // 2, size // 4 : size // 4 + 40] = 1
    # A slick is darker than the sea in both bands.
    image[:, mask > 0] -= np.array([[5.0], [4.0]], dtype=np.float32)
    if nodata:
        image[:, : size // 8, :] = NODATA_DB
    return image, mask


def build_fixture(root: Path, n_scenes: int = 4, nodata_scenes: int = 0) -> tuple[list, list]:
    rng = np.random.default_rng(11)
    images: list[Path] = []
    masks: list[Path] = []
    for index in range(n_scenes):
        image, mask = _fixture_scene(rng, nodata=index < nodata_scenes)
        image_path = root / "images" / f"{index:05d}.tif"
        mask_path = root / "masks" / f"{index:05d}.tif"
        _write_pair(image_path, mask_path, image, mask)
        images.append(image_path)
        masks.append(mask_path)
    return images, masks


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def test_profile_measures_the_known_sea_level(tmp_path: Path) -> None:
    images, masks = build_fixture(tmp_path)
    profile = profile_scenes(images, masks)
    assert profile.bands == 2
    assert profile.shape == (SCENE, SCENE)
    # The sea dominates the scene, so the pooled median is the sea level.
    assert profile.sea_level_db["VV"] == pytest.approx(-30.0, abs=1.0)
    assert profile.sea_level_db["VH"] == pytest.approx(-18.0, abs=1.0)


def test_profile_measures_the_known_oil_contrast(tmp_path: Path) -> None:
    images, masks = build_fixture(tmp_path)
    profile = profile_scenes(images, masks)
    assert profile.contrast_db["VV"] == pytest.approx(5.0, abs=1.0)
    assert profile.contrast_db["VH"] == pytest.approx(4.0, abs=1.0)


def test_profile_records_scene_to_scene_spread(tmp_path: Path) -> None:
    """These two fields were computed and then dropped by the constructor.

    Without them every generated scene had an identical sea level, which is a
    distribution mismatch the per-band histogram cannot see.
    """
    images, masks = build_fixture(tmp_path)
    profile = profile_scenes(images, masks)
    assert set(profile.sea_level_db_std) == {"VV", "VH"}
    assert set(profile.contrast_db_std) == {"VV", "VH"}


def test_equivalent_looks_are_per_band(tmp_path: Path) -> None:
    """VV and VH are different measurements; one shared value is wrong."""
    images, masks = build_fixture(tmp_path)
    profile = profile_scenes(images, masks)
    looks = profile.speckle["equivalent_looks"]
    assert isinstance(looks, dict)
    assert set(looks) == {"VV", "VH"}
    assert set(profile.speckle["speckle_sigma_db"]) == {"VV", "VH"}


def test_nodata_is_excluded_from_the_statistics(tmp_path: Path) -> None:
    """A 0.0 dB fill block is the brightest thing in the scene.

    Left in, it drags the 99th percentile to exactly 0.0 and reads as a large
    bright target. This is the check that it does not.
    """
    images, masks = build_fixture(tmp_path, nodata_scenes=4)
    profile = profile_scenes(images, masks)
    for band in profile.band_stats:
        assert band["percentiles"]["p99"] < -10.0, f"{band['name']} p99 still polluted"
        assert band["max"] < 5.0
    assert profile.nodata["value"] == NODATA_DB
    assert profile.nodata["scene_rate"] == pytest.approx(1.0)
    assert profile.nodata["fraction_mean"] > 0.0


def test_profile_save_load_round_trip(tmp_path: Path) -> None:
    images, masks = build_fixture(tmp_path)
    profile = profile_scenes(images, masks)
    path = tmp_path / "distribution.json"
    profile.save(path)
    reloaded = DistributionProfile.load(path)
    assert reloaded.shape == profile.shape
    assert reloaded.sea_level_db == profile.sea_level_db
    assert reloaded.bands == profile.bands
    payload = json.loads(path.read_text())
    assert "generated_utc" in payload


def test_profile_refuses_mismatched_inputs(tmp_path: Path) -> None:
    images, masks = build_fixture(tmp_path)
    with pytest.raises(ValueError, match="images vs"):
        profile_scenes(images, masks[:-1])


# ---------------------------------------------------------------------------
# Blob geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elongation", [1.0, 2.0, 4.0, 8.0])
def test_blob_area_and_elongation_are_reproduced(elongation: float) -> None:
    """The axis-swap bug made every slick round regardless of the request."""
    rng = np.random.default_rng(0)
    area_px = 2000.0
    patch = _blob_patch(rng, area_px, elongation)
    assert patch.any()

    rows, cols = np.nonzero(patch)
    assert float(patch.sum()) == pytest.approx(area_px, rel=0.25)

    if elongation >= 2.0:
        cov = np.cov(np.vstack([cols.astype(float), rows.astype(float)]))
        eigenvalues = np.linalg.eigvalsh(cov)
        measured = float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-9)))
        assert measured > 1.5, f"elongation {elongation} produced a round blob ({measured:.2f})"


def test_blob_elongation_increases_with_the_request() -> None:
    def measure(elongation: float) -> float:
        patch = _blob_patch(np.random.default_rng(9), 4000.0, elongation)
        rows, cols = np.nonzero(patch)
        cov = np.cov(np.vstack([cols.astype(float), rows.astype(float)]))
        eigenvalues = np.linalg.eigvalsh(cov)
        return float(np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-9)))

    assert measure(2.0) < measure(6.0)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_synthesize_scene_shapes_and_dtype(tmp_path: Path) -> None:
    images, masks = build_fixture(tmp_path)
    profile = profile_scenes(images, masks)
    image, mask, meta = synthesize_scene(profile, np.random.default_rng(0), size=64)
    assert image.shape == (2, 64, 64)
    assert image.dtype == np.float32
    assert mask.shape == (64, 64)
    assert set(np.unique(mask)).issubset({0, 1})
    assert "oil_fraction" in meta


def test_oil_budget_is_met(tmp_path: Path) -> None:
    """Coverage is drawn from the measured curve and filled, not left to chance.

    Letting the total emerge from a Poisson count of independently sampled areas
    missed the measured mean coverage by about 60%, because one 10^5 px slick
    swamps a hundred 176 px ones.
    """
    images, masks = build_fixture(tmp_path)
    profile = profile_scenes(images, masks)
    target = profile.oil_fraction["mean"]

    fractions = [
        synthesize_scene(profile, np.random.default_rng(seed), size=SCENE)[2]["oil_fraction"]
        for seed in range(12)
    ]
    measured = float(np.mean(fractions))
    # The fixture's coverage is ~6%; the point is that the generator lands in the
    # same region rather than overshooting by a factor.
    assert measured == pytest.approx(target, rel=0.6), f"{measured} vs target {target}"


def test_nodata_is_reproduced(tmp_path: Path) -> None:
    images, masks = build_fixture(tmp_path, nodata_scenes=4)
    profile = profile_scenes(images, masks)
    image, mask, meta = synthesize_scene(profile, np.random.default_rng(0), size=64)
    assert meta["nodata_fraction"] > 0.0
    assert np.isclose(image[0], NODATA_DB).sum() > 0
    # Ground truth must not claim oil inside a no-data block.
    nodata = image[0] == NODATA_DB
    assert not mask[nodata].any()


def test_single_pixel_extremes_cover_their_measured_fraction() -> None:
    """A radius-1 patch is one pixel, so its area is 1 and not pi.

    Using the disc area under-placed by a factor of pi, so the coverage came out
    at 32% of the measured value.
    """
    rng = np.random.default_rng(0)
    fraction = 0.01
    canvas = _extreme_canvas(rng, 512, 512, fraction, component_px=1.0)
    assert float(canvas.mean()) == pytest.approx(fraction, rel=0.05)


def test_extreme_canvas_is_empty_for_zero_inputs() -> None:
    rng = np.random.default_rng(0)
    assert not _extreme_canvas(rng, 64, 64, 0.0, 4.0).any()
    assert not _extreme_canvas(rng, 64, 64, 0.01, 0.0).any()


def test_nodata_canvas_is_empty_at_zero() -> None:
    assert not _nodata_canvas(np.random.default_rng(0), 64, 0.0).any()


def test_nodata_canvas_hits_its_target_roughly() -> None:
    canvas = _nodata_canvas(np.random.default_rng(0), 512, 0.05)
    assert float(canvas.mean()) == pytest.approx(0.05, abs=0.02)


def test_generate_dataset_writes_a_manifest_and_provenance(tmp_path: Path) -> None:
    images, masks = build_fixture(tmp_path / "fixture")
    profile = profile_scenes(images, masks)
    out = tmp_path / "synthetic"
    summary = generate_dataset(
        profile, out, GenerationConfig(n_scenes=6, size=64, seed=3, compression="deflate")
    )

    assert summary["n_rows"] == 6
    assert summary["provenance"] == "synthetic_mock"
    assert summary["split_counts"] == {"train": 5, "val": 1, "test": 0}

    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 6
    for row in rows:
        assert Path(row["image"]).is_file()
        assert Path(row["mask"]).is_file()
        assert row["provenance"] == "synthetic_mock"
    # Scene ids are unique across splits, so a case is always traceable.
    assert len({row["scene_id"] for row in rows}) == 6
    assert (out / "source.json").is_file()
    assert (out / "distribution.json").is_file()


def test_regenerating_rewrites_the_manifest_whole(tmp_path: Path) -> None:
    """The second generation's manifest must describe the second generation."""
    images, masks = build_fixture(tmp_path / "fixture")
    profile = profile_scenes(images, masks)
    out = tmp_path / "synthetic"

    generate_dataset(
        profile, out, GenerationConfig(n_scenes=8, size=64, seed=1, compression="deflate")
    )
    summary = generate_dataset(
        profile, out, GenerationConfig(n_scenes=4, size=64, seed=2, compression="deflate")
    )

    assert summary["n_rows"] == 4
    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 4


def test_stale_rasters_are_reported_rather_than_silently_included(
    tmp_path: Path,
) -> None:
    """A shorter re-generation leaves files behind; that must not be invisible.

    Generating 8 scenes and then 4 overwrites the first four stems and orphans
    the rest -- 4 images and 4 masks. They are reported so a directory that
    disagrees with its manifest is not a trap for whoever looks at it next.
    """
    images, masks = build_fixture(tmp_path / "fixture")
    profile = profile_scenes(images, masks)
    out = tmp_path / "synthetic"

    first = generate_dataset(
        profile, out, GenerationConfig(n_scenes=8, size=64, seed=1, compression="deflate")
    )
    assert first["stale_rasters"] == []

    before = {
        path
        for split in ("train", "val", "test")
        for kind in ("images", "masks")
        for path in (out / split / kind).glob("*.tif")
    }

    summary = generate_dataset(
        profile, out, GenerationConfig(n_scenes=4, size=64, seed=2, compression="deflate")
    )

    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    current = {Path(row["image"]) for row in rows} | {Path(row["mask"]) for row in rows}
    expected = before - current

    # The premise: a shorter re-generation genuinely leaves files behind. Which
    # ones depends on the split arithmetic -- run 2's val scene is stem 00003,
    # which does not collide with run 1's val scene at 00006 -- so the count is
    # derived rather than hardcoded.
    assert expected, "the premise of this test is that files are left behind"
    assert {Path(path) for path in summary["stale_rasters"]} == expected
    for path in expected:
        assert path.is_file()


def test_the_manifest_defines_the_dataset_not_the_directory(tmp_path: Path) -> None:
    """The defect this guards: 120 scenes at 2048 over 120 at 512 gave 123 pairs.

    A generation cannot clean up after an earlier one, so leftovers are
    inevitable. The dataset is defined by the manifest -- which is written
    atomically and completely -- so the leftovers must be inert.
    """
    images, masks = build_fixture(tmp_path / "fixture")
    profile = profile_scenes(images, masks)
    out = tmp_path / "synthetic"

    generate_dataset(
        profile, out, GenerationConfig(n_scenes=8, size=64, seed=1, compression="deflate")
    )
    generate_dataset(
        profile, out, GenerationConfig(n_scenes=4, size=128, seed=2, compression="deflate")
    )

    # The directory genuinely holds both generations -- that is the premise.
    on_disk = sum(
        len(list((out / split / "images").glob("*.tif"))) for split in ("train", "val", "test")
    )
    assert on_disk > 4

    settings = Settings(datasources=DataSourceSettings(synthetic_root=out))
    for rebuild in (False, True):
        spec = _build_synthetic(settings, rebuild=rebuild)
        assert spec.n_pairs == 4, f"rebuild={rebuild} indexed the leftovers"

    # And every pair the spec claims must be a real, single-resolution file.
    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    sizes = set()
    for row in rows:
        with rasterio.open(row["image"]) as src:
            sizes.add((src.width, src.height))
    assert sizes == {(128, 128)}, f"the dataset mixed resolutions: {sizes}"


def test_the_manifest_is_trusted_even_when_files_are_missing(tmp_path: Path) -> None:
    """With rebuild=False the manifest is the answer; it is not re-derived.

    Re-deriving on every call is what made the directory listing authoritative.
    A missing file is only escalated when the caller explicitly asked to rebuild.
    """
    images, masks = build_fixture(tmp_path / "fixture")
    profile = profile_scenes(images, masks)
    out = tmp_path / "synthetic"
    generate_dataset(
        profile, out, GenerationConfig(n_scenes=4, size=64, seed=1, compression="deflate")
    )

    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    Path(rows[0]["image"]).unlink()

    settings = Settings(datasources=DataSourceSettings(synthetic_root=out))
    assert _build_synthetic(settings, rebuild=False).n_pairs == 4
    # rebuild=True notices the gap and falls back to the directory listing.
    assert _build_synthetic(settings, rebuild=True).n_pairs == 3


# ---------------------------------------------------------------------------
# Extreme-value records
# ---------------------------------------------------------------------------


def test_profile_keeps_the_per_scene_extreme_records(tmp_path: Path) -> None:
    """The mean curve is not enough; the generator needs the spread.

    A pooled tail is set by the dataset's brightest scenes. An average over
    scenes has no brightest scene, so a generator fitted to the mean cannot
    reach the dataset's own p99.9.
    """
    images, masks = build_fixture(tmp_path, n_scenes=4)
    profile = profile_scenes(images, masks)
    for band in ("VV", "VH"):
        record = profile.extremes[band]
        assert "per_scene" in record
        assert len(record["per_scene"]) == 4
        for entry in record["per_scene"]:
            assert "sea_level_db" in entry
            assert "bright_fraction" in entry


def test_extreme_records_carry_more_than_three_knots(tmp_path: Path) -> None:
    """Three knots truncate the population at its 10th and 90th percentiles.

    The scene-level statistics that are checked live in the region those knots
    cannot describe, so the curve needs knots out towards the ends.
    """
    images, masks = build_fixture(tmp_path, n_scenes=4)
    profile = profile_scenes(images, masks)
    for band in ("VV", "VH"):
        entry = profile.extremes[band]["per_scene"][0]
        knots = sorted(key for key in entry if "_db_p" in key)
        assert "bright_db_p1" in knots
        assert "bright_db_p99" in knots
        assert "dark_db_p1" in knots
        assert "dark_db_p99" in knots


def test_the_generator_uses_the_extended_knot_set(tmp_path: Path) -> None:
    """The curve the generator samples must be the curve the profile recorded.

    It must span the population rather than stopping at p90: the scene's p99.9
    is around the population's p97, so a curve that ends at p90 cannot produce
    a pixel that bright, however the interpolation is done.
    """
    images, masks = build_fixture(tmp_path, n_scenes=4)
    profile = profile_scenes(images, masks)
    record = _pick_extreme_record(np.random.default_rng(0), profile, "VV")
    prefix_key = "bright_db_"
    curve = {
        key[len(prefix_key) :]: value for key, value in record.items() if key.startswith(prefix_key)
    }
    assert set(curve) == {"p1", "p10", "p25", "p50", "p75", "p90", "p99"}
    # The dense upper half is the point, not the count.
    assert float(curve["p99"]) >= float(curve["p90"]) >= float(curve["p75"])


def test_a_population_below_the_pixel_floor_records_no_coverage() -> None:
    """A coverage with no curve behind it is worse than no coverage at all.

    Every knot defaults to 0.0 dB, and 0.0 dB is this archive's no-data fill. A
    record that claims a bright population while leaving the whole curve at its
    defaults would make the generator write that many pixels at the fill value,
    which the profiler then excludes as fill -- so the population would be
    deleted from both sides of the comparison and the check would pass on a
    scene that never contained it. Measured on the real archive: 9 of 32 VH
    scenes had a bright population under the floor.
    """
    sea = -30.0
    band = np.full((64, 64), sea, dtype=np.float32)
    band[0, :3] = sea + 25.0  # three bright pixels, well under the floor

    record = _extremes(band, np.ones(band.shape, dtype=bool), sea)

    assert record["bright_fraction"] == 0.0
    assert all(value == NODATA_DB for key, value in record.items() if "_db_p" in key)


def test_a_population_at_the_pixel_floor_still_records_a_curve() -> None:
    """The floor is inclusive, so the fix above cannot silently widen.

    Zeroing the coverage is the right answer only when the population really is
    too small to describe. One pixel more than the floor must still be measured,
    or the guard would swallow every thin-but-real population in the archive.
    """
    sea = -30.0
    band = np.full((64, 64), sea, dtype=np.float32)
    band[0, :] = sea + 25.0  # exactly the floor

    record = _extremes(band, np.ones(band.shape, dtype=bool), sea)

    assert record["bright_fraction"] > 0.0
    assert record["bright_db_p50"] > sea
    assert record["bright_db_p50"] != NODATA_DB


def test_the_generator_refuses_a_coverage_with_an_all_nodata_curve() -> None:
    """A profile written before the invariant existed must not corrupt a scene.

    Such a record says "this many bright pixels" and then gives every knot the
    value 0.0 dB, which is the fill value. Writing them would inject fill pixels
    into the scene and buy the scene nothing, because the profiler excludes fill
    before it measures anything. Refusing the record is the safe direction: the
    scene is then merely missing a population it never had a curve for.
    """
    record = {
        "bright_fraction": 0.5,
        "bright_component_px": 1.0,
        "bright_db_p1": NODATA_DB,
        "bright_db_p10": NODATA_DB,
        "bright_db_p50": NODATA_DB,
        "bright_db_p90": NODATA_DB,
        "bright_db_p99": NODATA_DB,
    }
    band = np.full((32, 32), -32.0, dtype=np.float32)
    # ``_apply_extremes`` writes into the array it is given, so the returned
    # array is the same object. Comparing it to ``band`` would be comparing the
    # array to itself and would pass however many pixels were overwritten.
    before = band.copy()

    out = _apply_extremes(
        np.random.default_rng(0),
        band,
        record,
        "bright",
        32,
        32,
        1.0,
        scene_sea_level=-32.0,
    )

    # A 0.5 coverage over 1024 pixels would have been ~512 writes; none happened.
    np.testing.assert_array_equal(out, before)


def test_extremes_are_placed_relative_to_the_scene_sea_level() -> None:
    """The recorded offset must be preserved when the scene's sea moves.

    The record came from a scene whose sea sat at some level; writing its
    absolute brightness into a scene 2 dB darker would put a vessel 2 dB closer
    to the noise floor than the measurement says.
    """
    record = {
        "bright_fraction": 0.5,
        "bright_component_px": 1.0,
        "bright_db_p1": -10.0,
        "bright_db_p10": -10.0,
        "bright_db_p50": -10.0,
        "bright_db_p90": -10.0,
        "bright_db_p99": -10.0,
        "sea_level_db": -30.0,
    }
    band = np.full((32, 32), -32.0, dtype=np.float32)
    out = _apply_extremes(
        np.random.default_rng(0),
        band,
        record,
        "bright",
        32,
        32,
        1.0,
        scene_sea_level=-32.0,
    )
    # 20 dB above a sea level of -30 becomes 20 dB above -32.
    assert float(out.max()) == pytest.approx(-12.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_compare_profiles_against_itself_passes_every_check(tmp_path: Path) -> None:
    images, masks = build_fixture(tmp_path)
    profile = profile_scenes(images, masks)
    report = compare_profiles(profile, profile)
    assert report["n_failed"] == 0
    assert report["all_within_tolerance"] is True


def test_compare_profiles_reports_which_statistic_drifted(tmp_path: Path) -> None:
    """A boolean would not be actionable; the report names the quantity."""
    images, masks = build_fixture(tmp_path)
    real = profile_scenes(images, masks)

    shifted = profile_scenes(images, masks)
    shifted.band_stats[0]["percentiles"]["p50"] += 50.0

    report = compare_profiles(real, shifted)
    assert report["n_failed"] >= 1
    failed = {check["quantity"] for check in report["failed"]}
    assert "VV.p50" in failed
    entry = next(check for check in report["failed"] if check["quantity"] == "VV.p50")
    assert entry["real"] != entry["synthetic"]
    assert entry["delta"] > entry["tolerance"]


def test_compare_profiles_checks_the_standard_deviation(tmp_path: Path) -> None:
    """The spread is what the generator's field amplitude is calibrated from."""
    images, masks = build_fixture(tmp_path)
    real = profile_scenes(images, masks)
    wide = profile_scenes(images, masks)
    wide.band_stats[1]["std"] = real.band_stats[1]["std"] * 3.0
    report = compare_profiles(real, wide)
    assert "VH.std" in {check["quantity"] for check in report["failed"]}
