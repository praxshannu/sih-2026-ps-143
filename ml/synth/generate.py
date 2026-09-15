"""Generate a synthetic SAR oil-spill dataset fitted to a real profile.

The output is meant to be *substitutable* for the real archive, not merely
similar-looking. That means two separate obligations:

**Structure.** 2-band float32 GeoTIFFs in dB, 2048x2048, EPSG:4326 at the same
pixel size as the archive, a paired single-band mask per scene, the same
``{split}/images`` + ``{split}/masks`` layout, and the same ``manifest.jsonl``
contract. A loader that works on one must work on the other with no branch.

**Distribution.** Per-band dB statistics, sea level and its scene-to-scene
spread, oil-to-sea contrast, oil coverage *including the zero rate*, blob area
and elongation, and the effective number of looks. All of these are measured
from the real archive by :mod:`ml.synth.profile` and sampled here.

The physics that is modelled, and the physics that is not
---------------------------------------------------------
Modelled: multiplicative speckle with the measured equivalent number of looks,
a spatially correlated sea background, oil darker than sea by the measured
contrast, and elongated irregular slicks.

Not modelled: incidence-angle dependence, wind-shadow look-alikes, ships and
their wakes, land, or any real acquisition geometry. The generated scenes are
a statistical stand-in for training a pipeline, not a simulation of a
satellite. Anything derived from them is ``synthetic_mock`` and cannot support
a scientific claim — see :mod:`sentinel_core.provenance`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ml.synth.profile import DB_MAX, DB_MIN, NODATA_DB, DistributionProfile
from sentinel_core.provenance import DataProvenance, synthetic_warning

__all__ = ["GenerationConfig", "generate_dataset", "synthesize_scene"]

#: Pixel size of the real archive, in degrees. Matched so the synthetic scenes
#: have the same ground extent per pixel and the same geotransform shape.
DEFAULT_PIXEL_SIZE_DEG = 8.983152841195215e-05
DEFAULT_ORIGIN = (4.0, 55.0)

#: Fallback amplitude of the large-scale sea background, in dB, used only when
#: a profile predates the measurement. The real value is per band and comes from
#: ``profile.sea_field_db_std`` — see :func:`_field_amplitude`.
SEA_FIELD_AMPLITUDE_DB = 1.6

#: Softening applied to a blob edge, in pixels. A hard edge would give the
#: detector a boundary that no real SAR scene has.
BLOB_EDGE_SIGMA = 2.0

#: Raster suffixes a generation owns and may remove from its own split
#: directories. Deliberately narrow: see :func:`_clear_split_rasters`.
RASTER_SUFFIXES: tuple[str, ...] = (".tif", ".tiff")

#: The split directories a generated dataset is laid out in.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

#: Upper bound on the number of extreme patches stamped into one scene. VV's
#: clutter needs ~6000 single-pixel spikes at 2048^2; this only guards against
#: a profile that reports an implausible coverage.
MAX_EXTREME_PATCHES = 200_000

#: Upper bound on slicks per scene. A 2048^2 scene with 500 slicks is a texture,
#: not a slick field.
MAX_BLOBS_PER_SCENE = 60


def _looks_for(profile: DistributionProfile, name: str) -> float:
    """Effective looks for one band, from a per-band or legacy scalar profile."""
    speckle = profile.speckle or {}
    looks = speckle.get("equivalent_looks", 4.0)
    if isinstance(looks, dict):
        value = looks.get(name)
        if value is not None:
            return max(float(value), 1.0)
        return max(float(speckle.get("equivalent_looks_median", 4.0)), 1.0)
    return max(float(looks), 1.0)


def _field_amplitude(profile: DistributionProfile, name: str) -> float:
    """Amplitude of the sea background field for one band, in dB.

    Prefers the variance-calibrated value, because it is the one that makes the
    generated pooled spread match the measured one. Falls back to the measured
    block-median texture, then to the constant.
    """
    for key in ("sea_field_db_std", "sea_texture_db_std"):
        table = getattr(profile, key, None) or {}
        if name in table:
            return float(table[name])
    return SEA_FIELD_AMPLITUDE_DB


@dataclass
class GenerationConfig:
    """How much to generate, and how."""

    n_scenes: int = 120
    train_fraction: float = 0.8
    size: int = 2048
    seed: int = 42
    compression: str = "deflate"

    def split_counts(self) -> dict[str, int]:
        n_train = int(round(self.n_scenes * self.train_fraction))
        remainder = self.n_scenes - n_train
        n_val = remainder - remainder // 2
        n_test = remainder // 2
        return {"train": n_train, "val": n_val, "test": n_test}


# ---------------------------------------------------------------------------
# Scene synthesis
# ---------------------------------------------------------------------------


def _smooth_field(rng: np.random.Generator, size: int, coarse: int = 48, smooth: float = 2.0):
    """A unit-variance, spatially correlated random field.

    Generated coarse and interpolated up. The point is that neighbouring pixels
    are correlated the way a real sea surface is; independent per-pixel noise
    would look like noise, not like water.
    """
    from scipy.ndimage import gaussian_filter, zoom

    low = rng.normal(0.0, 1.0, (coarse, coarse)).astype(np.float32)
    low = gaussian_filter(low, smooth)
    field = zoom(low, size / coarse, order=1)[:size, :size].astype(np.float32)
    std = float(field.std())
    return field / std if std > 1e-6 else field


def _curve_arrays(percentiles: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """``(levels, values)`` from a ``{"p10": v, ...}`` mapping, sorted by level."""
    ordered = sorted(percentiles.items(), key=lambda item: float(item[0][1:]))
    levels = np.asarray([float(key[1:]) / 100.0 for key, _ in ordered])
    values = np.asarray([float(value) for _, value in ordered])
    return levels, values


def _sample_from_percentiles(rng: np.random.Generator, percentiles: dict[str, float]) -> float:
    """Draw a value from the empirical curve the profile recorded.

    Interpolating between the stored percentiles is a piecewise-linear
    approximation to the real distribution's inverse CDF. It reproduces the
    body and, more importantly, the tails — which is where a detector's
    threshold lives.
    """
    if not percentiles:
        return 0.0
    levels, values = _curve_arrays(percentiles)
    draw = rng.uniform(float(levels[0]), float(levels[-1]))
    return float(np.interp(draw, levels, values))


def _sample_many(rng: np.random.Generator, percentiles: dict[str, float], size: int) -> np.ndarray:
    """Vectorised form of :func:`_sample_from_percentiles`, for thousands of draws."""
    if not percentiles or size <= 0:
        return np.zeros(size, dtype=np.float32)
    levels, values = _curve_arrays(percentiles)
    draws = rng.uniform(float(levels[0]), float(levels[-1]), size)
    return np.interp(draws, levels, values).astype(np.float32)


def _oil_fraction_curve(profile: DistributionProfile) -> dict[str, float]:
    """The measured per-scene oil coverage, as an inverse CDF."""
    oil = profile.oil_fraction or {}
    return {
        "p10": float(oil.get("p10", 0.0)),
        "p50": float(oil.get("p50", 0.0)),
        "p90": float(oil.get("p90", 0.0)),
        "p99": float(oil.get("max", 0.0)),
    }


def _extreme_canvas(
    rng: np.random.Generator, height: int, width: int, fraction: float, component_px: float
) -> np.ndarray:
    """A coverage-matched mask of bright clutter or compact objects.

    The *component size* is what makes this honest. Measured on the archive, VV's
    bright extremes have a median component of 1.7 px — isolated spikes, i.e.
    heavy-tailed sea clutter — while VH's are 40 px, i.e. vessels. Reproducing
    only the coverage would put 0.17% of VH pixels down as single-pixel spikes
    and lose the objects entirely; reproducing only the component size would
    lose the clutter. Both numbers are measured and both are used.
    """
    canvas = np.zeros((height, width), dtype=bool)
    if fraction <= 0.0 or component_px <= 0.0:
        return canvas
    radius = max(1, int(round(math.sqrt(max(component_px, 1.0) / math.pi))))
    # A radius-1 patch is a single pixel, so its area is 1 and not pi. Using the
    # disc area here would under-place by a factor of pi — the coverage would
    # silently come out at 32% of the measured value.
    area = 1.0 if radius <= 1 else math.pi * radius**2
    count = int(round(fraction * height * width / area))
    if count <= 0:
        return canvas
    count = min(count, MAX_EXTREME_PATCHES)

    rows = rng.integers(0, height, count)
    cols = rng.integers(0, width, count)
    if radius <= 1:
        canvas[rows, cols] = True
        return canvas

    offsets = np.asarray(
        [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dy * dy + dx * dx <= radius * radius
        ],
        dtype=np.int64,
    )
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        # Wrapped, not clipped: a vessel at the scene edge is cut in two by the
        # swath boundary in reality, and wrapping keeps the coverage exact.
        canvas[(row + offsets[:, 0]) % height, (col + offsets[:, 1]) % width] = True
    return canvas


def _blob_patch(rng: np.random.Generator, area_px: float, elongation: float) -> np.ndarray:
    """A boolean patch holding one irregular, elongated slick.

    An ellipse would be wrong: real slicks are wind-driven ribbons with wavy
    edges and a slight arc. The patch is an ellipse whose semi-minor axis is
    modulated along the major axis, plus a gentle bend.

    Axis convention: the array is ``(length, thickness)`` and the slick runs
    along axis 0. Getting this backwards makes every slick round regardless of
    the elongation requested, which is silent — the mask is still plausible, it
    is just the wrong shape.
    """
    major = math.sqrt(max(area_px, 1.0) * max(elongation, 1.0) / math.pi)
    minor = max(math.sqrt(max(area_px, 1.0) / (math.pi * max(elongation, 1.0))), 1.0)

    length = int(2.4 * major) + 5
    thickness = int(2.4 * minor) + 5
    along, across = np.mgrid[0:length, 0:thickness].astype(np.float32)
    u = along - length / 2.0
    v = across - thickness / 2.0

    phase = rng.uniform(0.0, 2.0 * math.pi)
    waves = float(rng.integers(1, 4))
    amplitude = rng.uniform(0.10, 0.30)
    bend = rng.uniform(-0.25, 0.25)

    # Wavy half-thickness along the major axis; the sine averages to zero, so
    # the expected area stays close to the requested one.
    half_thickness = minor * (
        1.0 + amplitude * np.sin(2.0 * math.pi * waves * u / max(length, 1) + phase)
    )
    centreline = bend * major * (u / max(major, 1.0)) ** 2
    radius = (u / major) ** 2 + ((v - centreline) / np.maximum(half_thickness, 1.0)) ** 2
    return radius <= 1.0


def _add_blob(
    alpha: np.ndarray, rng: np.random.Generator, area_px: float, elongation: float
) -> int:
    """Render one blob into the soft-alpha canvas.

    Returns how many pixels it *newly* covered, or 0 if it fell outside. The
    count is measured rather than assumed because overlapping slicks merge
    (``np.maximum``) instead of summing, so the requested area and the added
    area differ — and the caller budgets oil coverage against the real number.
    """
    from scipy.ndimage import gaussian_filter

    patch = _blob_patch(rng, area_px, elongation)
    if not patch.any():
        return 0

    softened = gaussian_filter(patch.astype(np.float32), BLOB_EDGE_SIGMA)
    peak = float(softened.max())
    if peak <= 0:
        return 0
    softened /= peak

    height, width = softened.shape
    size = alpha.shape[0]
    if height >= size or width >= size:
        return 0
    top = int(rng.integers(0, size - height + 1))
    left = int(rng.integers(0, size - width + 1))
    region = alpha[top : top + height, left : left + width]
    before = int(np.count_nonzero(region > 0.5))
    # max() so overlapping slicks do not cancel; slicks merge in reality.
    np.maximum(region, softened, out=region)
    return max(int(np.count_nonzero(region > 0.5)) - before, 0)


def _nodata_canvas(rng: np.random.Generator, size: int, target_fraction: float) -> np.ndarray:
    """Rectangular no-data strips along scene edges, as the archive has.

    The real archive carries 0.0 dB fill blocks covering 2-19% of some scenes
    (swath edges and no-coverage borders). They are not cosmetic: sea sits near
    -25 dB, so a 0.0 dB block is the brightest object in the scene. A model
    trained only on clean scenes will read a real one as a target, so the
    nuisance is reproduced rather than hidden.
    """
    canvas = np.zeros((size, size), dtype=bool)
    if target_fraction <= 0.0:
        return canvas
    strips = int(rng.integers(1, 4))
    sides = rng.permutation(4)[:strips]
    share = target_fraction / strips
    for side in sides:
        thickness = max(1, min(size, int(round(share * size))))
        if side == 0:
            canvas[:thickness, :] = True
        elif side == 1:
            canvas[-thickness:, :] = True
        elif side == 2:
            canvas[:, :thickness] = True
        else:
            canvas[:, -thickness:] = True
    return canvas


def _pick_extreme_record(
    rng: np.random.Generator, profile: DistributionProfile, name: str
) -> dict[str, Any]:
    """One *measured scene's* extreme record, for one generated scene.

    Sampling a scene rather than averaging across scenes is the whole point. The
    pooled tail of a dataset is set by its brightest scenes; an averaged curve
    has no brightest scene, so a generator built on it can never reach the
    dataset's p99.9. Falls back to the mean aggregate for a profile written
    before the per-scene records existed.
    """
    record = (profile.extremes or {}).get(name) or {}
    per_scene = record.get("per_scene") or []
    if per_scene:
        return dict(per_scene[int(rng.integers(0, len(per_scene)))])
    return dict(record)


def _apply_extremes(
    rng: np.random.Generator,
    band_db: np.ndarray,
    record: dict[str, Any],
    prefix: str,
    height: int,
    width: int,
    area_scale: float,
    *,
    scene_sea_level: float,
) -> np.ndarray:
    """Overwrite the bright or dark extreme pixels with measured dB levels.

    The coverage is a fraction, so it transfers between resolutions unchanged.
    The component size is in pixels, so it is scaled with the scene area to keep
    objects the same size *relative to the scene* — with a one-pixel floor,
    because a component smaller than a pixel does not exist.

    Brightness is written as an *offset from the scene's own sea level*, not as
    an absolute level. The record came from a scene whose sea sat at some other
    level, and a vessel is bright relative to the water it is in — writing the
    recorded absolute value into a scene 2 dB darker would put the object 2 dB
    too close to the noise floor.
    """
    fraction = float(record.get(f"{prefix}_fraction", 0.0))
    component_px = max(float(record.get(f"{prefix}_component_px", 0.0)) * area_scale, 1.0)
    if fraction <= 0.0 or component_px <= 0.0:
        return band_db

    canvas = _extreme_canvas(rng, height, width, fraction, component_px)
    count = int(np.count_nonzero(canvas))
    if count == 0:
        return band_db

    # Every recorded knot, not a fixed trio, so extending the profile's knot set
    # does not silently leave the generator behind it. ``_curve_arrays`` expects
    # keys of the form ``p10``, so the band and direction prefix is stripped.
    prefix_key = f"{prefix}_db_"
    curve = {
        key[len(prefix_key) :]: float(value)
        for key, value in record.items()
        if key.startswith(prefix_key) and isinstance(value, int | float)
    }
    if not curve:
        return band_db
    # A profile written before the fraction/curve invariant was enforced can
    # carry a coverage with an all-default curve. Writing those pixels at their
    # default of 0.0 dB would inject this archive's no-data fill value into the
    # scene, and the profiler would then exclude them as fill rather than count
    # them. Refuse the record rather than corrupt the scene.
    if all(abs(value - NODATA_DB) <= 1e-6 for value in curve.values()):
        return band_db

    reference_sea = float(record.get("sea_level_db", scene_sea_level))
    shift = float(scene_sea_level) - reference_sea
    band_db[canvas] = _sample_many(rng, curve, count) + shift
    return band_db


def synthesize_scene(
    profile: DistributionProfile,
    rng: np.random.Generator,
    *,
    size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One synthetic scene: ``(image (C,H,W) float32 dB, mask (H,W) uint8, meta)``."""
    height, width = (size, size) if size else profile.shape
    band_names = [str(stat["name"]) for stat in profile.band_stats]
    bands = len(band_names)

    # Blob areas are measured in the real archive's pixels. Generating at a
    # different edge length without rescaling would make every slick the wrong
    # size relative to the scene — and the oil fraction with it.
    reference = float(profile.shape[0]) if profile.shape and profile.shape[0] else float(height)
    area_scale = (height / reference) ** 2

    has_oil = bool(rng.random() > float(profile.oil_fraction.get("zero_rate", 0.0)))

    alpha = np.zeros((height, width), dtype=np.float32)
    blob_count = 0
    if has_oil:
        # The oil budget is *drawn from the measured per-scene coverage* and
        # filled until it is spent, rather than emerging from a Poisson count of
        # independently sampled areas. The emergent total is dominated by the
        # extreme tail of the area distribution — one 10^5 px slick swamps a
        # hundred 176 px ones — and it missed the measured mean coverage by
        # about 60%. Budgeting targets the quantity that was actually measured.
        target_fraction = max(_sample_from_percentiles(rng, _oil_fraction_curve(profile)), 0.0)
        budget_px = target_fraction * height * width
        area_curve = profile.blobs.get("area_px", {}) or {"p50": 5000.0}
        elongation_curve = profile.blobs.get("elongation", {}) or {"p50": 3.0}
        placed = 0.0
        misses = 0
        while placed < budget_px and blob_count < MAX_BLOBS_PER_SCENE and misses < 8:
            want = _sample_from_percentiles(rng, area_curve) * area_scale
            want = max(min(want, budget_px - placed), 4.0)
            elongation = max(_sample_from_percentiles(rng, elongation_curve), 1.0)
            added = _add_blob(alpha, rng, want, elongation)
            if added:
                placed += added
                blob_count += 1
            else:
                # A blob larger than the scene can never be placed; count the
                # miss so a pathological area curve cannot spin forever.
                misses += 1

    nodata_target = 0.0
    if rng.random() < float(profile.nodata.get("scene_rate", 0.0)):
        nodata_target = float(profile.nodata.get("fraction_mean", 0.0))
    nodata = _nodata_canvas(rng, height, nodata_target)

    image = np.empty((bands, height, width), dtype=np.float32)
    for index, name in enumerate(band_names):
        base = float(profile.sea_level_db.get(name, -20.0))
        spread = float(profile.sea_level_db_std.get(name, 0.0))
        sea_level = base + (rng.normal(0.0, spread) if spread > 0 else 0.0)

        field_db = _smooth_field(rng, height) * _field_amplitude(profile, name)

        # Per band. VV and VH are separate measurements and their speckle
        # differs by more than a factor of four here (about 2.8 looks vs 12), so
        # one shared value makes one band systematically too noisy.
        looks = _looks_for(profile, name)
        # Multiplicative Gamma speckle on linear intensity, then to dB. Doing
        # this in dB directly would give symmetric noise; real speckle is not.
        speckle_linear = rng.gamma(shape=looks, scale=1.0 / looks, size=(height, width))
        speckle_db = 10.0 * np.log10(np.maximum(speckle_linear, 1e-6)).astype(np.float32)

        contrast = float(profile.contrast_db.get(name, 6.0))
        contrast_spread = float(profile.contrast_db_std.get(name, 0.0))
        per_blob = contrast + (rng.normal(0.0, contrast_spread) if contrast_spread > 0 else 0.0)

        band_db = sea_level + field_db + speckle_db - per_blob * alpha

        # Pixels Gamma speckle cannot produce. Overwritten rather than added:
        # the profile recorded the dB level of these pixels relative to their
        # own scene's sea, so reproducing that offset reproduces the tail.
        extreme_record = _pick_extreme_record(rng, profile, name)
        for prefix in ("bright", "dark"):
            band_db = _apply_extremes(
                rng,
                band_db,
                extreme_record,
                prefix,
                height,
                width,
                area_scale,
                scene_sea_level=sea_level,
            )

        image[index] = np.clip(band_db, DB_MIN, DB_MAX)

    if nodata.any():
        image[:, nodata] = NODATA_DB

    mask = (alpha > 0.5).astype(np.uint8)
    mask[nodata] = 0
    meta = {
        "has_oil": bool(mask.any()),
        "n_blobs": blob_count,
        "oil_fraction": float(mask.mean()),
        "nodata_fraction": float(nodata.mean()),
        "sea_level_db": {name: float(profile.sea_level_db.get(name, 0.0)) for name in band_names},
    }
    return image, mask, meta


# ---------------------------------------------------------------------------
# Dataset writing
# ---------------------------------------------------------------------------


def _write_scene(
    image_path: Path,
    mask_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    compression: str,
    origin: tuple[float, float],
    pixel_size: float,
) -> tuple[int, int]:
    import rasterio
    from rasterio.transform import from_origin

    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    bands, height, width = image.shape
    transform = from_origin(origin[0], origin[1], pixel_size, pixel_size)
    profile_kwargs: dict[str, Any] = {"tiled": True, "blockxsize": 256, "blockysize": 256}
    if compression != "none":
        profile_kwargs["compress"] = compression
        if compression == "deflate":
            # Predictor 3 is the floating-point horizontal differencing filter;
            # on smooth SAR data it roughly halves the file.
            profile_kwargs["predictor"] = 3

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
        **profile_kwargs,
    ) as dst:
        dst.write(image)
        for index, name in enumerate(("VV", "VH")[:bands], start=1):
            dst.set_band_description(index, name)

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
        **{k: v for k, v in profile_kwargs.items() if k != "predictor"},
    ) as dst:
        dst.write(mask[np.newaxis, ...])
        dst.set_band_description(1, "oil")

    return image_path.stat().st_size, mask_path.stat().st_size


def _stale_rasters(root: Path, written: Sequence[Path]) -> list[Path]:
    """Rasters in this dataset's split directories that this run did not write.

    ``generate_dataset`` rewrites every scene it is asked for and then rewrites
    the manifest whole, but it has no idea what an *earlier* run left behind.
    Generating 120 scenes at 2048 on top of an earlier run's 120 at 512 leaves
    the old files in place.

    They are reported rather than deleted. Deleting a few hundred files inside a
    user's data directory to tidy up after ourselves is a destructive operation
    that the operator did not ask for, and it is not necessary: the manifest is
    written atomically and completely, so it — not the directory listing — is
    what defines the dataset. ``sentinel_core.datasource`` now reads it that way,
    which is the actual fix. This function exists so the leftovers are not
    silent, because a directory that disagrees with its manifest is a trap for
    whoever looks at it next.
    """
    owned = {path.resolve() for path in written}
    stale: list[Path] = []
    for split in SPLIT_NAMES:
        for kind in ("images", "masks"):
            directory = root / split / kind
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() in RASTER_SUFFIXES and path.resolve() not in owned:
                    stale.append(path)
    return stale


def generate_dataset(
    profile: DistributionProfile,
    root: Path,
    config: GenerationConfig,
    *,
    pixel_size: float = DEFAULT_PIXEL_SIZE_DEG,
    origin: tuple[float, float] = DEFAULT_ORIGIN,
    band_names: Sequence[str] = ("VV", "VH"),
) -> dict[str, Any]:
    """Write a complete synthetic dataset and return a summary of what it did.

    The manifest written here is the definition of the dataset. Rasters left by
    a previous generation are reported in the summary as ``stale_rasters`` and
    excluded from it; see :func:`_stale_rasters`.
    """
    counts = config.split_counts()
    rng = np.random.default_rng(config.seed)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scene_index = 0
    per_split_stats: dict[str, list[dict[str, Any]]] = {name: [] for name in counts}

    for split, count in counts.items():
        for _ in range(count):
            image, mask, meta = synthesize_scene(profile, rng, size=config.size)
            # Global numbering, so a scene id is unique across splits — a split
            # that reused 00000.tif in all three would be impossible to audit.
            stem = f"{scene_index:05d}"
            image_path = root / split / "images" / f"{stem}.tif"
            mask_path = root / split / "masks" / f"{stem}.tif"
            image_bytes, mask_bytes = _write_scene(
                image_path,
                mask_path,
                image,
                mask,
                compression=config.compression,
                origin=(origin[0] + scene_index * 0.2, origin[1]),
                pixel_size=pixel_size,
            )
            rows.append(
                {
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "split": split,
                    "scene_id": stem,
                    "provenance": DataProvenance.SYNTHETIC.value,
                    "source": "synthetic",
                    "image_bytes": image_bytes,
                    "mask_bytes": mask_bytes,
                }
            )
            per_split_stats[split].append(meta)
            scene_index += 1
            if scene_index % 20 == 0:
                logger.info("  generated {}/{} scene(s)", scene_index, config.n_scenes)

    manifest = root / "manifest.jsonl"
    tmp_manifest = manifest.with_suffix(".jsonl.tmp")
    with tmp_manifest.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_manifest.replace(manifest)

    written = [Path(row["image"]) for row in rows] + [Path(row["mask"]) for row in rows]
    stale = _stale_rasters(root, written)
    if stale:
        logger.warning(
            "{} raster(s) in {} were not written by this run and are NOT part of the "
            "dataset; the manifest defines it. They are left in place rather than "
            "deleted — remove them yourself if you want the directory to match. "
            "First few: {}",
            len(stale),
            root,
            ", ".join(path.name for path in stale[:5]),
        )

    oil_fractions = [meta["oil_fraction"] for stats in per_split_stats.values() for meta in stats]
    nodata_fractions = [
        meta["nodata_fraction"] for stats in per_split_stats.values() for meta in stats
    ]
    summary: dict[str, Any] = {
        "kind": "synthetic",
        "bands": len(band_names),
        "shape": list(profile.shape if not config.size else (config.size, config.size)),
        "dtype": "float32",
        "n_rows": len(rows),
        "split_counts": counts,
        "seed": config.seed,
        "compression": config.compression,
        "split_strategy": "directory",
        "provenance": DataProvenance.SYNTHETIC.value,
        "note": synthetic_warning("synthetic dataset"),
        "fitted_from": profile.source,
        #: Rasters present in the split directories that this run did not write.
        #: Excluded from the dataset (the manifest defines it), reported so a
        #: directory that disagrees with its manifest is not a silent trap.
        "stale_rasters": [str(path) for path in stale],
        "oil_fraction_mean": round(float(np.mean(oil_fractions)), 6) if oil_fractions else 0.0,
        "oil_fraction_zero_rate": round(
            float(np.mean([value <= 0.0 for value in oil_fractions])), 6
        )
        if oil_fractions
        else 0.0,
        "nodata_fraction_mean": round(float(np.mean(nodata_fractions)), 6)
        if nodata_fractions
        else 0.0,
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "warnings": [
            "Statistical stand-in, not a simulation. Not modelled: incidence-angle "
            "dependence, acquisition geometry, land, or the spatial structure of "
            "vessels and their wakes — the bright and dark extremes are reproduced "
            "in coverage and brightness, but placed as compact patches rather than "
            "as real objects on real tracks.",
        ],
    }

    (root / "source.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    profile.save(root / "distribution.json")
    logger.info("wrote {} synthetic scene(s) to {}", len(rows), root)
    return summary
