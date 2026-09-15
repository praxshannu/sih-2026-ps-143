"""Measure a SAR dataset so synthetic data can be fitted to it.

What is measured, and why each one matters
------------------------------------------
* **Per-band dB histogram.** The headline "does it look the same" check. A
  synthetic scene whose 5th percentile is 8 dB off the real one will produce a
  detector trained on a threshold that does not transfer.
* **Sea level and oil-to-sea contrast.** Oil is dark in SAR. If the generator
  picks a contrast the real data never shows, the task becomes trivially easier
  or impossible, and the resulting IoU says nothing about the real one.
* **Oil coverage per scene, including the zero rate.** Roughly half of a real
  spill archive has no oil in it. Generating only positive scenes would make
  the model hallucinate slicks.
* **Blob geometry.** Slicks are elongated, not circular. Area and elongation
  percentiles are sampled directly.
* **Effective number of looks.** Derived from the local coefficient of
  variation in homogeneous water. This is what makes the speckle texture right
  rather than merely noisy.

The same function profiles real and synthetic data, so the comparison in
``compare_profiles`` is measuring both sides with one ruler.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

__all__ = [
    "DB_MAX",
    "DB_MIN",
    "HIST_BINS",
    "NODATA_DB",
    "DistributionProfile",
    "compare_profiles",
    "profile_from_spec",
    "profile_scenes",
    "read_bands",
]

#: dB window for the shared histogram. The real archive spans about
#: -54 .. +13 dB, so this clips almost nothing and both sides share the bins.
DB_MIN = -60.0
DB_MAX = 20.0
HIST_BINS = 256

_PERCENTILES = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)

#: Fill value used by the archive for no-data. Several scenes carry large
#: blocks of exactly 0.0 dB (measured: 1.8%, 6.1% and 19.0% of pixels in three
#: of twelve sampled scenes) — swath edges and no-coverage borders.
#:
#: This matters more than it looks. Sea sits near -25 dB, so a 0.0 dB block is
#: the *brightest* thing in the scene: left in, it becomes a spike in the
#: histogram, it drags the 99th percentile to exactly 0.0, and it looks to a
#: detector like a large bright target. It is excluded from every statistic
#: here, recorded as its own measurement, and reproduced by the generator so
#: the model meets the same nuisance at training time.
NODATA_DB = 0.0
_NODATA_TOLERANCE = 1e-6

#: Below this the "oil" pixels are treated as the sea mode's own left tail
#: rather than as a slick. A mask is binary ground truth, so this only guards
#: the contrast estimate against an empty or near-empty mask.
_MIN_OIL_PIXELS = 64

#: How far above (or below) the scene's sea level a pixel has to sit before it
#: counts as an "extreme". 8 dB is well outside what Gamma speckle produces:
#: measured on this archive, the brightest 0.1% of VV pixels is 16 dB above sea
#: level, and no number of looks gets there.
EXTREME_THRESHOLD_DB = 8.0

#: Fewer pixels than this and the percentile estimate is noise, not a
#: measurement.
_MIN_EXTREME_PIXELS = 64

#: Percentiles recorded for the bright and dark extreme populations.
#:
#: Five knots rather than three, because the generator samples this curve as an
#: inverse CDF and the scene-level statistics that are checked live *outside*
#: the body of the extreme population. Measured on this archive, VV's brightest
#: 0.1% of pixels is the top 22% of its bright population, and VH's is the top
#: 40% — both inside [p10, p90], but only just, and a three-knot piecewise
#: linear curve straightens out exactly the curvature that decides where they
#: land. The tails are where a detector's threshold sits, so they are worth the
#: extra two numbers.
_EXTREME_PERCENTILES = (1.0, 10.0, 50.0, 90.0, 99.0)


def read_bands(path: Path) -> np.ndarray:
    """Read a raster as ``(C, H, W)`` float32 without touching global state."""
    import warnings

    import rasterio

    from sentinel_core.logging import quiet_logger

    with quiet_logger("rasterio"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            return src.read().astype(np.float32, copy=False)


def _read_mask(path: Path) -> np.ndarray:
    """Read a mask as a boolean ``(H, W)`` array."""
    data = read_bands(path)
    if data.ndim == 3:
        data = data[0] if data.shape[0] == 1 else data.max(axis=0)
    return data > 0.5


def _percentiles(values: np.ndarray) -> dict[str, float]:
    computed = np.percentile(values, _PERCENTILES)
    return {f"p{p:g}": round(float(v), 4) for p, v in zip(_PERCENTILES, computed, strict=True)}


def _histogram(values: np.ndarray) -> dict[str, Any]:
    counts, edges = np.histogram(values, bins=HIST_BINS, range=(DB_MIN, DB_MAX))
    total = int(counts.sum())
    probabilities = (counts / total).tolist() if total else [0.0] * HIST_BINS
    return {
        "bin_edges": [round(float(e), 4) for e in edges],
        "probabilities": [round(float(p), 10) for p in probabilities],
        "clipped_below": int((values < DB_MIN).sum()),
        "clipped_above": int((values > DB_MAX).sum()),
    }


def _blob_geometry(mask: np.ndarray) -> tuple[list[float], list[float]]:
    """Areas (px) and elongations of each connected component.

    Elongation is ``sqrt(lambda1 / lambda2)`` of the component's second-moment
    matrix — 1.0 is a disc, and a Fay-type linear slick runs well above 3.
    """
    from scipy import ndimage

    labelled, count = ndimage.label(mask)
    if count == 0:
        return [], []

    areas: list[float] = []
    elongations: list[float] = []
    # ndimage.find_objects gives the bounding box per label; the moments are
    # computed on the slice, which is much cheaper than over the whole scene.
    for index, box in enumerate(ndimage.find_objects(labelled), start=1):
        if box is None:
            continue
        region = labelled[box] == index
        area = float(region.sum())
        if area < 4:
            continue
        rows, cols = np.nonzero(region)
        rows = rows.astype(np.float64)
        cols = cols.astype(np.float64)
        rows -= rows.mean()
        cols -= cols.mean()
        cov = np.array(
            [
                [(cols * cols).mean(), (cols * rows).mean()],
                [(cols * rows).mean(), (rows * rows).mean()],
            ]
        )
        eigenvalues = np.linalg.eigvalsh(cov)
        major = float(eigenvalues[-1])
        minor = float(eigenvalues[0])
        # A one-pixel-wide line has a zero second moment across; cap rather
        # than divide by zero.
        elongation = 50.0 if minor <= 1e-9 else float(math.sqrt(major / minor))
        areas.append(area)
        elongations.append(min(elongation, 50.0))
    return areas, elongations


def _component_median_px(mask: np.ndarray) -> float:
    """Median connected-component size, in pixels. 0.0 when there is nothing."""
    from scipy import ndimage

    labelled, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = np.bincount(labelled.ravel())[1:]
    return float(np.median(sizes)) if sizes.size else 0.0


def _extremes(band: np.ndarray, valid: np.ndarray, sea_level: float) -> dict[str, float]:
    """Measure the pixels Gamma speckle cannot explain.

    A Gaussian sea field plus Gamma speckle produces a symmetric-ish dB
    distribution with tails set by the number of looks. The real archive is not
    that: measured here, VV's brightest 0.1% sits 16 dB above sea level, which
    is roughly 40x the mean intensity and far outside any plausible looks
    count. Two separate mechanisms are responsible, and the *component size*
    is what tells them apart:

    * **VV, median component 1.7 px** — isolated single pixels. This is
      spiky sea clutter: real sea surface returns are K-distributed, with a
      heavier tail than the Gamma law a single-look model assumes.
    * **VH, median component 40 px** — compact objects. Vessels, rigs and
      their wakes. A detector trained on scenes without them will flag every
      real one.

    Dark extremes are the same measurement with the opposite sign: wind
    shadows and low-backscatter patches, 2.2% of VV pixels and 1.2% of VH.

    Recording all three numbers — coverage, brightness and component size —
    is what lets the generator reproduce the tail instead of approximating it
    with more noise, which would widen the body as well.

    Brightness is recorded as a five-knot inverse CDF rather than a mean, and
    the scene's own sea level is recorded alongside it, so the generator can
    place these pixels at the measured *offset from their scene's sea* instead
    of at an absolute level. Writing absolute levels into every scene is what
    decouples the extremes from the sea they sit on.
    """
    out: dict[str, float] = {"sea_level_db": round(float(sea_level), 4)}
    for prefix in ("bright", "dark"):
        out[f"{prefix}_fraction"] = 0.0
        out[f"{prefix}_component_px"] = 0.0
        for level in _EXTREME_PERCENTILES:
            out[f"{prefix}_db_p{level:g}"] = 0.0

    for prefix, selection in (
        ("bright", valid & (band > sea_level + EXTREME_THRESHOLD_DB)),
        ("dark", valid & (band < sea_level - EXTREME_THRESHOLD_DB)),
    ):
        out[f"{prefix}_fraction"] = float(selection.mean())
        if int(selection.sum()) < _MIN_EXTREME_PIXELS:
            continue
        values = np.percentile(band[selection], _EXTREME_PERCENTILES)
        for level, value in zip(_EXTREME_PERCENTILES, values, strict=True):
            out[f"{prefix}_db_p{level:g}"] = round(float(value), 4)
        out[f"{prefix}_component_px"] = round(_component_median_px(selection), 3)
    return out


def _effective_looks(band: np.ndarray, valid: np.ndarray) -> float:
    """Equivalent number of looks, from the local coefficient of variation.

    For a fully developed speckle field the intensity CV is ``1/sqrt(L)``. The
    estimate uses 16x16 blocks and the median CV, so a slick or a ship does not
    drag the answer around.

    Blocks that are mostly no-data are skipped rather than clamped: a constant
    fill block has zero variance, which would read as an infinite number of
    looks and pull the estimate upwards.
    """
    linear = np.power(10.0, band / 10.0)
    block = 16
    height, width = linear.shape
    rows = (height // block) * block
    cols = (width // block) * block
    if rows == 0 or cols == 0:
        return 1.0

    def _blocks(array: np.ndarray) -> np.ndarray:
        return array[:rows, :cols].reshape(rows // block, block, cols // block, block).transpose(
            0, 2, 1, 3
        )

    blocks = _blocks(linear)
    valid_fraction = _blocks(valid.astype(np.float32)).mean(axis=(2, 3))
    means = blocks.mean(axis=(2, 3))
    stds = blocks.std(axis=(2, 3))

    usable = (valid_fraction >= 0.95) & (means > 0)
    if not usable.any():
        return 1.0
    cv = stds[usable] / means[usable]
    cv = cv[cv > 1e-6]
    if cv.size == 0:
        return 64.0
    median_cv = float(np.median(cv))
    return float(min(1.0 / (median_cv**2), 64.0))


def speckle_sigma_db(looks: float) -> float:
    """Standard deviation of dB speckle for ``looks`` looks, exactly.

    Gamma speckle with ``L`` looks has ``E[ln Y] = psi(L) - ln L`` and
    ``Var[ln Y] = psi'(L)``, so the dB spread is ``10/ln(10) * sqrt(psi'(L))``.

    The usual shortcut ``4.3429/sqrt(L)`` is the small-CV limit and it
    *understates* the spread at low looks — at L = 2.8 it gives 2.60 dB where
    the exact value is 2.85 dB. VV here really is low-look, so the difference is
    large enough to matter.
    """
    from scipy.special import polygamma

    looks = max(float(looks), 1e-3)
    return float(4.3429448190325175 * math.sqrt(float(polygamma(1, looks))))


def _sea_texture(band: np.ndarray, valid: np.ndarray, mask: np.ndarray, block: int = 32) -> float:
    """Spatial variation of the sea background, in dB.

    Computed as the standard deviation of per-block *medians* over blocks that
    are almost entirely valid and almost entirely oil-free. The median over a
    32x32 block averages speckle down by roughly the number of pixels, so what
    remains is the large-scale structure of the sea surface rather than noise.

    This is a *lower bound* on the field amplitude: block medians also average
    out structure finer than one block. ``profile_scenes`` therefore also
    records a variance-calibrated amplitude, and the generator uses that one.
    """
    height, width = band.shape
    rows = (height // block) * block
    cols = (width // block) * block
    if rows < block or cols < block:
        return 0.0

    def _blocks(array: np.ndarray) -> np.ndarray:
        return array[:rows, :cols].reshape(rows // block, block, cols // block, block).transpose(
            0, 2, 1, 3
        )

    medians = np.median(_blocks(band), axis=(2, 3))
    valid_fraction = _blocks(valid.astype(np.float32)).mean(axis=(2, 3))
    oil_fraction = _blocks(mask.astype(np.float32)).mean(axis=(2, 3))
    usable = (valid_fraction > 0.95) & (oil_fraction < 0.05)
    if int(usable.sum()) < 16:
        return 0.0
    return float(medians[usable].std())



@dataclass
class DistributionProfile:
    """Everything the generator needs, plus the evidence it was measured."""

    bands: int
    shape: tuple[int, int]
    dtype: str
    n_scenes_sampled: int
    band_stats: list[dict[str, Any]] = field(default_factory=list)
    sea_level_db: dict[str, float] = field(default_factory=dict)
    sea_level_db_std: dict[str, float] = field(default_factory=dict)
    #: Measured spatial variation of the sea background (block medians).
    sea_texture_db_std: dict[str, float] = field(default_factory=dict)
    #: Amplitude the generator should use: the part of the pooled variance that
    #: speckle and scene-to-scene level changes do not already explain.
    sea_field_db_std: dict[str, float] = field(default_factory=dict)
    contrast_db: dict[str, float] = field(default_factory=dict)
    contrast_db_std: dict[str, float] = field(default_factory=dict)
    oil_fraction: dict[str, float] = field(default_factory=dict)
    blobs: dict[str, Any] = field(default_factory=dict)
    speckle: dict[str, Any] = field(default_factory=dict)
    #: Per band: coverage, brightness and typical component size of the pixels
    #: that Gamma speckle cannot explain (bright clutter and objects, dark
    #: shadows). See :func:`_extremes`.
    extremes: dict[str, Any] = field(default_factory=dict)
    nodata: dict[str, float] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["generated_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DistributionProfile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload.pop("generated_utc", None)
        payload["shape"] = tuple(payload["shape"])
        return cls(**payload)


def _extreme_variance(
    records: Sequence[dict[str, float]], sea_levels: Sequence[float]
) -> float:
    """Mean variance the measured extremes contribute to the pooled spread.

    ``sum_i p_i * (level_i - sea)^2`` per scene, averaged over scenes. This is
    the term that has to be removed from the sea-field budget: those pixels are
    generated separately, and the pooled standard deviation they were measured
    from already includes them.
    """
    if not records or len(records) != len(sea_levels):
        return 0.0
    per_scene: list[float] = []
    for record, sea_level in zip(records, sea_levels, strict=True):
        variance = 0.0
        for prefix in ("bright", "dark"):
            fraction = record.get(f"{prefix}_fraction", 0.0)
            if fraction <= 0.0:
                continue
            deltas = [
                record.get(f"{prefix}_db_p10", sea_level) - sea_level,
                record.get(f"{prefix}_db_p50", sea_level) - sea_level,
                record.get(f"{prefix}_db_p90", sea_level) - sea_level,
            ]
            variance += fraction * float(np.mean(np.square(deltas)))
        per_scene.append(variance)
    return float(np.mean(per_scene)) if per_scene else 0.0


def profile_scenes(
    images: Sequence[Path],
    masks: Sequence[Path],
    *,
    source: dict[str, Any] | None = None,
    band_names: Sequence[str] = ("VV", "VH"),
) -> DistributionProfile:
    """Measure a paired image/mask set.

    Reads every scene in full — this is the one place where the pixels matter,
    and a sampled profile would understate the tails, which are exactly the
    part a detector thresholds on.
    """
    if len(images) != len(masks):
        raise ValueError(f"{len(images)} images vs {len(masks)} masks")
    if not images:
        raise ValueError("nothing to profile")

    logger.info("profiling {} scene(s) — reading pixels", len(images))

    per_band_values: list[list[np.ndarray]] = [[] for _ in band_names]
    sea_levels: list[list[float]] = [[] for _ in band_names]
    contrasts: list[list[float]] = [[] for _ in band_names]
    looks_by_band: list[list[float]] = [[] for _ in band_names]
    texture_by_band: list[list[float]] = [[] for _ in band_names]
    extremes_by_band: list[list[dict[str, float]]] = [[] for _ in band_names]
    oil_fractions: list[float] = []
    areas: list[float] = []
    elongations: list[float] = []
    blobs_per_scene: list[float] = []
    nodata_fractions: list[float] = []
    shape: tuple[int, int] | None = None
    dtype = "float32"

    for index, (image_path, mask_path) in enumerate(zip(images, masks, strict=True)):
        bands = read_bands(image_path)
        mask = _read_mask(mask_path)
        if shape is None:
            shape = (int(bands.shape[1]), int(bands.shape[2]))
            dtype = str(bands.dtype)

        scene_valid: np.ndarray | None = None
        for band_index in range(min(bands.shape[0], len(band_names))):
            band = bands[band_index]
            # No-data is excluded from every statistic. Left in, a 0.0 dB block
            # (sea is near -25 dB) would be the brightest thing in the scene.
            valid = np.abs(band - NODATA_DB) > _NODATA_TOLERANCE
            if scene_valid is None:
                scene_valid = valid
                nodata_fractions.append(float(1.0 - valid.mean()))
            if not valid.any():
                continue
            sea_pixels = band[valid]
            per_band_values[band_index].append(sea_pixels)
            sea_level = float(np.median(sea_pixels))
            sea_levels[band_index].append(sea_level)
            oil_pixels = mask & valid
            if int(oil_pixels.sum()) >= _MIN_OIL_PIXELS:
                contrasts[band_index].append(sea_level - float(np.median(band[oil_pixels])))
            # Per band, not just the first one. VV and VH are not the same
            # measurement at the same resolution: measured here, VV runs at
            # ~2.8 looks and VH at ~12, so reusing VV's speckle for VH made the
            # synthetic VH roughly 1.3 dB too noisy in every scene.
            looks_by_band[band_index].append(_effective_looks(band, valid))
            texture_by_band[band_index].append(_sea_texture(band, valid, mask))
            extremes_by_band[band_index].append(_extremes(band, valid, sea_level))

        fraction = float(mask.mean())
        oil_fractions.append(fraction)
        if mask.any():
            scene_areas, scene_elongations = _blob_geometry(mask)
            areas.extend(scene_areas)
            elongations.extend(scene_elongations)
            blobs_per_scene.append(float(len(scene_areas)))
        else:
            blobs_per_scene.append(0.0)

        if (index + 1) % 8 == 0:
            logger.info("  profiled {}/{}", index + 1, len(images))

    band_stats: list[dict[str, Any]] = []
    sea_level_db: dict[str, float] = {}
    sea_level_db_std: dict[str, float] = {}
    sea_texture_db_std: dict[str, float] = {}
    sea_field_db_std: dict[str, float] = {}
    contrast_db: dict[str, float] = {}
    contrast_db_std: dict[str, float] = {}
    looks_per_band: dict[str, float] = {}
    speckle_sigma_per_band: dict[str, float] = {}
    extremes: dict[str, Any] = {}
    for band_index, name in enumerate(band_names[: len(per_band_values)]):
        pooled = np.concatenate(per_band_values[band_index])
        levels = np.asarray(sea_levels[band_index], dtype=np.float64)
        band_stats.append(
            {
                "name": name,
                "mean": round(float(pooled.mean()), 4),
                "std": round(float(pooled.std()), 4),
                "min": round(float(pooled.min()), 4),
                "max": round(float(pooled.max()), 4),
                "percentiles": _percentiles(pooled),
                "histogram": _histogram(pooled),
            }
        )
        sea_level_db[name] = round(float(levels.mean()), 4)
        # Scene-to-scene spread of the sea level. Without it the generator would
        # emit 120 scenes at one identical brightness, which is a distribution
        # mismatch the per-band histogram cannot see.
        sea_level_db_std[name] = round(float(levels.std()), 4)
        band_contrasts = np.asarray(contrasts[band_index], dtype=np.float64)
        contrast_db[name] = round(float(band_contrasts.mean()), 4) if band_contrasts.size else 0.0
        contrast_db_std[name] = (
            round(float(band_contrasts.std()), 4) if band_contrasts.size else 0.0
        )

        band_looks = (
            float(np.median(looks_by_band[band_index])) if looks_by_band[band_index] else 1.0
        )
        looks_per_band[name] = round(band_looks, 4)
        sigma = speckle_sigma_db(band_looks)
        speckle_sigma_per_band[name] = round(sigma, 4)

        texture = (
            float(np.mean(texture_by_band[band_index]))
            if texture_by_band[band_index]
            else 0.0
        )
        sea_texture_db_std[name] = round(texture, 4)

        # Calibrate the spatial field so the *observable* matches: whatever
        # pooled variance speckle, scene-to-scene level changes and the measured
        # extremes do not already explain is attributed to the sea background.
        #
        # The extremes term matters. The pooled standard deviation above is
        # computed over *all* valid pixels, so it already contains the variance
        # the bright and dark extremes contribute. The generator adds those
        # pixels separately, so leaving their variance inside the field budget
        # would count it twice — measured on this archive that inflated the
        # field by a factor of 2.4 on VV and 1.4 on VH.
        #
        # This is a deliberate modelling choice, not a measurement. The
        # block-median texture above is a lower bound (a 32x32 median also
        # averages out structure finer than one block), and the residual also
        # absorbs the non-Gaussian tails the generator does not model
        # separately. All of the numbers are recorded so the choice is
        # auditable.
        residual = (
            float(pooled.std()) ** 2
            - sigma**2
            - float(levels.std()) ** 2
            - _extreme_variance(extremes_by_band[band_index], sea_levels[band_index])
        )
        sea_field_db_std[name] = round(math.sqrt(max(residual, 0.0)), 4)

        # Extremes are aggregated as the mean over scenes, except the component
        # size which is pooled: a scene with no bright object must not drag the
        # typical object size towards zero.
        records = extremes_by_band[band_index]
        if records:
            _extreme_keys = [
                key
                for key in records[0]
                if key.endswith("_fraction")
                or "_db_p" in key
                or key.endswith("_component_px")
                or key == "sea_level_db"
            ]
            extremes[name] = {
                key: round(float(np.mean([rec[key] for rec in records])), 6)
                for key in _extreme_keys
                if key != "sea_level_db"
            }
            # The per-scene records are kept, and they are what the generator
            # samples from. A mean curve carries no between-scene variation, and
            # the between-scene variation of extreme brightness is most of what
            # decides the pooled tail: VV's brightest 0.1% of pixels comes from
            # whichever scenes had bright objects, not from the average scene.
            # Collapsing to the mean is why a generated VV p99.9 came out 3.4 dB
            # too dim while its own bright-extreme p90 matched.
            extremes[name]["per_scene"] = [
                {key: float(rec[key]) for key in _extreme_keys} for rec in records
            ]
            for prefix in ("bright", "dark"):
                sizes = [
                    rec[f"{prefix}_component_px"]
                    for rec in records
                    if rec[f"{prefix}_component_px"] > 0.0
                ]
                extremes[name][f"{prefix}_component_px"] = (
                    round(float(np.median(sizes)), 3) if sizes else 0.0
                )
            extremes[name]["threshold_db"] = EXTREME_THRESHOLD_DB
            extremes[name]["scene_rate"] = round(
                float(
                    np.mean(
                        [
                            float(rec["bright_fraction"] > 0.0 or rec["dark_fraction"] > 0.0)
                            for rec in records
                        ]
                    )
                ),
                6,
            )


    fractions = np.asarray(oil_fractions, dtype=np.float64)
    return DistributionProfile(
        bands=len(band_stats),
        shape=shape or (0, 0),
        dtype=dtype,
        n_scenes_sampled=len(images),
        band_stats=band_stats,
        sea_level_db=sea_level_db,
        sea_level_db_std=sea_level_db_std,
        sea_texture_db_std=sea_texture_db_std,
        sea_field_db_std=sea_field_db_std,
        contrast_db=contrast_db,
        contrast_db_std=contrast_db_std,
        oil_fraction={
            "mean": round(float(fractions.mean()), 6),
            "std": round(float(fractions.std()), 6),
            "p10": round(float(np.percentile(fractions, 10)), 6),
            "p50": round(float(np.percentile(fractions, 50)), 6),
            "p90": round(float(np.percentile(fractions, 90)), 6),
            "max": round(float(fractions.max()), 6),
            "zero_rate": round(float((fractions <= 0.0).mean()), 6),
        },
        blobs={
            "per_scene_mean": round(float(np.mean(blobs_per_scene)), 4) if blobs_per_scene else 0.0,
            "n_blobs": len(areas),
            "area_px": _percentiles(np.asarray(areas)) if areas else {},
            "area_px_max": round(float(max(areas)), 2) if areas else 0.0,
            "elongation": _percentiles(np.asarray(elongations)) if elongations else {},
        },
        speckle={
            "equivalent_looks": looks_per_band,
            "equivalent_looks_median": round(
                float(np.median(list(looks_per_band.values()))), 4
            )
            if looks_per_band
            else 1.0,
            "speckle_sigma_db": speckle_sigma_per_band,
        },
        extremes=extremes,
        nodata={
            "value": NODATA_DB,
            "fraction_mean": round(float(np.mean(nodata_fractions)), 6)
            if nodata_fractions
            else 0.0,
            "fraction_max": round(float(np.max(nodata_fractions)), 6) if nodata_fractions else 0.0,
            "scene_rate": round(
                float(np.mean([value > 0.0 for value in nodata_fractions])), 6
            )
            if nodata_fractions
            else 0.0,
        },
        source=dict(source or {}),
    )


def profile_from_spec(
    spec: Any,
    *,
    sample: int | None = 32,
    seed: int = 0,
    rebuild_index: bool = False,
) -> DistributionProfile:
    """Profile a resolved :class:`DataSourceSpec`, sampling ``sample`` pairs.

    Sampling is deterministic (sorted names, fixed stride) so two runs profile
    the same scenes and a reported difference is a real difference.
    """
    from sentinel_core.datasource import MANIFEST_NAME

    manifest = Path(spec.prepared_root) / MANIFEST_NAME
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"{manifest} is empty; resolve the source first")

    rows.sort(key=lambda row: str(row["image"]))
    if sample is not None and sample < len(rows):
        stride = len(rows) / sample
        rows = [rows[int(i * stride)] for i in range(sample)]

    logger.info(
        "sampling {} of {} scene(s) from {} ({})",
        len(rows),
        len(rows) if sample is None else sample,
        spec.kind,
        spec.provenance.value,
    )
    return profile_scenes(
        [Path(row["image"]) for row in rows],
        [Path(row["mask"]) for row in rows],
        source={
            "kind": spec.kind,
            "provenance": spec.provenance.value,
            "n_pairs_total": spec.n_pairs,
            "n_sampled": len(rows),
            "seed": seed,
            "images_dir": str(spec.images_dir),
        },
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _histogram_total_variation(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Total variation distance between two probability vectors, in [0, 1]."""
    pa = np.asarray(a["probabilities"], dtype=np.float64)
    pb = np.asarray(b["probabilities"], dtype=np.float64)
    return float(0.5 * np.abs(pa - pb).sum())


def compare_profiles(
    real: DistributionProfile,
    synthetic: DistributionProfile,
    *,
    percentile_tolerance_db: float = 2.0,
    tv_tolerance: float = 0.15,
    std_tolerance_db: float = 1.0,
) -> dict[str, Any]:
    """Measure how closely synthetic statistics reproduce the real ones.

    Returns a structured report rather than a boolean: the interesting output
    is *which* statistic drifted, not that something did.
    """
    checks: list[dict[str, Any]] = []

    for real_band, synth_band in zip(real.band_stats, synthetic.band_stats, strict=False):
        name = str(real_band["name"])
        for key, real_value in real_band["percentiles"].items():
            synth_value = float(synth_band["percentiles"].get(key, float("nan")))
            delta = abs(synth_value - float(real_value))
            checks.append(
                {
                    "quantity": f"{name}.{key}",
                    "real": round(float(real_value), 4),
                    "synthetic": round(synth_value, 4),
                    "delta": round(delta, 4),
                    "tolerance": percentile_tolerance_db,
                    "unit": "dB",
                    "ok": bool(delta <= percentile_tolerance_db),
                }
            )
        # The per-band standard deviation, checked explicitly. It is the
        # quantity the generator's sea-field amplitude is calibrated from, and
        # percentiles alone can pass while the spread is wrong.
        real_std = float(real_band.get("std", 0.0))
        synth_std = float(synth_band.get("std", 0.0))
        std_delta = abs(synth_std - real_std)
        checks.append(
            {
                "quantity": f"{name}.std",
                "real": round(real_std, 4),
                "synthetic": round(synth_std, 4),
                "delta": round(std_delta, 4),
                "tolerance": std_tolerance_db,
                "unit": "dB",
                "ok": bool(std_delta <= std_tolerance_db),
            }
        )
        tv = _histogram_total_variation(real_band["histogram"], synth_band["histogram"])
        checks.append(
            {
                "quantity": f"{name}.histogram_tv_distance",
                "real": 0.0,
                "synthetic": round(tv, 4),
                "delta": round(tv, 4),
                "tolerance": tv_tolerance,
                "unit": "TV",
                "ok": bool(tv <= tv_tolerance),
            }
        )

    for name in real.sea_level_db:
        delta = abs(synthetic.sea_level_db.get(name, 0.0) - real.sea_level_db[name])
        checks.append(
            {
                "quantity": f"{name}.sea_level_db",
                "real": real.sea_level_db[name],
                "synthetic": synthetic.sea_level_db.get(name, 0.0),
                "delta": round(delta, 4),
                "tolerance": percentile_tolerance_db,
                "unit": "dB",
                "ok": bool(delta <= percentile_tolerance_db),
            }
        )

    for name in real.contrast_db:
        delta = abs(synthetic.contrast_db.get(name, 0.0) - real.contrast_db[name])
        checks.append(
            {
                "quantity": f"{name}.oil_contrast_db",
                "real": real.contrast_db[name],
                "synthetic": synthetic.contrast_db.get(name, 0.0),
                "delta": round(delta, 4),
                "tolerance": percentile_tolerance_db,
                "unit": "dB",
                "ok": bool(delta <= percentile_tolerance_db),
            }
        )

    oil_delta = abs(synthetic.oil_fraction["mean"] - real.oil_fraction["mean"])
    checks.append(
        {
            "quantity": "oil_fraction.mean",
            "real": real.oil_fraction["mean"],
            "synthetic": synthetic.oil_fraction["mean"],
            "delta": round(oil_delta, 6),
            "tolerance": 0.02,
            "unit": "fraction",
            "ok": bool(oil_delta <= 0.02),
        }
    )
    zero_delta = abs(synthetic.oil_fraction["zero_rate"] - real.oil_fraction["zero_rate"])
    checks.append(
        {
            "quantity": "oil_fraction.zero_rate",
            "real": real.oil_fraction["zero_rate"],
            "synthetic": synthetic.oil_fraction["zero_rate"],
            "delta": round(zero_delta, 6),
            "tolerance": 0.20,
            "unit": "fraction",
            "ok": bool(zero_delta <= 0.20),
        }
    )

    passed = sum(1 for check in checks if check["ok"])
    return {
        "n_checks": len(checks),
        "n_passed": passed,
        "n_failed": len(checks) - passed,
        "match_fraction": round(passed / len(checks), 4) if checks else 0.0,
        "all_within_tolerance": passed == len(checks),
        "checks": checks,
        "failed": [check for check in checks if not check["ok"]],
    }
