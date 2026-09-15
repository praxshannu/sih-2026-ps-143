"""SAR augmentations that operate on what the data actually is.

Why this module exists
----------------------
The previous augmentation module wrapped albumentations behind a
``try: import albumentations / except ImportError: return None``. albumentations
is not installed here, so every caller silently received ``None`` and the
``if transform is not None`` guard in the dataset skipped augmentation entirely.
Training "with augmentation" was training without it, and nothing said so. The
same module then normalized 2-band dB SAR with ImageNet RGB statistics
(``mean=[0.485, 0.456, 0.406]``), which is a three-channel prior applied to a
two-channel physical quantity.

This module has no optional dependency, never returns ``None``, and works in the
units the archive is stored in.

What the transforms assume
--------------------------
* Images are ``(C, H, W)`` ``float32`` **in dB** — the native form of both the
  real archive and the synthetic generator. No conversion happens on the way in.
* Masks are ``(H, W)`` or ``(1, H, W)`` and binary; the spatial axes are the
  last two in either case. A mask is a label, so it is *never* interpolated:
  every geometric transform moves it with the same integer index arithmetic it
  applies to the image, and ``Transform.__call__`` refuses a pair whose spatial
  extents differ.
* Geometry and radiometry are separate concerns. Geometric transforms are exact
  and lossless; radiometric ones change pixel values and are applied in the
  order a physical scene would produce them (speckle, then contrast, then a
  scene-level brightness offset).

The physics, briefly
--------------------
Speckle in SAR is **multiplicative** on intensity and follows a Gamma
distribution with ``L`` looks. Adding Gaussian noise to dB values, which is
what the old module did, produces symmetric noise that no real sensor makes.
:class:`SpeckleNoise` therefore goes dB -> linear -> multiply by
``Gamma(L, 1/L)`` -> dB, which is symmetric in the *ratio* and skewed in dB,
as the real thing is.

Determinism
-----------
Every transform draws from one seeded generator. Under
``DataLoader(num_workers>0)`` each worker inherits the parent's generator state
through ``fork``, so workers would otherwise emit identical augmentations.
:meth:`Compose.__call__` reseeds per worker on first use, which keeps a run
reproducible *and* makes the workers differ.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

__all__ = [
    "Compose",
    "RandomCrop",
    "RandomFlip",
    "RandomRotate90",
    "RadiometricJitter",
    "SpeckleNoise",
    "Transform",
    "build_eval_transforms",
    "build_train_transforms",
    "db_to_linear",
    "linear_to_db",
]

#: dB window shared with the profile/generator so a clipped value means the same
#: thing everywhere.
DB_MIN = -60.0
DB_MAX = 20.0

_EPS = 1e-6


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """dB -> linear intensity. The inverse of :func:`linear_to_db`."""
    return np.power(10.0, np.asarray(db, dtype=np.float32) / 10.0)


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    """Linear intensity -> dB, with a floor so ``log10(0)`` cannot happen."""
    return (10.0 * np.log10(np.maximum(np.asarray(linear, dtype=np.float32), _EPS))).astype(
        np.float32
    )


class Transform:
    """Base class for a joint image/mask transform.

    Subclasses implement :meth:`apply`, which receives ``(image, mask, rng)`` and
    returns the pair. Splitting the RNG out means a transform cannot quietly
    create its own stream and become unreproducible.
    """

    #: Human-readable name, used in the run report.
    name: str = "transform"

    def apply(
        self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __call__(
        self, image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if image.shape[-2:] != mask.shape[-2:]:
            # A geometric transform moves the mask by the *same index
            # arithmetic* as the image, so the two have to be the same size.
            # Checking here rather than in each subclass means the invariant
            # cannot be forgotten by a transform added later.
            raise ValueError(
                f"{self.name}: image spatial extent {tuple(image.shape[-2:])} does not "
                f"match mask {tuple(mask.shape[-2:])}; a mask that is not the size of "
                "its image cannot be moved with it."
            )
        return self.apply(image, mask, rng)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": _public_fields(self)}


def _public_fields(obj: Any) -> dict[str, Any]:
    if not hasattr(obj, "__dataclass_fields__"):
        return {}
    return {
        key: value
        for key, value in vars(obj).items()
        if not key.startswith("_") and isinstance(value, (int, float, bool, str, tuple, list))
    }


# ---------------------------------------------------------------------------
# Geometric — exact, mask moved with identical index arithmetic
# ---------------------------------------------------------------------------


@dataclass
class RandomFlip(Transform):
    """Independent horizontal and vertical flips.

    Flips are the safest augmentation available for SAR: the scene is a top-down
    view of a surface, so mirroring changes nothing physical. Rotations by
    arbitrary angles do not have that property — they resample, and a resampled
    dB field is not the same measurement.

    The mask is indexed with ``...`` rather than ``:``. It reaches here either as
    ``(H, W)`` or as ``(1, H, W)`` — the loader keeps a channel axis because the
    model wants one — and ``mask[:, ::-1]`` on the 3-D layout flips the *height*
    while the image flips the *width*. The mask would then be silently
    desynchronised from the image it labels, which trains on wrong targets
    rather than failing.
    """

    p_horizontal: float = 0.5
    p_vertical: float = 0.5
    name: str = field(default="random_flip", init=False)

    def apply(self, image, mask, rng):
        if rng.random() < self.p_horizontal:
            image = image[:, :, ::-1]
            mask = mask[..., ::-1]
        if rng.random() < self.p_vertical:
            image = image[:, ::-1, :]
            mask = mask[..., ::-1, :]
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


@dataclass
class RandomRotate90(Transform):
    """Rotate by k * 90 degrees. Exact — no interpolation is involved.

    ``axes=(-2, -1)`` for the mask, for the same reason as :class:`RandomFlip`:
    the spatial axes are the last two whatever the mask's leading layout is.
    """

    p: float = 0.5
    name: str = field(default="random_rotate90", init=False)

    def apply(self, image, mask, rng):
        if rng.random() >= self.p:
            return image, mask
        k = int(rng.integers(1, 4))
        return np.ascontiguousarray(np.rot90(image, k, axes=(1, 2))), np.ascontiguousarray(
            np.rot90(mask, k, axes=(-2, -1))
        )


@dataclass
class RandomCrop(Transform):
    """Crop to ``size`` x ``size`` at a uniformly random offset.

    A scene is far larger than a training patch (2048 vs 512), so cropping is
    what actually feeds the network — it is not an optional extra. It raises
    rather than padding when the scene is smaller than the patch: silently
    zero-padding dB data would invent a black region that reads as open water.

    ``mask[..., rows, cols]``, not ``mask[rows, cols]``: on a ``(1, H, W)`` mask
    the two-argument form means ``mask[rows, cols, :]`` and slices the channel
    axis against the rows, which yields an empty first axis rather than a crop.
    """

    size: int
    name: str = field(default="random_crop", init=False)

    def apply(self, image, mask, rng):
        _, height, width = image.shape
        if height < self.size or width < self.size:
            raise ValueError(
                f"cannot crop {self.size}x{self.size} from a {height}x{width} scene; "
                "raise --image-size or prepare larger scenes"
            )
        top = int(rng.integers(0, height - self.size + 1))
        left = int(rng.integers(0, width - self.size + 1))
        rows = slice(top, top + self.size)
        cols = slice(left, left + self.size)
        return (
            np.ascontiguousarray(image[:, rows, cols]),
            np.ascontiguousarray(mask[..., rows, cols]),
        )


# ---------------------------------------------------------------------------
# Radiometric — these change pixel values
# ---------------------------------------------------------------------------


@dataclass
class SpeckleNoise(Transform):
    """Multiplicative Gamma speckle with the dataset's measured number of looks.

    ``L`` is the equivalent number of looks from the profile (real archive:
    measured, not assumed). Resampling it over a range teaches the detector that
    the same slick can be noisier or cleaner than the scene it was labelled in,
    which is the main source of domain shift between two acquisitions.
    """

    looks: float = 4.0
    p: float = 0.5
    #: Multiplicative spread applied to ``looks``. 1.0 disables the spread and
    #: always uses the measured value.
    looks_jitter: float = 0.5
    name: str = field(default="speckle_noise", init=False)

    def apply(self, image, mask, rng):
        if rng.random() >= self.p:
            return image, mask
        base = max(float(self.looks), 1.0)
        if self.looks_jitter > 0:
            base *= math.exp(float(rng.normal(0.0, self.looks_jitter)))
        looks = float(np.clip(base, 1.0, 64.0))

        linear = db_to_linear(image)
        gain = rng.gamma(shape=looks, scale=1.0 / looks, size=image.shape).astype(np.float32)
        return np.clip(linear_to_db(linear * gain), DB_MIN, DB_MAX), mask


@dataclass
class RadiometricJitter(Transform):
    """Per-scene dB gain and offset.

    Two different acquisitions of the same water are not at the same absolute
    level: the sea reference moves with incidence angle, wind and calibration.
    Modelling that as a scene-wide affine change in dB is crude but it is the
    right *shape* of nuisance — a per-pixel brightness wobble (what the old
    module did) is not, because it leaves the sea level where it was.

    The gain is drawn multiplicatively about 1.0 so it cannot go negative, and
    the offset is in dB, which is the unit the archive is in.
    """

    gain_std: float = 0.03
    offset_std_db: float = 0.5
    p: float = 0.5
    name: str = field(default="radiometric_jitter", init=False)

    def apply(self, image, mask, rng):
        if rng.random() >= self.p:
            return image, mask
        gain = float(math.exp(rng.normal(0.0, self.gain_std))) if self.gain_std > 0 else 1.0
        offset = float(rng.normal(0.0, self.offset_std_db)) if self.offset_std_db > 0 else 0.0
        return np.clip(image * gain + offset, DB_MIN, DB_MAX), mask


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class Compose:
    """Apply transforms in order with one seeded generator.

    The call signature is ``(image, mask) -> (image, mask)`` on ``(C, H, W)``
    float32 and a mask that is either ``(H, W)`` or ``(1, H, W)`` binary. That is
    a deliberate change from the albumentations ``(H, W, C)`` dict contract the
    old module used: the dataset reads CHW, the model wants CHW, and transposing
    twice per sample to satisfy a library that is not installed was pure
    overhead.

    The mask's *spatial axes are always the last two*, and every geometric
    transform indexes them as such. The two mask layouts are not a convenience:
    the loader reads the mask as ``(1, H, W)`` and hands it straight here, so a
    transform that assumes ``(H, W)`` is wrong for the only caller that exists.
    """

    def __init__(
        self,
        transforms: Sequence[Transform] = (),
        *,
        seed: int = 0,
        enabled: bool = True,
    ) -> None:
        self.transforms = list(transforms)
        self.seed = int(seed)
        self.enabled = bool(enabled)
        self._rng = np.random.default_rng(self.seed)
        self._worker_signature: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.transforms)

    def _reseed_for_worker(self) -> None:
        """Give each DataLoader worker its own stream.

        Workers are forked, so without this every worker would start from the
        parent's state and emit the same augmentations for different samples.
        """
        try:
            from torch.utils.data import get_worker_info
        except ImportError:  # torch is optional for pure-numpy use
            return
        info = get_worker_info()
        signature = (0, 0) if info is None else (info.id, info.num_workers)
        if signature != self._worker_signature:
            self._worker_signature = signature
            self._rng = np.random.default_rng(self.seed + 1000 * signature[0])

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask)
        if not self.enabled:
            return image, mask
        self._reseed_for_worker()
        for transform in self.transforms:
            image, mask = transform(image, mask, self._rng)
        return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask)

    def describe(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "enabled": self.enabled,
            "steps": [transform.describe() for transform in self.transforms],
        }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_train_transforms(
    image_size: int = 512,
    *,
    seed: int = 0,
    equivalent_looks: float = 4.0,
    enabled: bool = True,
    flip_p: float = 0.5,
    random: bool = True,
) -> Compose:
    """The training pipeline: crop, then geometrically flip, then perturb.

    Order is deliberate. Cropping first keeps every later operation on the
    patch that is actually trained on, and radiometric jitter comes last so the
    speckle it is layered on is the speckle the model sees.

    ``random=False`` keeps the crop but makes it deterministic and drops every
    stochastic step. That is what ``--deterministic`` means, and it is *not* the
    same as ``enabled=False``: a 2048x2048 scene still has to be cropped to
    ``image_size`` or the encoder sees an input it was never shaped for.
    """
    if not random:
        return Compose([_CenterCrop(image_size)], seed=seed, enabled=True)
    return Compose(
        [
            RandomCrop(image_size),
            RandomFlip(p_horizontal=flip_p, p_vertical=flip_p * 0.5),
            RandomRotate90(p=flip_p * 0.5),
            SpeckleNoise(looks=equivalent_looks, p=0.5),
            RadiometricJitter(p=0.5),
        ],
        seed=seed,
        enabled=enabled,
    )


def build_eval_transforms(
    image_size: int = 512,
    *,
    seed: int = 0,
    center: bool = True,
) -> Compose:
    """The evaluation pipeline: one deterministic crop and nothing else.

    Validation and test must measure the model, not the augmentation. A random
    crop here would make the val metric depend on the seed, which is exactly how
    a "best epoch" gets picked by luck.
    """
    return Compose([_CenterCrop(image_size) if center else _Identity()], seed=seed, enabled=True)


@dataclass
class _CenterCrop(Transform):
    """Deterministic centre crop — the evaluation counterpart of RandomCrop."""

    size: int
    name: str = field(default="center_crop", init=False)

    def apply(self, image, mask, rng):
        _, height, width = image.shape
        if height < self.size or width < self.size:
            raise ValueError(
                f"cannot centre-crop {self.size}x{self.size} from a {height}x{width} scene"
            )
        top = (height - self.size) // 2
        left = (width - self.size) // 2
        rows = slice(top, top + self.size)
        cols = slice(left, left + self.size)
        return (
            np.ascontiguousarray(image[:, rows, cols]),
            np.ascontiguousarray(mask[..., rows, cols]),
        )


@dataclass
class _Identity(Transform):
    name: str = field(default="identity", init=False)

    def apply(self, image, mask, rng):
        return image, mask


def describe_augmentation(compose: Compose | None) -> dict[str, Any]:
    """Report form, so a run records what augmentation it actually used."""
    if compose is None:
        logger.warning("no augmentation configured; the run will report augmentation: none")
        return {"enabled": False, "steps": [], "note": "no transform configured"}
    return compose.describe()
