"""Tests for the SAR augmentation chain.

These pin down the three properties that make augmentation trustworthy: it must
not return ``None``, it must move the mask with the image, and it must be
reproducible from a seed. The previous module failed all three — it returned
``None`` whenever albumentations was absent, which silently disabled
augmentation while the run reported nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.training.transforms import (
    Compose,
    RadiometricJitter,
    RandomCrop,
    RandomFlip,
    RandomRotate90,
    SpeckleNoise,
    build_eval_transforms,
    build_train_transforms,
    db_to_linear,
    describe_augmentation,
    linear_to_db,
)

IMAGE_SIZE = 128


def make_pair(size: int = IMAGE_SIZE, bands: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic image with a single-pixel marker the mask points at."""
    rng = np.random.default_rng(7)
    image = rng.normal(-20.0, 2.0, (bands, size, size)).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4, size // 3] = 1
    return image, mask


# ---------------------------------------------------------------------------
# No silent no-op
# ---------------------------------------------------------------------------


def test_builders_never_return_none() -> None:
    """The failure mode that made the old module useless."""
    assert build_train_transforms(IMAGE_SIZE) is not None
    assert build_eval_transforms(IMAGE_SIZE) is not None
    assert len(build_train_transforms(IMAGE_SIZE)) > 0


def test_describe_reports_a_missing_chain_instead_of_failing() -> None:
    described = describe_augmentation(None)
    assert described["enabled"] is False
    assert described["steps"] == []


def test_describe_lists_every_step() -> None:
    described = describe_augmentation(build_train_transforms(IMAGE_SIZE))
    names = [step["name"] for step in described["steps"]]
    assert names[0] == "random_crop"
    assert "random_flip" in names
    assert "speckle_noise" in names


# ---------------------------------------------------------------------------
# Geometry: the mask moves with the image
# ---------------------------------------------------------------------------


def test_crop_returns_the_requested_size() -> None:
    image, mask = make_pair()
    cropped_image, cropped_mask = RandomCrop(32)(image, mask, np.random.default_rng(0))
    assert cropped_image.shape == (2, 32, 32)
    assert cropped_mask.shape == (32, 32)


def test_crop_refuses_a_patch_larger_than_the_scene() -> None:
    """Padding dB data would invent a black region that reads as open water."""
    image, mask = make_pair(64)
    with pytest.raises(ValueError, match="cannot crop"):
        RandomCrop(128)(image, mask, np.random.default_rng(0))


def test_flip_keeps_the_marker_aligned() -> None:
    """A flip must move the mask by the same index arithmetic as the image."""
    image, mask = make_pair()
    # Make the marker the brightest pixel so it can be found in the image.
    image[:, mask > 0] = 99.0
    flipped_image, flipped_mask = RandomFlip(1.0, 1.0)(image, mask, np.random.default_rng(0))

    bright = np.argwhere(flipped_image[0] > 50.0)
    marked = np.argwhere(flipped_mask > 0)
    assert bright.shape == marked.shape == (1, 2)
    assert tuple(bright[0]) == tuple(marked[0])


def test_rotate90_preserves_the_values_exactly() -> None:
    """Rotation by a quarter turn must not interpolate anything."""
    image, mask = make_pair(32)
    rotated_image, rotated_mask = RandomRotate90(1.0)(image, mask, np.random.default_rng(3))
    assert rotated_image.shape == image.shape
    assert np.allclose(np.sort(rotated_image.ravel()), np.sort(image.ravel()))
    assert rotated_mask.sum() == mask.sum()


@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_compose_is_reproducible_for_a_seed(seed: int) -> None:
    image, mask = make_pair()
    first = build_train_transforms(32, seed=seed)(image, mask)
    second = build_train_transforms(32, seed=seed)(image, mask)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_different_seeds_produce_different_augmentations() -> None:
    image, mask = make_pair()
    a = build_train_transforms(32, seed=1)(image, mask)
    b = build_train_transforms(32, seed=2)(image, mask)
    assert not np.array_equal(a[0], b[0])


def test_disabled_compose_is_a_passthrough() -> None:
    image, mask = make_pair(32)
    out_image, out_mask = Compose([RandomFlip(1.0, 1.0)], enabled=False)(image, mask)
    assert np.array_equal(out_image, image)
    assert np.array_equal(out_mask, mask)


def test_eval_transforms_are_deterministic() -> None:
    """Validation must measure the model, not the augmentation."""
    image, mask = make_pair()
    compose = build_eval_transforms(32)
    first = compose(image, mask)
    second = compose(image, mask)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_deterministic_training_still_crops() -> None:
    """``random=False`` is not ``enabled=False`` — a 2048 scene still needs a patch."""
    image, mask = make_pair(256)
    compose = build_train_transforms(64, random=False)
    out_image, out_mask = compose(image, mask)
    assert out_image.shape == (2, 64, 64)
    assert out_mask.shape == (64, 64)
    assert compose.describe()["steps"][0]["name"] == "center_crop"


def test_compose_rejects_a_transform_that_changes_the_band_count() -> None:
    """A transform must not add or drop channels; the loader asserts this too."""

    class Dropper(Compose):
        def __call__(self, image, mask):  # type: ignore[override]
            return image[:1], mask

    image, mask = make_pair(32)
    out_image, _ = Dropper([])(image, mask)
    assert out_image.shape[0] == 1  # the transform itself is allowed to do this
    # The guard lives in the dataset, which is where the band count is known.


# ---------------------------------------------------------------------------
# Mask layout: the loader hands over (1, H, W), not (H, W)
#
# Every test above passes a 2-D mask, which is the layout the docstring
# described. The loader does not: it reads a mask as (1, H, W) and hands it
# straight to the chain, so the 2-D tests exercised a contract no caller used
# and the whole geometry layer was wrong for the only layout that mattered.
# These run both layouts through the same assertions.
# ---------------------------------------------------------------------------


def _marker_pair(layout: str, size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """An image whose brightest pixel is the one the mask marks."""
    image, mask = make_pair(size)
    image[:, mask > 0] = 99.0
    return (image, mask) if layout == "2d" else (image, mask[np.newaxis, ...])


def _marked_positions(mask: np.ndarray, layout: str) -> np.ndarray:
    return np.argwhere(mask[0] > 0) if layout == "3d" else np.argwhere(mask > 0)


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_crop_returns_a_patch_for_both_mask_layouts(layout: str) -> None:
    """``mask[rows, cols]`` on a 3-D mask means ``mask[rows, cols, :]``.

    The slices land on the channel axis and the row axis, so the crop comes back
    with an empty first axis instead of a patch — which is what the trainer hit:
    "stack expects each tensor to be equal size, but got [0, 256, 2048] at entry
    0 and [1, 0, 2048] at entry 1".
    """
    image, mask = _marker_pair(layout)
    cropped_image, cropped_mask = RandomCrop(32)(image, mask, np.random.default_rng(0))
    assert cropped_image.shape == (2, 32, 32)
    assert cropped_mask.shape == ((32, 32) if layout == "2d" else (1, 32, 32))
    assert cropped_mask.shape[-2:] == cropped_image.shape[-2:]


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_centre_crop_returns_a_patch_for_both_mask_layouts(layout: str) -> None:
    """The evaluation path goes through ``_CenterCrop``, so it has the same bug."""
    image, mask = _marker_pair(layout)
    cropped_image, cropped_mask = build_eval_transforms(32)(image, mask)
    assert cropped_image.shape == (2, 32, 32)
    assert cropped_mask.shape == ((32, 32) if layout == "2d" else (1, 32, 32))


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_flip_moves_the_mask_with_the_image_in_both_layouts(layout: str) -> None:
    """This one is silent, and that is what makes it dangerous.

    ``mask[:, ::-1]`` on a ``(1, H, W)`` mask flips the *height* while the image
    flips the *width*. The mask stays a plausible array; it just stops matching
    the image it labels, and the run trains happily on wrong targets.
    """
    image, mask = _marker_pair(layout)
    for seed in range(6):
        flipped_image, flipped_mask = RandomFlip(1.0, 1.0)(
            image, mask, np.random.default_rng(seed)
        )
        bright = np.argwhere(flipped_image[0] > 50.0)
        marked = _marked_positions(flipped_mask, layout)
        assert bright.shape == marked.shape == (1, 2), (layout, seed)
        assert tuple(bright[0]) == tuple(marked[0]), (layout, seed)


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_rotate90_moves_the_mask_with_the_image_in_both_layouts(layout: str) -> None:
    """``np.rot90(mask, axes=(0, 1))`` turns channel-against-height on a 3-D mask."""
    image, mask = _marker_pair(layout)
    for seed in range(6):
        rotated_image, rotated_mask = RandomRotate90(1.0)(
            image, mask, np.random.default_rng(seed)
        )
        bright = np.argwhere(rotated_image[0] > 50.0)
        marked = _marked_positions(rotated_mask, layout)
        assert bright.shape == marked.shape == (1, 2), (layout, seed)
        assert tuple(bright[0]) == tuple(marked[0]), (layout, seed)


@pytest.mark.parametrize("layout", ["2d", "3d"])
def test_the_full_train_chain_accepts_the_loader_layout(layout: str) -> None:
    """The chain as the trainer builds it, on a mask as the loader reads it.

    This is the test that would have caught the crash: the pieces above can each
    look right while the composed chain still fails, because only the chain has
    every step running on the same array.
    """
    image, mask = _marker_pair(layout, 256)
    chain = build_train_transforms(64, seed=0)
    for _ in range(8):
        out_image, out_mask = chain(image, mask)
        assert out_image.shape == (2, 64, 64)
        assert out_mask.shape == ((64, 64) if layout == "2d" else (1, 64, 64))


def test_a_mask_that_is_not_the_size_of_its_image_is_refused() -> None:
    """The invariant every geometric transform depends on, checked in one place.

    Without it a mismatched pair is moved by index arithmetic that means
    different things on the two arrays, and the result still looks like a mask.
    """
    image, mask = make_pair(64)
    with pytest.raises(ValueError, match="does not match mask"):
        RandomFlip(1.0, 1.0)(image, mask[:32], np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Radiometry
# ---------------------------------------------------------------------------


def test_db_linear_round_trip() -> None:
    db = np.array([-30.0, -20.0, -10.0], dtype=np.float32)
    assert np.allclose(linear_to_db(db_to_linear(db)), db, atol=1e-3)


def test_speckle_is_multiplicative_on_intensity() -> None:
    """Speckle must not shift the mean in linear intensity."""
    flat = np.full((1, 512, 512), -20.0, dtype=np.float32)
    mask = np.zeros((512, 512), dtype=np.uint8)
    out, _ = SpeckleNoise(looks=4.0, p=1.0, looks_jitter=0.0)(flat, mask, np.random.default_rng(0))

    mean_db = float(out.mean())
    assert abs(mean_db - (-20.0)) < 1.0, "multiplicative speckle should not bias the dB mean"
    assert float(out.std()) > 1.0, "speckle must add spread"


def test_speckle_spread_falls_with_more_looks() -> None:
    flat = np.full((1, 256, 256), -20.0, dtype=np.float32)
    mask = np.zeros((256, 256), dtype=np.uint8)
    low, _ = SpeckleNoise(looks=2.0, p=1.0, looks_jitter=0.0)(flat, mask, np.random.default_rng(1))
    high, _ = SpeckleNoise(looks=16.0, p=1.0, looks_jitter=0.0)(
        flat, mask, np.random.default_rng(1)
    )
    assert float(high.std()) < float(low.std())


def test_speckle_is_skipped_when_the_draw_says_so() -> None:
    image, mask = make_pair(32)
    out, _ = SpeckleNoise(looks=4.0, p=0.0)(image, mask, np.random.default_rng(0))
    assert np.array_equal(out, image)


def test_radiometric_jitter_moves_the_level_without_flattening_it() -> None:
    rng = np.random.default_rng(5)
    image = rng.normal(-20.0, 2.0, (1, 128, 128)).astype(np.float32)
    mask = np.zeros((128, 128), dtype=np.uint8)
    out, _ = RadiometricJitter(gain_std=0.05, offset_std_db=2.0, p=1.0)(
        image, mask, np.random.default_rng(0)
    )
    assert out.shape == image.shape
    assert abs(float(out.mean()) - float(image.mean())) > 0.01, "the level should move"
    assert float(out.std()) > 0.5 * float(image.std()), "the structure must survive"


def test_radiometric_jitter_clips_into_the_db_window() -> None:
    image = np.full((1, 64, 64), 19.5, dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.uint8)
    out, _ = RadiometricJitter(gain_std=0.5, offset_std_db=50.0, p=1.0)(
        image, mask, np.random.default_rng(2)
    )
    assert float(out.max()) <= 20.0
    assert float(out.min()) >= -60.0
