"""Dataset loader for prepared Zenodo SAR oil-spill data.

The contract this module enforces — every one of these was a silent failure
mode in the previous version:

* **2-band SAR input.** A Sentinel-1 scene is VV + VH. Nothing is padded to a
  fictional 6-channel input, and nothing is truncated to 3: a band count other
  than the expected one is an error naming the file, not a silent reshape.
* **Auxiliary channels are opt-in.** Wind and incidence angle are *not* in the
  Zenodo image archive. When enabled they are appended explicitly and recorded;
  when disabled (default) an image with extra bands is rejected rather than
  quietly cropped.
* **Deterministic pairing.** Images and masks are sorted and asserted equal in
  length; the resulting order is recorded on the dataset and in the run config,
  so a metric can always be traced back to the exact tensors that produced it.
* **Empty or missing data is an error.** Never a synthetic fallback.

Expected layouts::

    data/zenodo_prepared/{train,val,test}/{images,masks}/*.tif   (preferred)
    data/zenodo_prepared/manifest.jsonl                          (provenance)
    data/dartis/{images,masks}/*.tif                             (legacy)

Usage:
    from dataset import SAROilSpillDataset
    ds = SAROilSpillDataset("data/zenodo_prepared/train", expected_bands=2)
"""

from __future__ import annotations

import json
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

_TRAINING_DIR = Path(__file__).resolve().parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from metrics import (  # noqa: E402
    NormalizationStats,
    SceneStatistics,
    compute_scene_statistics,
)
from splits import default_scene_id  # noqa: E402

IMAGE_SUFFIXES: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy")
SAR_BAND_NAMES: tuple[str, ...] = ("VV", "VH")
#: Auxiliary channels are physically meaningful but absent from the archive.
AUX_CHANNEL_NAMES: tuple[str, ...] = ("wind_speed", "wind_dir", "incidence_angle")

#: A joint image/mask transform: ``(C, H, W) float32`` + ``(H, W)`` binary in,
#: the same shapes out. See :mod:`transforms` for why this is not the
#: albumentations ``(H, W, C)`` dict contract.
JointTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class DatasetUnavailableError(FileNotFoundError):
    """Raised when a training/validation directory is missing or empty."""


# ---------------------------------------------------------------------------
# Raster IO
# ---------------------------------------------------------------------------


def read_raster(path: Path) -> np.ndarray:
    """Read a raster as ``(C, H, W)`` float32. No CRS is required or assumed."""
    if path.suffix == ".npy":
        array = np.load(path).astype(np.float32, copy=False)
    else:
        try:
            import rasterio

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with rasterio.open(path) as src:
                    array = src.read().astype(np.float32)
        except Exception:  # rasterio cannot open PNG/JPG, and may not be installed
            from PIL import Image

            with Image.open(path) as img:
                array = np.asarray(img, dtype=np.float32)
                array = array[np.newaxis, ...] if array.ndim == 2 else array.transpose(2, 0, 1)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"{path}: expected a 2-D or 3-D raster, got shape {array.shape}")
    return array


def _sorted_rasters(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _find_subdir(root: Path, *names: str) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_manifest(root: Path, split: str | None = None) -> list[dict[str, Any]]:
    """Load ``manifest.jsonl`` rows, optionally filtered to one split."""
    path = Path(root) / "manifest.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if split is not None and payload.get("split") != split:
            continue
        rows.append(payload)
    return rows


def infer_provenance(root: Path, rows: Sequence[dict[str, Any]] = ()) -> tuple[str, bool]:
    """Return ``(provenance, scientifically_valid)`` for a dataset root.

    A fixture tree whose name advertises ``synthetic`` is labelled as such and
    marked not scientifically valid — a number that looks authoritative and
    means nothing is worse than no number at all.
    """
    lowered = {part.lower() for part in Path(root).parts}
    for row in rows:
        if str(row.get("provenance", "")).startswith("synthetic"):
            return "synthetic_fixture", False
    if any("synthetic" in part for part in lowered):
        return "synthetic_fixture", False
    return "real", True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairingReport:
    """How images were matched to masks — recorded in every run config."""

    strategy: str
    n_pairs: int
    image_names: tuple[str, ...] = ()
    mask_names: tuple[str, ...] = ()
    band_count: int = 0
    band_names: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_pairs": self.n_pairs,
            "band_count": self.band_count,
            "band_names": list(self.band_names),
            "note": self.note,
            "image_names": list(self.image_names),
            "mask_names": list(self.mask_names),
        }


class SAROilSpillDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Paired SAR image / binary mask dataset.

    Args:
        root: directory with ``images/`` + ``masks/`` (or a prepared root with
            ``manifest.jsonl`` plus ``<split>/images``).
        indices: optional subset of pair indices (train/val subsets).
        transform: joint ``(image, mask) -> (image, mask)`` transform on CHW
            arrays, applied in dB *before* normalization, or None. See
            :mod:`transforms`.
        image_size: advisory target size; used by the transform, not by the loader.
        split: when the root holds several splits, select one (``train``/``val``/``test``).
        expected_bands: required band count, or ``None`` to infer it from the
            first image and then enforce consistency across the dataset.
        use_auxiliary_channels: keep channels beyond VV/VH (wind, incidence).
            Off by default; when on, the extra channels are recorded.
        scene_regex: optional override for deriving a scene id from a filename.
    """

    def __init__(
        self,
        root: str | Path,
        indices: Sequence[int] | None = None,
        transform: JointTransform | None = None,
        image_size: int = 512,
        *,
        split: str | None = None,
        expected_bands: int | None = None,
        use_auxiliary_channels: bool = False,
        scene_regex: str | None = None,
        normalization: NormalizationStats | None = None,
        manifest: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.expected_bands = expected_bands
        self.use_auxiliary_channels = use_auxiliary_channels
        self.scene_regex = scene_regex
        self.normalization = normalization
        self.rows: list[dict[str, Any]] = list(manifest) if manifest else []
        self.scene_ids: tuple[str, ...] = ()
        self.band_names: tuple[str, ...] = SAR_BAND_NAMES
        self.provenance, self.scientifically_valid = "real", True

        if not self.root.exists():
            raise DatasetUnavailableError(
                f"Dataset root {self.root} does not exist. Prepare real data first "
                "(scripts/prepare_zenodo_dataset.py); this loader never invents samples."
            )

        if not self.rows:
            self.rows = load_manifest(self.root, split)
        self.provenance, self.scientifically_valid = infer_provenance(self.root, self.rows)

        manifest_scene_ids: list[str] = []
        if self.rows:
            (
                self.image_paths,
                self.mask_paths,
                pairing_strategy,
                manifest_scene_ids,
            ) = self._pair_from_manifest()
        else:
            (
                self.image_paths,
                self.mask_paths,
                pairing_strategy,
                _,
            ) = self._pair_from_directories()

        if not self.image_paths:
            raise DatasetUnavailableError(
                f"No image/mask pairs under {self.root}"
                + (f" for split {split!r}" if split else "")
                + ". Refusing to train on an empty dataset; prepare real labelled "
                "data or pass an explicit --allow-synthetic-fixtures run."
            )
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(
                f"Image/mask count mismatch under {self.root}: "
                f"{len(self.image_paths)} images vs {len(self.mask_paths)} masks. "
                "Pairing is stem-based and sorted; check for a stray file."
            )

        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]
            if manifest_scene_ids:
                manifest_scene_ids = [manifest_scene_ids[i] for i in indices]

        # The manifest is the authority on scene membership when it carries a
        # scene id. The index builder may have grouped tiles by footprint, and
        # re-deriving a scene from the file path would throw that away: for a
        # flat archive of tiles the path says every tile belongs to the same
        # scene, which makes the leakage check vacuous and hides a real
        # train/val overlap. `strict=True` so a misalignment between the
        # manifest and the paired paths is an error rather than a silently
        # wrong scene id.
        if manifest_scene_ids:
            self.scene_ids = tuple(
                str(scene) or default_scene_id(path, self.scene_regex)
                for scene, path in zip(manifest_scene_ids, self.image_paths, strict=True)
            )
        else:
            self.scene_ids = tuple(
                default_scene_id(path, self.scene_regex) for path in self.image_paths
            )
        bands = self._infer_bands()
        self.band_names = self._band_names(bands)
        self.pairing = PairingReport(
            strategy=pairing_strategy,
            n_pairs=len(self.image_paths),
            image_names=tuple(p.name for p in self.image_paths),
            mask_names=tuple(p.name for p in self.mask_paths),
            band_count=bands,
            band_names=self.band_names,
            note=(
                "sorted, stem-based pairing asserted equal length; order recorded"
                if pairing_strategy == "directory"
                else "manifest order (prepared dataset)"
            ),
        )

    # -- pairing ----------------------------------------------------------

    def _pair_from_manifest(self) -> tuple[list[Path], list[Path], str, list[str]]:
        """Paths and scene ids from the manifest, in manifest order.

        The scene ids come back aligned with the paths, and ``""`` where a row
        does not carry one — the caller falls back to deriving it from the path.
        """
        images: list[Path] = []
        masks: list[Path] = []
        scenes: list[str] = []
        for row in self.rows:
            image = self.root / str(row["image"])
            mask = self.root / str(row["mask"])
            if not image.exists() or not mask.exists():
                raise DatasetUnavailableError(
                    f"Manifest row points at a missing file under {self.root}: "
                    f"{row['image']} / {row['mask']}. Re-run the preparation step."
                )
            images.append(image)
            masks.append(mask)
            scenes.append(str(row.get("scene_id") or ""))
        return images, masks, "manifest", scenes

    def _pair_from_directories(self) -> tuple[list[Path], list[Path], str, list[str]]:
        base = self.root
        if self.split is not None:
            candidate = base / self.split
            if candidate.is_dir():
                base = candidate
        image_dir = _find_subdir(base, "images", "image", "imgs")
        mask_dir = _find_subdir(base, "masks", "mask", "labels", "gt")
        if image_dir is None or mask_dir is None:
            raise DatasetUnavailableError(
                f"{base} has neither a manifest.jsonl nor images/ + masks/ "
                "subdirectories. Nothing to train on."
            )
        images = _sorted_rasters(image_dir)
        masks = _sorted_rasters(mask_dir)
        if not images or not masks:
            raise DatasetUnavailableError(
                f"No rasters found under {image_dir} or {mask_dir} "
                f"({', '.join(IMAGE_SUFFIXES)}). Refusing to train on an empty dataset."
            )
        images, masks = self._match_by_stem(images, masks)
        return images, masks, "directory", []

    @staticmethod
    def _match_by_stem(images: list[Path], masks: list[Path]) -> tuple[list[Path], list[Path]]:
        """Sorted, stem-based pairing with the ``_mask`` convention tolerated."""
        mask_index: dict[str, Path] = {}
        for mask in masks:
            mask_index.setdefault(mask.stem.lower(), mask)
            for suffix in ("_mask", "-mask", "_label", "_gt", "_seg"):
                if mask.stem.lower().endswith(suffix):
                    mask_index.setdefault(mask.stem.lower()[: -len(suffix)], mask)
        paired_images: list[Path] = []
        paired_masks: list[Path] = []
        missing: list[str] = []
        for image in images:
            found = mask_index.get(image.stem.lower())
            if found is None:
                missing.append(image.name)
                continue
            paired_images.append(image)
            paired_masks.append(found)
        if missing:
            raise DatasetUnavailableError(
                f"{len(missing)} image(s) have no mask (e.g. {missing[:5]}). "
                "Refusing to drop labels silently."
            )
        return paired_images, paired_masks

    # -- bands ------------------------------------------------------------

    def _infer_bands(self) -> int:
        probe = read_raster(self.image_paths[0])
        bands = int(probe.shape[0])
        if self.expected_bands is not None and bands != self.expected_bands:
            raise ValueError(
                f"{self.image_paths[0].name}: {bands} bands found, {self.expected_bands} "
                "expected (VV, VH). Bands are never dropped or padded implicitly — "
                "either fix the preparation step or enable auxiliary channels."
            )
        if self.expected_bands is None:
            for path in self.image_paths[1:]:
                if int(read_raster(path).shape[0]) != bands:
                    raise ValueError(
                        f"Inconsistent band count: {self.image_paths[0].name} has "
                        f"{bands}, {path.name} has {int(read_raster(path).shape[0])}."
                    )
        return bands

    def _band_names(self, bands: int) -> tuple[str, ...]:
        if bands <= len(SAR_BAND_NAMES):
            return SAR_BAND_NAMES[:bands]
        extra = AUX_CHANNEL_NAMES[: bands - len(SAR_BAND_NAMES)]
        tail = tuple(f"aux_{i}" for i in range(bands - len(SAR_BAND_NAMES) - len(extra)))
        return SAR_BAND_NAMES + extra + tail

    # -- torch ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = read_raster(image_path)
        mask = read_raster(mask_path)
        if mask.shape[0] > 1:
            mask = mask[:1]
        if image.shape[1:] != mask.shape[1:]:
            raise ValueError(
                f"{image_path.name} {image.shape[1:]} and {mask_path.name} "
                f"{mask.shape[1:]} differ in extent; refusing to resize silently."
            )

        bands = int(image.shape[0])
        if self.expected_bands is not None and bands != self.expected_bands:
            raise ValueError(
                f"{image_path.name}: {bands} bands, expected {self.expected_bands} "
                "(VV, VH). Bands are never dropped implicitly."
            )
        if not self.use_auxiliary_channels and bands > len(SAR_BAND_NAMES):
            raise ValueError(
                f"{image_path.name}: {bands} bands but auxiliary channels are disabled. "
                "Pass use_auxiliary_channels=True (trainer: --aux-channels) if these "
                "extra channels are genuinely wind/incidence data, not padding."
            )

        # Augmentation runs *before* normalization, and in the units the archive
        # stores (dB). Speckle is multiplicative on linear intensity, so it must
        # be applied to a field it can be inverted from — normalizing first
        # would leave a shifted, scaled quantity with no way back.
        #
        # The contract is native CHW: (C, H, W) float32 in, (C, H, W) float32
        # out, mask (H, W) binary. The old albumentations-style HWC/dict
        # contract existed only to satisfy a library that is not installed, and
        # its ImportError handler turned a missing dependency into a silent
        # no-op augmentation.
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            image = np.asarray(image, dtype=np.float32)
            mask = np.asarray(mask)
            if image.shape[0] != bands:
                raise ValueError(
                    f"{image_path.name}: transform changed the band count "
                    f"{bands} -> {image.shape[0]}; a transform must not add or drop channels"
                )

        if self.normalization is not None:
            stats = self.normalization
            for band in range(min(bands, len(stats.mean))):
                mean = stats.mean[band]
                std = stats.std[band] if stats.std[band] > 0 else 1.0
                image[band] = (image[band] - mean) / std

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray((mask > 0.5).astype(np.float32)))
        if mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[:1]
        return image_tensor, mask_tensor

    # -- introspection ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "split": self.split,
            "n_pairs": len(self.image_paths),
            "band_count": self.pairing.band_count,
            "band_names": list(self.band_names),
            "expected_bands": self.expected_bands,
            "use_auxiliary_channels": self.use_auxiliary_channels,
            "provenance": self.provenance,
            "scientifically_valid": self.scientifically_valid,
            "scenes": sorted(set(self.scene_ids)),
            "pairing": self.pairing.to_dict(),
        }


def compute_normalization_statistics(
    dataset: SAROilSpillDataset,
    max_items: int | None = 64,
    percentiles: tuple[float, ...] = (1.0, 50.0, 99.0),
) -> NormalizationStats:
    """Per-band mean/std/percentiles over a deterministic subset of a dataset.

    These are the statistics an out-of-distribution check later compares
    against, so they must come from the training split only.
    """
    if len(dataset) == 0:
        raise DatasetUnavailableError("Cannot compute normalization statistics: dataset empty.")
    items = range(min(len(dataset), max_items) if max_items else len(dataset))
    # Sized zero and grown on the first item so the accumulators are never
    # Optional — an `Optional[ndarray]` narrowed by an `if` inside a loop is a
    # type error waiting to happen, and the band count is not known until then.
    sums = np.zeros(0)
    sums_sq = np.zeros(0)
    counts = 0
    samples: list[np.ndarray] = []
    for index in items:
        image, _ = dataset[index]
        array = image.numpy().astype(np.float64)
        bands = array.shape[0]
        flat = array.reshape(bands, -1)
        if sums.size == 0:
            sums = np.zeros(bands)
            sums_sq = np.zeros(bands)
        sums += flat.sum(axis=1)
        sums_sq += (flat**2).sum(axis=1)
        counts += flat.shape[1]
        if len(samples) < 8:
            samples.append(flat[:, :: max(1, flat.shape[1] // 4096)])
    if sums.size == 0:
        raise DatasetUnavailableError("Cannot compute normalization statistics: no items read.")
    mean = sums / counts
    std = np.sqrt(np.maximum(sums_sq / counts - mean**2, 0.0))
    pooled = np.concatenate(samples, axis=1) if samples else np.zeros((len(mean), 1))
    pcts = np.percentile(pooled, percentiles, axis=1).T
    return NormalizationStats(
        mean=tuple(float(v) for v in mean),
        std=tuple(float(v) for v in std),
        percentiles=tuple(tuple(float(v) for v in row) for row in pcts),
        band_names=tuple(dataset.band_names[: len(mean)]),
        percentile_levels=percentiles,
    )


def scene_statistics_from_tensor(
    image: torch.Tensor | np.ndarray,
    band_names: tuple[str, ...] = SAR_BAND_NAMES,
) -> SceneStatistics:
    """Convenience wrapper used by the OOD check at inference time."""
    return compute_scene_statistics(image, band_names=band_names)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------


def get_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    image_size: int = 512,
    val_split: float = 0.15,
    test_split: float = 0.1,
    num_workers: int = 0,
    train_transform: JointTransform | None = None,
    val_transform: JointTransform | None = None,
    *,
    expected_bands: int | None = None,
    use_auxiliary_channels: bool = False,
    manifest: Sequence[dict[str, Any]] | None = None,
) -> tuple[Any, Any, Any]:
    """Build train/val/test dataloaders.

    If ``data_dir`` is a prepared root (``manifest.jsonl`` + ``<split>/``), the
    prepared splits are used verbatim — they were made per *scene*. Otherwise the
    split is still computed per scene (never per tile) from the file names; the
    randomness that used to live here is gone, because a random tile split
    leaks the same acquisition across train and test.

    ``manifest`` overrides the rows read from disk. That exists so a caller can
    cap a run (``max_scenes``) or hold out a split without rewriting the index —
    the index is an input, and training must not write to it.
    """
    from torch.utils.data import DataLoader

    root = Path(data_dir)
    rows_in = list(manifest) if manifest is not None else load_manifest(root)
    if rows_in:
        splits = {str(row["split"]) for row in rows_in}
        if {"train", "val", "test"} & splits:
            loaders: list[Any] = []
            for split in ("train", "val", "test"):
                rows = [r for r in rows_in if r.get("split") == split]
                if not rows:
                    # A split with no rows is legitimate: test_fraction defaults
                    # to 0.0, so there is usually no test split at all. The
                    # DataLoader was guarded against this but the dataset was
                    # not, and building it with an empty manifest sends __init__
                    # down the pair-from-directories path against a root that
                    # holds nothing but a manifest — which raises
                    # DatasetUnavailableError for a run that has everything it
                    # needs. The caller gets None and skips the split.
                    loaders.append(None)
                    continue
                transform = train_transform if split == "train" else val_transform
                dataset = SAROilSpillDataset(
                    root,
                    split=split,
                    transform=transform,
                    image_size=image_size,
                    expected_bands=expected_bands,
                    use_auxiliary_channels=use_auxiliary_channels,
                    manifest=rows,
                )
                loaders.append(
                    DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=split == "train",
                        num_workers=num_workers,
                        pin_memory=False,
                        drop_last=split == "train",
                    )
                )
            logger.info("prepared-root splits: {}", sorted(splits))
            return loaders[0], loaders[1], loaders[2]

    full = SAROilSpillDataset(
        root,
        transform=None,
        image_size=image_size,
        expected_bands=expected_bands,
        use_auxiliary_channels=use_auxiliary_channels,
    )
    scene_ids = list(full.scene_ids)
    from splits import assign_scene_splits

    assignment = assign_scene_splits(scene_ids, val_fraction=val_split)
    index_by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, scene in enumerate(scene_ids):
        index_by_split[assignment.by_scene[scene]].append(idx)

    from transforms import build_eval_transforms, build_train_transforms

    train_tf = (
        train_transform if train_transform is not None else build_train_transforms(image_size)
    )
    val_tf = val_transform if val_transform is not None else build_eval_transforms(image_size)

    def _loader(indices: list[int], transform: JointTransform | None, shuffle: bool) -> Any:
        subset = SAROilSpillDataset(
            root,
            indices=indices,
            transform=transform,
            image_size=image_size,
            expected_bands=expected_bands,
            use_auxiliary_channels=use_auxiliary_channels,
        )
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=shuffle,
        )

    logger.info(
        "scene-aware split: train={} val={} test={}",
        len(index_by_split["train"]),
        len(index_by_split["val"]),
        len(index_by_split["test"]),
    )
    return (
        _loader(index_by_split["train"], train_tf, True),
        _loader(index_by_split["val"], val_tf, False),
        _loader(index_by_split["test"], val_tf, False),
    )


def split_indices_by_scene(
    scene_ids: Iterable[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Indices per split, grouped by scene id (no tile-level randomness)."""
    from splits import assign_scene_splits

    ids = list(scene_ids)
    assignment = assign_scene_splits(ids, val_fraction=val_fraction + test_fraction, seed=seed)
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, scene in enumerate(ids):
        out[assignment.by_scene[scene]].append(idx)
    if out["val"] and test_fraction > 0:
        cut = max(1, int(round(len(out["val"]) * test_fraction / (val_fraction + test_fraction))))
        out["test"] = out["val"][:cut]
        out["val"] = out["val"][cut:]
    return out


@dataclass
class DatasetSummary:
    root: str
    n_pairs: int
    provenance: str
    scientifically_valid: bool
    bands: int
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "n_pairs": self.n_pairs,
            "provenance": self.provenance,
            "scientifically_valid": self.scientifically_valid,
            "bands": self.bands,
            "fields": self.fields,
        }
