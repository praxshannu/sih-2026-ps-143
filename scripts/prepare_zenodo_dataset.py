#!/usr/bin/env python3
"""Prepare the *extracted* Zenodo Sentinel-1 oil-spill archives for training.

Zenodo concept DOI **10.5281/zenodo.8320179**:

* Part I   (record 8346860, ``01_Train_Val_Oil_Spill_images.7z``)  — oil spills
* Part II  (record 8253899, ``01_Train_Val_Lookalike_images.7z``)  — look-alikes / no-oil
* Part III (record 13761290, ``02_Test_images_and_ground_truth.7z``) — **held-out test only**

This script does not download anything (see ``download_zenodo_dataset.py`` for
that, and for why the archives are too large for a laptop). It reads an
already-extracted tree and writes::

    data/zenodo_prepared/
      train/images/  train/masks/
      val/images/    val/masks/
      test/images/   test/masks/
      manifest.jsonl  manifest.csv  preparation_report.json

Guarantees, all enforced in code and asserted by ``ml/tests/test_prepare_zenodo.py``:

* every image is paired with a mask by identifier, deterministically (sorted);
  an unmatched image is a hard error, never a silent skip;
* tiles from one scene never straddle two splits — splitting is per scene;
* Part III can only ever land in ``test``;
* both SAR bands (VV, VH) are preserved and counted; a band count other than
  two is reported, never silently truncated;
* masks are label rasters: a missing CRS is recorded, never a rejection;
* the disk is measured *before* a byte is written; over budget is a refusal.

Examples::

    python scripts/prepare_zenodo_dataset.py --source-root data/zenodo --limit 20
    python scripts/prepare_zenodo_dataset.py --source-root data/zenodo --max-bytes 2GiB
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "training"))

from splits import (  # noqa: E402
    CONCEPT_DOI,
    PART_1,
    PART_2,
    PART_3,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TEST_ONLY_PARTS,
    assert_no_scene_leakage,
    assign_scene_splits,
    default_scene_id,
    limit_scenes,
    normalise_part,
    part_doi,
    scene_leakage,
)

IMAGE_SUFFIXES: frozenset[str] = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"})
MASK_DIR_NAMES: frozenset[str] = frozenset(
    {"mask", "masks", "labels", "label", "gt", "ground_truth", "groundtruth", "annotations"}
)
DEFAULT_MASK_SUFFIXES: tuple[str, ...] = (
    "_mask",
    "-mask",
    "_label",
    "-label",
    "_gt",
    "-gt",
    "_seg",
    "-seg",
    "_target",
    "-target",
)

#: The single most important caveat about this dataset, restated in every
#: report and on stdout. Part I is oil-only; without Part II the model has
#: never seen a look-alike.
PART_ONE_WARNING = (
    "Part I contains oil-spill examples but no look-alike / no-oil diversity "
    "(that is Part II). Training on Part I alone teaches 'dark patch = oil', "
    "so the network will fire on low-wind patches, rain cells, biogenic films "
    "and ship wakes. An IoU measured on a Part-I-only split is not detection "
    "skill and must not be reported as such."
)

BYTE_UNITS: tuple[tuple[str, int], ...] = (
    ("kib", 1024),
    ("mib", 1024**2),
    ("gib", 1024**3),
    ("kb", 1000),
    ("mb", 1000**2),
    ("gb", 1000**3),
    ("b", 1),
)


class PreparationError(RuntimeError):
    """Any condition that must stop the run before a byte is written."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_bytes(value: str) -> int:
    """Parse ``512MiB`` / ``2GiB`` / plain integer into a byte count."""
    text = str(value).strip().lower().replace(" ", "")
    for suffix, factor in BYTE_UNITS:
        if text.endswith(suffix):
            number = text[: -len(suffix)]
            if not number:
                raise argparse.ArgumentTypeError(f"missing number in byte value {value!r}")
            return int(float(number) * factor)
    try:
        return int(float(text))
    except ValueError as exc:  # pragma: no cover - argparse path
        raise argparse.ArgumentTypeError(f"cannot parse byte value {value!r}") from exc


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_mask_candidate(path: Path, mask_suffixes: tuple[str, ...]) -> bool:
    """A file is a mask if it lives under a mask-ish directory or is suffixed."""
    lowered = path.stem.lower()
    if any(parent.name.lower() in MASK_DIR_NAMES for parent in path.parents):
        return True
    return any(lowered.endswith(suffix.lower()) for suffix in mask_suffixes)


def mask_key(path: Path, mask_suffixes: tuple[str, ...]) -> str:
    """Identifier used to match a mask back to its image."""
    stem = path.stem
    for suffix in sorted(mask_suffixes, key=len, reverse=True):
        if stem.lower().endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
            break
    return stem.lower()


@dataclass(frozen=True)
class BandInfo:
    count: int
    height: int
    width: int
    crs: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "height": self.height,
            "width": self.width,
            "crs": self.crs,
        }


def read_band_info(path: Path) -> BandInfo:
    """Read geometry/band metadata. Masks never need a CRS — see module docs.

    GDAL's ``NotGeoreferencedWarning`` is suppressed on purpose: label rasters
    are legitimately un-georeferenced and requiring a CRS here would reject
    valid training data. The CRS (or its absence) is recorded per row instead.
    """
    if path.suffix == ".npy":
        import numpy as np

        array = np.load(path, mmap_mode="r")
        if array.ndim == 2:
            return BandInfo(1, int(array.shape[0]), int(array.shape[1]), None)
        return BandInfo(int(array.shape[0]), int(array.shape[1]), int(array.shape[2]), None)
    try:
        import rasterio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path) as src:
                crs = src.crs.to_string() if src.crs else None
                return BandInfo(int(src.count), int(src.height), int(src.width), crs)
    except Exception:  # rasterio cannot open PNG/JPG, and may not be installed
        from PIL import Image

        with Image.open(path) as img:
            array_shape = (img.height, img.width)
            bands = len(img.getbands())
            return BandInfo(bands, array_shape[0], array_shape[1], None)


# ---------------------------------------------------------------------------
# Discovery and pairing
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    image: Path
    mask: Path
    part: str
    source_id: str
    scene_id: str
    image_sha256: str = ""
    mask_sha256: str = ""
    bands: int = 0
    height: int = 0
    width: int = 0
    mask_crs: str | None = None
    n_bytes: int = 0
    split: str = SPLIT_TRAIN


def discover_parts(source_root: Path, overrides: dict[str, str]) -> dict[str, list[Path]]:
    """Classify the directories under ``source_root`` into Zenodo parts."""
    if not source_root.exists():
        raise PreparationError(
            f"Source tree {source_root} does not exist. Extract the Zenodo archives "
            f"({CONCEPT_DOI}) first: "
            "python scripts/download_zenodo_dataset.py --record part3 --download --extract"
        )
    if not source_root.is_dir():
        raise PreparationError(f"Source root {source_root} is not a directory.")

    parts: dict[str, list[Path]] = {}
    unknown: list[str] = []
    candidates = sorted(p for p in source_root.iterdir() if p.is_dir())
    for candidate in candidates:
        part = overrides.get(candidate.name) or normalise_part(candidate.name)
        if part is None:
            unknown.append(candidate.name)
            continue
        parts.setdefault(part, []).append(candidate)

    if not parts:
        # The operator may point straight at a single extracted part.
        solo = overrides.get(source_root.name) or normalise_part(source_root.name)
        if solo:
            parts[solo] = [source_root]
    if not parts:
        raise PreparationError(
            f"No Zenodo part directories found under {source_root}. "
            f"Seen: {[p.name for p in candidates] or 'nothing'}. "
            "Pass e.g. --part-dir my_folder:part3 to map them explicitly."
        )
    if unknown:
        raise PreparationError(
            f"Could not classify directory/ies {unknown} under {source_root}. "
            "Refusing to guess which Zenodo part they are: pass "
            "--part-dir NAME:part1 (or :part2 / :part3) explicitly."
        )
    return parts


def collect_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                files.append(path)
    return sorted(files)


def pair_images_and_masks(
    files: list[Path],
    mask_suffixes: tuple[str, ...],
    part: str,
    source_id: str,
    scene_regex: str | None,
) -> tuple[list[Sample], list[str], list[str]]:
    """Deterministically pair every image with its mask.

    Returns ``(samples, unmatched_images, unused_masks)``. Unmatched images are
    returned as errors; the caller must turn them into a hard failure.
    """
    images: list[Path] = []
    masks: list[Path] = []
    for path in files:
        if is_mask_candidate(path, mask_suffixes):
            masks.append(path)
        else:
            images.append(path)

    index: dict[str, list[Path]] = {}
    for mask in masks:
        index.setdefault(mask_key(mask, mask_suffixes), []).append(mask)
        index.setdefault(mask.stem.lower(), []).append(mask)

    samples: list[Sample] = []
    unmatched: list[str] = []
    for image in images:
        # A mask may be indexed twice (stripped stem and plain stem); de-dup.
        distinct = sorted({p.resolve() for p in index.get(image.stem.lower(), [])}, key=str)
        if len(distinct) > 1:
            unmatched.append(f"{image}: ambiguous mask match {[str(p) for p in distinct]}")
            continue
        if not distinct:
            unmatched.append(
                f"{image}: no mask found "
                f"(looked for stem {image.stem!r} with suffixes {list(mask_suffixes)})"
            )
            continue
        mask = Path(distinct[0])
        samples.append(
            Sample(
                image=image,
                mask=mask,
                part=part,
                source_id=source_id,
                scene_id=default_scene_id(image, scene_regex),
            )
        )

    used = {sample.mask.resolve() for sample in samples}
    unused = [str(mask) for mask in masks if mask.resolve() not in used]
    return samples, unmatched, unused


# ---------------------------------------------------------------------------
# Disk guard
# ---------------------------------------------------------------------------


def check_disk_budget(
    output_dir: Path,
    required_bytes: int,
    max_bytes: int | None,
    min_free_bytes: int,
) -> dict[str, Any]:
    usage = shutil.disk_usage(output_dir if output_dir.exists() else Path(output_dir.anchor))
    budget = usage.free - min_free_bytes
    if max_bytes is not None:
        budget = min(budget, max_bytes)
    denied = {
        "free_bytes": int(usage.free),
        "required_bytes": int(required_bytes),
        "max_bytes": int(max_bytes) if max_bytes is not None else None,
        "min_free_bytes": int(min_free_bytes),
        "budget_bytes": int(budget),
    }
    if required_bytes > budget:
        raise PreparationError(
            f"Refusing to prepare: {human_bytes(required_bytes)} of images+masks needed, "
            f"but the budget is {human_bytes(budget)} "
            f"(free {human_bytes(usage.free)} - reserve {human_bytes(min_free_bytes)}"
            + (f", capped at --max-bytes {human_bytes(max_bytes)}" if max_bytes else "")
            + "). Reduce the subset with --limit N, or raise --max-bytes."
        )
    denied["ok"] = True
    return denied


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


@dataclass
class PreparedDataset:
    output_dir: Path
    samples: list[Sample] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def prepare(
    source_root: Path,
    output_dir: Path,
    *,
    part_overrides: dict[str, str] | None = None,
    mask_suffixes: tuple[str, ...] = DEFAULT_MASK_SUFFIXES,
    scene_regex: str | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
    limit: int | None = None,
    max_bytes: int | None = None,
    min_free_bytes: int = 2 * 1024**3,
    archive_manifests: dict[str, dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> PreparedDataset:
    """Plan and (unless ``dry_run``) write the prepared dataset."""
    overrides = part_overrides or {}
    parts = discover_parts(source_root, overrides)

    all_samples: list[Sample] = []
    unmatched: list[str] = []
    unused_masks: list[str] = []
    for part, roots in sorted(parts.items()):
        for root in roots:
            files = collect_files([root])
            if not files:
                raise PreparationError(
                    f"Part {part} directory {root} contains no images "
                    f"({sorted(IMAGE_SUFFIXES)}). Nothing was written."
                )
            samples, missing, unused = pair_images_and_masks(
                files, mask_suffixes, part, root.name, scene_regex
            )
            unmatched.extend(missing)
            unused_masks.extend(unused)
            all_samples.extend(samples)

    if unmatched:
        preview = "\n  ".join(unmatched[:10])
        raise PreparationError(
            f"{len(unmatched)} image(s) have no mask — refusing to prepare a dataset "
            "with silently dropped labels:\n  "
            f"{preview}\n"
            "Fix the pairing convention (--mask-suffix / --part-dir) and re-run."
        )
    if not all_samples:
        raise PreparationError(
            f"No image/mask pairs could be formed from {source_root}. "
            "Nothing was written; check the mask suffix convention (--mask-suffix)."
        )

    # --- metadata: bands, checksums (never truncate a band) -----------------
    for sample in all_samples:
        info = read_band_info(sample.image)
        sample.bands = info.count
        sample.height = info.height
        sample.width = info.width
        sample.mask_crs = read_band_info(sample.mask).crs
        sample.image_sha256 = sha256_file(sample.image)
        sample.mask_sha256 = sha256_file(sample.mask)
        sample.n_bytes = sample.image.stat().st_size + sample.mask.stat().st_size

    # --- scene-aware splits; Part III is test-only --------------------------
    test_scenes = sorted({s.scene_id for s in all_samples if s.part in TEST_ONLY_PARTS})
    other_scenes = sorted({s.scene_id for s in all_samples if s.part not in TEST_ONLY_PARTS})
    if limit is not None:
        keep = limit_scenes([*other_scenes, *test_scenes], limit, pinned=test_scenes)
        keep_set = set(keep)
        dropped = len({s.scene_id for s in all_samples} - keep_set)
        all_samples = [s for s in all_samples if s.scene_id in keep_set]
        if not all_samples:
            raise PreparationError(f"--limit {limit} selected no samples; nothing written.")
    else:
        dropped = 0

    assignment = assign_scene_splits(
        [s.scene_id for s in all_samples],
        val_fraction=val_fraction,
        seed=seed,
        test_only_scenes=test_scenes,
    )
    for sample in all_samples:
        sample.split = assignment.by_scene[sample.scene_id]

    assert_no_scene_leakage((s.scene_id, s.split) for s in all_samples)
    leaked_parts = sorted(
        {s.part for s in all_samples if s.part in TEST_ONLY_PARTS and s.split != SPLIT_TEST}
    )
    if leaked_parts:
        raise PreparationError(
            f"Part III ({', '.join(leaked_parts)}) leaked into a non-test split. "
            "Part III is the held-out archive and must never be trained on."
        )

    # --- disk guard before writing anything --------------------------------
    required = sum(s.n_bytes for s in all_samples)
    disk = check_disk_budget(output_dir, required, max_bytes, min_free_bytes)

    warnings: list[str] = list(assignment.warnings)
    present_parts = {s.part for s in all_samples}
    if PART_1 in present_parts and PART_2 not in present_parts:
        warnings.append(PART_ONE_WARNING)
    elif PART_1 in present_parts:
        warnings.append(
            "Part I is oil-spill imagery only; Part II supplies the look-alike / "
            "no-oil diversity. Verify both are present and balanced before "
            "trusting any metric produced from this prepared set."
        )
    band_histogram: dict[str, int] = {}
    for sample in all_samples:
        band_histogram[str(sample.bands)] = band_histogram.get(str(sample.bands), 0) + 1
    if any(count != 2 for count in (int(k) for k in band_histogram)):
        warnings.append(
            "Some images do not have exactly 2 bands (VV, VH). Band counts seen: "
            f"{band_histogram}. Bands are preserved as-is and recorded per row — "
            "they were NOT truncated."
        )
    if unused_masks:
        warnings.append(
            f"{len(unused_masks)} mask file(s) matched no image (e.g. "
            f"{unused_masks[:3]}); they were left out of the prepared set."
        )
    if dropped:
        warnings.append(f"--limit {limit} dropped {dropped} scene(s) from the source tree.")

    result = PreparedDataset(output_dir=output_dir, samples=all_samples)

    report: dict[str, Any] = {
        "generated_utc": utc_now(),
        "source_root": str(source_root),
        "output_root": str(output_dir),
        "concept_doi": CONCEPT_DOI,
        "part_dois": {part: part_doi(part) for part in sorted(present_parts)},
        "archive_checksums": archive_manifests or {},
        "counts": {
            split: sum(1 for s in all_samples if s.split == split)
            for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)
        }
        | {"total": len(all_samples)},
        "scene_counts": assignment.counts(),
        "scenes": {
            scene: {
                "split": split,
                "parts": sorted({s.part for s in all_samples if s.scene_id == scene}),
                "images": sum(1 for s in all_samples if s.scene_id == scene),
            }
            for scene, split in sorted(assignment.by_scene.items())
        },
        "band_counts": band_histogram,
        "mask_crs_present": sum(1 for s in all_samples if s.mask_crs),
        "mask_crs_absent": sum(1 for s in all_samples if not s.mask_crs),
        "part_iii_test_only": True,
        "part_iii_violations": 0,
        "scene_leakage": {},
        "split_params": {"val_fraction": val_fraction, "seed": seed, "limit": limit},
        "disk": disk | {"written_bytes": 0},
        "warnings": warnings,
        "dry_run": dry_run,
    }
    report["scene_leakage"] = {
        scene: sorted(splits)
        for scene, splits in scene_leakage((s.scene_id, s.split) for s in all_samples).items()
    }

    if dry_run:
        print(f"[prepare] DRY RUN — would write {len(all_samples)} pairs to {output_dir}")
        result.report = report
        return result

    written = _write_dataset(output_dir, all_samples)
    report["disk"]["written_bytes"] = written
    _write_manifests(output_dir, all_samples, archive_manifests or {})
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    result.report = report
    return result


def _out_names(sample: Sample) -> tuple[str, str]:
    """Deterministic output names: ``<scene>__<stem>`` and ``<scene>__<stem>_mask``."""
    scene = sample.scene_id.replace("/", "_").replace(" ", "_")
    image_name = f"{scene}__{sample.image.stem}{sample.image.suffix.lower()}"
    mask_name = f"{scene}__{sample.image.stem}_mask{sample.mask.suffix.lower()}"
    return image_name, mask_name


def _write_dataset(output_dir: Path, samples: list[Sample]) -> int:
    written = 0
    for sample in samples:
        image_name, mask_name = _out_names(sample)
        dst_image = output_dir / sample.split / "images" / image_name
        dst_mask = output_dir / sample.split / "masks" / mask_name
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image, dst_image)
        shutil.copy2(sample.mask, dst_mask)
        written += dst_image.stat().st_size + dst_mask.stat().st_size
    return written


def _manifest_row(sample: Sample, archive_manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    archive = archive_manifests.get(sample.part, {})
    return {
        "split": sample.split,
        "image": str(Path(sample.split) / "images" / _out_names(sample)[0]),
        "mask": str(Path(sample.split) / "masks" / _out_names(sample)[1]),
        "part": sample.part,
        "scene_id": sample.scene_id,
        "source_id": sample.source_id,
        "source_doi": part_doi(sample.part),
        "concept_doi": CONCEPT_DOI,
        "archive_checksum": archive.get("sha256"),
        "archive_checksum_kind": "sha256" if archive.get("sha256") else None,
        "image_sha256": sample.image_sha256,
        "mask_sha256": sample.mask_sha256,
        "bands": sample.bands,
        "height": sample.height,
        "width": sample.width,
        "bytes": sample.n_bytes,
        "mask_crs": sample.mask_crs,
    }


def _write_manifests(
    output_dir: Path,
    samples: list[Sample],
    archive_manifests: dict[str, dict[str, Any]] | None = None,
) -> None:
    manifests = archive_manifests or {}
    rows = [_manifest_row(s, manifests) for s in samples]
    with (output_dir / "manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    if rows:
        with (output_dir / "manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def load_archive_manifests(source_root: Path) -> dict[str, dict[str, Any]]:
    """Read ``<part>.manifest.json`` written by ``download_zenodo_dataset.py``."""
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(source_root.glob("*.manifest.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        record = str(payload.get("record", ""))
        part = normalise_part(record) or normalise_part(path.stem)
        if part:
            found[part] = {
                "file": str(payload.get("file", path.name)),
                "zenodo_record": payload.get("zenodo_record"),
                "md5": payload.get("md5"),
                "sha256": payload.get("sha256"),
                "bytes": payload.get("bytes"),
                "source": payload.get("source"),
            }
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=Path("data/zenodo"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/zenodo_prepared"))
    parser.add_argument(
        "--part-dir",
        action="append",
        default=[],
        metavar="NAME:PART",
        help="explicitly classify a source directory, e.g. --part-dir my_dump:part3",
    )
    parser.add_argument(
        "--mask-suffix",
        action="append",
        default=[],
        help=f"mask filename suffix (repeatable); defaults to {DEFAULT_MASK_SUFFIXES}",
    )
    parser.add_argument("--scene-regex", default=None, help="regex with a 'scene' group")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="keep at most N scenes")
    parser.add_argument("--max-bytes", type=parse_bytes, default=None, help="e.g. 2GiB")
    parser.add_argument("--min-free-bytes", type=parse_bytes, default=2 * 1024**3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, str] = {}
    for item in args.part_dir:
        if ":" not in item:
            print(f"error: --part-dir expects NAME:PART, got {item!r}", file=sys.stderr)
            return 2
        name, part = item.split(":", 1)
        part = part.strip().lower()
        if part not in (PART_1, PART_2, PART_3):
            print(f"error: unknown part {part!r} (use part1/part2/part3)", file=sys.stderr)
            return 2
        overrides[name.strip()] = part

    suffixes = tuple(dict.fromkeys([*args.mask_suffix, *DEFAULT_MASK_SUFFIXES]))
    try:
        result = prepare(
            args.source_root,
            args.output_dir,
            part_overrides=overrides,
            mask_suffixes=suffixes,
            scene_regex=args.scene_regex,
            val_fraction=args.val_fraction,
            seed=args.seed,
            limit=args.limit,
            max_bytes=args.max_bytes,
            min_free_bytes=args.min_free_bytes,
            archive_manifests=load_archive_manifests(args.source_root),
            dry_run=args.dry_run,
        )
    except PreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = result.report
    counts = report["counts"]
    print(f"[prepare] source: {report['source_root']}")
    print(f"[prepare] output: {report['output_root']}")
    print(
        f"[prepare] pairs: {counts['total']} "
        f"(train {counts[SPLIT_TRAIN]}, val {counts[SPLIT_VAL]}, test {counts[SPLIT_TEST]})"
    )
    print(f"[prepare] scenes: {report['scene_counts']}")
    print(f"[prepare] band counts: {report['band_counts']}  (bands preserved, never truncated)")
    print(f"[prepare] Part III is test-only: {report['part_iii_test_only']}")
    for warning in report["warnings"]:
        print(f"[prepare] WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
