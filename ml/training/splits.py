"""Scene- and source-aware splitting for SAR oil-spill datasets.

A Sentinel-1 scene is one acquisition. Tiles cut from the same scene are
near-duplicates: they share the calibration, the incidence angle, the wind
regime and the speckle realisation. Splitting a dataset by *tiles* therefore
leaks information from train into val/test and produces an IoU that is
measurably optimistic and scientifically meaningless.

This module splits by **scene id**, deterministically:

* the assignment is a pure function of ``(scene_id, seed, val_fraction)``;
* it uses ``hashlib.sha256`` (never ``hash()``, which is salted per process
  for ``str`` and would make runs irreproducible);
* the Zenodo "Part III" held-out archive is test-only, always.

No I/O, no randomness, no side effects.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLITS: tuple[str, str, str] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

PART_1 = "part1"
PART_2 = "part2"
PART_3 = "part3"

#: Zenodo DOI 10.5281/zenodo.8320179, Part III, is the held-out test archive.
#: It must never contribute a training or validation tile.
TEST_ONLY_PARTS: frozenset[str] = frozenset({PART_3})

#: Per-part DOIs of the Zenodo concept record 10.5281/zenodo.8320179.
PART_RECORD_IDS: Mapping[str, int] = {
    PART_1: 8346860,  # 01_Train_Val_Oil_Spill_images.7z
    PART_2: 8253899,  # 01_Train_Val_Lookalike_images.7z
    PART_3: 13761290,  # 02_Test_images_and_ground_truth.7z
}
CONCEPT_DOI = "10.5281/zenodo.8320179"

#: Directory names that carry no scene identity (they are containers, or a
#: split/part label), so they must not be used as a scene id.
GENERIC_DIR_NAMES: frozenset[str] = frozenset(
    {
        "images",
        "image",
        "imgs",
        "img",
        "masks",
        "mask",
        "msks",
        "labels",
        "label",
        "gt",
        "ground_truth",
        "groundtruth",
        "ground truth",
        "annotations",
        "annotation",
        "tiles",
        "patches",
        "patch",
        "chips",
        "data",
        "train",
        "val",
        "valid",
        "validation",
        "test",
        "trainval",
        "train_val",
        "part1",
        "part2",
        "part3",
        "part_i",
        "part_ii",
        "part_iii",
    }
)

#: Trailing tile indices that must be stripped before the remainder is used as
#: a scene id: ``S1A_..._20200810_0012`` -> ``S1A_..._20200810``.
_TILE_SUFFIX_RE = re.compile(
    r"^(?P<scene>.+?)[_-](?:\d{1,6}|tile_?\d+|\d+[_-]\d+|r\d+c\d+)$",
    re.IGNORECASE,
)
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}


def normalise_part(name: str) -> str | None:
    """Map an archive/directory name onto ``part1``/``part2``/``part3``.

    Returns ``None`` when the name cannot be classified; callers must then ask
    the operator for an explicit ``--part-dir NAME:PART`` mapping rather than
    guessing, because a misclassified part silently breaks the Part III
    held-out guarantee.
    """
    token = re.sub(r"[^a-z0-9]+", "", str(name).lower())

    explicit = re.fullmatch(r"part([0-9]+|[ivx]+)", token)
    if explicit:
        raw = explicit.group(1)
        index = int(raw) if raw.isdigit() else _ROMAN.get(raw, 0)
        if index in (1, 2, 3):
            return f"part{index}"

    if "lookalike" in token or "lookalikes" in token or "nooil" in token:
        return PART_2
    if "groundtruth" in token or "groundtruths" in token:
        return PART_3
    if "oilspill" in token:
        return PART_1
    if "test" in token and "train" not in token:
        return PART_3
    return None


def part_doi(part: str) -> str:
    """Per-part Zenodo DOI, falling back to the concept DOI."""
    record = PART_RECORD_IDS.get(part)
    return f"10.5281/zenodo.{record}" if record else CONCEPT_DOI


def default_scene_id(path: str | Path, scene_regex: str | None = None) -> str:
    """Derive a stable scene/source id from a file path.

    Resolution order:

    1. ``scene_regex`` — if it matches, use group ``scene`` (or group 1).
    2. The nearest non-generic parent directory (scene-per-directory layouts).
    3. The file stem with any trailing tile index stripped.

    Everything is deterministic: no timestamps, no random state.
    """
    p = Path(path)
    if scene_regex:
        pattern = re.compile(scene_regex)
        for candidate in (p.stem, p.name, str(p)):
            match = pattern.search(candidate)
            if match:
                groups = match.groupdict()
                if "scene" in groups:
                    return str(groups["scene"])
                if match.groups():
                    return str(match.group(1))
    for parent in p.parents:
        name = parent.name.lower()
        if not name or name in GENERIC_DIR_NAMES:
            continue
        if normalise_part(name) is not None:
            continue
        return parent.name
    stem = p.stem
    match = _TILE_SUFFIX_RE.match(stem)
    return match.group("scene") if match else stem


@dataclass(frozen=True)
class SplitAssignment:
    """Result of a scene-aware split.

    ``by_scene`` maps scene id -> split. ``seed``/``val_fraction`` are echoed so
    the assignment can be reproduced from a report alone.
    """

    by_scene: dict[str, str] = field(default_factory=dict)
    val_fraction: float = 0.2
    seed: int = 0
    warnings: tuple[str, ...] = ()

    def scenes(self, split: str) -> list[str]:
        return sorted(s for s, value in self.by_scene.items() if value == split)

    def counts(self) -> dict[str, int]:
        return {split: len(self.scenes(split)) for split in SPLITS}

    def split_of(self, scene: str) -> str | None:
        return self.by_scene.get(scene)

    def to_dict(self) -> dict[str, object]:
        return {
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "counts": self.counts(),
            "by_scene": dict(sorted(self.by_scene.items())),
            "warnings": list(self.warnings),
        }


def scene_rank(scene: str, seed: int) -> str:
    """Deterministic, process-stable ranking key for a scene id."""
    return hashlib.sha256(f"{seed}|{scene}".encode()).hexdigest()


def assign_scene_splits(
    scene_ids: Iterable[str],
    val_fraction: float = 0.2,
    seed: int = 0,
    test_scenes: Iterable[str] = (),
    test_only_scenes: Iterable[str] = (),
) -> SplitAssignment:
    """Split *scenes* (not tiles) into train/val/test.

    ``test_scenes`` are pre-declared test scenes (e.g. everything from a
    held-out part). ``test_only_scenes`` are additionally pinned so that a
    later call cannot promote them — used to enforce Part III.

    A scene lands in exactly one split; that is structural here, not a check
    applied afterwards.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    forced_test = {str(s) for s in test_only_scenes} | {str(s) for s in test_scenes}
    pool = sorted({str(s) for s in scene_ids} - forced_test)
    by_scene: dict[str, str] = dict.fromkeys(sorted(forced_test), SPLIT_TEST)

    warnings: list[str] = []
    n_val = int(round(len(pool) * val_fraction))
    if pool and n_val == 0 and val_fraction > 0:
        if len(pool) > 1:
            n_val = 1
            warnings.append(
                f"val_fraction={val_fraction} rounds to 0 for {len(pool)} scenes; "
                "forced 1 scene into val so validation is not empty."
            )
        else:
            warnings.append(
                "Only one scene is available for train/val: it must all go to "
                "train, so validation is empty and no generalisation claim can "
                "be made from this run."
            )
    if pool and n_val >= len(pool):
        n_val = max(len(pool) - 1, 0)
        warnings.append(
            f"val_fraction={val_fraction} would leave no training scene; "
            f"clamped val to {n_val} of {len(pool)} scenes."
        )
    if not pool:
        warnings.append("No train/val scenes at all — the source tree holds test data only.")

    ranked = sorted(pool, key=lambda scene: scene_rank(scene, seed))
    for scene in ranked[:n_val]:
        by_scene[scene] = SPLIT_VAL
    for scene in ranked[n_val:]:
        by_scene[scene] = SPLIT_TRAIN

    return SplitAssignment(
        by_scene=by_scene,
        val_fraction=val_fraction,
        seed=seed,
        warnings=tuple(warnings),
    )


def scene_leakage(pairs: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Return ``{scene_id: {splits}}`` for every scene seen in >1 split.

    Empty dict == no leakage. Kept as a standalone function so tests and the
    preparation report can assert the invariant independently of the code that
    produced the assignment.
    """
    seen: dict[str, set[str]] = {}
    for scene, split in pairs:
        seen.setdefault(str(scene), set()).add(str(split))
    return {scene: splits for scene, splits in seen.items() if len(splits) > 1}


def assert_no_scene_leakage(pairs: Iterable[tuple[str, str]]) -> None:
    leaked = scene_leakage(pairs)
    if leaked:
        detail = ", ".join(f"{scene}: {sorted(v)}" for scene, v in sorted(leaked.items())[:5])
        raise ValueError(
            f"Scene leakage detected — {len(leaked)} scene(s) appear in more than "
            f"one split ({detail}). Splits must be made per scene, never per tile."
        )


def limit_scenes(
    scene_ids: Sequence[str],
    limit: int | None,
    pinned: Iterable[str] = (),
) -> list[str]:
    """Deterministically subsample scenes for a laptop-sized run.

    ``pinned`` (e.g. held-out test scenes) survive the cut; the remainder is
    taken from the sorted scene list so the subset is reproducible.
    """
    ordered = sorted({str(s) for s in scene_ids})
    if limit is None or limit <= 0 or limit >= len(ordered):
        return ordered
    keep = {str(s) for s in pinned}
    selected = [s for s in ordered if s in keep]
    for scene in ordered:
        if len(selected) >= limit:
            break
        if scene not in keep:
            selected.append(scene)
    return sorted(selected)[: max(limit, len(keep))]
