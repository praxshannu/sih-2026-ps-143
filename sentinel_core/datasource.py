"""Runtime switching between the real and synthetic SAR datasets.

The contract
------------
A *data source* is not a directory. It is a directory plus the claims you are
allowed to make about the data in it. This module resolves one of two sources
and returns both halves together, so a caller cannot obtain the pixels while
losing the provenance:

* ``real`` — the Sentinel-1 archive, read **in place** where it lives. Nothing
  is copied, moved or rewritten. On this machine that is
  ``/Volumes/Ventoy/Oil``, an external disk; on another it is whatever
  ``SENTINEL_DATASOURCES__REAL_IMAGES_DIR`` says.
* ``synthetic`` — a generated dataset that matches the real one's structure and
  distribution, labelled ``synthetic_mock`` at every layer.

Neither falls back to the other. Asking for ``real`` when the archive is not
mounted is an error, because the alternative is a run that completes and
reports a number computed on invented data. A caller that genuinely wants a
substitute sets ``SENTINEL_DATASOURCES__ALLOW_SYNTHETIC_FALLBACK=true`` and
gets the substitution **plus** a synthetic label on everything downstream.

Why a manifest index rather than two directories
------------------------------------------------
``ml/training/dataset.py`` already defines a prepared root as "a directory with
a ``manifest.jsonl`` whose rows name an image and a mask". The real archive has
images and masks in two unrelated absolute directories, so the cheapest honest
adapter is an index that *points at* them. The index is a few hundred kilobytes
of JSON; it records absolute paths, so ``Path(root) / "/abs/path"`` resolves to
``/abs/path`` and the bytes are still read from the external disk. The index is
an index, not a copy — and it is also where per-pair provenance is recorded,
which is the thing that makes the switch auditable.

Splitting without scene ids
---------------------------
The archive carries no scene identifier: every GeoTIFF has only generic TIFF
tags, and the files are numbered ``00000.tif`` … ``01339.tif``. What the files
*do* carry is a real geotransform, and the tiles are scattered across the North
Sea. So the split groups tiles by the coarse grid cell containing their
centroid — a spatial proxy for "same acquisition" — and records
``split_strategy`` plus a warning that it is a proxy. A per-file hash split
would put two 20 km tiles of the same pass on opposite sides of the train/test
boundary and report an IoU that means nothing; this at least does not do that
silently.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from sentinel_core.config import DataSourceKind, Settings, get_settings
from sentinel_core.errors import (
    DatasetUnavailableError,
    DataSourceMismatchError,
    DataSourceUnavailableError,
)
from sentinel_core.provenance import DataProvenance, synthetic_warning
from sentinel_core.raster import header_info

__all__ = [
    "DataSourceRegistry",
    "DataSourceSpec",
    "active_source",
    "get_registry",
    "resolve_source",
    "source_report",
    "switch_source",
]

RASTER_SUFFIXES: tuple[str, ...] = (".tif", ".tiff")
MANIFEST_NAME = "manifest.jsonl"
SOURCE_META_NAME = "source.json"

#: Coarse grid used to group tiles that probably came from the same pass.
DEFAULT_SCENE_GRID_DEG = 1.0

#: macOS writes an AppleDouble sidecar next to every file on a non-native
#: filesystem. They end in ``.tif`` too and would otherwise be indexed as
#: 4 KB "scenes" that fail to open.
_APPLEDOUBLE_PREFIX = "._"


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSourceSpec:
    """A resolved data source: where the bytes are, and what they may support."""

    kind: DataSourceKind
    prepared_root: Path
    images_dir: Path
    masks_dir: Path
    n_pairs: int
    bands: int
    provenance: DataProvenance
    in_place: bool
    read_only: bool
    split_strategy: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def scientifically_valid(self) -> bool:
        """True when metrics from this source can support a claim."""
        return self.provenance.supports_scientific_claim

    @property
    def is_synthetic(self) -> bool:
        return self.provenance.is_synthetic

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "prepared_root": str(self.prepared_root),
            "images_dir": str(self.images_dir),
            "masks_dir": str(self.masks_dir),
            "n_pairs": self.n_pairs,
            "bands": self.bands,
            "in_place": self.in_place,
            "read_only": self.read_only,
            "split_strategy": self.split_strategy,
            "note": self.note,
            "warnings": list(self.warnings),
            **self.provenance.describe(),
        }


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------


def _list_rasters(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in RASTER_SUFFIXES
        and not p.name.startswith(_APPLEDOUBLE_PREFIX)
    )


def _probe(path: Path) -> tuple[int, str]:
    """Band count and a coarse geolocation key, without reading any pixels.

    Cheap because :mod:`sentinel_core.raster` turns off GDAL's per-open
    directory scan — 268 ms down to 1.8 ms per file on the external archive,
    which is why validating all 1200 scenes costs seconds rather than minutes.
    """
    header = header_info(path)
    return header.bands, header.geo_key


def _scene_id(path: Path, geo: str, grid_deg: float) -> str:
    """Spatial proxy for a scene id; falls back to the file stem.

    Two tiles whose centroids fall in the same ``grid_deg`` cell are treated as
    the same "scene" for splitting purposes. That is a heuristic, not a fact —
    which is why it is recorded on the spec rather than assumed downstream.
    """
    if not geo:
        return path.stem
    try:
        lon, lat = (float(part) for part in geo.split(","))
    except ValueError:  # pragma: no cover - _probe always emits "x,y"
        return path.stem
    return f"g{round(lon / grid_deg):+d}_{round(lat / grid_deg):+d}"


# ---------------------------------------------------------------------------
# Pairing and splitting
# ---------------------------------------------------------------------------


def _pair_by_stem(
    images: Sequence[Path], masks: Sequence[Path], *, source: str
) -> list[tuple[Path, Path]]:
    """Pair images to masks by file stem, refusing to drop anything.

    Masks win ties on the ``_mask`` convention, matching
    :meth:`SAROilSpillDataset._match_by_stem`, so both halves of the pipeline
    agree on what a pair is.
    """
    if not images:
        raise DataSourceUnavailableError(
            f"{source}: no images found",
            reason="no_images",
            context={"source": source},
        )
    if not masks:
        raise DataSourceUnavailableError(
            f"{source}: no masks found",
            reason="no_masks",
            context={"source": source},
        )

    index: dict[str, Path] = {}
    for mask in masks:
        index.setdefault(mask.stem.lower(), mask)
        for suffix in ("_mask", "-mask", "_label", "_gt", "_seg"):
            if mask.stem.lower().endswith(suffix):
                index.setdefault(mask.stem.lower()[: -len(suffix)], mask)

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for image in images:
        mask = index.get(image.stem.lower())
        if mask is None:
            missing.append(image.name)
            continue
        pairs.append((image, mask))

    if missing:
        raise DataSourceUnavailableError(
            f"{source}: {len(missing)} image(s) have no mask (e.g. {', '.join(missing[:5])})",
            reason="unpaired_images",
            context={"source": source, "n_missing": len(missing)},
        )

    orphans = {m.stem.lower() for m in masks} - {i.stem.lower() for i in images}
    if orphans:
        logger.warning(
            "{}: {} mask(s) have no image and were ignored (e.g. {})",
            source,
            len(orphans),
            ", ".join(sorted(orphans)[:5]),
        )
    return pairs


def _assign_splits(
    entries: Sequence[tuple[str, Path, Path]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[tuple[str, Path, Path]]]:
    """Deterministic per-scene split. Reuses ``splits.assign_scene_splits``."""
    import sys

    training_dir = Path(__file__).resolve().parents[1] / "ml" / "training"
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))

    from splits import assign_scene_splits  # type: ignore[import-not-found]

    scene_ids = sorted({scene for scene, _, _ in entries})
    combined = val_fraction + test_fraction
    assignment = assign_scene_splits(scene_ids, val_fraction=combined, seed=seed)

    out: dict[str, list[tuple[str, Path, Path]]] = {"train": [], "val": [], "test": []}
    for entry in entries:
        out[assignment.by_scene[entry[0]]].append(entry)

    # Re-partition the held-out group so `test` is not empty when asked for.
    if test_fraction > 0 and out["val"]:
        held = sorted(out["val"], key=lambda e: e[0])
        cut = max(1, int(round(len(held) * test_fraction / combined)))
        out["test"], out["val"] = held[:cut], held[cut:]
    return out


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def _write_manifest(
    prepared_root: Path,
    rows: Iterable[dict[str, Any]],
    meta: dict[str, Any],
) -> int:
    """Write ``manifest.jsonl`` + ``source.json`` atomically.

    Atomic because a half-written manifest is worse than no manifest: the
    dataset loader would pair a prefix of the archive and report a smaller
    ``n_pairs`` as though that were the dataset.
    """
    prepared_root.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared_root / MANIFEST_NAME
    meta_path = prepared_root / SOURCE_META_NAME

    materialised = list(rows)
    tmp_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_manifest.open("w", encoding="utf-8") as sink:
        for row in materialised:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp_manifest, manifest_path)

    meta = dict(meta)
    meta["n_rows"] = len(materialised)
    meta["generated_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    return len(materialised)


def _entries_with_geo(
    pairs: Sequence[tuple[Path, Path]], grid_deg: float, require_bands: int
) -> tuple[list[tuple[str, Path, Path]], int, str]:
    """Read every scene header and group tiles by their coarse spatial cell.

    This is the expensive path. On the external archive a *cold* file open costs
    ~300 ms (measured: 23 ms just to open and read 16 bytes, the rest is GDAL
    parsing the IFD), so 1200 files is roughly six minutes the first time. The
    result is written into the index, so the cost is paid once.
    """
    logger.info(
        "reading {} scene header(s) to group tiles spatially — this is slow on "
        "cold removable media (about 0.3 s per never-before-opened file)",
        len(pairs),
    )
    entries: list[tuple[str, Path, Path]] = []
    band_counts: set[int] = set()
    for index, (image, mask) in enumerate(pairs, start=1):
        bands, geo = _probe(image)
        band_counts.add(bands)
        entries.append((_scene_id(image, geo, grid_deg), image, mask))
        if index % 100 == 0:
            logger.info("  header {}/{}", index, len(pairs))

    if len(band_counts) != 1:
        raise DataSourceMismatchError(
            f"real archive mixes band counts {sorted(band_counts)}; a model cannot "
            "be trained on inconsistent input channels",
            reason="inconsistent_bands",
            context={"bands_seen": sorted(band_counts)},
        )
    bands = next(iter(band_counts))
    if bands != require_bands:
        raise DataSourceMismatchError(
            f"real archive has {bands} band(s), configuration requires {require_bands}",
            reason="band_count_mismatch",
            context={"found": bands, "required": require_bands},
        )
    return entries, bands, f"all {len(pairs)} headers read"


def _build_real(settings: Settings, *, rebuild: bool, grid_deg: float) -> DataSourceSpec:
    cfg = settings.datasources
    images_dir = cfg.real_images_dir
    masks_dir = cfg.real_masks_dir
    prepared_root = settings.paths.data_dir / "index" / "real"

    images = _list_rasters(images_dir)
    if not images:
        raise DataSourceUnavailableError(
            f"real SAR archive not found at {images_dir}",
            reason="real_images_missing",
            context={"images_dir": str(images_dir), "exists": images_dir.is_dir()},
        )
    masks = _list_rasters(masks_dir)
    if not masks:
        raise DataSourceUnavailableError(
            f"real masks not found at {masks_dir}. The image archive is read in "
            "place from the external disk, but its masks must be extracted once "
            "from the local archive: `python scripts/prepare_real_masks.py`.",
            reason="real_masks_missing",
            context={"masks_dir": str(masks_dir), "exists": masks_dir.is_dir()},
        )

    manifest_path = prepared_root / MANIFEST_NAME
    if manifest_path.is_file() and not rebuild:
        meta = json.loads((prepared_root / SOURCE_META_NAME).read_text(encoding="utf-8"))
        if meta.get("images_dir") == str(images_dir) and meta.get("n_rows"):
            return DataSourceSpec(
                kind="real",
                prepared_root=prepared_root,
                images_dir=images_dir,
                masks_dir=masks_dir,
                n_pairs=int(meta["n_rows"]),
                bands=int(meta.get("bands", cfg.require_bands)),
                provenance=DataProvenance.REAL,
                in_place=True,
                read_only=not os.access(images_dir, os.W_OK),
                split_strategy=str(meta.get("split_strategy", "unknown")),
                warnings=tuple(meta.get("warnings", ())),
                note="cached index; pass rebuild=True to re-enumerate",
            )

    pairs = _pair_by_stem(images, masks, source="real")

    if cfg.spatial_scene_grouping:
        entries, bands, band_note = _entries_with_geo(pairs, grid_deg, cfg.require_bands)
        split_strategy = f"spatial_grid_{grid_deg}deg"
    else:
        # No header scan: index by file stem. This is instant, which matters
        # because the archive is on removable media where touching all 1200
        # files costs minutes. Band count is not lost — the dataset loader
        # asserts it on every scene it actually reads.
        entries = [(image.stem, image, mask) for image, mask in pairs]
        bands = cfg.require_bands
        band_note = (
            "not scanned: the archive is on removable media where a cold file open "
            "costs ~0.3 s, so 1200 header reads is minutes. The loader asserts the "
            "band count on every scene it reads. Set "
            "SENTINEL_DATASOURCES__SPATIAL_SCENE_GROUPING=true to pay it once and "
            "cache it."
        )
        split_strategy = "per_file_hash"

    warnings: list[str] = []
    if not cfg.spatial_scene_grouping:
        warnings.append(
            "Tiles were split per file, not per scene. This archive carries no scene "
            "identifier (every GeoTIFF has only generic TIFF tags), so tiles from one "
            "acquisition can straddle the train/test boundary and cross-split metrics "
            "are optimistic. Enable spatial scene grouping to reduce this."
        )
    elif len({entry[0] for entry in entries}) == len(entries):
        warnings.append(
            "Spatial scene grouping put every tile in its own grid cell, so the split "
            "is still effectively per file. Cross-split metrics are optimistic."
        )

    by_split = _assign_splits(
        entries,
        val_fraction=settings.training.val_fraction,
        test_fraction=settings.training.test_fraction,
        seed=settings.training.seed,
    )

    rows: list[dict[str, Any]] = []
    for split, items in by_split.items():
        for scene, image, mask in sorted(items, key=lambda e: e[1].name):
            rows.append(
                {
                    "image": str(image),
                    "mask": str(mask),
                    "split": split,
                    "scene_id": scene,
                    "provenance": DataProvenance.REAL.value,
                    "source": "real",
                    "image_bytes": image.stat().st_size,
                    "mask_bytes": mask.stat().st_size,
                }
            )

    n = _write_manifest(
        prepared_root,
        rows,
        {
            "kind": "real",
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
            "in_place": True,
            "bands": bands,
            "band_validation": band_note,
            "split_strategy": split_strategy,
            "scene_grid_deg": grid_deg if cfg.spatial_scene_grouping else None,
            "n_scenes": len({e[0] for e in entries}),
            "split_counts": {k: len(v) for k, v in by_split.items()},
            "warnings": warnings,
            "provenance": DataProvenance.REAL.value,
        },
    )

    logger.info(
        "real source indexed: {} pairs, {} bands, scenes={}, splits={}",
        n,
        bands,
        len({e[0] for e in entries}),
        {k: len(v) for k, v in by_split.items()},
    )
    return DataSourceSpec(
        kind="real",
        prepared_root=prepared_root,
        images_dir=images_dir,
        masks_dir=masks_dir,
        n_pairs=n,
        bands=bands,
        provenance=DataProvenance.REAL,
        in_place=True,
        read_only=not os.access(images_dir, os.W_OK),
        split_strategy=split_strategy,
        warnings=tuple(warnings),
        note="index points at the archive in place; no raster was copied",
    )


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Read a manifest's rows, or ``[]`` if it is missing or unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return []
        if isinstance(row, dict) and row.get("image"):
            rows.append(row)
    return rows


def _synthetic_spec(
    root: Path,
    meta: dict[str, Any],
    *,
    n_pairs: int,
    bands: int,
    split_strategy: str,
) -> DataSourceSpec:
    return DataSourceSpec(
        kind="synthetic",
        prepared_root=root,
        images_dir=root,
        masks_dir=root,
        n_pairs=n_pairs,
        bands=bands,
        provenance=DataProvenance.SYNTHETIC,
        in_place=False,
        read_only=False,
        split_strategy=split_strategy,
        warnings=tuple(meta.get("warnings", ())),
        note=str(meta.get("note", synthetic_warning("synthetic dataset"))),
    )


def _build_synthetic(settings: Settings, *, rebuild: bool) -> DataSourceSpec:
    cfg = settings.datasources
    root = cfg.synthetic_root
    if not root.is_dir():
        raise DataSourceUnavailableError(
            f"synthetic dataset not found at {root}. Generate it with "
            "`python scripts/generate_synthetic.py`.",
            reason="synthetic_missing",
            context={"synthetic_root": str(root), "exists": False},
        )

    meta_path = root / SOURCE_META_NAME
    manifest_path = root / MANIFEST_NAME
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}

    # The manifest defines the dataset. The directory listing does not.
    #
    # Enumerating the split directories is what turned a real defect into a
    # corrupt dataset: a re-generation rewrites the scenes it was asked for and
    # rewrites the manifest whole, but it does not remove what an earlier run
    # left behind. Generating 120 scenes at 2048 over an earlier run's 120 at
    # 512 therefore produced a spec claiming 123 pairs at two different
    # resolutions, and the run reported success. Reading the manifest makes
    # those leftovers inert instead of authoritative.
    if manifest_path.is_file():
        manifest_rows = _load_manifest_rows(manifest_path)
        if manifest_rows:
            bands = int(meta.get("bands", cfg.require_bands))
            if not rebuild:
                return _synthetic_spec(
                    root,
                    meta,
                    n_pairs=len(manifest_rows),
                    bands=bands,
                    split_strategy="manifest",
                )
            missing = [row for row in manifest_rows if not Path(row["image"]).is_file()]
            if not missing:
                return _synthetic_spec(
                    root,
                    meta,
                    n_pairs=len(manifest_rows),
                    bands=bands,
                    split_strategy="manifest",
                )
            logger.warning(
                "{} of {} row(s) in {} name files that no longer exist; "
                "re-enumerating the split directories instead",
                len(missing),
                len(manifest_rows),
                manifest_path,
            )

    # No usable manifest: index the split directories directly.
    images: list[Path] = []
    masks: list[Path] = []
    for split in ("train", "val", "test"):
        images.extend(_list_rasters(root / split / "images"))
        masks.extend(_list_rasters(root / split / "masks"))
    if not images:
        raise DataSourceUnavailableError(
            f"{root} has no {{train,val,test}}/{{images,masks}} rasters and no {MANIFEST_NAME}",
            reason="synthetic_empty",
            context={"synthetic_root": str(root)},
        )

    pairs = _pair_by_stem(images, masks, source="synthetic")
    bands, _ = _probe(pairs[0][0])

    rows: list[dict[str, Any]] = []
    for image, mask in pairs:
        split = image.parent.parent.name
        rows.append(
            {
                "image": str(image),
                "mask": str(mask),
                "split": split,
                "scene_id": image.stem,
                "provenance": DataProvenance.SYNTHETIC.value,
                "source": "synthetic",
            }
        )

    _write_manifest(
        root,
        rows,
        {
            "kind": "synthetic",
            "bands": bands,
            "split_strategy": "directory",
            "provenance": DataProvenance.SYNTHETIC.value,
            "note": synthetic_warning("synthetic dataset"),
        },
    )
    return _synthetic_spec(
        root, meta, n_pairs=len(rows), bands=bands, split_strategy="directory"
    )


# ---------------------------------------------------------------------------
# Resolution and runtime switching
# ---------------------------------------------------------------------------


def resolve_source(
    kind: DataSourceKind | None = None,
    *,
    settings: Settings | None = None,
    rebuild: bool = False,
    grid_deg: float = DEFAULT_SCENE_GRID_DEG,
) -> DataSourceSpec:
    """Resolve ``kind`` (default: the configured ``data_source``) to a spec.

    ``real`` is attempted first when configured, and a failure is only answered
    with synthetic data when ``allow_synthetic_fallback`` is explicitly on — in
    which case the returned spec is labelled synthetic and carries the reason
    for the substitution in its warnings.
    """
    settings = settings or get_settings()
    requested: DataSourceKind = kind or settings.data_source

    if requested == "real":
        try:
            return _build_real(settings, rebuild=rebuild, grid_deg=grid_deg)
        except DataSourceUnavailableError as exc:
            if not settings.datasources.allow_synthetic_fallback:
                raise
            logger.warning(
                "real data source unavailable [{}] and ALLOW_SYNTHETIC_FALLBACK is on; "
                "substituting the synthetic dataset — every metric from this run is "
                "synthetic",
                exc.reason,
            )
            spec = _build_synthetic(settings, rebuild=rebuild)
            return DataSourceSpec(
                **{
                    **spec.__dict__,
                    "warnings": (*spec.warnings, f"substituted_for_real:{exc.reason}"),
                    "note": (
                        f"SUBSTITUTED for real data ({exc.reason}). "
                        f"{synthetic_warning('synthetic dataset')}"
                    ),
                }
            )

    return _build_synthetic(settings, rebuild=rebuild)


class DataSourceRegistry:
    """Process-wide holder for the active data source, switchable at runtime.

    Long-lived processes (a notebook, an API worker, a REPL) need to change
    source without restarting. ``set_active`` re-resolves and caches, so the
    switch is a real switch: the next dataset built reads the other tree.

    Not thread-safe by design — the alternative is a lock held across disk
    enumeration, and the realistic use is a single operator flipping a switch,
    not concurrent mutation.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._active: DataSourceSpec | None = None

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    def active(self, *, rebuild: bool = False) -> DataSourceSpec:
        if self._active is None or rebuild:
            self._active = resolve_source(settings=self.settings, rebuild=rebuild)
        return self._active

    def set_active(self, kind: DataSourceKind, *, rebuild: bool = False) -> DataSourceSpec:
        """Switch the active source. Returns the newly active spec."""
        spec = resolve_source(kind, settings=self.settings, rebuild=rebuild)
        self._active = spec
        logger.info(
            "data source switched -> {} ({}, {} pairs, {})",
            spec.kind,
            spec.n_pairs,
            spec.provenance.value,
            "in place" if spec.in_place else "local",
        )
        return spec

    def probe(self, kind: DataSourceKind) -> dict[str, Any]:
        """Describe a source without making it active, for health endpoints."""
        try:
            return resolve_source(kind, settings=self.settings).describe()
        except Exception as exc:  # noqa: BLE001 - a probe must never raise
            return {
                "kind": kind,
                "available": False,
                "error": type(exc).__name__,
                "reason": getattr(exc, "reason", "unknown"),
                "message": str(exc),
            }

    def report(self) -> dict[str, Any]:
        """Both sources plus which one is active. Never raises."""
        active = None
        try:
            active = self.active().describe()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            active = {
                "kind": self.settings.data_source,
                "available": False,
                "reason": getattr(exc, "reason", "unknown"),
                "message": str(exc),
            }
        return {
            "configured": self.settings.data_source,
            "active": active,
            "available": {
                "real": self.probe("real"),
                "synthetic": self.probe("synthetic"),
            },
        }


_REGISTRY: DataSourceRegistry | None = None


def get_registry(settings: Settings | None = None) -> DataSourceRegistry:
    """The process-wide registry."""
    global _REGISTRY
    if _REGISTRY is None or settings is not None:
        _REGISTRY = DataSourceRegistry(settings)
    return _REGISTRY


def switch_source(kind: DataSourceKind, *, rebuild: bool = False) -> DataSourceSpec:
    """Switch the active data source at runtime."""
    return get_registry().set_active(kind, rebuild=rebuild)


def active_source(*, rebuild: bool = False) -> DataSourceSpec:
    """The currently active data source."""
    return get_registry().active(rebuild=rebuild)


def source_report() -> dict[str, Any]:
    """Availability of both sources and the active one. Never raises."""
    return get_registry().report()


# ``DatasetUnavailableError`` is re-exported for callers that build datasets
# straight from a spec; keeps the import surface in one place.
_ = DatasetUnavailableError

SplitName = Literal["train", "val", "test"]
