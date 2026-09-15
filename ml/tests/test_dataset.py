"""Tests for the paired SAR loader.

The loader has one job that is easy to get subtly wrong: decide what the dataset
*is*, and which splits it has. These pin the decision that was wrong — a split
with no rows is a normal state, not a fatal one — because it only shows up on
the real archive, where ``test_fraction`` defaults to 0.0 and there is therefore
no test split to build.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ml.training.dataset import get_dataloaders

SIZE = 16


def _write_raster(path: Path, bands: int, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((bands, SIZE, SIZE), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(4.0, 55.0, 1e-4, 1e-4),
    ) as dst:
        dst.write(data)


def _prepared_root(tmp_path: Path, splits: dict[str, int]) -> tuple[Path, list[dict]]:
    """A prepared root: manifest.jsonl plus <split>/{images,masks}."""
    root = tmp_path / "prepared"
    rows: list[dict[str, str]] = []
    index = 0
    for split, count in splits.items():
        for _ in range(count):
            name = f"{index:05d}.tif"
            image = Path(split) / "images" / name
            mask = Path(split) / "masks" / name
            _write_raster(root / image, 2, -20.0)
            _write_raster(root / mask, 1, 0.0)
            rows.append({"image": str(image), "mask": str(mask), "split": split})
            index += 1
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return root, rows


def test_a_split_with_no_rows_yields_no_loader_instead_of_raising(tmp_path: Path) -> None:
    """``test_fraction`` defaults to 0.0, so most runs have no test split.

    The prepared-root branch guarded the DataLoader against an empty split but
    built the dataset unconditionally. With an empty manifest the dataset falls
    through to pair-from-directories, and a prepared root holds nothing but a
    manifest and the split directories — so a real-archive run died with
    ``DatasetUnavailableError: ... has neither a manifest.jsonl nor images/ +
    masks/ subdirectories`` for a split it never needed.
    """
    root, rows = _prepared_root(tmp_path, {"train": 3, "val": 2})

    train_loader, val_loader, test_loader = get_dataloaders(
        str(root), batch_size=1, image_size=8, num_workers=0, manifest=rows
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is None


def test_all_three_splits_are_loaded_when_they_are_present(tmp_path: Path) -> None:
    """The empty-split fix must not turn a present split into a None."""
    root, rows = _prepared_root(tmp_path, {"train": 3, "val": 2, "test": 2})

    train_loader, val_loader, test_loader = get_dataloaders(
        str(root), batch_size=1, image_size=8, num_workers=0, manifest=rows
    )

    assert train_loader is not None and len(train_loader.dataset) == 3
    assert val_loader is not None and len(val_loader.dataset) == 2
    assert test_loader is not None and len(test_loader.dataset) == 2


def test_an_empty_train_split_is_still_fatal(tmp_path: Path) -> None:
    """A run with no training data is a usage error, not something to paper over.

    Only a *missing* split becomes None. An empty train split would otherwise
    reach the trainer as a None loader and be reported as "absent", which reads
    like a configuration choice rather than the mistake it is.
    """
    root, rows = _prepared_root(tmp_path, {"val": 2, "test": 2})

    train_loader, val_loader, test_loader = get_dataloaders(
        str(root), batch_size=1, image_size=8, num_workers=0, manifest=rows
    )

    assert train_loader is None
    assert val_loader is not None
    assert test_loader is not None
