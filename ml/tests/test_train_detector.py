"""Tests for the detector training pipeline.

Three things are being pinned down here, and they are the three that were
actually broken:

1. ``align_targets`` — the old trainer unsqueezed masks the dataset had already
   returned as ``(B, 1, H, W)``, producing ``(B, 1, 1, H, W)``, which broadcast
   against ``(B, 1, H, W)`` logits into a ``(B, B, 1, H, W)`` cross-comparison.
   The loss decreased. The model learned nothing. These tests fail if the shape
   guard is ever relaxed.
2. The in-place contract — ``SourceFingerprint`` must actually detect a change
   to the source directory, and ``assert_outputs_are_outside_the_source`` must
   actually refuse an output path inside it. An assertion that cannot fail is
   not evidence.
3. Checkpoint atomicity — an interrupted run must not leave a file that
   ``--resume`` will try to load.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
import torch

from ml.training.train_detector import (
    BCEDiceLoss,
    DiceLoss,
    SourceFingerprint,
    SplitMetrics,
    _cap_scenes,
    align_targets,
    assert_outputs_are_outside_the_source,
    atomic_torch_save,
    load_checkpoint,
    resolve_device,
    seed_everything,
)
from sentinel_core.datasource import DataSourceSpec
from sentinel_core.errors import CheckpointError, TrainingError
from sentinel_core.provenance import DataProvenance

BATCH, CHANNELS, HEIGHT, WIDTH = 4, 1, 8, 8


def logits() -> torch.Tensor:
    return torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)


def make_spec(root: Path) -> DataSourceSpec:
    """A minimal real-source spec pointing at ``root``."""
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    return DataSourceSpec(
        kind="real",
        prepared_root=root,
        images_dir=images,
        masks_dir=masks,
        n_pairs=1,
        bands=2,
        provenance=DataProvenance.REAL,
        in_place=True,
        read_only=True,
        split_strategy="scene",
    )


# ---------------------------------------------------------------------------
# align_targets -- the double-unsqueeze guard
# ---------------------------------------------------------------------------


def test_align_targets_accepts_bchw_unchanged() -> None:
    masks = torch.zeros(BATCH, 1, HEIGHT, WIDTH)
    targets = align_targets(masks, logits())
    assert targets.shape == (BATCH, 1, HEIGHT, WIDTH)


def test_align_targets_promotes_bhw_to_bchw() -> None:
    masks = torch.zeros(BATCH, HEIGHT, WIDTH)
    targets = align_targets(masks, logits())
    assert targets.shape == (BATCH, 1, HEIGHT, WIDTH)


def test_align_targets_returns_float() -> None:
    masks = torch.zeros(BATCH, 1, HEIGHT, WIDTH, dtype=torch.uint8)
    assert align_targets(masks, logits()).dtype == torch.float32


def test_align_targets_rejects_the_historical_double_unsqueeze() -> None:
    """The exact shape the old trainer produced must now be an error.

    ``(B, 1, 1, H, W)`` broadcasts against ``(B, 1, H, W)``. This is the one
    case where "it did not crash" was the bug.
    """
    double = torch.zeros(BATCH, 1, 1, HEIGHT, WIDTH)
    with pytest.raises(TrainingError):
        align_targets(double, logits())


def test_align_targets_documents_why_broadcasting_is_the_hazard() -> None:
    """Show the broadcast the guard exists to prevent.

    Without the shape assertion this is what the loss would have been computed
    against: every sample scored against every other sample's mask.
    """
    double = torch.zeros(BATCH, 1, 1, HEIGHT, WIDTH)
    broadcast = double * torch.ones(BATCH, 1, HEIGHT, WIDTH)
    assert broadcast.shape == (BATCH, BATCH, 1, HEIGHT, WIDTH)


def test_align_targets_rejects_spatial_mismatch() -> None:
    masks = torch.zeros(BATCH, 1, HEIGHT + 1, WIDTH)
    with pytest.raises(TrainingError):
        align_targets(masks, logits())


def test_align_targets_rejects_multiple_channels() -> None:
    masks = torch.zeros(BATCH, 3, HEIGHT, WIDTH)
    with pytest.raises(TrainingError):
        align_targets(masks, logits())


def test_align_targets_rejects_two_dimensional_input() -> None:
    with pytest.raises(TrainingError):
        align_targets(torch.zeros(HEIGHT, WIDTH), logits())


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_dice_loss_is_near_zero_for_a_perfect_prediction() -> None:
    target = torch.zeros(BATCH, 1, HEIGHT, WIDTH)
    target[:, :, :4, :] = 1.0
    confident = torch.where(target > 0.5, 10.0, -10.0)
    assert float(DiceLoss()(confident, target)) < 0.01


def test_dice_loss_is_near_one_for_an_empty_prediction() -> None:
    target = torch.zeros(BATCH, 1, HEIGHT, WIDTH)
    target[:, :, :4, :] = 1.0
    empty = torch.full((BATCH, 1, HEIGHT, WIDTH), -10.0)
    assert float(DiceLoss()(empty, target)) > 0.9


def test_bce_dice_loss_is_bounded() -> None:
    target = torch.zeros(BATCH, 1, HEIGHT, WIDTH)
    target[:, :, :4, :] = 1.0
    value = float(BCEDiceLoss()(torch.randn(BATCH, 1, HEIGHT, WIDTH), target))
    assert 0.0 <= value <= 2.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_split_metrics_pool_counts_rather_than_averaging_rates() -> None:
    metrics = SplitMetrics()
    from ml.training.metrics import BinaryCounts

    metrics.add(BinaryCounts(tp=1, fp=1, fn=0, tn=8))  # tiny, iou 0.5
    metrics.add(BinaryCounts(tp=100, fp=0, fn=0, tn=0))  # large, iou 1.0
    payload = metrics.to_dict()
    # A mean of the two IoUs would be 0.75; the pooled value is 101/102.
    assert payload["iou"] == pytest.approx(101 / 102, abs=1e-6)


def test_split_metrics_reports_counts_and_rates() -> None:
    metrics = SplitMetrics()
    payload = metrics.to_dict()
    assert {"loss", "n_batches", "iou", "precision", "recall", "f1"} <= set(payload)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def test_resolve_device_honours_an_explicit_cpu() -> None:
    device, reason = resolve_device("cpu")
    assert device.type == "cpu"
    assert reason


def test_resolve_device_auto_returns_a_usable_device() -> None:
    device, reason = resolve_device("auto")
    assert device.type in {"cpu", "mps", "cuda"}
    assert reason.startswith("auto")


def test_resolve_device_refuses_cuda_when_it_is_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(TrainingError):
        resolve_device("cuda")


def test_seed_everything_is_reproducible() -> None:
    seed_everything(11)
    first = torch.randn(4)
    seed_everything(11)
    assert torch.equal(first, torch.randn(4))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_atomic_save_writes_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "detector_best.pth"
    atomic_torch_save({"model_state_dict": {"w": torch.zeros(2)}}, path)
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_save_overwrites_an_existing_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "detector_best.pth"
    atomic_torch_save({"model_state_dict": {"w": torch.zeros(2)}, "epoch": 1}, path)
    atomic_torch_save({"model_state_dict": {"w": torch.ones(2)}, "epoch": 2}, path)
    payload = load_checkpoint(path, torch.device("cpu"))
    assert payload["epoch"] == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_save_leaves_the_previous_checkpoint_intact_on_failure(
    tmp_path: Path,
) -> None:
    """The atomicity guarantee, stated as the property that matters.

    Whatever goes wrong, the destination is either the old checkpoint or the new
    one -- never a truncation. A payload ``torch.save`` cannot serialise is the
    easiest way to interrupt a save mid-write. A lock is used rather than a
    lambda because a lambda's failure mode depends on the enclosing scope.
    """
    path = tmp_path / "detector_best.pth"
    atomic_torch_save({"model_state_dict": {"w": torch.zeros(2)}, "epoch": 1}, path)
    before = path.read_bytes()

    with pytest.raises(TypeError):
        atomic_torch_save({"model_state_dict": {"w": threading.Lock()}}, path)

    assert path.read_bytes() == before
    assert load_checkpoint(path, torch.device("cpu"))["epoch"] == 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_checkpoint_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "nope.pth", torch.device("cpu"))


def test_load_checkpoint_reports_a_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "truncated.pth"
    path.write_bytes(b"\x80\x02not-a-torch-file")
    with pytest.raises(CheckpointError):
        load_checkpoint(path, torch.device("cpu"))


def test_load_checkpoint_reports_a_malformed_payload(tmp_path: Path) -> None:
    path = tmp_path / "malformed.pth"
    torch.save({"something_else": 1}, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, torch.device("cpu"))


# ---------------------------------------------------------------------------
# The read-only proof
# ---------------------------------------------------------------------------


def test_fingerprint_verifies_an_untouched_directory(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    source.mkdir()
    (source / "a.tif").write_bytes(b"x")
    fingerprint = SourceFingerprint.capture(source)
    ok, detail = fingerprint.verify(source)
    assert ok, detail


def test_fingerprint_detects_an_added_file(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    source.mkdir()
    (source / "a.tif").write_bytes(b"x")
    fingerprint = SourceFingerprint.capture(source)
    (source / "b.tif").write_bytes(b"y")
    ok, detail = fingerprint.verify(source)
    assert not ok
    assert "entry count" in detail or "mtime" in detail


def test_fingerprint_detects_a_removed_file(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    source.mkdir()
    for name in ("a.tif", "b.tif"):
        (source / name).write_bytes(b"x")
    fingerprint = SourceFingerprint.capture(source)
    (source / "a.tif").unlink()
    ok, _ = fingerprint.verify(source)
    assert not ok


def test_fingerprint_records_the_device_it_was_taken_on(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    source.mkdir()
    payload = SourceFingerprint.capture(source).to_dict()
    assert {"path", "mtime_ns", "size", "n_entries", "st_dev", "captured_utc"} <= set(payload)


# ---------------------------------------------------------------------------
# Outputs must be outside the source
# ---------------------------------------------------------------------------


def test_output_inside_the_source_is_refused(tmp_path: Path) -> None:
    spec = make_spec(tmp_path / "archive")
    with pytest.raises(TrainingError):
        assert_outputs_are_outside_the_source(
            spec, {"checkpoints": spec.prepared_root / "runs"}
        )


def test_output_equal_to_the_source_is_refused(tmp_path: Path) -> None:
    spec = make_spec(tmp_path / "archive")
    with pytest.raises(TrainingError):
        assert_outputs_are_outside_the_source(spec, {"run_dir": spec.images_dir})


def test_output_outside_the_source_is_allowed(tmp_path: Path) -> None:
    spec = make_spec(tmp_path / "archive")
    assert_outputs_are_outside_the_source(
        spec,
        {
            "checkpoints": tmp_path / "checkpoints",
            "run_dir": tmp_path / "runs" / "abc",
            "log_dir": tmp_path / "logs",
        },
    )


def test_sibling_directory_with_a_shared_prefix_is_allowed(tmp_path: Path) -> None:
    """``/data/archive-backup`` must not be mistaken for ``/data/archive``."""
    spec = make_spec(tmp_path / "archive")
    assert_outputs_are_outside_the_source(spec, {"run_dir": tmp_path / "archive-backup"})


# ---------------------------------------------------------------------------
# Scene capping
# ---------------------------------------------------------------------------


def make_rows(n_train: int, n_val: int, n_test: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        rows.extend({"split": split, "scene": f"{split}-{i}"} for i in range(count))
    return rows


def test_cap_scenes_is_a_no_op_when_under_the_cap() -> None:
    rows = make_rows(8, 1, 1)
    kept, note = _cap_scenes(rows, 100)
    assert kept == rows
    assert note is None


def test_cap_scenes_is_a_no_op_when_the_cap_is_none() -> None:
    rows = make_rows(8, 1, 1)
    kept, note = _cap_scenes(rows, None)
    assert len(kept) == len(rows)
    assert note is None


def test_cap_scenes_keeps_every_split_represented() -> None:
    """A capped run must still have something to validate against."""
    rows = make_rows(96, 12, 12)
    kept, note = _cap_scenes(rows, 20)
    assert note is not None
    splits = {row["split"] for row in kept}
    assert splits == {"train", "val", "test"}
    assert len(kept) < len(rows)


def test_cap_scenes_reports_what_it_did() -> None:
    rows = make_rows(96, 12, 12)
    _, note = _cap_scenes(rows, 20)
    assert note is not None
    assert "max_scenes=20" in note
    assert "of 120" in note


def test_cap_scenes_does_not_mutate_the_input() -> None:
    rows = make_rows(10, 1, 1)
    original = list(rows)
    _cap_scenes(rows, 4)
    assert rows == original
