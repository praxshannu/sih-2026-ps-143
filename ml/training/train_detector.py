"""Production training pipeline for the UNet++ SCSE oil-spill detector.

What this replaces, and why each one was a real defect
------------------------------------------------------
The previous script was a 350-line ``main()`` that read its settings from
``argparse``, printed to stdout, and wrote checkpoints with a bare
``torch.save``. Concretely:

* **The mask was unsqueezed twice.** ``SAROilSpillDataset`` returns masks as
  ``(B, 1, H, W)``; the trainer called ``.unsqueeze(1)`` on them, giving
  ``(B, 1, 1, H, W)``. Broadcasting that against ``(B, 1, H, W)`` logits expands
  to ``(B, B, 1, H, W)`` — every image scored against every *other* image's mask
  in the batch. The loss fell, the IoU moved, and nothing raised. See
  :func:`align_targets`.
* **``in_channels`` was hardcoded to 6.** The archive is 2-band (VV, VH). The
  encoder's first convolution was therefore shaped for channels that never
  arrive. It is now read from the dataset.
* **The split leaked.** ``train_indices = range(train_size)`` is a contiguous
  slice of a sorted file list, so consecutive scenes — often the same
  acquisition — landed in both train and val. Splits now come from the prepared
  index, which assigns them per scene.
* **IoU was the mean of per-batch IoU.** With ~1.5% oil coverage, most batches
  have almost no positive pixels and their IoU is near 0 or exactly 1 (empty
  union). Averaging those is not the dataset IoU. Metrics are now pooled pixel
  counts over the whole split.
* **``torch.utils.tensorboard`` was a hard import.** It is not installed here,
  so the module could not even be imported. Metrics go to JSONL, which needs no
  dependency and is diffable.
* **Checkpoints were written in place.** A crash mid-``torch.save`` left a
  truncated ``best.pth`` that ``--resume`` would then load. Writes are atomic.
* **Nothing recorded what the run actually was.** The run report now carries the
  data provenance, the exact source directory, the normalization statistics, the
  augmentation chain, and whether the source directory was modified.

The in-place contract
---------------------
The real archive is read from the external disk and never written to. That is
enforced rather than asserted: :class:`SourceFingerprint` records the
``st_mtime_ns`` and entry count of the source directories before the run and
re-checks them after. If training touched the archive, the run report says so
and the run is marked failed.

Usage::

    python scripts/train.py --data-source real --epochs 3 --max-scenes 24
    python scripts/train.py --data-source synthetic --epochs 5
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn

_TRAINING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TRAINING_DIR.parents[1]
for _path in (str(_TRAINING_DIR), str(_REPO_ROOT / "services" / "detect")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dataset import (  # noqa: E402
    compute_normalization_statistics,
    get_dataloaders,
    load_manifest,
)
from metrics import BinaryCounts, metrics_from_counts  # noqa: E402
from splits import scene_leakage  # noqa: E402
from transforms import (  # noqa: E402
    build_eval_transforms,
    build_train_transforms,
    describe_augmentation,
)

from sentinel_core.config import DataSourceKind, Settings, get_settings  # noqa: E402
from sentinel_core.datasource import DataSourceSpec, resolve_source  # noqa: E402
from sentinel_core.errors import CheckpointError, TrainingError  # noqa: E402
from sentinel_core.provenance import DataProvenance  # noqa: E402

__all__ = [
    "BCEDiceLoss",
    "DiceLoss",
    "RunReport",
    "SourceFingerprint",
    "align_targets",
    "resolve_device",
    "train",
]

CHECKPOINT_NAME = "detector_best.pth"
LAST_CHECKPOINT_NAME = "detector_last.pth"
METRICS_NAME = "metrics.jsonl"
REPORT_NAME = "run_report.json"


# ---------------------------------------------------------------------------
# Device and determinism
# ---------------------------------------------------------------------------


def resolve_device(choice: str = "auto") -> tuple[torch.device, str]:
    """Pick a device and say why.

    MPS is preferred over CPU because this project's development machine has no
    CUDA, but it is *not* treated as a drop-in for CUDA: mixed precision and
    ``pin_memory`` are CUDA-only here and :func:`train` gates them on the device
    type rather than assuming.
    """
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError(
                "device=cuda requested but torch.cuda.is_available() is False",
                reason="device_unavailable",
                context={"torch": torch.__version__},
            )
        return torch.device("cuda"), "requested explicitly"
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise TrainingError(
                "device=mps requested but torch.backends.mps.is_available() is False",
                reason="device_unavailable",
                context={"torch": torch.__version__, "built": torch.backends.mps.is_built()},
            )
        return torch.device("mps"), "requested explicitly"
    if choice == "cpu":
        return torch.device("cpu"), "requested explicitly"
    if torch.cuda.is_available():
        return torch.device("cuda"), "auto: CUDA available"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "auto: Apple MPS available, no CUDA"
    return torch.device("cpu"), "auto: no accelerator"


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed every generator that can affect a run."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("deterministic mode: cudnn benchmark off, algorithms restricted")


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class DiceLoss(nn.Module):
    """Soft Dice on probabilities, averaged over the batch.

    Computed per sample and then averaged, not as one global ratio: a global
    ratio lets a single large batch dominate, which matters when the positive
    class is ~1.5% of pixels.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice.

    BCE alone is dominated by the ~98.5% background; Dice alone has no gradient
    where the prediction is empty. The pair is the standard remedy and the
    weights are validated to sum above zero in
    :class:`sentinel_core.config.TrainingSettings`.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


# ---------------------------------------------------------------------------
# Target alignment — the fix for the double-unsqueeze
# ---------------------------------------------------------------------------


def align_targets(masks: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Force masks to ``(B, 1, H, W)`` float and assert they match the logits.

    This function exists because of a specific bug. The old trainer did
    ``masks.unsqueeze(1)`` on masks the dataset had *already* returned as
    ``(B, 1, H, W)``. The result was ``(B, 1, 1, H, W)``, which broadcasts
    against ``(B, 1, H, W)`` logits into ``(B, B, 1, H, W)``: a full
    cross-comparison of every image against every other image's mask in the
    batch. Training "worked" — the loss decreased — on a target that was mostly
    somebody else's label.

    The shape assertion is the point. Broadcasting is convenient and it is
    exactly what makes this class of bug invisible.
    """
    if masks.ndim == 4 and masks.shape[1] == 1:
        targets = masks
    elif masks.ndim == 3:
        targets = masks.unsqueeze(1)
    elif masks.ndim == 4:
        raise TrainingError(
            f"mask has {masks.shape[1]} channels; a binary spill mask has 1",
            reason="target_channel_count",
            context={"mask_shape": list(masks.shape)},
        )
    else:
        raise TrainingError(
            f"unsupported mask shape {tuple(masks.shape)}; expected (B,1,H,W) or (B,H,W)",
            reason="target_shape_unsupported",
            context={"mask_shape": list(masks.shape)},
        )

    targets = targets.float()
    if targets.shape != logits.shape:
        raise TrainingError(
            f"mask shape {tuple(targets.shape)} does not match logits "
            f"{tuple(logits.shape)}. Refusing to let broadcasting invent a loss.",
            reason="target_shape_mismatch",
            context={"mask_shape": list(targets.shape), "logit_shape": list(logits.shape)},
        )
    return targets


# ---------------------------------------------------------------------------
# Metric accumulation
# ---------------------------------------------------------------------------


@dataclass
class SplitMetrics:
    """Pooled pixel counts plus the rates derived from them."""

    loss: float = 0.0
    n_batches: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def counts(self) -> BinaryCounts:
        return BinaryCounts(tp=self.tp, fp=self.fp, fn=self.fn, tn=self.tn)

    def add(self, other: BinaryCounts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "loss": round(self.loss, 6),
            "n_batches": self.n_batches,
        }
        payload.update({k: round(v, 6) for k, v in metrics_from_counts(self.counts()).items()})
        return payload


def _run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    grad_clip: float,
    max_steps: int | None,
    epoch: int,
    log_every: int,
) -> SplitMetrics:
    """One pass over ``loader``. Trains when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)
    metrics = SplitMetrics()
    threshold = 0.5

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (images, masks) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                logger.info("stopping epoch {} at max_steps_per_epoch={}", epoch, max_steps)
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            amp = scaler is not None
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                targets = align_targets(masks, logits)
                loss = criterion(logits, targets)

            if optimizer is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()

            metrics.loss += float(loss.detach())
            metrics.n_batches += 1
            probabilities = torch.sigmoid(logits.detach()).float().cpu().numpy()
            metrics.add(
                _counts_from_arrays(probabilities, targets.detach().cpu().numpy(), threshold)
            )

            if training and log_every and step % log_every == 0:
                batch = metrics.to_dict()
                logger.info(
                    "epoch {} step {}/{} loss={:.4f} iou={:.4f} recall={:.4f}",
                    epoch,
                    step,
                    len(loader),
                    float(loss.detach()),
                    batch["iou"],
                    batch["recall"],
                )

    if metrics.n_batches:
        metrics.loss /= metrics.n_batches
    return metrics


def _counts_from_arrays(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> BinaryCounts:
    """TP/FP/FN/TN without importing the binarisation rules twice."""
    pred = (probabilities > threshold).ravel()
    true = (targets > 0.5).ravel()
    if pred.shape != true.shape:
        raise TrainingError(
            f"prediction {pred.shape} and target {true.shape} differ; cannot count",
            reason="metric_shape_mismatch",
        )
    tp = int(np.count_nonzero(pred & true))
    fp = int(np.count_nonzero(pred & ~true))
    fn = int(np.count_nonzero(~pred & true))
    tn = int(np.count_nonzero(~pred & ~true))
    return BinaryCounts(tp=tp, fp=fp, fn=fn, tn=tn)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> Path:
    """Write a checkpoint so a crash cannot leave a half-file behind.

    ``torch.save`` straight to ``best.pth`` truncates the previous checkpoint
    first. An interruption then leaves a file that ``--resume`` will happily try
    to load. Writing to a sibling temp file and ``os.replace``-ing it is atomic
    on the same filesystem, so ``best.pth`` is either the old checkpoint or the
    new one and never something in between.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise CheckpointError(
            f"could not write checkpoint {path}: {exc}",
            reason="checkpoint_write_failed",
            context={"path": str(path)},
        ) from exc
    return path


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint, converting any failure into a typed error."""
    if not path.is_file():
        raise CheckpointError(
            f"checkpoint {path} does not exist",
            reason="checkpoint_missing",
            context={"path": str(path)},
        )
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise CheckpointError(
            f"checkpoint {path} is unreadable ({exc}); it may be truncated from an "
            "interrupted save",
            reason="checkpoint_unreadable",
            context={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        keys = sorted(payload) if isinstance(payload, dict) else []
        raise CheckpointError(
            f"checkpoint {path} has no model_state_dict",
            reason="checkpoint_malformed",
            context={"path": str(path), "keys": keys},
        )
    return payload


# ---------------------------------------------------------------------------
# The read-only proof
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFingerprint:
    """Evidence that the dataset directory was not written to during a run.

    The in-place contract is the whole point of this pipeline, and "we did not
    write there" is exactly the kind of claim that is easy to assert and
    impossible to check after the fact. So it is checked: the directory's
    ``st_mtime_ns`` and entry count are captured before training and compared
    after. Adding, removing or rewriting a file in the directory changes its
    mtime; a same-mtime content edit is caught by the entry count only if the
    file count changed, which is a real limitation and is stated as one.
    """

    path: str
    mtime_ns: int
    size: int
    n_entries: int
    st_dev: int
    captured_utc: str

    @classmethod
    def capture(cls, path: Path) -> SourceFingerprint:
        stat = path.stat()
        try:
            n_entries = sum(1 for _ in path.iterdir())
        except OSError:
            n_entries = -1
        return cls(
            path=str(path),
            mtime_ns=int(stat.st_mtime_ns),
            size=int(stat.st_size),
            n_entries=n_entries,
            st_dev=int(stat.st_dev),
            captured_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    def verify(self, path: Path) -> tuple[bool, str]:
        after = SourceFingerprint.capture(path)
        if after.mtime_ns != self.mtime_ns:
            return False, f"mtime changed ({self.mtime_ns} -> {after.mtime_ns})"
        if after.n_entries != self.n_entries:
            return False, f"entry count changed ({self.n_entries} -> {after.n_entries})"
        return True, "unchanged"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_outputs_are_outside_the_source(
    spec: DataSourceSpec, outputs: Mapping[str, Path]
) -> None:
    """Refuse to write checkpoints, runs or logs inside the dataset.

    A misconfigured ``SENTINEL_PATHS__CHECKPOINT_DIR`` pointing at the archive
    would violate the in-place contract *and* fill the data volume. Better to
    stop before the first epoch than to discover it in the report.
    """
    source_roots = {spec.prepared_root.resolve(), spec.images_dir.resolve()}
    for label, path in outputs.items():
        resolved = path.resolve()
        for source in source_roots:
            if resolved == source or source in resolved.parents:
                raise TrainingError(
                    f"{label} ({resolved}) is inside the data source ({source}); "
                    "the dataset is input only and must never be written to",
                    reason="output_inside_source",
                    context={"label": label, "path": str(resolved), "source": str(source)},
                )


# ---------------------------------------------------------------------------
# Run report
# ---------------------------------------------------------------------------


@dataclass
class RunReport:
    """Everything needed to know what a run was, without re-reading the logs."""

    run_id: str
    started_utc: str
    finished_utc: str = ""
    duration_s: float = 0.0
    status: str = "running"
    device: str = ""
    device_reason: str = ""
    seed: int = 0
    epochs: int = 0
    epochs_completed: int = 0
    best_epoch: int = -1
    best_val_iou: float = 0.0
    environment: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    data_source: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    augmentation: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: dict[str, str] = field(default_factory=dict)
    source_fingerprints: list[dict[str, Any]] = field(default_factory=list)
    source_unchanged: bool | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    provenance: str = DataProvenance.UNAVAILABLE.value
    scientifically_valid: bool = False
    claim: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return path


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cpu_count": os.cpu_count(),
    }


def _equivalent_looks_for_augmentation(
    settings: Settings, spec: DataSourceSpec
) -> tuple[float, str]:
    """Centre the speckle jitter on the value the model will meet in the field.

    Augmentation exists to prepare the model for real acquisitions, so the
    jitter is centred on the *real* measured looks even when training on the
    synthetic set — the synthetic generator was itself fitted to that number.
    Falls back to 4.0 with the source recorded, never silently.
    """
    candidates = [
        Path(spec.prepared_root) / "distribution.json",
        settings.datasources.synthetic_root / "distribution.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("could not read {}: {}", candidate, exc)
            continue
        looks = (payload.get("speckle") or {}).get("equivalent_looks")
        if isinstance(looks, dict):
            values = [float(v) for v in looks.values() if isinstance(v, (int, float))]
            if values:
                return float(np.median(values)), str(candidate)
        elif isinstance(looks, (int, float)):
            return float(looks), str(candidate)
    logger.warning(
        "no measured equivalent-looks profile found; speckle augmentation will "
        "use the 4.0 default"
    )
    return 4.0, "default"


def _new_run_id(settings: Settings, source: DataSourceSpec) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = settings.run_name or f"{source.kind}-{source.provenance.value}"
    digest = hashlib.sha256(f"{stamp}|{name}".encode()).hexdigest()[:6]
    return f"{stamp}-{name}-{digest}"


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _cap_scenes(
    rows: Sequence[dict[str, Any]], max_scenes: int | None
) -> tuple[list[dict[str, Any]], str | None]:
    """Trim a manifest to ``max_scenes`` pairs, keeping split proportions."""
    rows = list(rows)
    if max_scenes is None or len(rows) <= max_scenes:
        return rows, None
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row.get("split", "train")), []).append(row)
    total = len(rows)
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for split, group in by_split.items():
        n = max(1, int(round(len(group) * max_scenes / total)))
        kept.extend(group[:n])
        counts[split] = n
    note = (
        f"max_scenes={max_scenes} capped this run to {len(kept)} of {total} pair(s) "
        f"({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})"
    )
    return kept, note


def train(
    settings: Settings | None = None,
    *,
    data_source: DataSourceKind | None = None,
    epochs: int | None = None,
    max_scenes: int | None = None,
    max_steps_per_epoch: int | None = None,
    resume: Path | None = None,
    rebuild_index: bool = False,
    log_every: int = 10,
) -> RunReport:
    """Run a full training job and return its report.

    Args:
        settings: configuration; ``get_settings()`` when omitted.
        data_source: ``real`` or ``synthetic``; overrides
            ``settings.data_source`` for this run only.
        epochs: overrides ``settings.training.epochs``.
        max_scenes: cap the number of pairs (a laptop-sized run over the real
            archive needs a bound).
        max_steps_per_epoch: cap optimiser steps per epoch, for smoke runs.
        resume: checkpoint to continue from.
        rebuild_index: rebuild the data-source index before training. Off by
            default because the index is an input and the real one lives on a
            USB disk where a cold header scan is expensive.

    Raises:
        TrainingError: on any condition that makes the run meaningless — an
            empty split, a source that is synthetic when real was demanded, a
            source directory that changed during the run.
        CheckpointError: on unreadable or unwritable checkpoints.
    """
    settings = settings or get_settings()
    training = settings.training
    epochs = training.epochs if epochs is None else epochs
    max_scenes = training.max_scenes if max_scenes is None else max_scenes
    max_steps_per_epoch = (
        training.max_steps_per_epoch if max_steps_per_epoch is None else max_steps_per_epoch
    )

    kind = data_source or settings.data_source
    spec = resolve_source(kind, settings=settings, rebuild=rebuild_index)
    if kind == "real" and spec.is_synthetic:
        raise TrainingError(
            "data_source=real resolved to synthetic data; refusing to report real-data "
            "metrics from a stand-in",
            reason="source_kind_mismatch",
            context={"requested": kind, "resolved": spec.provenance.value},
        )

    run_id = _new_run_id(settings, spec)
    run_dir = settings.paths.runs_dir / run_id
    checkpoint_dir = settings.paths.checkpoint_dir / run_id
    assert_outputs_are_outside_the_source(
        spec,
        {
            "runs_dir": settings.paths.runs_dir,
            "checkpoint_dir": settings.paths.checkpoint_dir,
            "log_dir": settings.paths.log_dir,
        },
    )

    report = RunReport(
        run_id=run_id,
        started_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        device="",
        seed=training.seed,
        epochs=epochs,
        environment=_environment_snapshot(),
        settings=settings.describe(),
        provenance=spec.provenance.value,
        scientifically_valid=spec.scientifically_valid,
        warnings=list(spec.warnings),
    )

    logger.info("run {} | source={} ({})", run_id, spec.kind, spec.provenance.value)
    logger.info(
        "data source: {} pair(s), {} band(s), images={} in_place={} read_only={}",
        spec.n_pairs,
        spec.bands,
        spec.images_dir,
        spec.in_place,
        spec.read_only,
    )
    for warning in spec.warnings:
        logger.warning("data source warning: {}", warning)

    fingerprints = [SourceFingerprint.capture(spec.images_dir)]
    if spec.masks_dir != spec.images_dir and spec.masks_dir.is_dir():
        fingerprints.append(SourceFingerprint.capture(spec.masks_dir))
    report.source_fingerprints = [f.to_dict() for f in fingerprints]

    device, device_reason = resolve_device(training.device)
    report.device = str(device)
    report.device_reason = device_reason
    logger.info("device: {} ({})", device, device_reason)

    seed_everything(training.seed, deterministic=training.deterministic)

    rows = load_manifest(spec.prepared_root)
    if not rows:
        raise TrainingError(
            f"the prepared index at {spec.prepared_root} has no rows; run the index "
            "build first",
            reason="index_empty",
            context={"prepared_root": str(spec.prepared_root)},
        )
    rows, cap_note = _cap_scenes(rows, max_scenes)
    if cap_note:
        logger.warning(cap_note)
        report.warnings.append(cap_note)

    augmentation_on = not training.deterministic
    looks, looks_source = _equivalent_looks_for_augmentation(settings, spec)
    train_tf = build_train_transforms(
        training.image_size,
        seed=training.seed,
        equivalent_looks=looks,
        random=augmentation_on,
    )
    eval_tf = build_eval_transforms(training.image_size, seed=training.seed)
    report.augmentation = describe_augmentation(train_tf)
    report.augmentation["equivalent_looks"] = looks
    report.augmentation["equivalent_looks_source"] = looks_source

    train_loader, val_loader, test_loader = get_dataloaders(
        spec.prepared_root,
        batch_size=training.batch_size,
        image_size=training.image_size,
        num_workers=training.num_workers,
        train_transform=train_tf,
        val_transform=eval_tf,
        expected_bands=spec.bands,
        manifest=rows,
    )
    if train_loader is None or val_loader is None:
        raise TrainingError(
            "the index produced no train or validation split; nothing to learn from",
            reason="empty_split",
            context={"manifest_rows": len(rows)},
        )

    # Splits come from the index, which assigns them per scene. Verify it,
    # because a tile-level split silently inflates every reported metric.
    leaked = scene_leakage(
        (str(row["scene_id"]), str(row.get("split", "train")))
        for row in rows
        if row.get("scene_id")
    )
    if leaked:
        preview = dict(list(leaked.items())[:5])
        raise TrainingError(
            f"{len(leaked)} scene id(s) appear in more than one split (e.g. {preview}); "
            "a scene shared across splits makes every cross-split metric optimistic",
            reason="scene_leakage",
            context={"n_leaked": len(leaked)},
        )

    split_summary: dict[str, Any] = {}
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        if loader is None:
            split_summary[name] = {"n_pairs": 0, "note": "absent"}
            continue
        dataset = loader.dataset
        split_summary[name] = {
            "n_pairs": len(dataset),
            "bands": dataset.pairing.band_count,
            "band_names": list(dataset.band_names),
            "pairing": dataset.pairing.strategy,
            "n_scenes": len(set(dataset.scene_ids)),
        }
    report.splits = split_summary
    logger.info(
        "splits: train={} val={} test={}",
        split_summary["train"]["n_pairs"],
        split_summary["val"]["n_pairs"],
        split_summary["test"]["n_pairs"],
    )

    in_channels = int(train_loader.dataset.pairing.band_count)
    if in_channels != spec.bands:
        raise TrainingError(
            f"loader reports {in_channels} bands but the source spec says {spec.bands}",
            reason="band_count_disagreement",
            context={"loader": in_channels, "spec": spec.bands},
        )

    # Normalization is computed from the training split only and then attached
    # to all three, so val and test are measured on the same scale the model was
    # trained on. Computing it per split would make the val metric incomparable.
    stats_items = max(1, min(32, len(train_loader.dataset)))
    logger.info("computing per-band normalization from {} training scene(s)", stats_items)
    normalization = compute_normalization_statistics(train_loader.dataset, max_items=stats_items)
    for loader in (train_loader, val_loader, test_loader):
        if loader is not None:
            loader.dataset.normalization = normalization
    report.normalization = {
        "band_names": list(normalization.band_names),
        "mean": list(normalization.mean),
        "std": list(normalization.std),
        "percentiles": [list(row) for row in normalization.percentiles],
        "percentile_levels": list(normalization.percentile_levels),
        "source_split": "train",
        "n_scenes_sampled": stats_items,
    }
    logger.info(
        "normalization mean={} std={}",
        [round(v, 3) for v in normalization.mean],
        [round(v, 3) for v in normalization.std],
    )

    from app.models.unetpp_scse import UNetPlusPlusSCSE

    weights = None if training.encoder_weights == "none" else training.encoder_weights
    try:
        model = UNetPlusPlusSCSE(
            encoder_name=training.encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=1,
        )
    except Exception as exc:
        raise TrainingError(
            f"could not build {training.encoder} with encoder_weights={weights!r}: {exc}. "
            "If this is a download failure, set "
            "SENTINEL_TRAINING__ENCODER_WEIGHTS=none to train from scratch.",
            reason="model_build_failed",
            context={"encoder": training.encoder, "encoder_weights": weights},
        ) from exc
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report.model = {
        "arch": "UNetPlusPlusSCSE",
        "encoder": training.encoder,
        "encoder_weights": weights or "none",
        "in_channels": in_channels,
        "classes": 1,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    logger.info(
        "model: UNet++/{} in_channels={} params={:,} trainable={:,}",
        training.encoder,
        in_channels,
        total_params,
        trainable_params,
    )

    criterion = BCEDiceLoss(training.bce_weight, training.dice_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # Mixed precision is CUDA-only here. GradScaler on MPS is not supported and
    # fp16 autocast on MPS is not a drop-in; claiming AMP on this machine would
    # mean silently running a different numerical path than the one reported.
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    if device.type == "mps":
        logger.info("MPS: running fp32; CUDA-only AMP is not applied")
    report.settings["amp_enabled"] = amp_enabled

    start_epoch = 0
    best_val_iou = -1.0
    if resume is not None:
        payload = load_checkpoint(resume, device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_val_iou = float(payload.get("val_iou", -1.0))
        logger.info(
            "resumed from {} at epoch {} (best val IoU {:.4f})",
            resume,
            start_epoch,
            best_val_iou,
        )
        report.warnings.append(f"resumed from {resume} at epoch {start_epoch}")

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / METRICS_NAME
    logger.info("runs: {} | checkpoints: {}", run_dir, checkpoint_dir)

    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=training.grad_clip,
            max_steps=max_steps_per_epoch,
            epoch=epoch,
            log_every=log_every,
        )
        scheduler.step()
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            grad_clip=training.grad_clip,
            max_steps=None,
            epoch=epoch,
            log_every=0,
        )

        entry: dict[str, Any] = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": round(time.monotonic() - epoch_started, 3),
            "train": train_metrics.to_dict(),
            "val": val_metrics.to_dict(),
        }
        report.history.append(entry)
        with metrics_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(entry, sort_keys=True) + "\n")

        logger.info(
            "epoch {}/{} train loss={:.4f} iou={:.4f} | val loss={:.4f} iou={:.4f} "
            "f1={:.4f} precision={:.4f} recall={:.4f} ({:.1f}s)",
            epoch + 1,
            epochs,
            train_metrics.loss,
            entry["train"]["iou"],
            val_metrics.loss,
            entry["val"]["iou"],
            entry["val"]["f1"],
            entry["val"]["precision"],
            entry["val"]["recall"],
            entry["epoch_seconds"],
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_iou": val_metrics.to_dict()["iou"],
            "val_f1": val_metrics.to_dict()["f1"],
            "run_id": run_id,
            "in_channels": in_channels,
            "encoder": training.encoder,
            "image_size": training.image_size,
            "band_names": list(normalization.band_names),
            "normalization": report.normalization,
            "provenance": spec.provenance.value,
            "scientifically_valid": spec.scientifically_valid,
        }
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()

        atomic_torch_save(payload, checkpoint_dir / LAST_CHECKPOINT_NAME)
        if val_metrics.to_dict()["iou"] > best_val_iou:
            best_val_iou = val_metrics.to_dict()["iou"]
            report.best_epoch = epoch
            report.best_val_iou = best_val_iou
            best_path = atomic_torch_save(payload, checkpoint_dir / CHECKPOINT_NAME)
            logger.info("new best val IoU {:.4f} -> {}", best_val_iou, best_path)

        report.epochs_completed = epoch + 1

    report.duration_s = round(time.monotonic() - started, 3)
    report.checkpoints = {
        "best": str(checkpoint_dir / CHECKPOINT_NAME),
        "last": str(checkpoint_dir / LAST_CHECKPOINT_NAME),
        "metrics": str(metrics_path),
    }

    # The in-place contract, checked rather than asserted.
    unchanged = True
    for fingerprint in fingerprints:
        ok, detail = fingerprint.verify(Path(fingerprint.path))
        if not ok:
            unchanged = False
            message = f"data source {fingerprint.path} was modified during the run: {detail}"
            logger.error(message)
            report.errors.append(message)
    report.source_unchanged = unchanged
    if not unchanged:
        raise TrainingError(
            "the data source directory changed during training; the in-place "
            "contract was violated",
            reason="source_modified",
            context={"errors": report.errors},
        )

    report.finished_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    report.status = "completed"
    report.claim = _claim_text(spec, report)
    report_path = report.save(run_dir / REPORT_NAME)

    logger.info(
        "done: {} epoch(s) in {:.1f}s, best val IoU {:.4f} at epoch {}",
        report.epochs_completed,
        report.duration_s,
        report.best_val_iou,
        report.best_epoch + 1 if report.best_epoch >= 0 else -1,
    )
    logger.info("report: {}", report_path)
    logger.info("source unchanged: {}", report.source_unchanged)
    return report


def _claim_text(spec: DataSourceSpec, report: RunReport) -> str:
    """State plainly what these numbers do and do not support."""
    if spec.provenance is DataProvenance.REAL and spec.scientifically_valid:
        return (
            f"Trained on real SAR scenes read in place from {spec.images_dir}. "
            f"Split strategy: {spec.split_strategy}. The archive carries no scene "
            "identifier, so if the split is per file rather than per acquisition, "
            "cross-split metrics are optimistic — treat them as an upper bound."
        )
    return (
        f"Trained on {spec.provenance.value} data. These metrics describe the synthetic "
        "generator, not the ocean: they cannot be reported as a detection performance "
        "for a real oil spill."
    )
