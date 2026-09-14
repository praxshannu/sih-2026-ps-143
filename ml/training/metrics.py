"""Segmentation metrics and out-of-distribution diagnostics.

Every function here is pure: numpy/torch in, plain Python out, no I/O, no
logging, no global state. That matters because a metric that quietly writes a
file or mutates its input cannot be trusted as evidence.

Degenerate-case policy (stated explicitly, because it is the difference
between an honest 1.0 and a fabricated one):

* ``iou`` of two identical masks is ``1.0``; of two disjoint masks ``0.0``.
* when *both* prediction and target are empty the result is vacuously
  perfect (``1.0``) — so ``support`` (the number of target positives) is
  always reported alongside it. A 1.0 with ``support == 0`` must never be
  read as skill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, torch is optional at runtime
    import torch

ArrayLike = Union[np.ndarray, "torch.Tensor", list[Any], tuple[Any, ...]]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Array interop
# ---------------------------------------------------------------------------


def to_numpy(values: ArrayLike) -> np.ndarray:
    """Accept numpy arrays, torch tensors (any device/grad) or nested lists."""
    if hasattr(values, "detach"):
        values = values.detach()  # type: ignore[union-attr]
    if hasattr(values, "cpu"):
        values = values.cpu()  # type: ignore[union-attr]
    if hasattr(values, "numpy"):
        values = values.numpy()  # type: ignore[union-attr]
    return np.asarray(values)


def binarise(
    values: ArrayLike,
    threshold: float = 0.5,
    logits: bool = False,
) -> np.ndarray:
    """Convert probabilities or raw logits to a 0/1 mask.

    ``logits=True`` applies a sigmoid first; passing logits with
    ``logits=False`` is the classic silent bug that makes a threshold of 0.5
    label half the ocean as oil, so it is an explicit argument.
    """
    arr = to_numpy(values).astype(np.float64, copy=False)
    if logits:
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))
    return (arr > threshold).astype(np.int64)


# ---------------------------------------------------------------------------
# Confusion matrix and binary metrics
# ---------------------------------------------------------------------------


def confusion_matrix(
    pred: ArrayLike,
    target: ArrayLike,
    num_classes: int = 2,
    threshold: float = 0.5,
    logits: bool = False,
) -> np.ndarray:
    """Dense confusion matrix, shape ``(num_classes, num_classes)``.

    ``matrix[i, j]`` counts pixels whose *true* class is ``i`` and whose
    *predicted* class is ``j``. ``matrix.sum()`` is therefore exactly the
    number of pixels compared.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    p = (
        binarise(pred, threshold, logits)
        if num_classes == 2
        else to_numpy(pred).ravel().astype(np.int64)
    ).ravel()
    t = (
        binarise(target, threshold, logits)
        if num_classes == 2
        else to_numpy(target).ravel().astype(np.int64)
    ).ravel()
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    if p.size == 0:
        raise ValueError("cannot build a confusion matrix from empty arrays")
    if p.min() < 0 or t.min() < 0 or p.max() >= num_classes or t.max() >= num_classes:
        raise ValueError(
            "label values outside [0, num_classes); got "
            f"pred [{p.min()}, {p.max()}], target [{t.min()}, {t.max()}]"
        )
    counts = np.bincount(t * num_classes + p, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).astype(np.int64)


@dataclass(frozen=True)
class BinaryCounts:
    """Raw pixel counts — the evidence that every rate below is derived from."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n_pixels(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def support(self) -> int:
        """Number of true positive pixels (target positives)."""
        return self.tp + self.fn

    @property
    def predicted_positives(self) -> int:
        return self.tp + self.fp


def binary_counts(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    logits: bool = False,
) -> BinaryCounts:
    p = binarise(pred, threshold, logits).ravel()
    t = binarise(target, threshold, logits).ravel()
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs target {t.shape}")
    tp = int(np.count_nonzero((p == 1) & (t == 1)))
    fp = int(np.count_nonzero((p == 1) & (t == 0)))
    fn = int(np.count_nonzero((p == 0) & (t == 1)))
    tn = int(np.count_nonzero((p == 0) & (t == 0)))
    return BinaryCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def _safe_ratio(numerator: float, denominator: float, default: float) -> float:
    return float(numerator / denominator) if denominator > 0 else default


def binary_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    logits: bool = False,
) -> dict[str, float]:
    """IoU / precision / recall / F1 / accuracy for a binary segmentation."""
    counts = binary_counts(pred, target, threshold, logits)
    return metrics_from_counts(counts)


def metrics_from_counts(counts: BinaryCounts) -> dict[str, float]:
    """Derive every rate from raw counts in one place, so they cannot diverge."""
    union = counts.tp + counts.fp + counts.fn
    degenerate = union == 0  # nothing predicted and nothing labelled
    perfect = 1.0
    precision = _safe_ratio(counts.tp, counts.tp + counts.fp, perfect if degenerate else 0.0)
    recall = _safe_ratio(counts.tp, counts.tp + counts.fn, perfect if degenerate else 0.0)
    f1 = _safe_ratio(2 * counts.tp, 2 * counts.tp + counts.fp + counts.fn, perfect if degenerate else 0.0)
    return {
        "iou": _safe_ratio(counts.tp, union, perfect),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": _safe_ratio(counts.tp + counts.tn, counts.n_pixels, 0.0),
        "specificity": _safe_ratio(counts.tn, counts.tn + counts.fp, 0.0),
        "tp": float(counts.tp),
        "fp": float(counts.fp),
        "fn": float(counts.fn),
        "tn": float(counts.tn),
        "support": float(counts.support),
        "n_pixels": float(counts.n_pixels),
        "degenerate_no_positive_evidence": 1.0 if degenerate else 0.0,
    }


def per_class_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    num_classes: int = 2,
    class_names: tuple[str, ...] | None = None,
    threshold: float = 0.5,
    logits: bool = False,
) -> dict[str, dict[str, float]]:
    """Per-class IoU/precision/recall/F1 plus macro and mean (mIoU) averages."""
    matrix = confusion_matrix(pred, target, num_classes, threshold, logits)
    names = class_names or tuple(f"class_{i}" for i in range(num_classes))
    if len(names) != num_classes:
        raise ValueError(f"class_names has {len(names)} entries, expected {num_classes}")

    per_class: dict[str, dict[str, float]] = {}
    ious: list[float] = []
    for index, name in enumerate(names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        tn = int(matrix.sum() - tp - fp - fn)
        stats = metrics_from_counts(BinaryCounts(tp=tp, fp=fp, fn=fn, tn=tn))
        per_class[name] = stats
        if tp + fn > 0:  # only classes present in the target enter the mean
            ious.append(stats["iou"])

    per_class["mean_iou"] = {"iou": float(np.mean(ious)) if ious else 0.0}
    return per_class


def segmentation_report(
    pred: ArrayLike,
    target: ArrayLike,
    num_classes: int = 2,
    class_names: tuple[str, ...] | None = None,
    threshold: float = 0.5,
    logits: bool = False,
) -> dict[str, Any]:
    """Full report: binary metrics, per-class metrics, confusion matrix."""
    matrix = confusion_matrix(pred, target, num_classes, threshold, logits)
    return {
        "binary": binary_metrics(pred, target, threshold, logits),
        "per_class": per_class_metrics(pred, target, num_classes, class_names, threshold, logits),
        "confusion_matrix": matrix.astype(int).tolist(),
        "threshold": threshold,
        "num_classes": num_classes,
    }


# ---------------------------------------------------------------------------
# Scene statistics and out-of-distribution detection
# ---------------------------------------------------------------------------


def amplitude_to_db(amplitude: ArrayLike, epsilon: float = 1e-10) -> np.ndarray:
    """Linear SAR amplitude -> decibels. No clamping beyond the epsilon floor."""
    arr = to_numpy(amplitude).astype(np.float64, copy=False)
    return 10.0 * np.log10(np.clip(arr, epsilon, None))


@dataclass(frozen=True)
class SceneStatistics:
    """Per-band summary of one scene, used for the OOD check."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    percentiles: tuple[tuple[float, ...], ...]
    band_names: tuple[str, ...]
    percentile_levels: tuple[float, ...] = (1.0, 50.0, 99.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_names": list(self.band_names),
            "mean": list(self.mean),
            "std": list(self.std),
            "percentile_levels": list(self.percentile_levels),
            "percentiles": [list(p) for p in self.percentiles],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneStatistics:
        return cls(
            mean=tuple(float(v) for v in payload["mean"]),
            std=tuple(float(v) for v in payload["std"]),
            percentiles=tuple(tuple(float(v) for v in row) for row in payload["percentiles"]),
            band_names=tuple(str(v) for v in payload["band_names"]),
            percentile_levels=tuple(float(v) for v in payload.get("percentile_levels", (1, 50, 99))),
        )

    def unavailable(self) -> bool:
        """True when a statistic could not be computed (non-finite values)."""
        return any(not math.isfinite(v) for v in (*self.mean, *self.std))


#: Normalization stats are structurally the same object as scene statistics;
#: the alias documents intent at the call site.
NormalizationStats = SceneStatistics


def compute_scene_statistics(
    image: ArrayLike,
    band_names: tuple[str, ...] = ("VV", "VH"),
    percentiles: tuple[float, ...] = (1.0, 50.0, 99.0),
    apply_db: bool = False,
) -> SceneStatistics:
    """Per-band mean/std/percentiles of a ``(C, H, W)`` or ``(H, W)`` image.

    Bands are never collapsed: a 2-band VV/VH scene yields two entries. If a
    band is entirely non-finite the statistic is reported as ``nan`` rather
    than being dropped — an unavailable value with a reason beats a number
    nobody can audit.
    """
    arr = to_numpy(image)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"expected a (C, H, W) or (H, W) array, got shape {arr.shape}")
    if apply_db:
        arr = amplitude_to_db(arr)

    means: list[float] = []
    stds: list[float] = []
    pcts: list[tuple[float, ...]] = []
    for band in arr:
        flat = band.reshape(-1).astype(np.float64, copy=False)
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            means.append(float("nan"))
            stds.append(float("nan"))
            pcts.append(tuple(float("nan") for _ in percentiles))
            continue
        means.append(float(finite.mean()))
        stds.append(float(finite.std()))
        pcts.append(tuple(float(v) for v in np.percentile(finite, percentiles)))

    names = band_names if len(band_names) == arr.shape[0] else tuple(
        f"band_{i}" for i in range(arr.shape[0])
    )
    return SceneStatistics(
        mean=tuple(means),
        std=tuple(stds),
        percentiles=tuple(pcts),
        band_names=names,
        percentile_levels=tuple(percentiles),
    )


@dataclass(frozen=True)
class OodReport:
    """Result of comparing a scene against the training normalization stats."""

    is_ood: bool
    threshold_sigma: float
    max_sigma: float
    per_band_sigma: tuple[float, ...]
    band_names: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    confidence_multiplier: float = 1.0
    evaluated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ood": self.is_ood,
            "evaluated": self.evaluated,
            "threshold_sigma": self.threshold_sigma,
            "max_sigma": self.max_sigma,
            "per_band_sigma": list(self.per_band_sigma),
            "band_names": list(self.band_names),
            "reasons": list(self.reasons),
            "confidence_multiplier": self.confidence_multiplier,
        }


def check_out_of_distribution(
    scene: SceneStatistics,
    reference: SceneStatistics,
    sigma_threshold: float = 3.0,
    confidence_penalty: float = 0.5,
) -> OodReport:
    """Flag a scene whose per-band statistics sit far from the training set.

    Two independent z-scores are computed per band and the worse one wins:

    * mean vs ``reference.std`` (the training spread);
    * median-vs-median vs a quarter of the reference p1..p99 range, which
      catches a shift in *level* even when the std happens to match.

    Anything above ``sigma_threshold`` marks the scene OOD and multiplies the
    reported confidence by ``confidence_penalty`` — the platform must never
    present a confident number for a scene the model has never seen.
    """
    if sigma_threshold <= 0:
        raise ValueError("sigma_threshold must be positive")
    if scene.unavailable() or reference.unavailable():
        return OodReport(
            is_ood=False,
            threshold_sigma=sigma_threshold,
            max_sigma=float("nan"),
            per_band_sigma=(),
            band_names=(),
            reasons=(
                "OOD check unavailable: NaN in scene or reference statistics "
                "(a band contained no finite pixels).",
            ),
            confidence_multiplier=1.0,
            evaluated=False,
        )

    n = min(len(scene.mean), len(reference.mean))
    if n == 0:
        return OodReport(
            is_ood=False,
            threshold_sigma=sigma_threshold,
            max_sigma=float("nan"),
            per_band_sigma=(),
            band_names=(),
            reasons=("OOD check unavailable: no comparable bands.",),
            confidence_multiplier=1.0,
            evaluated=False,
        )

    zs: list[float] = []
    reasons: list[str] = []
    names = scene.band_names[:n]
    for index in range(n):
        band = names[index] if index < len(names) else f"band_{index}"
        spread = max(reference.std[index], _EPS)
        z_mean = abs(scene.mean[index] - reference.mean[index]) / spread

        scene_p = scene.percentiles[index] if index < len(scene.percentiles) else ()
        ref_p = reference.percentiles[index] if index < len(reference.percentiles) else ()
        z_pct = 0.0
        if scene_p and ref_p:
            half = min(len(scene_p), len(ref_p)) // 2
            scale = max((ref_p[-1] - ref_p[0]) / 4.0, _EPS)
            z_pct = abs(scene_p[half] - ref_p[half]) / scale

        z = max(z_mean, z_pct)
        zs.append(z)
        if z > sigma_threshold:
            reasons.append(
                f"{band}: {z:.2f} sigma from training statistics "
                f"(mean z={z_mean:.2f}, median z={z_pct:.2f}, limit {sigma_threshold:g})"
            )

    is_ood = bool(zs) and max(zs) > sigma_threshold
    return OodReport(
        is_ood=is_ood,
        threshold_sigma=sigma_threshold,
        max_sigma=float(max(zs)) if zs else float("nan"),
        per_band_sigma=tuple(zs),
        band_names=tuple(names),
        reasons=tuple(reasons),
        confidence_multiplier=confidence_penalty if is_ood else 1.0,
        evaluated=True,
    )


def adjust_confidence(confidence: float, report: OodReport) -> tuple[float, str]:
    """Apply the OOD penalty to a confidence, returning ``(value, note)``."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    if not report.evaluated:
        return confidence, "OOD check unavailable — confidence not adjusted."
    if not report.is_ood:
        return confidence, "Scene within training distribution."
    adjusted = max(0.0, min(1.0, confidence * report.confidence_multiplier))
    return adjusted, (
        f"Scene flagged out-of-distribution ({report.max_sigma:.2f} sigma > "
        f"{report.threshold_sigma:g}); confidence lowered "
        f"{confidence:.3f} -> {adjusted:.3f}."
    )


@dataclass(frozen=True)
class MetricBundle:
    """Container used by the trainers to persist metrics with provenance."""

    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: str = "real"
    scientifically_valid: bool = True
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.metrics)
        payload["provenance"] = self.provenance
        payload["scientifically_valid"] = self.scientifically_valid
        payload["notes"] = list(self.notes)
        return payload
