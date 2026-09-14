"""Explainability for the deterministic Tier-A detector.

**This is not Grad-CAM, and it does not pretend to be.** Grad-CAM is a
gradient-weighted activation map: it needs a trained network, a forward pass
whose class score you can differentiate, and a backward pass to get
:dL/dA. The Tier-A detector is a deterministic operator — a Lee-sigma filter,
two adaptive-threshold gates, morphology and a connected-components pass.
There is no loss, no activation tensor and no gradient, so there is nothing for
Grad-CAM to compute. Emitting a blurry heatmap and calling it Grad-CAM would be
exactly the kind of authoritative-looking nothing this platform exists to
avoid.

What a deterministic operator *can* explain honestly, this module produces:

* **A per-pixel evidence map** — the dB contrast of every pixel against its
  local sea background, normalised to 0–1, masked to the pixels that passed
  **both** threshold gates. A pixel that failed a gate has evidence 0, and the
  response says which gate it failed.
* **A per-detection audit** — which gates passed, every penalty that was
  applied (or would have rejected the blob), and the exact contribution each
  term made to the confidence score. The numbers in the audit are the same
  numbers used to compute the confidence, not a post-hoc story.

The real Grad-CAM implementation lives in
:mod:`app.processors.unet_path` and is used only when a trained UNet++
checkpoint actually exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Public name of this explanation method. Deliberately not "grad-cam".
EVIDENCE_METHOD = "deterministic_evidence_map"

#: A pixel this many dB below its local background gets full contrast evidence.
DEFAULT_CONTRAST_SCALE_DB = 6.0

#: Default size of the downsampled evidence grid shipped in the JSON response.
DEFAULT_GRID_SIDE = 48


@dataclass(frozen=True)
class EvidenceConfig:
    contrast_scale_dB: float = DEFAULT_CONTRAST_SCALE_DB
    grid_side: int = DEFAULT_GRID_SIDE


@dataclass
class EvidenceMap:
    """Per-pixel evidence for a whole scene (float32, 0..1, NaN outside ocean)."""

    values: np.ndarray
    gate_local: np.ndarray
    gate_scene: np.ndarray
    contrast_dB: np.ndarray

    def inside(self, mask: np.ndarray) -> np.ndarray:
        return self.values[mask]

    def grid(self, block: int | None = None, side: int = DEFAULT_GRID_SIDE) -> np.ndarray:
        """Block-max downsample of the evidence for JSON transport."""
        return block_reduce_max(self.values, block or _block_for(self.values.shape, side))


def _block_for(shape: tuple[int, ...], side: int) -> int:
    return max(1, int(np.ceil(max(shape[0], shape[1]) / max(side, 1))))


def block_reduce_max(values: np.ndarray, block: int) -> np.ndarray:
    """Max-pool ``values`` into ``block``-sized cells (NaN -> 0).

    Max (not mean) is used because the question the grid answers is "was there
    evidence anywhere in this cell", not "what was the average".
    """
    if block <= 1:
        return np.nan_to_num(values, nan=0.0)
    h, w = values.shape
    hh, ww = h // block, w // block
    if hh == 0 or ww == 0:
        return np.nan_to_num(values, nan=0.0)
    trimmed = values[: hh * block, : ww * block]
    pooled = trimmed.reshape(hh, block, ww, block).max(axis=(1, 3))
    return np.nan_to_num(pooled, nan=0.0)


def build_evidence_map(
    vv_db: np.ndarray,
    ocean: np.ndarray,
    local_bg: np.ndarray,
    *,
    noise_sigma_dB: float,
    sea_baseline_dB: float,
    k: float,
    min_scene_contrast_dB: float,
    config: EvidenceConfig | None = None,
) -> EvidenceMap:
    """Per-pixel evidence: contrast vs local sea, gated by the detector's rules.

    ``evidence = clip((local_bg - dB) / scale, 0, 1)`` for pixels passing both
    Solberg gates, ``0`` everywhere else.
    """
    cfg = config or EvidenceConfig()
    with np.errstate(invalid="ignore"):
        contrast = np.where(ocean, local_bg - vv_db, np.nan).astype(np.float32)
    gate_local = ocean & np.isfinite(contrast) & (contrast >= k * noise_sigma_dB)
    gate_scene = ocean & np.isfinite(vv_db) & ((sea_baseline_dB - vv_db) >= min_scene_contrast_dB)
    score = np.clip(contrast / max(cfg.contrast_scale_dB, 1e-6), 0.0, 1.0)
    values = np.where(gate_local & gate_scene, score, 0.0).astype(np.float32)
    values[~ocean] = 0.0
    return EvidenceMap(
        values=values,
        gate_local=gate_local,
        gate_scene=gate_scene,
        contrast_dB=contrast,
    )


@dataclass
class Penalty:
    """One named influence on a detection's confidence. Never anonymous."""

    name: str
    value: float
    effect: str  # "reject" | "reduce" | "flag" | "support"
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(float(self.value), 4),
            "effect": self.effect,
            "detail": self.detail,
        }


@dataclass
class DetectionEvidence:
    """Full audit trail for one detection."""

    method: str = EVIDENCE_METHOD
    gates: dict[str, bool] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)
    penalties: list[Penalty] = field(default_factory=list)
    pixel_evidence: dict[str, float | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "gates": dict(self.gates),
            "components": {k: round(float(v), 4) for k, v in self.components.items()},
            "penalties": [p.as_dict() for p in self.penalties],
            "pixel_evidence": dict(self.pixel_evidence),
        }


def detection_evidence(
    *,
    contrast_dB: float,
    scene_contrast_dB: float,
    noise_sigma_dB: float,
    area_km2: float,
    elongation: float,
    valid_fraction: float,
    wind_viability: str,
    confidence_terms: dict[str, float],
    min_dB_contrast: float,
    max_elongation: float,
    min_area_km2: float,
    pixel_evidence: dict[str, float | None],
) -> DetectionEvidence:
    """Assemble the audit for one surviving detection.

    ``confidence_terms`` must be the *same* terms used to build the confidence
    score (contrast / validity / size) so the explanation can never drift from
    the number it explains.
    """
    gates = {
        "gate_local_k_sigma_below_local_background": bool(contrast_dB >= min_dB_contrast),
        "gate_scene_below_clean_sea_baseline": bool(scene_contrast_dB > 0.0),
        "gate_min_area_km2": bool(area_km2 >= min_area_km2),
        "gate_max_elongation": bool(elongation <= max_elongation),
    }
    penalties: list[Penalty] = [
        Penalty(
            name="local_contrast",
            value=contrast_dB,
            effect="support",
            detail=(
                f"{contrast_dB:.2f} dB below the local sea background "
                f"(speckle sigma {noise_sigma_dB:.2f} dB)"
            ),
        ),
        Penalty(
            name="scene_contrast",
            value=scene_contrast_dB,
            effect="support",
            detail=f"{scene_contrast_dB:.2f} dB below the scene clean-sea baseline (p75)",
        ),
    ]
    if contrast_dB < min_dB_contrast:
        penalties.append(
            Penalty(
                name="contrast_shortfall",
                value=min_dB_contrast - contrast_dB,
                effect="reject",
                detail=(
                    f"contrast {contrast_dB:.2f} dB is below the {min_dB_contrast:.2f} dB "
                    "minimum — indistinguishable from speckle"
                ),
            )
        )
    if elongation > max_elongation:
        penalties.append(
            Penalty(
                name="elongation",
                value=elongation,
                effect="reject",
                detail=(
                    f"elongation {elongation:.2f} exceeds {max_elongation:.2f} — shape is a "
                    "wind streak, not a slick"
                ),
            )
        )
    else:
        penalties.append(
            Penalty(
                name="elongation",
                value=elongation,
                effect="support",
                detail=f"elongation {elongation:.2f} is within the {max_elongation:.2f} limit",
            )
        )
    if area_km2 < min_area_km2:
        penalties.append(
            Penalty(
                name="area_below_minimum",
                value=area_km2,
                effect="reject",
                detail=f"area {area_km2:.4f} km² is below the {min_area_km2:.4f} km² floor",
            )
        )
    if wind_viability != "OK":
        penalties.append(
            Penalty(
                name="wind_forcing",
                value=0.0,
                effect="flag",
                detail=(
                    f"wind viability {wind_viability}: look-alike discrimination needs a wind "
                    "speed in the 2–10 m/s band and none was supplied"
                ),
            )
        )
    if valid_fraction < 0.5:
        penalties.append(
            Penalty(
                name="low_valid_fraction",
                value=valid_fraction,
                effect="reduce",
                detail=(
                    f"only {valid_fraction:.1%} of the scene is usable ocean; the validity "
                    "term lowers confidence accordingly"
                ),
            )
        )
    return DetectionEvidence(
        gates=gates,
        components={k: float(v) for k, v in confidence_terms.items()},
        penalties=penalties,
        pixel_evidence=pixel_evidence,
    )


def explanation_block(
    *,
    evidence: EvidenceMap | None,
    grid: np.ndarray | None,
    bbox_wsen: list[float] | None,
    per_detection: list[dict[str, Any]],
    gradcam_used: bool,
) -> dict[str, Any]:
    """The ``explanation`` block of an inference response.

    ``gradcam_used`` is reported explicitly so a consumer can never mistake the
    deterministic evidence map for a gradient-based saliency map.
    """
    block: dict[str, Any] = {
        "method": "gradcam" if gradcam_used else EVIDENCE_METHOD,
        "gradcam": gradcam_used,
        "not_gradcam_reason": (
            None
            if gradcam_used
            else (
                "Grad-CAM requires gradients of a class score through a trained network. "
                "The active detector is the deterministic Tier-A operator and no trained "
                "UNet++ checkpoint exists on this host, so no gradient exists to weight. "
                "This map is per-pixel dB contrast against the local sea, gated by the same "
                "two thresholds the detector used."
            )
        ),
        "components": [
            "db_contrast_vs_local_sea",
            "gate_local_k_sigma",
            "gate_scene_baseline",
        ],
        "per_detection": per_detection,
    }
    if grid is not None and evidence is not None:
        block["per_pixel"] = {
            "grid": {
                "shape": [int(grid.shape[0]), int(grid.shape[1])],
                "block_reduction": "max",
                "value_range": [0.0, 1.0],
                "bbox_wsen": bbox_wsen,
                "values": [[round(float(v), 3) for v in row] for row in grid],
            },
            "coverage": {
                "evidence_pixels": int(np.count_nonzero(evidence.values > 0)),
                "gate_local_pixels": int(np.count_nonzero(evidence.gate_local)),
                "gate_scene_pixels": int(np.count_nonzero(evidence.gate_scene)),
                "max_evidence": (
                    float(np.nanmax(evidence.values)) if evidence.values.size else 0.0
                ),
                "mean_evidence": (
                    float(np.nanmean(evidence.values)) if evidence.values.size else 0.0
                ),
            },
        }
    return block
