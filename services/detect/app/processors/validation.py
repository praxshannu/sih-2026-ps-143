"""Pre-flight validation of a SAR scene, before any inference runs.

Nothing in SENTINEL may emit a detection from a scene that was never checked:
a silently-empty result and a raw traceback are both worse than an honest
refusal. This module answers, in one call, whether a GeoTIFF is a scene the
deterministic Tier-A detector can legitimately run on — and when it is not,
it names the reason with a machine-readable code.

Invalidity is reported as ``state == "invalid_scene"`` with a ``reason_code``
from :data:`REASON_CODES`, never as an exception and never as an empty
detection list.

Checks, in order (first failure wins):

``file_not_found``       path does not exist (or is not a regular file)
``unreadable_raster``    rasterio cannot open it (corrupt / not a raster)
``insufficient_bands``   fewer than 3 bands (needs sigma0 VV, VH, dataMask)
``missing_crs``          no coordinate reference system — not georeferenced
``missing_geotransform``  affine transform is degenerate (zero pixel size)
``scene_too_small``      smaller than ``min_side_px`` on either axis
``scene_too_large``      more than ``max_pixels`` (memory guard, see below)
``unsupported_dtype``    bands are not numeric
``no_valid_pixels``      (almost) no finite, unmasked sigma0 samples

The ``scene_too_large`` guard exists because the detector needs *global* scene
statistics (the clean-sea dB percentile and the speckle sigma). Those cannot be
computed from a tiled pass without changing the verified numbers, so the
raster is held once in memory; the guard is what makes that safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

STATE_OK = "ok"
STATE_INVALID_SCENE = "invalid_scene"

#: The deterministic detector needs sigma0 VV, sigma0 VH and a data mask.
REQUIRED_BANDS = 3

REASON_CODES = (
    "file_not_found",
    "not_a_file",
    "unreadable_raster",
    "insufficient_bands",
    "missing_crs",
    "missing_geotransform",
    "scene_too_small",
    "scene_too_large",
    "unsupported_dtype",
    "no_valid_pixels",
)

#: Resolution of the decimated probe used for the valid-pixel check.
PROBE_SIDE = 512

_DEFAULT_MIN_SIDE_PX = 64
_DEFAULT_MAX_PIXELS = 80_000_000  # ~320 MB per float32 band
_DEFAULT_MIN_VALID_FRACTION = 0.01


@dataclass(frozen=True)
class SceneValidation:
    """Outcome of the pre-flight check. Serialisable to JSON."""

    valid: bool
    state: str = STATE_OK
    reason_code: str | None = None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "state": self.state,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "details": dict(self.details),
        }


def _invalid(code: str, reason: str, **details: Any) -> SceneValidation:
    return SceneValidation(
        valid=False,
        state=STATE_INVALID_SCENE,
        reason_code=code,
        reason=reason,
        details=details,
    )


def validate_scene(
    tif_path: str | Path,
    *,
    min_bands: int = REQUIRED_BANDS,
    min_side_px: int = _DEFAULT_MIN_SIDE_PX,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    min_valid_fraction: float = _DEFAULT_MIN_VALID_FRACTION,
) -> SceneValidation:
    """Validate a GeoTIFF *before* any inference is attempted.

    Args:
        tif_path: path to the scene.
        min_bands: required band count (VV, VH, dataMask = 3).
        min_side_px: reject anything smaller than this on either axis.
        max_pixels: memory guard; larger scenes are refused, not truncated.
        min_valid_fraction: reject scenes with less than this share of finite,
            unmasked sigma0 samples.

    Returns:
        A :class:`SceneValidation`. ``valid`` is the only field a caller needs
        to branch on; ``reason_code`` is stable and machine-readable.
    """
    path = Path(tif_path)

    if not path.exists():
        return _invalid("file_not_found", f"scene file not found: {path}", path=str(path))
    if not path.is_file():
        return _invalid("not_a_file", f"scene path is not a regular file: {path}", path=str(path))

    try:
        with rasterio.open(path) as src:
            count = int(src.count)
            width = int(src.width)
            height = int(src.height)
            crs = src.crs
            transform = src.transform
            dtypes = list(src.dtypes)
            descriptions = [d or "" for d in (src.descriptions or ())]

            if count < min_bands:
                return _invalid(
                    "insufficient_bands",
                    (
                        f"scene has {count} band(s); the detector needs {min_bands} "
                        "(sigma0_VV_linear, sigma0_VH_linear, dataMask)"
                    ),
                    band_count=count,
                    required_bands=min_bands,
                )
            if crs is None:
                return _invalid(
                    "missing_crs",
                    "scene has no coordinate reference system — it is not georeferenced, "
                    "so detections cannot be placed on the globe",
                    band_count=count,
                )
            if not (math.isfinite(transform.a) and math.isfinite(transform.e)) or (
                transform.a == 0.0 or transform.e == 0.0
            ):
                return _invalid(
                    "missing_geotransform",
                    "scene has a degenerate affine transform (zero pixel size) — "
                    "pixels cannot be mapped to ground coordinates",
                    transform=tuple(float(v) for v in transform)[:6],
                )
            if min(width, height) < min_side_px:
                return _invalid(
                    "scene_too_small",
                    f"scene is {width}x{height} px; the smallest usable side is "
                    f"{min_side_px} px",
                    width=width,
                    height=height,
                    min_side_px=min_side_px,
                )
            if width * height > max_pixels:
                return _invalid(
                    "scene_too_large",
                    f"scene is {width}x{height} = {width * height} px, above the "
                    f"{max_pixels} px guard; split it before running detection",
                    width=width,
                    height=height,
                    max_pixels=max_pixels,
                )
            if any(not np.issubdtype(np.dtype(d), np.number) for d in dtypes):
                return _invalid(
                    "unsupported_dtype",
                    f"scene band dtypes {dtypes} are not numeric",
                    dtypes=dtypes,
                )

            valid_fraction = _probe_valid_fraction(
                src, min(width, PROBE_SIDE), min(height, PROBE_SIDE)
            )
            if valid_fraction < min_valid_fraction:
                return _invalid(
                    "no_valid_pixels",
                    f"only {valid_fraction:.4%} of probe pixels carry finite, unmasked "
                    "sigma0 — there is nothing to detect in this scene",
                    valid_fraction=round(valid_fraction, 6),
                    min_valid_fraction=min_valid_fraction,
                )

    except RasterioIOError as exc:
        return _invalid("unreadable_raster", f"rasterio could not open the scene: {exc}")
    except Exception as exc:  # noqa: BLE001 - validation must never raise
        return _invalid(
            "unreadable_raster",
            f"unexpected error while inspecting the scene: {type(exc).__name__}: {exc}",
        )

    return SceneValidation(
        valid=True,
        state=STATE_OK,
        details={
            "path": str(path.resolve()),
            "width": width,
            "height": height,
            "band_count": count,
            "crs": crs.to_string() if crs is not None else None,
            "epsg": (crs.to_epsg() if crs is not None else None),
            "pixel_size": (abs(float(transform.a)), abs(float(transform.e))),
            "dtypes": dtypes,
            "band_descriptions": descriptions,
            "valid_fraction_probe": round(valid_fraction, 6),
        },
    )


def _probe_valid_fraction(src: rasterio.DatasetReader, probe_w: int, probe_h: int) -> float:
    """Fraction of probe pixels that are finite sigma0 AND inside the data mask.

    The probe is a decimated read, so a 2048² or a 40000² scene costs the same.
    ``dataMask`` is 1 for valid ocean and 0 for land/nodata, and band 1 carries
    the calibrated sigma0 the detector actually thresholds.
    """
    vv = src.read(1, out_shape=(probe_h, probe_w))
    mask = src.read(src.count, out_shape=(probe_h, probe_w))
    finite = np.isfinite(vv) & (vv > 0.0)
    masked = np.isfinite(mask) & (mask > 0)
    return float(np.count_nonzero(finite & masked) / max(finite.size, 1))
