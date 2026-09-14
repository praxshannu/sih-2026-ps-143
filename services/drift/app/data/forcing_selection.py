"""Per-field forcing selection record + synthetic-confidence accounting.

Why this module exists
----------------------
A drift answer is only as good as the forcing behind it, and the drift service
can be driven by CMEMS currents, ERA5 wind, GFS wind, a local NetCDF, or —
when nothing real is reachable — a labelled synthetic field. Those are not
interchangeable, so every response has to state, **per field**, which source
was chosen, whether it is real, and what time span it actually covers.

The second job is the confidence penalty. A synthetic field does not quietly
shrink the error bars; it visibly downgrades the confidence score *and* emits
a literal ``SYNTHETIC FORCING`` warning so the UI can detect it without
parsing prose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

# Literal marker the UI keys off. Never paraphrase it.
SYNTHETIC_WARNING = "SYNTHETIC FORCING"
SYNTHETIC_SOURCE = "synthetic_mock"

# Confidence is multiplicative-free: we subtract, then clamp. Two synthetic
# fields (wind AND current) must never leave a run looking defensible.
PENALTY_PER_SYNTHETIC_FIELD = 0.35
MAX_SYNTHETIC_PENALTY = 0.70

# status vocabulary — kept small so the UI can switch on it.
REAL_STAGED = "real_staged"  # dataset open, coverage known
REAL_CONFIGURED = "real_configured"  # credentials/mirror present, fetched on demand
SYNTHETIC = "synthetic_mock"  # labelled mock — never evidence
UNAVAILABLE = "unavailable"  # nothing served; callers must fail or degrade


def _iso(value: Any) -> str | None:
    """Best-effort ISO-8601 rendering of a numpy/py datetime."""
    if value is None:
        return None
    try:
        dt = value.item() if hasattr(value, "item") else value
    except Exception:  # pragma: no cover - defensive
        dt = value
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


@dataclass
class FieldSelection:
    """Which source won for one forcing field, and how far we trust it."""

    field: str  # "wind" | "current"
    source: str  # era5 | gfs | cmems | local_file | synthetic_mock
    provider: str  # concrete provider class name
    is_real: bool
    status: str  # REAL_STAGED | REAL_CONFIGURED | SYNTHETIC | UNAVAILABLE
    reason: str = ""
    dataset_id: str | None = None
    coverage_start_utc: str | None = None
    coverage_end_utc: str | None = None

    @property
    def is_synthetic(self) -> bool:
        """True when this field is not real data (mock, or nothing at all)."""
        return not self.is_real

    def coverage(self) -> dict[str, str | None]:
        return {
            "start_utc": self.coverage_start_utc,
            "end_utc": self.coverage_end_utc,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForcingSelection:
    """The two field selections plus the derived confidence downgrade."""

    wind: FieldSelection
    current: FieldSelection
    forcing_source: str  # legacy top-level label, kept stable for the UI
    extra_notes: list[str] = field(default_factory=list)

    @property
    def synthetic_fields(self) -> list[str]:
        return [f.field for f in (self.wind, self.current) if f.is_synthetic]

    @property
    def any_synthetic(self) -> bool:
        return bool(self.synthetic_fields)

    @property
    def confidence_penalty(self) -> float:
        n = len(self.synthetic_fields)
        return min(n * PENALTY_PER_SYNTHETIC_FIELD, MAX_SYNTHETIC_PENALTY)

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.any_synthetic:
            out.append(
                f"{SYNTHETIC_WARNING}: "
                + ", ".join(f"{f.field}={f.source}" for f in (self.wind, self.current) if f.is_synthetic)
                + " — drift is illustrative, not evidence."
            )
        out.extend(self.extra_notes)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Flat record: new per-field detail + the legacy keys the UI already reads."""
        return {
            "forcing_source": self.forcing_source,
            # legacy keys (stable contract)
            "wind_origin": self.wind.source,
            "current_origin": self.current.source,
            # per-field provenance
            "wind": self.wind.to_dict(),
            "current": self.current.to_dict(),
            "wind_is_real": self.wind.is_real,
            "current_is_real": self.current.is_real,
            "any_synthetic": self.any_synthetic,
            "synthetic_fields": self.synthetic_fields,
            "confidence_penalty": round(self.confidence_penalty, 4),
            "warnings": self.warnings,
        }


def apply_confidence_penalty(
    base_confidence: float,
    selection: ForcingSelection | dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Downgrade a 0..1 confidence score by the synthetic-forcing penalty.

    Returns (confidence, basis) where `basis` is the audit trail: what the
    score started at, what was subtracted and why. Never returns a number
    without saying how it was obtained.
    """
    if isinstance(selection, dict):
        penalty = float(selection.get("confidence_penalty", 0.0))
        fields = list(selection.get("synthetic_fields") or [])
    else:
        penalty = selection.confidence_penalty
        fields = selection.synthetic_fields

    clamped_base = min(max(float(base_confidence), 0.0), 1.0)
    adjusted = min(max(clamped_base - penalty, 0.0), 1.0)
    basis = {
        "base_confidence": round(clamped_base, 4),
        "synthetic_penalty": round(penalty, 4),
        "synthetic_fields": fields,
        "final_confidence": round(adjusted, 4),
    }
    return round(adjusted, 4), basis


def coverage_from_dataset(ds: Any) -> tuple[str | None, str | None]:
    """(start_utc, end_utc) from an xarray dataset's time coordinate, if any."""
    try:
        if ds is None or "time" not in ds.coords:
            return None, None
        t = ds["time"].values
        if len(t) == 0:
            return None, None
        return _iso(t[0]), _iso(t[-1])
    except Exception:  # pragma: no cover - defensive, coverage is metadata only
        return None, None
