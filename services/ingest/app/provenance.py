"""One shared provenance + env-gate helper for every SENTINEL ingest source.

WHY THIS MODULE EXISTS
----------------------
``docs/VERIFICATION.md`` §9.2 records the worst failure this platform has had: an
ERA5 pull silently degraded to synthetic wind, and *the run succeeded and
returned invented numbers*. The failure was not the fallback itself — it was
that three different sources each re-implemented the decision, so they drifted,
and one of them decided "unavailable" meant "make something up".

This module is the single place that decides. CMEMS, ERA5 and AIS all call it,
so they cannot drift apart again.

Rules enforced here
-------------------
1. **Unset means off.** ``ALLOW_SYNTHETIC_FORCING``, ``ALLOW_SYNTHETIC_AIS`` and
   ``SENTINEL_DEMO_MODE`` are opt-in. An unset or unrecognised value is False;
   ``false``/``0``/``no``/``off`` are also False.
2. **Geography is a gate, not a hint.** Synthetic AIS additionally requires that
   the AOI sit inside ``NO_REAL_COVERAGE_BOXES`` and outside
   ``KNOWN_COASTAL_COVERAGE_BOXES``. That logic lives in ``sources/ais_synthetic.py``
   and is passed in as ``geo_allowed`` — it is never re-implemented here.
3. **Every payload carries its own label.** ``envelope()`` emits
   ``provenance`` / ``is_synthetic`` / ``warning`` on the data itself, so a
   record that leaks out of the API still identifies itself.
4. **Partial coverage is reported, never blended.** An AOI that straddles a
   source boundary yields ``status="partial"`` plus the uncovered sub-box; real
   and synthetic rows are never merged into one undifferentiated list.

Deliberately dependency-free: standard library only, so ``scripts/`` can import
it without pulling in FastAPI, xarray or the DB driver.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ── Provenance values ─────────────────────────────────────────────────────
# These strings are the contract with the UI. Changing one changes what the
# operator sees, so they are named here and referenced everywhere else.

PROVENANCE_REAL = "real"
PROVENANCE_SYNTHETIC = "synthetic_mock"
PROVENANCE_UNAVAILABLE = "unavailable"
PROVENANCE_NO_COVERAGE = "no_real_coverage"

# ── Environment gates ─────────────────────────────────────────────────────

ENV_ALLOW_SYNTHETIC_FORCING = "ALLOW_SYNTHETIC_FORCING"
ENV_ALLOW_SYNTHETIC_AIS = "ALLOW_SYNTHETIC_AIS"
ENV_SENTINEL_DEMO_MODE = "SENTINEL_DEMO_MODE"

# Gates required before *any* synthetic forcing (currents, winds) may be used.
FORCING_GATES: tuple[str, ...] = (ENV_ALLOW_SYNTHETIC_FORCING,)

# Gates required before synthetic *vessel tracks* may be used. Demo mode is a
# separate switch from the data flag on purpose: an operator can enable
# synthetic forcing for a physics demo without also inventing ships.
AIS_GATES: tuple[str, ...] = (ENV_SENTINEL_DEMO_MODE, ENV_ALLOW_SYNTHETIC_AIS)

_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on"})

SYNTHETIC_WARNING = (
    "SYNTHETIC DATA — {source} is not real. Generated for demonstration only "
    "and MUST NOT be used as evidence in any investigation or enforcement action."
)


def env_flag(name: str, env: Mapping[str, str] | None = None) -> bool:
    """True only when `name` is explicitly set to a truthy value.

    Unset, empty, and unrecognised values are all False — a typo in
    ``ALLOW_SYNTHETIC_FORCING`` must never open the gate.
    """
    raw = (os.environ if env is None else env).get(name, "")
    return str(raw).strip().lower() in _TRUTHY


def provenance_envelope(
    source: str,
    provenance: str,
    *,
    synthetic: bool | None = None,
    warning: str | None = None,
    error: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The payload every source returns, whatever happened upstream.

    ``synthetic`` defaults to ``provenance == PROVENANCE_SYNTHETIC`` so a caller
    cannot label a payload ``synthetic_mock`` and leave ``is_synthetic`` False.
    """
    if synthetic is None:
        synthetic = provenance == PROVENANCE_SYNTHETIC
    if synthetic and warning is None:
        warning = SYNTHETIC_WARNING.format(source=source)
    payload: dict[str, Any] = {
        "source": source,
        "provenance": provenance,
        "is_synthetic": synthetic,
        "warning": warning,
        "error": error,
        "coverage": coverage,
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if extra:
        payload.update(extra)
    return payload


class ProvenanceError(RuntimeError):
    """Raised when a source cannot supply real data and may not invent any.

    Carries a machine-readable ``reason`` so an operator (or a log shipper) can
    act on the failure without string-matching a human sentence.
    """

    def __init__(
        self,
        source: str,
        reason: str,
        detail: str = "",
        *,
        required_env: Sequence[str] = (),
        provenance: str = PROVENANCE_UNAVAILABLE,
        coverage: dict[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.reason = reason
        self.detail = detail
        self.required_env = tuple(required_env)
        self.provenance = provenance
        self.coverage = coverage
        super().__init__(self.message)

    @property
    def message(self) -> str:
        text = f"{self.source} unavailable [{self.reason}]"
        if self.detail:
            text = f"{text}: {self.detail}"
        if self.required_env:
            text = (
                f"{text} — synthetic data is refused. "
                f"Set {' and '.join(self.required_env)} to permit it explicitly."
            )
        return text

    def to_dict(self) -> dict[str, Any]:
        return provenance_envelope(
            self.source,
            self.provenance,
            synthetic=False,
            error={
                "reason": self.reason,
                "detail": self.detail or None,
                "required_env": list(self.required_env),
            },
            coverage=self.coverage,
        )


@dataclass(frozen=True)
class SyntheticDecision:
    """The verdict: may this source emit synthetic data right now, and why."""

    source: str
    allowed: bool
    provenance: str
    reason: str
    failed_gates: tuple[str, ...] = ()
    warning: str | None = None

    @property
    def message(self) -> str:
        if self.allowed:
            return f"{self.source}: synthetic data permitted ({self.reason})"
        return f"{self.source}: synthetic data refused ({self.reason})"

    def envelope(self, **extra: Any) -> dict[str, Any]:
        return provenance_envelope(
            self.source,
            self.provenance,
            synthetic=self.allowed,
            warning=self.warning,
            extra=extra,
        )

    def require(self, cause: BaseException | None = None) -> None:
        """Raise :class:`ProvenanceError` unless synthetic data was allowed."""
        if self.allowed:
            return
        raise ProvenanceError(
            self.source,
            self.reason,
            str(cause) if cause is not None else "",
            required_env=self.failed_gates,
            provenance=self.provenance,
        ) from cause


def decide_synthetic(
    source: str,
    *,
    required_env: Sequence[str] = FORCING_GATES,
    geo_allowed: bool = True,
    geo_reason: str = "outside_permitted_box",
    env: Mapping[str, str] | None = None,
) -> SyntheticDecision:
    """Decide whether `source` may fall back to synthetic data.

    Args:
        source: short source name used in logs and payloads, e.g. ``"cmems"``.
        required_env: every gate in this sequence must be truthy.
        geo_allowed: the caller's geographic verdict (for AIS this is
            ``ais_synthetic.should_use_synthetic(bbox)``).
        geo_reason: machine-readable code recorded when ``geo_allowed`` is False.
        env: override the environment (tests inject an explicit mapping).

    Returns:
        A :class:`SyntheticDecision`. When ``allowed`` is True the caller may
        generate synthetic data and **must** stamp the result with
        ``decision.provenance`` and surface ``decision.warning``.
    """
    failed = tuple(name for name in required_env if not env_flag(name, env))
    reasons: list[str] = [f"{name}_not_enabled" for name in failed]
    if not geo_allowed:
        reasons.append(geo_reason)

    if not reasons:
        return SyntheticDecision(
            source=source,
            allowed=True,
            provenance=PROVENANCE_SYNTHETIC,
            reason="synthetic_permitted",
            failed_gates=(),
            warning=SYNTHETIC_WARNING.format(source=source),
        )

    return SyntheticDecision(
        source=source,
        allowed=False,
        provenance=PROVENANCE_UNAVAILABLE if failed else PROVENANCE_NO_COVERAGE,
        reason="+".join(reasons),
        failed_gates=failed,
        warning=None,
    )


# ── Coverage reporting ────────────────────────────────────────────────────


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def bbox_overlap_fraction(
    aoi: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
) -> float:
    """Fraction of `aoi` area covered by `box`, in [0, 1]."""
    aw, as_, ae, an = aoi
    bw, bs, be, bn = box
    ow = _clip(min(ae, be) - max(aw, bw), 0.0, float("inf"))
    oh = _clip(min(an, bn) - max(as_, bs), 0.0, float("inf"))
    if ow <= 0 or oh <= 0:
        return 0.0
    aoi_area = (ae - aw) * (an - as_)
    if aoi_area <= 0:
        return 0.0
    return float(_clip((ow * oh) / aoi_area, 0.0, 1.0))


def coverage_report(
    aoi: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
    label: str,
) -> dict[str, Any]:
    """Honest geographic coverage of `aoi` against one source box.

    Returns ``status`` as ``full`` / ``partial`` / ``none``. Callers must treat
    ``partial`` as exactly that: return the covered part, name the rest, and
    never fill the gap with a different kind of data.
    """
    fraction = bbox_overlap_fraction(aoi, box)
    if fraction >= 1.0:
        status = "full"
    elif fraction <= 0.0:
        status = "none"
    else:
        status = "partial"
    return {
        "source": label,
        "status": status,
        "covered_fraction": round(fraction, 6),
        "source_box": list(box),
        "aoi": list(aoi),
    }


def window_coverage(
    requested: tuple[datetime, datetime],
    returned: tuple[datetime, datetime] | None,
) -> dict[str, Any]:
    """Honest temporal coverage: what was asked for vs what upstream returned.

    Used by the forcing sources, where a truncated NetCDF is the realistic
    partial-coverage case (and, per VERIFICATION §10, the one that otherwise
    looks exactly like an upstream outage).
    """
    req_start, req_end = requested
    payload: dict[str, Any] = {
        "status": "none",
        "requested": [req_start.isoformat(), req_end.isoformat()],
        "returned": None,
        "missing_before_hours": None,
        "missing_after_hours": None,
    }
    if returned is None:
        return payload
    got_start, got_end = returned
    before = (got_start - req_start).total_seconds() / 3600.0
    after = (req_end - got_end).total_seconds() / 3600.0
    payload["returned"] = [got_start.isoformat(), got_end.isoformat()]
    payload["missing_before_hours"] = round(max(before, 0.0), 3)
    payload["missing_after_hours"] = round(max(after, 0.0), 3)
    if before <= 0 and after <= 0:
        payload["status"] = "full"
    else:
        payload["status"] = "partial"
    return payload
