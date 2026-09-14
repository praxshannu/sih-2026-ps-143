"""AIS coverage policy — the gate that decides whether a ranking is allowed.

WHY THIS MODULE EXISTS
----------------------
There is no free AIS source covering the open Indian Ocean (see
``docs/VERIFICATION.md`` §2.7 and §6):

  * AISStream (the feed configured in ``.env``) is a **terrestrial** receiver
    network. A global bbox produced 48 real messages in 13 s; an Indian Ocean
    box produced **zero** in 40 s.
  * MarineCadastre is US-waters only, so it never covers an Indian Ocean AOI.
  * The Danish Maritime Authority archive is unreachable from this machine.
  * Satellite AIS (Spire / ORBCOMM) is a paid subscription.

So an AOI in that region has **no real AIS coverage**. SENTINEL may serve a
clearly labelled synthetic vessel layer there for situational awareness (the
ingest service owns that decision), but attribution must never rank suspects
from it: a ranking built on invented tracks is worse than no ranking, because
it looks like evidence.

THE RULE ENFORCED HERE
----------------------
``is_rankable(...)`` is the only place that decides. It returns ``False`` for
anything that is not provably real AIS, and callers must then emit an
empty-with-reason response instead of a suspect list.

AOI PARAMETERS — NAMED AXES ONLY
--------------------------------
The positional ``west,south,east,north`` string is accepted for backwards
compatibility and nothing else. It is unverifiable: a lat-first caller sends
four numbers that are all legal in either slot, so no validator can detect the
swap, the AOI is silently relocated, and the real/synthetic verdict flips.
Named axes (``min_lon``/``min_lat``/``max_lon``/``max_lat``) cannot be
misordered, so they are the primary contract. A **partial** named set is a
422 — never guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

# Machine-readable reason codes. Every "no ranking" response carries one.
REASON_NO_REAL_COVERAGE = "NO_REAL_AIS_COVERAGE"  # open-ocean region, no receivers
REASON_SYNTHETIC_SOURCE = "AIS_SOURCE_SYNTHETIC"  # rows labelled synthetic_mock
REASON_NO_POSITIONS = "NO_AIS_POSITIONS_MATCHED"  # real region, nothing matched
REASON_NO_AOI = "AIS_COVERAGE_UNKNOWN_NO_AOI"  # cannot prove coverage
REASON_UNVERIFIED = "AIS_PROVENANCE_UNVERIFIED"  # provenance absent/unknown
REASON_PARTIAL_BBOX = "BBOX_PARTIAL"  # some axes given, others missing
REASON_BBOX_RANGE = "BBOX_OUT_OF_RANGE"  # west>=east, south>=north, or off-globe
REASON_BBOX_PARSE = "BBOX_UNPARSEABLE"  # deprecated string malformed

# Provenance labels that count as real receiver data. Anything else is not
# rankable — an unrecognised label is treated as unverified, never as real.
REAL_AIS_PROVENANCE = frozenset(
    {
        "live_terrestrial",
        "live_satellite",
        "aisstream",
        "marinecadastre",
        "dma",
        "real",
    }
)

SYNTHETIC_AIS_PROVENANCE = frozenset({"synthetic_mock", "synthetic", "simulated"})

# Regions where no free real AIS coverage exists (docs/VERIFICATION.md §6).
NO_REAL_COVERAGE_BOXES: tuple[tuple[float, float, float, float], ...] = (
    # Indian Ocean, Arabian Sea, Bay of Bengal, S. of Mauritius — all
    # open-ocean regions with no AISStream terrestrial coverage.
    (35.0, -30.0, 110.0, 35.0),
)

# Coastal carve-outs inside the box above where terrestrial receivers DO exist.
# Ordering matters: if an AOI touches one of these, real coverage wins and
# synthetic data must never mask it.
KNOWN_COASTAL_COVERAGE_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (72.60, 18.80, 73.10, 19.30),  # Mumbai / JNPT approaches
    (80.10, 12.90, 80.45, 13.25),  # Chennai
    (56.00, 26.40, 56.90, 27.10),  # Strait of Hormuz
    (43.20, 12.40, 45.10, 13.20),  # Gulf of Aden / Bab-el-Mandeb
    (79.70, 6.80, 80.00, 7.10),  # Colombo
    (67.90, 24.70, 68.20, 25.10),  # Karachi
    (88.00, 21.60, 88.40, 22.10),  # Kolkata / Haldia approaches
    (103.50, 1.10, 104.30, 1.50),  # Singapore Strait (eastern edge)
)


class AoiError(ValueError):
    """Raised when an AOI cannot be resolved. Callers map this to HTTP 422.

    ``code`` is one of the ``REASON_*`` constants so the API can return a
    machine-readable reason instead of a sentence.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """True when two (west, south, east, north) boxes intersect."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _validate_box(w: float, s: float, e: float, n: float) -> tuple[float, float, float, float]:
    if not (-180.0 <= w < e <= 180.0):
        raise AoiError(
            REASON_BBOX_RANGE,
            "bbox longitudes must satisfy -180 <= min_lon < max_lon <= 180",
        )
    if not (-90.0 <= s < n <= 90.0):
        raise AoiError(
            REASON_BBOX_RANGE,
            "bbox latitudes must satisfy -90 <= min_lat < max_lat <= 90",
        )
    return (w, s, e, n)


def _parse_bbox_string(raw: str) -> tuple[float, float, float, float]:
    parts = (raw or "").split(",")
    if len(parts) != 4:
        raise AoiError(REASON_BBOX_PARSE, "bbox must be 'west,south,east,north' in decimal degrees")
    try:
        w, s, e, n = (float(p) for p in parts)
    except ValueError as exc:
        raise AoiError(
            REASON_BBOX_PARSE,
            "bbox must be 'west,south,east,north' in decimal degrees",
        ) from exc
    return _validate_box(w, s, e, n)


def resolve_aoi(
    min_lon: float | None,
    min_lat: float | None,
    max_lon: float | None,
    max_lat: float | None,
    bbox: str | None = None,
) -> tuple[float, float, float, float] | None:
    """Resolve an AOI from named axes, falling back to the deprecated string.

    Returns ``None`` when the caller supplied no AOI at all (the caller then
    has to decide whether coverage can be proven some other way).

    Raises:
        AoiError: on a partial named set, an out-of-range box, or an
            unparseable deprecated string. Never guesses a missing axis.
    """
    named = (min_lon, min_lat, max_lon, max_lat)
    if any(v is not None for v in named):
        if any(v is None for v in named):
            raise AoiError(
                REASON_PARTIAL_BBOX,
                "min_lon, min_lat, max_lon and max_lat must be supplied together",
            )
        return _validate_box(
            float(min_lon),  # type: ignore[arg-type]
            float(min_lat),  # type: ignore[arg-type]
            float(max_lon),  # type: ignore[arg-type]
            float(max_lat),  # type: ignore[arg-type]
        )
    if bbox:
        # Loud, once per call: the positional form cannot be validated against
        # axis-order mistakes, so we want it visible in the logs.
        logger.warning(
            "Deprecated 'bbox' string used for an AOI — use min_lon/min_lat/max_lon/max_lat; "
            "a lat-first string is four numbers that are all legal in either slot, so a swap "
            "cannot be detected and the real/synthetic verdict flips silently"
        )
        return _parse_bbox_string(bbox)
    return None


@dataclass(frozen=True)
class CoverageVerdict:
    """Whether an AOI has real AIS coverage, and why."""

    coverage: str  # "real" | "synthetic" | "none"
    provenance: str  # e.g. live_terrestrial | synthetic_mock | unknown
    reason: str  # machine-readable REASON_* code
    message: str
    bbox: tuple[float, float, float, float] | None = None

    @property
    def rankable(self) -> bool:
        """True only when suspect ranking is permitted."""
        return self.coverage == "real"

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "provenance": self.provenance,
            "reason": self.reason,
            "message": self.message,
            "bbox": list(self.bbox) if self.bbox else None,
            "rankable": self.rankable,
        }


def coverage_verdict(box: tuple[float, float, float, float] | None) -> CoverageVerdict:
    """Decide whether ``box`` has real AIS receiver coverage.

    ``(35, -30, 110, 35)`` is the documented no-coverage region; the coastal
    carve-outs inside it win, so real coverage is never masked by synthetic.
    """
    if box is None:
        return CoverageVerdict(
            coverage="none",
            provenance="unknown",
            reason=REASON_NO_AOI,
            message=(
                "No AOI supplied, so AIS coverage cannot be proven. Supply "
                "min_lon/min_lat/max_lon/max_lat so the coverage verdict is explicit."
            ),
        )

    in_no_coverage = any(_overlaps(box, b) for b in NO_REAL_COVERAGE_BOXES)
    in_coastal = any(_overlaps(box, b) for b in KNOWN_COASTAL_COVERAGE_BOXES)

    if in_no_coverage and not in_coastal:
        return CoverageVerdict(
            coverage="none",
            provenance="synthetic_mock",
            reason=REASON_NO_REAL_COVERAGE,
            message=(
                "No real AIS coverage for this AOI: AISStream is a terrestrial "
                "receiver network with zero coverage over the open Indian Ocean "
                "(verified 0 messages in 40 s), and no free historical source "
                "covers it. Suspect ranking is withheld; synthetic tracks are "
                "never scored."
            ),
            bbox=box,
        )

    return CoverageVerdict(
        coverage="real",
        provenance="live_terrestrial",
        reason="",
        message="Live terrestrial AIS coverage is assumed available for this AOI.",
        bbox=box,
    )


def provenance_verdict(provenance: str | None, matched: int = 0) -> CoverageVerdict:
    """Decide rankability from the provenance label on the AIS rows themselves.

    Used when the AOI is unknown but the data carries its own provenance — the
    strongest evidence available. Unknown labels are treated as unverified,
    never as real.
    """
    label = (provenance or "unknown").strip().lower()
    if label in REAL_AIS_PROVENANCE:
        if matched <= 0:
            return CoverageVerdict(
                coverage="none",
                provenance=label,
                reason=REASON_NO_POSITIONS,
                message=(
                    f"AIS coverage is real ({label}) but no positions matched the "
                    "origin/time window, so there is nothing to rank."
                ),
            )
        return CoverageVerdict(
            coverage="real",
            provenance=label,
            reason="",
            message=f"Real AIS coverage ({label}); {matched} positions matched.",
        )
    if label in SYNTHETIC_AIS_PROVENANCE:
        return CoverageVerdict(
            coverage="none",
            provenance=label,
            reason=REASON_SYNTHETIC_SOURCE,
            message=(
                f"AIS provenance is '{label}'. Synthetic vessel tracks are never "
                "scored: a suspect ranking built on them would look like evidence."
            ),
        )
    return CoverageVerdict(
        coverage="none",
        provenance=label or "unknown",
        reason=REASON_UNVERIFIED,
        message=(
            f"AIS provenance '{label}' is not a recognised real source, so "
            "coverage cannot be proven and no ranking is produced."
        ),
    )
