"""Synthetic AIS for regions with no real receiver coverage.

WHY THIS EXISTS
---------------
There is no free AIS source covering the open Indian Ocean. Verified 2026-09-13:

  * AISStream (the live feed in `.env`) is a **terrestrial** network — a global
    bounding box yielded 48 real messages in 13 s, while the Indian Ocean AOI
    (lat -20..30, lon 50..100) yielded **zero** in 40 s.
  * MarineCadastre is US-waters only, so it never covers an Indian Ocean bbox.
  * The Danish Maritime Authority archive is unreachable from this machine.
  * Satellite AIS (Spire / ORBCOMM) is a paid subscription.

So for Indian Ocean cases the choice is: no vessel layer at all, or a clearly
labelled synthetic one. This module provides the latter.

THE NON-NEGOTIABLE RULE
-----------------------
Every record this module produces carries ``provenance = "synthetic_mock"`` and
every API response that includes them carries ``SYNTHETIC_NOTICE``. SENTINEL
never lets these reach the screen unlabelled: the UI must render the warning
banner exported by ``get_synthetic_disclaimer()``. Tracks are deterministic
(seeded), so they are reproducible for demos and never mistaken for a live feed.

This is ONLY used when real coverage is absent. ``should_use_synthetic()``
encodes that decision so it cannot be applied silently elsewhere.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

# Regions where no free real AIS coverage exists. Synthetic data may be used
# here, and ONLY here, and only with the notice attached.
NO_REAL_COVERAGE_BOXES: tuple[tuple[float, float, float, float], ...] = (
    # Indian Ocean, Arabian Sea, Bay of Bengal, and S. of Mauritius — all
    # open-ocean regions with no AISStream terrestrial coverage.
    (35.0, -30.0, 110.0, 35.0),
)

# Carve-outs: coastal stretches inside the box above where AISStream *does*
# have real terrestrial receivers. If an AOI touches one of these we must use
# the live feed, never synthetic — otherwise we would be hiding real data.
KNOWN_COASTAL_COVERAGE_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (72.60, 18.80, 73.10, 19.30),   # Mumbai / JNPT approaches
    (80.10, 12.90, 80.45, 13.25),   # Chennai
    (56.00, 26.40, 56.90, 27.10),   # Strait of Hormuz
    (43.20, 12.40, 45.10, 13.20),   # Gulf of Aden / Bab-el-Mandeb
    (79.70, 6.80, 80.00, 7.10),     # Colombo
    (67.90, 24.70, 68.20, 25.10),   # Karachi
    (88.00, 21.60, 88.40, 22.10),   # Kolkata / Haldia approaches
    (103.50, 1.10, 104.30, 1.50),   # Singapore Strait (eastern edge)
)

SYNTHETIC_NOTICE = (
    "SYNTHETIC AIS — no live vessel coverage for this region. "
    "These tracks are simulated for demonstration and must not be used "
    "as evidence in any investigation or enforcement action."
)

PROVENANCE = "synthetic_mock"


def _overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """True when two (w, s, e, n) boxes share any area."""
    aw, as_, ae, an = a
    bw, bs, be, bn = b
    return aw < be and ae > bw and as_ < bn and an > bs


def should_use_synthetic(bbox: tuple[float, float, float, float]) -> bool:
    """True only when the AOI has no real AIS coverage (open Indian Ocean).

    Logic: the AOI must overlap a known no-coverage region AND must not overlap
    any coastal carve-out where live terrestrial receivers exist. When both
    apply, real coverage wins — SENTINEL never substitutes synthetic data over
    a region where a real feed is available.
    """
    in_open = any(_overlaps(bbox, b) for b in NO_REAL_COVERAGE_BOXES)
    if not in_open:
        return False
    in_coastal = any(_overlaps(bbox, b) for b in KNOWN_COASTAL_COVERAGE_BOXES)
    return not in_coastal


def coverage_verdict(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """Explain, for the UI, *why* an AOI is real or synthetic.

    The response is always returned so the frontend can render the reason, not
    just a boolean — the operator needs to know the basis for the label.
    """
    w, s, e, n = bbox
    in_open = any(_overlaps(bbox, b) for b in NO_REAL_COVERAGE_BOXES)
    coastal_hits = [
        list(b) for b in KNOWN_COASTAL_COVERAGE_BOXES if _overlaps(bbox, b)
    ]
    synthetic = in_open and not coastal_hits

    if synthetic:
        basis = (
            "AOI lies in the open Indian Ocean. AISStream is a terrestrial "
            "receiver network and has no coverage here; verified 0 messages "
            "over an Indian Ocean bbox in a 40 s live test."
        )
        mode = "synthetic_only"
    elif coastal_hits:
        basis = (
            "AOI overlaps a coastal zone with live AISStream terrestrial "
            "receivers. Real AIS is available and synthetic data is suppressed."
        )
        mode = "live_terrestrial"
    else:
        basis = (
            "AOI is outside the Indian Ocean no-coverage region. Live "
            "terrestrial AIS coverage is assumed available."
        )
        mode = "live_terrestrial"

    return {
        "bbox": [w, s, e, n],
        "mode": mode,
        "use_synthetic": synthetic,
        "coastal_carveouts": coastal_hits,
        "basis": basis,
        "provenance": PROVENANCE if synthetic else "live_terrestrial",
        "notice": SYNTHETIC_NOTICE if synthetic else None,
    }


@dataclass
class SyntheticVessel:
    mmsi: str
    name: str
    vessel_type: str
    flag: str
    imo: str
    track: list[dict[str, Any]]
    gaps: list[dict[str, Any]]
    provenance: str = PROVENANCE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Deterministic roster so demos are reproducible across runs.
_ROSTER: tuple[tuple[str, str, str, str, str], ...] = (
    ("477218700", "MV WAKASHIO", "BULK CARRIER", "PA", "9448745"),
    ("371456000", "PACIFIC VOYAGER", "TANKER", "PA", "9384521"),
    ("636019234", "SEA HARVESTER", "FISHING", "ID", "8812345"),
    ("563123456", "ORIENTAL SPIRIT", "CARGO", "SG", "9523456"),
    ("215678900", "ARABIAN DAWN", "TANKER", "AE", "9678901"),
    ("538008765", "SOUTHERN CROSS", "CARGO", "MT", "9345678"),
    ("356789012", "BLUE MARLIN", "FISHING", "LK", "8734562"),
    ("477334511", "INDIAN TRADER", "BULK CARRIER", "IN", "9456789"),
)

_NAV_STATUSES = (
    "UNDER_WAY_USING_ENGINE",
    "UNDER_WAY_SAILING",
    "AT_ANCHOR",
    "UNDER_WAY_USING_ENGINE",
)


def _step(
    lon: float, lat: float, course_deg: float, speed_kn: float, hours: float
) -> tuple[float, float]:
    """Great-circle-ish dead reckoning: move `hours` at `speed_kn` along `course_deg`."""
    dist_nm = speed_kn * hours
    dlat = dist_nm * math.cos(math.radians(course_deg)) / 60.0
    dlon = (
        dist_nm
        * math.sin(math.radians(course_deg))
        / (60.0 * max(math.cos(math.radians(lat)), 1e-3))
    )
    return lon + dlon, lat + dlat


def generate_synthetic_ais(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    n_vessels: int = 6,
    seed: int = 20200725,
    sample_minutes: int = 10,
    dark_vessel_indices: tuple[int, ...] = (2, 5),
) -> dict[str, Any]:
    """Deterministically generate synthetic AIS tracks inside `bbox`.

    Args:
        bbox: (west, south, east, north) in decimal degrees.
        start / end: UTC window.
        n_vessels: how many tracks to emit (capped at the roster size).
        seed: RNG seed — same seed always produces the same tracks.
        sample_minutes: reporting cadence.
        dark_vessel_indices: which roster entries get a deliberate AIS gap.

    Returns:
        A dict whose ``vessels`` carry ``provenance='synthetic_mock'`` and whose
        top level repeats ``SYNTHETIC_NOTICE``. Nothing here is real.
    """
    rng = random.Random(seed)
    w, s, e, n = bbox
    vessels: list[dict[str, Any]] = []

    for i in range(min(n_vessels, len(_ROSTER))):
        mmsi, name, vtype, flag, imo = _ROSTER[i]
        # Start each vessel at a random point inside the AOI.
        lon = rng.uniform(w + 0.1 * (e - w), w + 0.9 * (e - w))
        lat = rng.uniform(s + 0.1 * (n - s), s + 0.9 * (n - s))
        course = rng.uniform(0.0, 360.0)
        base_speed = rng.uniform(8.0, 14.0)

        track: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        t = start
        dt = timedelta(minutes=sample_minutes)
        dark_from: datetime | None = None

        if i in dark_vessel_indices:
            # Deliberate AIS gap in the middle third of the window.
            span = (end - start).total_seconds()
            dark_from = start + timedelta(seconds=span * 0.45)
            dark_until = start + timedelta(seconds=span * 0.62)

        while t <= end:
            in_gap = (
                dark_from is not None and dark_from <= t <= dark_until
            )
            if in_gap:
                if not gaps or gaps[-1].get("end") is not None:
                    gaps.append(
                        {
                            "start": t.isoformat().replace("+00:00", "Z"),
                            "end": None,
                            "provenance": PROVENANCE,
                        }
                    )
                t += dt
                continue

            if gaps and gaps[-1].get("end") is None:
                gaps[-1]["end"] = t.isoformat().replace("+00:00", "Z")

            speed = max(0.0, base_speed + rng.gauss(0.0, 1.2))
            course = (course + rng.gauss(0.0, 4.0)) % 360.0
            lon, lat = _step(lon, lat, course, speed, sample_minutes / 60.0)
            # Keep vessels inside the AOI by reflecting at the edges.
            if not (w <= lon <= e):
                course = (180.0 - course) % 360.0
                lon = min(max(lon, w), e)
            if not (s <= lat <= n):
                course = (-course) % 360.0
                lat = min(max(lat, s), n)

            track.append(
                {
                    "timestamp": t.isoformat().replace("+00:00", "Z"),
                    "longitude": round(lon, 5),
                    "latitude": round(lat, 5),
                    "sog": round(speed, 1),
                    "cog": round(course, 1),
                    "heading": round(course, 1),
                    "nav_status": rng.choice(_NAV_STATUSES),
                    "source": "SYNTHETIC",
                    "provenance": PROVENANCE,
                }
            )
            t += dt

        vessels.append(
            SyntheticVessel(
                mmsi=mmsi,
                name=name,
                vessel_type=vtype,
                flag=flag,
                imo=imo,
                track=track,
                gaps=gaps,
            ).to_dict()
        )

    return {
        "provenance": PROVENANCE,
        "is_synthetic": True,
        "notice": SYNTHETIC_NOTICE,
        "reason": (
            "No free AIS source covers this region. AISStream is a terrestrial "
            "network with no Indian Ocean receivers; MarineCadastre is US-waters "
            "only; satellite AIS requires a paid subscription."
        ),
        "bbox": list(bbox),
        "window": [
            start.isoformat().replace("+00:00", "Z"),
            end.isoformat().replace("+00:00", "Z"),
        ],
        "count": len(vessels),
        "vessels": vessels,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def synthetic_disclaimer() -> dict[str, str]:
    """The payload the UI must render whenever synthetic AIS is on screen."""
    return {
        "severity": "warning",
        "provenance": PROVENANCE,
        "title": "SIMULATED VESSEL DATA",
        "message": SYNTHETIC_NOTICE,
        "short": "SYNTHETIC",
    }
