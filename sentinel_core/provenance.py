"""Provenance labels for datasets, mirroring the ingest contract.

The wire strings here are **not** a new vocabulary. ``services/ingest/app/
provenance.py`` already defines them for forcing and AIS data, and the UI
already renders them. This module re-declares them for the training/data side
rather than importing them, because that module lives inside a service package
that is only importable with the service directory on ``sys.path``.

Re-declaring a contract is how the two copies drift, so the duplication is
policed: ``sentinel_core/tests/test_provenance_contract.py`` loads the ingest
module by file path and asserts the string values are identical. If someone
changes one side, the test fails rather than the UI quietly mislabelling a
synthetic scene as real.

The glyphs match the UI legend — ``■`` real, ``▲`` synthetic, ``□``
unavailable, ``✕`` failed — so a log line and a badge say the same thing.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "PROVENANCE_NO_COVERAGE",
    "PROVENANCE_REAL",
    "PROVENANCE_SYNTHETIC",
    "PROVENANCE_UNAVAILABLE",
    "DataProvenance",
    "synthetic_warning",
]

#: Wire values. Must equal the constants in services/ingest/app/provenance.py.
PROVENANCE_REAL = "real"
PROVENANCE_SYNTHETIC = "synthetic_mock"
PROVENANCE_UNAVAILABLE = "unavailable"
PROVENANCE_NO_COVERAGE = "no_real_coverage"

_SYNTHETIC_WARNING = (
    "SYNTHETIC DATA — {source} is not real. Generated for demonstration only "
    "and MUST NOT be used as evidence in any investigation or enforcement action."
)


class DataProvenance(str, Enum):
    """Where a dataset came from, and whether it can support a claim.

    ``str``-valued so it serialises to the same JSON the API and UI already
    expect, without a custom encoder.
    """

    REAL = PROVENANCE_REAL
    SYNTHETIC = PROVENANCE_SYNTHETIC
    UNAVAILABLE = PROVENANCE_UNAVAILABLE
    NO_REAL_COVERAGE = PROVENANCE_NO_COVERAGE

    @property
    def glyph(self) -> str:
        return _GLYPHS[self]

    @property
    def is_synthetic(self) -> bool:
        return self is DataProvenance.SYNTHETIC

    @property
    def supports_scientific_claim(self) -> bool:
        """True only for real data.

        Every metric computed on a synthetic split is a check that the code
        runs, not evidence that the model works. Anything that reports a
        number to a human should consult this first.
        """
        return self is DataProvenance.REAL

    def describe(self) -> dict[str, Any]:
        return {
            "provenance": self.value,
            "glyph": self.glyph,
            "is_synthetic": self.is_synthetic,
            "supports_scientific_claim": self.supports_scientific_claim,
        }


_GLYPHS: dict[DataProvenance, str] = {
    DataProvenance.REAL: "■",
    DataProvenance.SYNTHETIC: "▲",
    DataProvenance.UNAVAILABLE: "□",
    DataProvenance.NO_REAL_COVERAGE: "□",
}


def synthetic_warning(source: str) -> str:
    """The sentence that must accompany any synthetic artefact."""
    return _SYNTHETIC_WARNING.format(source=source)
