"""Error types for the SENTINEL drift service."""

from __future__ import annotations


class EnvironmentalDataError(Exception):
    """Raised when environmental forcing data is unavailable.

    Per the No-Fabrication Policy, missing data is never silently
    synthesized — callers must degrade explicitly and label the source.
    """

    pass
