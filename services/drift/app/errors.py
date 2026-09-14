"""Error types for the SENTINEL drift service.

Every failure carries a machine-readable `code` and `reason` so callers (and
the UI) can render an explicit unavailable state instead of guessing — and so
a missing forcing field can never be silently replaced by synthetic data.
"""

from __future__ import annotations


class EnvironmentalDataError(Exception):
    """Raised when environmental forcing data is unavailable.

    Per the No-Fabrication Policy, missing data is never silently
    synthesized — callers must degrade explicitly and label the source.
    """

    code = "environmental_data_unavailable"
    field = "unknown"

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason or message

    def to_dict(self) -> dict[str, str]:
        return {
            "error": type(self).__name__,
            "code": self.code,
            "field": self.field,
            "reason": self.reason,
            "message": str(self),
        }


class ForcingUnavailableError(EnvironmentalDataError):
    """Base class for a single missing forcing field (wind or current)."""

    code = "forcing_unavailable"

    def __init__(self, message: str, *, field: str, reason: str = "") -> None:
        super().__init__(message, reason=reason)
        self.field = field


class MissingWindForcingError(ForcingUnavailableError):
    """No real 10 m wind forcing available and synthetic fallback refused."""

    code = "missing_wind_forcing"
    field = "wind"

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message, field="wind", reason=reason)


class MissingCurrentForcingError(ForcingUnavailableError):
    """No real surface current forcing available and synthetic fallback refused."""

    code = "missing_current_forcing"
    field = "current"

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message, field="current", reason=reason)


class RegimeUndeterminedError(EnvironmentalDataError):
    """The K_ij diffusion regime could not be classified from the inputs.

    Raised only where a regime is strictly required. The backward SDE instead
    records `regime=None` + a reason, because AGENTS.md forbids inventing one.
    """

    code = "regime_undetermined"
    field = "diffusivity_regime"
