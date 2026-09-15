"""Typed error hierarchy for SENTINEL.

Why a hierarchy rather than bare exceptions
-------------------------------------------
The failure this project cares most about is the *quiet* one: a run that
completes and returns a number nobody can trace. ``docs/VERIFICATION.md`` §9.2
records an ERA5 pull that silently degraded to synthetic wind and succeeded.
Every error here therefore carries:

* a stable machine-readable ``reason`` code, so a log shipper or an operator
  can branch on the failure without matching a human sentence, and
* a ``context`` mapping of the values that produced the failure, so the
  message does not have to be reconstructed from memory.

The second rule is that a missing input is an **error**, never a fallback.
:class:`DataSourceUnavailableError` exists precisely so that "the real dataset
is not mounted" cannot be answered with invented data — a caller that wants
that has to ask for it explicitly via
:attr:`sentinel_core.config.DataSourceSettings.allow_synthetic_fallback`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "CheckpointError",
    "ConfigError",
    "DataSourceError",
    "DataSourceMismatchError",
    "DataSourceUnavailableError",
    "DatasetError",
    "DatasetUnavailableError",
    "ProvenanceError",
    "SentinelError",
    "TrainingError",
]


class SentinelError(Exception):
    """Base class for every error SENTINEL raises deliberately.

    Args:
        message: human-readable one-liner. Should name the thing that failed,
            not just the category.
        reason: machine-readable code. Defaults to the class-level ``reason``.
        context: values that produced the failure (paths, shapes, counts).
            Serialised into structured logs; keep it small and non-secret.
    """

    #: Default machine-readable code; subclasses override.
    reason: str = "sentinel_error"
    #: Process exit code used by the CLI entry points.
    exit_code: int = 1

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message
        if reason is not None:
            self.reason = reason
        self.context: dict[str, Any] = dict(context or {})
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Structured form for logs and API error payloads."""
        return {
            "error": type(self).__name__,
            "reason": self.reason,
            "message": self.message,
            "context": self.context,
        }

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({detail})"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(SentinelError):
    """Configuration is missing, contradictory, or unparseable."""

    reason = "config_error"
    exit_code = 78  # EX_CONFIG


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


class DataSourceError(SentinelError):
    """Base class for data-source resolution and access failures."""

    reason = "data_source_error"


class DataSourceUnavailableError(DataSourceError):
    """The selected data source cannot be read.

    Raised instead of falling back. If the real dataset is not mounted, that is
    the answer; the caller must opt in to a substitute.
    """

    reason = "data_source_unavailable"


class DataSourceMismatchError(DataSourceError):
    """The data does not have the shape the pipeline requires.

    Covers band-count, dtype, extent and pairing mismatches. These are never
    repaired implicitly — padding a 2-band SAR scene to 3 channels, or dropping
    a channel to make a shape fit, changes what the model sees without changing
    anything the operator can see.
    """

    reason = "data_source_mismatch"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DatasetError(SentinelError):
    """Base class for dataset construction and sampling failures."""

    reason = "dataset_error"


class DatasetUnavailableError(DatasetError, FileNotFoundError):
    """A dataset directory is missing, empty, or unpaired.

    Also a :class:`FileNotFoundError` because callers that only want to know
    "is the data there" should not have to import this package to ask.
    """

    reason = "dataset_unavailable"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainingError(SentinelError):
    """A training run failed in a way the operator must act on."""

    reason = "training_error"


class CheckpointError(TrainingError):
    """A checkpoint could not be written, read, or matched to a model."""

    reason = "checkpoint_error"


class ProvenanceError(SentinelError):
    """Data would have to be labelled dishonestly to proceed.

    Raised when a synthetic artefact is about to be used somewhere that claims
    to be real, or vice versa.
    """

    reason = "provenance_error"
