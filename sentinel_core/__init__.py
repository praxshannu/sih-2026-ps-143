"""SENTINEL shared core: configuration, logging, errors, provenance, data sources.

This package is deliberately small and dependency-light. It is the one place
that answers "what is configured", "where does data come from", and "what does
this failure mean" — questions that were previously answered independently in
each service and, on at least one occasion, answered differently.

It is importable from the repository root (``pythonpath = ["."]`` in
``pyproject.toml``) and is consumed by ``ml/``, ``scripts/`` and, where it
matters, the services.
"""

from __future__ import annotations

from sentinel_core.config import Settings, get_settings, reload_settings
from sentinel_core.errors import (
    ConfigError,
    DatasetError,
    DatasetUnavailableError,
    DataSourceError,
    DataSourceMismatchError,
    DataSourceUnavailableError,
    SentinelError,
    TrainingError,
)
from sentinel_core.logging import (
    bind_context,
    configure_logging,
    get_logger,
    quiet_logger,
)
from sentinel_core.provenance import DataProvenance

__all__ = [
    "ConfigError",
    "DataProvenance",
    "DataSourceError",
    "DataSourceMismatchError",
    "DataSourceUnavailableError",
    "DatasetError",
    "DatasetUnavailableError",
    "SentinelError",
    "Settings",
    "TrainingError",
    "bind_context",
    "configure_logging",
    "get_logger",
    "get_settings",
    "quiet_logger",
    "reload_settings",
]
