"""Typed, validated configuration for SENTINEL's data and training layers.

Before this module every knob was a bare ``os.getenv(...)`` call at the point
of use. That has three costs, all of which have already been paid once in this
repository:

1. **No single place to see the contract.** ``SENTINEL_DATA_DIR`` existed in
   ``.env.example`` and was read by nobody; the SAR directory was computed in
   two files and the two disagreed.
2. **Typos are silent.** ``os.getenv("TRAIN_EPOCHS", "5")`` returns the default
   for ``TRAIN_EPOCH`` and the run quietly trains for 5 epochs. Here a mistyped
   *field* is ignored (``extra="ignore"``, so a stray ``.env`` key cannot break
   startup) but a mistyped *value* raises.
3. **Cross-field constraints are unenforceable.** ``image_size`` must be
   divisible by 32 for a 5-level UNet++ encoder; that is checked once here
   instead of failing inside a conv layer four hours into a run.

Layout: a flat ``data_source`` switch (``SENTINEL_DATA_SOURCE=real|synthetic``)
because that is the knob an operator actually turns, plus grouped settings for
the parts that are configured once. Nested groups use the ``__`` delimiter:
``SENTINEL_TRAINING__EPOCHS=10``.

Relative paths are resolved against the repository root, not the process
working directory. ``SENTINEL_DATA_DIR=./data`` means the same thing whether it
is read by a CLI at the repo root or by a service that chdir'd into its own
directory — which is exactly the bug that made a hardcoded ``/app/data/sar``
look correct in one place and silently wrong in another.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentinel_core.errors import ConfigError

__all__ = [
    "DataSourceKind",
    "DataSourceSettings",
    "LoggingSettings",
    "Settings",
    "TrainingSettings",
    "get_settings",
    "reload_settings",
    "repo_root",
]

DataSourceKind = Literal["real", "synthetic"]
Environment = Literal["dev", "ci", "prod"]

_ROOT_MARKER = "pyproject.toml"
#: Where the bulk SAR archive is expected to be mounted. Only a default: the
#: real data source is fully overridable via ``SENTINEL_DATASOURCES__REAL_IMAGES_DIR``.
DEFAULT_EXTERNAL_ROOT = Path("/Volumes/Ventoy")

#: A 5-level encoder (resnet34 out_indices 0..4) downsamples by 32, so a size
#: that is not a multiple of 32 loses pixels at the last up-sample.
_ENCODER_STRIDE = 32

_VALID_LOG_LEVELS = frozenset({"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"})


def repo_root() -> Path:
    """Repository root, located by marker file rather than by hop count."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    # Fallback: sentinel_core/config.py -> repo root.
    return Path(__file__).resolve().parents[1]


def _resolve(path: Path | str, base: Path) -> Path:
    """Absolute-ise ``path``; relative paths are anchored to ``base``."""
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


class PathSettings(BaseModel):
    """Filesystem locations. All optional; each has a computed default."""

    model_config = ConfigDict(extra="ignore", validate_default=True)

    data_dir: Path = Field(
        default_factory=lambda: repo_root() / "data",
        description="Root for derived data. Overridable with SENTINEL_DATA_DIR.",
    )
    checkpoint_dir: Path = Field(default_factory=lambda: repo_root() / "checkpoints")
    runs_dir: Path = Field(default_factory=lambda: repo_root() / "runs")
    log_dir: Path = Field(default_factory=lambda: repo_root() / "logs")

    @field_validator("data_dir", "checkpoint_dir", "runs_dir", "log_dir", mode="after")
    @classmethod
    def _anchor(cls, value: Path) -> Path:
        return _resolve(value, repo_root())


class DataSourceSettings(BaseModel):
    """Where the real and synthetic datasets live, and how far we may substitute.

    ``allow_synthetic_fallback`` is the whole safety model in one flag. It is
    **off** by default, so asking for the real dataset and not finding it is an
    error. Turning it on permits a labelled substitution — the run is then
    stamped ``synthetic_mock`` and
    :attr:`sentinel_core.provenance.DataProvenance.supports_scientific_claim`
    is False for everything derived from it. The point is that the substitution
    is a decision someone made, not a branch the code took quietly.
    """

    model_config = ConfigDict(extra="ignore", validate_default=True)

    real_images_dir: Path = Field(
        default_factory=lambda: DEFAULT_EXTERNAL_ROOT / "Oil",
        description="Flat directory of real 2-band SAR GeoTIFFs. Read in place.",
    )
    real_masks_dir: Path = Field(
        default_factory=lambda: repo_root() / "data" / "oil_spill_masks",
        description="Flat directory of real masks, paired to images by file stem.",
    )
    synthetic_root: Path = Field(
        default_factory=lambda: repo_root() / "data" / "synthetic",
        description="Synthetic dataset root with {train,val,test}/{images,masks}.",
    )
    allow_synthetic_fallback: bool = False
    require_bands: int = Field(default=2, ge=1, le=64)
    #: Group tiles by the coarse grid cell containing their centroid, as a proxy
    #: for "same acquisition", so train/test do not share a pass. Off by default
    #: because it requires reading the header of every scene, and on the external
    #: archive a *cold* file open costs ~0.3 s — roughly six minutes for 1200
    #: files. The result is cached in the index, so the cost is paid once.
    spatial_scene_grouping: bool = False
    #: Refuse to start if the real dataset is on a filesystem we may write to.
    #: Guards the "train in place" contract: the source archive is input only.
    real_read_only_expected: bool = True

    @field_validator("real_images_dir", "real_masks_dir", "synthetic_root", mode="after")
    @classmethod
    def _anchor(cls, value: Path) -> Path:
        # The external mount is absolute in practice; anchoring is a no-op for
        # it and fixes relative overrides.
        return _resolve(value, repo_root())


class TrainingSettings(BaseModel):
    """Everything the trainer needs that is not a path."""

    model_config = ConfigDict(extra="ignore", validate_default=True)

    epochs: int = Field(default=2, ge=1, le=1000)
    batch_size: int = Field(default=4, ge=1, le=256)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    image_size: int = Field(default=512, ge=64, le=4096)
    num_workers: int = Field(default=0, ge=0, le=32)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = 42
    val_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    encoder: str = "resnet34"
    encoder_weights: Literal["imagenet", "none"] = "imagenet"
    bce_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    dice_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    #: Cap the number of scenes. The real archive is 1200 x 32 MB and streams
    #: off a USB disk at ~47 MB/s; a laptop-sized run needs a bound, and an
    #: unbounded default would make `--epochs 1` take an hour.
    max_scenes: int | None = Field(default=None, ge=1)
    #: Hard cap on optimiser steps per epoch, for smoke runs.
    max_steps_per_epoch: int | None = Field(default=None, ge=1)
    #: Deterministic training: disables the shuffle RNG and augmentation jitter.
    deterministic: bool = False

    @field_validator("image_size")
    @classmethod
    def _divisible_by_encoder_stride(cls, value: int) -> int:
        if value % _ENCODER_STRIDE != 0:
            raise ValueError(
                f"image_size must be a multiple of {_ENCODER_STRIDE} for a 5-level "
                f"UNet++ encoder (got {value}); otherwise the final up-sample "
                "cannot reproduce the input resolution."
            )
        return value

    @model_validator(mode="after")
    def _check_fractions_and_loss(self) -> TrainingSettings:
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                "val_fraction + test_fraction must leave at least one scene for "
                f"training (got {self.val_fraction} + {self.test_fraction})."
            )
        if self.bce_weight + self.dice_weight <= 0.0:
            raise ValueError("bce_weight + dice_weight must be > 0; the loss would be constant.")
        return self


class LoggingSettings(BaseModel):
    """Structured-logging configuration."""

    model_config = ConfigDict(extra="ignore", validate_default=True)

    level: str = "INFO"
    json_output: bool = False
    #: When False, no file sink is added even though ``paths.log_dir`` exists.
    to_file: bool = False

    @field_validator("level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        upper = value.strip().upper()
        if upper not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"unknown log level {value!r}; expected one of "
                f"{', '.join(sorted(_VALID_LOG_LEVELS))}"
            )
        return upper


class Settings(BaseSettings):
    """The root configuration object.

    Environment variables are prefixed ``SENTINEL_``; nested groups use ``__``::

        SENTINEL_DATA_SOURCE=real
        SENTINEL_TRAINING__EPOCHS=10
        SENTINEL_LOGGING__JSON_OUTPUT=true
        SENTINEL_DATASOURCES__REAL_IMAGES_DIR=/Volumes/Ventoy/Oil

    Unknown keys are ignored rather than fatal, so the shared ``.env`` (which
    carries database URLs, API keys and CDSE credentials for other services)
    can be loaded without every service having to agree on one schema.
    """

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_nested_delimiter="__",
        env_file=str(repo_root() / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    environment: Environment = "dev"
    #: The runtime switch. `real` reads the archive in place; `synthetic` reads
    #: the generated dataset. Neither falls back to the other.
    data_source: DataSourceKind = "synthetic"
    run_name: str | None = None

    paths: PathSettings = Field(default_factory=PathSettings)
    datasources: DataSourceSettings = Field(default_factory=DataSourceSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    def resolved_log_dir(self) -> Path | None:
        """Log directory when file logging is on, else ``None``."""
        return self.paths.log_dir if self.logging.to_file else None

    def describe(self) -> dict[str, object]:
        """A log-safe view of the configuration.

        Paths and switches only. There are no secrets in this object by
        construction, and the shared ``.env`` values that *are* secret are
        ignored by ``extra="ignore"`` and never enter it.
        """
        return {
            "environment": self.environment,
            "data_source": self.data_source,
            "run_name": self.run_name,
            "paths": {
                "data_dir": str(self.paths.data_dir),
                "checkpoint_dir": str(self.paths.checkpoint_dir),
                "runs_dir": str(self.paths.runs_dir),
                "log_dir": str(self.paths.log_dir),
            },
            "datasources": {
                "real_images_dir": str(self.datasources.real_images_dir),
                "real_masks_dir": str(self.datasources.real_masks_dir),
                "synthetic_root": str(self.datasources.synthetic_root),
                "allow_synthetic_fallback": self.datasources.allow_synthetic_fallback,
                "require_bands": self.datasources.require_bands,
                "spatial_scene_grouping": self.datasources.spatial_scene_grouping,
            },
            "training": self.training.model_dump(),
            "logging": self.logging.model_dump(),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, parsed and validated once.

    Raises:
        ConfigError: when the environment is present but invalid. The original
            pydantic error is attached as ``__cause__`` and its per-field detail
            is folded into the message, because "validation failed" without the
            field name is not actionable.
    """
    try:
        return Settings()
    except Exception as exc:  # pydantic ValidationError, or a bad .env encoding
        raise ConfigError(
            f"SENTINEL configuration is invalid: {exc}",
            reason="config_invalid",
        ) from exc


def reload_settings() -> Settings:
    """Re-read the environment. Intended for tests and long-lived processes."""
    get_settings.cache_clear()
    return get_settings()
