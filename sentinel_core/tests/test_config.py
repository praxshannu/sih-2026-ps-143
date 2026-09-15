"""Configuration tests.

The point of these is not that pydantic works. It is that the *contract* this
project relies on holds: a mistyped value fails loudly, a mistyped key does not
break startup, relative paths mean the same thing everywhere, and the log-safe
view of the configuration contains no secrets.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from sentinel_core.config import (
    DataSourceSettings,
    LoggingSettings,
    Settings,
    TrainingSettings,
    get_settings,
    reload_settings,
    repo_root,
)
from sentinel_core.errors import ConfigError


def test_repo_root_is_the_directory_holding_pyproject() -> None:
    assert (repo_root() / "pyproject.toml").is_file()
    assert (repo_root() / "sentinel_core").is_dir()


def test_defaults_are_the_laptop_profile() -> None:
    settings = Settings()
    assert settings.environment == "dev"
    # Synthetic is the default on purpose: the real archive is on removable
    # media, and a default that reads removable media fails on a fresh clone.
    assert settings.data_source == "synthetic"
    assert settings.datasources.allow_synthetic_fallback is False
    assert settings.datasources.require_bands == 2
    assert settings.training.num_workers == 0  # macOS + MPS + fork is a footgun


def test_env_override_switches_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_DATA_SOURCE", "real")
    monkeypatch.setenv("SENTINEL_TRAINING__EPOCHS", "7")
    settings = reload_settings()
    assert settings.data_source == "real"
    assert settings.training.epochs == 7
    monkeypatch.delenv("SENTINEL_DATA_SOURCE")
    monkeypatch.delenv("SENTINEL_TRAINING__EPOCHS")
    reload_settings()


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("SENTINEL_DATA_SOURCE", "reel"),
        ("SENTINEL_TRAINING__EPOCHS", "zero"),
        ("SENTINEL_TRAINING__EPOCHS", "0"),
        ("SENTINEL_TRAINING__BATCH_SIZE", "-1"),
        ("SENTINEL_TRAINING__LEARNING_RATE", "0"),
        ("SENTINEL_TRAINING__IMAGE_SIZE", "500"),
        ("SENTINEL_LOGGING__LEVEL", "CHATTY"),
        ("SENTINEL_DATASOURCES__REQUIRE_BANDS", "99"),
    ],
)
def test_bad_values_are_rejected(monkeypatch: pytest.MonkeyPatch, var: str, value: str) -> None:
    """A typo in a value must not silently train something else."""
    monkeypatch.setenv(var, value)
    with pytest.raises(ConfigError):
        reload_settings()
    monkeypatch.delenv(var)
    reload_settings()


def test_unknown_keys_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared .env carries other services' keys; they must not be fatal."""
    monkeypatch.setenv("SENTINEL_SECRET_KEY", "not-ours")
    monkeypatch.setenv("SENTINEL_DEMO_MODE", "true")
    settings = reload_settings()
    assert settings.data_source in {"real", "synthetic"}
    monkeypatch.delenv("SENTINEL_SECRET_KEY")
    monkeypatch.delenv("SENTINEL_DEMO_MODE")
    reload_settings()


def test_relative_paths_anchor_to_the_repo_not_the_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """`SENTINEL_DATA_DIR=./data` must not depend on where the process started."""
    monkeypatch.setenv("SENTINEL_PATHS__DATA_DIR", "./data")
    settings = reload_settings()
    assert settings.paths.data_dir == (repo_root() / "data").resolve()
    monkeypatch.delenv("SENTINEL_PATHS__DATA_DIR")
    reload_settings()


def test_image_size_must_suit_the_encoder() -> None:
    with pytest.raises(ValidationError) as caught:
        TrainingSettings(image_size=500)
    assert "multiple of" in str(caught.value)


def test_splits_must_leave_something_to_train_on() -> None:
    with pytest.raises(ValidationError) as caught:
        TrainingSettings(val_fraction=0.6, test_fraction=0.5)
    assert "at least one scene" in str(caught.value)


def test_loss_weights_cannot_both_be_zero() -> None:
    with pytest.raises(ValidationError) as caught:
        TrainingSettings(bce_weight=0.0, dice_weight=0.0)
    assert "constant" in str(caught.value)


def test_describe_contains_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env` holds a real SENTINEL_SECRET_KEY; it must not leak into logs."""
    monkeypatch.setenv("SENTINEL_SECRET_KEY", "super-secret-value")
    settings = reload_settings()
    rendered = json.dumps(settings.describe())
    assert "super-secret-value" not in rendered
    assert "secret" not in rendered.lower()
    monkeypatch.delenv("SENTINEL_SECRET_KEY")
    reload_settings()


def test_get_settings_is_cached_until_reloaded() -> None:
    first = get_settings()
    assert first is get_settings()
    assert reload_settings() is not first


def test_log_dir_is_none_unless_file_logging_is_on() -> None:
    assert Settings(logging=LoggingSettings(to_file=False)).resolved_log_dir() is None
    assert Settings(logging=LoggingSettings(to_file=True)).resolved_log_dir() is not None


def test_datasource_settings_reject_absurd_band_counts() -> None:
    with pytest.raises(ValidationError):
        DataSourceSettings(require_bands=0)
    with pytest.raises(ValidationError):
        DataSourceSettings(require_bands=1000)
