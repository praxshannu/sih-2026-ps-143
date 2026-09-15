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


def test_scene_grouping_defaults_to_per_file() -> None:
    """The default must not read removable media: a fresh clone has no archive."""
    assert DataSourceSettings().resolved_scene_grouping() == "per_file"


def test_the_legacy_spatial_flag_resolves_to_the_grid() -> None:
    cfg = DataSourceSettings(spatial_scene_grouping=True)
    assert cfg.resolved_scene_grouping() == "grid"
    # The alias is not rewritten in place; both keys survive into the dump so a
    # reader can see which one was set.
    assert cfg.spatial_scene_grouping is True


def test_an_explicit_mode_beats_the_legacy_flag() -> None:
    cfg = DataSourceSettings(spatial_scene_grouping=True, scene_grouping="footprint")
    assert cfg.resolved_scene_grouping() == "footprint"


def test_the_legacy_flag_does_not_override_an_explicit_per_file() -> None:
    """A stale boolean must not overrule a mode the operator actually wrote.

    This is why the resolver reads ``model_fields_set`` rather than comparing
    ``scene_grouping`` against its default: the two cases are indistinguishable
    from the value alone, and treating a deliberate ``per_file`` as "unset"
    would silently re-enable a header scan the operator had turned off.
    """
    cfg = DataSourceSettings(spatial_scene_grouping=True, scene_grouping="per_file")
    assert cfg.resolved_scene_grouping() == "per_file"


def test_scene_grouping_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        DataSourceSettings(scene_grouping="spatial")  # type: ignore[arg-type]


def test_scene_grouping_is_settable_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTINEL_DATASOURCES__SCENE_GROUPING", "footprint")
    settings = reload_settings()
    assert settings.datasources.resolved_scene_grouping() == "footprint"
    datasources = settings.describe()["datasources"]
    assert isinstance(datasources, dict)
    assert datasources["scene_grouping"] == "footprint"
    monkeypatch.delenv("SENTINEL_DATASOURCES__SCENE_GROUPING")
    reload_settings()


def test_an_env_supplied_mode_also_beats_the_legacy_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``model_fields_set`` must be populated from the environment, not just kwargs.

    The whole point of reading it is that a value arriving from a deployment's
    env file counts as deliberate. If it were only populated by constructor
    arguments, every env-configured deployment would still be silently
    overridden by a stale boolean.
    """
    monkeypatch.setenv("SENTINEL_DATASOURCES__SPATIAL_SCENE_GROUPING", "true")
    monkeypatch.setenv("SENTINEL_DATASOURCES__SCENE_GROUPING", "per_file")
    settings = reload_settings()
    assert settings.datasources.spatial_scene_grouping is True
    assert settings.datasources.resolved_scene_grouping() == "per_file"
    monkeypatch.delenv("SENTINEL_DATASOURCES__SPATIAL_SCENE_GROUPING")
    monkeypatch.delenv("SENTINEL_DATASOURCES__SCENE_GROUPING")
    reload_settings()
