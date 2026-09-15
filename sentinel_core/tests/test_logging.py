"""Structured-logging tests.

Three properties are load-bearing:

* configuring twice must not double every line (a CLI configures logging, then
  imports a module that configures it again);
* the JSON sink must keep ``extra`` as fields, because that is the entire
  reason for having it;
* the noisy third-party loggers must actually be quiet, because a 1200-file
  index produced 1200 warning lines and buried the one that mattered.
"""

from __future__ import annotations

import io
import json
import logging

from loguru import logger

from sentinel_core.logging import (
    DEFAULT_QUIET_LOGGERS,
    bind_context,
    configure_logging,
    get_logger,
    quiet_logger,
)


def _json_sink() -> io.StringIO:
    buffer = io.StringIO()
    logger.add(buffer, serialize=True)
    return buffer


def _records(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().strip().splitlines() if line.strip()]


def test_configure_logging_does_not_duplicate_records(capsys) -> None:
    configure_logging(service="dup", level="INFO", intercept_stdlib=False)
    configure_logging(service="dup", level="INFO", intercept_stdlib=False)
    logger.bind(service="dup").info("exactly-once-marker")
    captured = capsys.readouterr()
    assert captured.err.count("exactly-once-marker") == 1


def test_human_sink_includes_the_service_name(capsys) -> None:
    configure_logging(service="myservice", level="INFO", intercept_stdlib=False)
    logger.bind(service="myservice").info("hello")
    assert "myservice" in capsys.readouterr().err


def test_json_sink_keeps_extra_as_fields() -> None:
    configure_logging(service="json", json_output=True, intercept_stdlib=False)
    buffer = _json_sink()
    logger.bind(service="json", run_id="r-1").info("structured", epoch=3)
    logger.remove()

    payload = _records(buffer)[-1]["record"]
    assert payload["message"] == "structured"
    assert payload["extra"]["epoch"] == 3
    assert payload["extra"]["run_id"] == "r-1"
    assert payload["extra"]["service"] == "json"


def test_default_extra_makes_the_format_total() -> None:
    """A bare logger.info must not KeyError on the missing `service` field."""
    configure_logging(service="t", level="INFO", intercept_stdlib=False)
    buffer = _json_sink()
    logger.info("no explicit bind")
    logger.remove()
    assert _records(buffer)[-1]["record"]["extra"]["service"] == "-"


def test_bind_context_attaches_and_then_releases_fields() -> None:
    configure_logging(service="ctx", json_output=True, intercept_stdlib=False)
    buffer = _json_sink()
    with bind_context(run_id="run-7", data_source="real"):
        logger.bind(service="ctx").info("inside")
    logger.bind(service="ctx").info("outside")
    logger.remove()

    by_message = {row["record"]["message"]: row["record"] for row in _records(buffer)}
    assert by_message["inside"]["extra"]["run_id"] == "run-7"
    assert by_message["inside"]["extra"]["data_source"] == "real"
    # Outside the block the binding is gone, not sticky.
    assert by_message["outside"]["extra"]["run_id"] == "-"


def test_get_logger_binds_the_service() -> None:
    configure_logging(service="bound", json_output=True, intercept_stdlib=False)
    buffer = _json_sink()
    get_logger("bound", phase="smoke").info("hi")
    logger.remove()
    assert _records(buffer)[-1]["record"]["extra"]["phase"] == "smoke"


def test_quiet_logger_silences_then_restores() -> None:
    target = logging.getLogger("rasterio")
    before = target.level
    with quiet_logger("rasterio"):
        assert target.level == logging.ERROR
    assert target.level == before


def test_configure_logging_applies_the_quiet_list() -> None:
    configure_logging(service="q", level="INFO", intercept_stdlib=False)
    for name, expected in DEFAULT_QUIET_LOGGERS.items():
        assert logging.getLogger(name).level == expected, name


def test_rasterio_chatter_does_not_reach_the_sink(capsys) -> None:
    """The exact failure: one CPL warning per scene, 1200 scenes."""
    configure_logging(service="q", level="INFO", intercept_stdlib=True)
    logging.getLogger("rasterio._env").warning(
        "CPL_AppDefined in 00000.tif: Sum of Photometric type-related color channels"
    )
    assert "CPL_AppDefined" not in capsys.readouterr().err


def test_stdlib_records_are_intercepted(capsys) -> None:
    configure_logging(service="q", level="INFO", intercept_stdlib=True)
    logging.getLogger("some.library").warning("a library said something")
    assert "a library said something" in capsys.readouterr().err
