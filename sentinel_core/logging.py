"""Structured logging for SENTINEL.

The project rule (``AGENTS.md``) is *loguru, never print*. The rule the code
actually needed and did not have is that a log line should be **machine
readable when a machine is reading it**. Two sinks solve both halves:

* a human sink for a terminal — compact, coloured, one line per event;
* a JSON sink for a collector — ``serialize=True``, so ``extra`` fields
  survive as fields instead of being interpolated into a sentence and lost.

Everything that runs binds a ``service`` name and, where relevant, a ``run_id``
and ``data_source``, so a line can be traced back to the run that produced it
without correlating timestamps by eye.

Standard-library logging is intercepted rather than ignored: uvicorn, httpx and
torch all log through ``logging``, and letting those bypass the sink is how a
structured log ends up with unstructured holes.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

from loguru import logger

__all__ = [
    "DEFAULT_QUIET_LOGGERS",
    "InterceptHandler",
    "bind_context",
    "configure_logging",
    "get_logger",
    "logger",
    "quiet_logger",
]

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{extra[service]}</cyan> "
    "<dim>|</dim> <level>{message}</level>"
)

#: loguru's default ``extra`` is empty, so a format referencing ``extra[service]``
#: raises KeyError on the first line logged. A default makes the format total.
_DEFAULT_EXTRA: dict[str, Any] = {"service": "-", "run_id": "-", "data_source": "-"}

#: Third-party loggers that are correct and useless at volume.
#:
#: ``rasterio`` is the one that matters here. GDAL emits
#: ``CPL_AppDefined ... Sum of Photometric type-related color channels and
#: ExtraSamples doesn't match SamplesPerPixel`` once per 2-band float32
#: GeoTIFF. The file is fine — GDAL is describing how it will *interpret* the
#: extra sample — but indexing 1200 scenes produced 1200 warning lines and
#: buried the one line that mattered. Intercepting stdlib logging at the root
#: (which is the point of this module) means every library's chatter arrives
#: unless it is named here.
DEFAULT_QUIET_LOGGERS: Mapping[str, int] = {
    "rasterio": logging.ERROR,
    "matplotlib": logging.WARNING,
    "urllib3": logging.WARNING,
    "PIL": logging.WARNING,
    "h5py": logging.WARNING,
    "numexpr": logging.WARNING,
    "asyncio": logging.WARNING,
    "fiona": logging.ERROR,
}


class InterceptHandler(logging.Handler):
    """Route ``logging`` records into loguru, preserving level and caller."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # ``currentframe()`` is typed as returning a frame, but the walk
        # terminates on ``f_back`` being None, so the variable has to admit it.
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(
    *,
    service: str = "sentinel",
    level: str = "INFO",
    json_output: bool = False,
    log_dir: Path | str | None = None,
    log_filename: str | None = None,
    intercept_stdlib: bool = True,
    quiet: Mapping[str, int] | None = None,
    colorize: bool | None = None,
) -> None:
    """Install SENTINEL's log sinks, replacing any existing configuration.

    Idempotent on purpose: calling it twice (a CLI that configures logging, then
    imports a module that also configures it) must not double every line.

    Args:
        service: bound to every record as ``extra["service"]``.
        level: minimum level for the console sink.
        json_output: emit JSON lines on stderr instead of the human format.
        log_dir: when given, also write ``<log_dir>/<log_filename>`` with
            rotation. Created if absent.
        log_filename: defaults to ``<service>.log``.
        intercept_stdlib: forward ``logging`` records into loguru.
        quiet: logger-name -> level overrides. Defaults to
            :data:`DEFAULT_QUIET_LOGGERS`; pass ``{}`` to hear everything.
        colorize: force colour on/off; ``None`` lets loguru decide from the
            stream (it correctly disables colour when stderr is a pipe).
    """
    logger.remove()
    logger.configure(extra=dict(_DEFAULT_EXTRA))

    sink_kwargs: dict[str, Any] = {"level": level}
    if colorize is not None:
        sink_kwargs["colorize"] = colorize

    if json_output:
        # serialize=True emits the whole record as JSON, extra fields included.
        logger.add(sys.stderr, serialize=True, backtrace=False, diagnose=False, **sink_kwargs)
    else:
        logger.add(
            sys.stderr,
            format=_CONSOLE_FORMAT,
            backtrace=False,
            diagnose=False,
            **sink_kwargs,
        )

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        logger.add(
            directory / (log_filename or f"{service}.log"),
            level=level,
            rotation="64 MB",
            retention=8,
            compression="gz",
            # JSON on disk regardless of the console format: the file is for
            # machines, the terminal is for people.
            serialize=True,
            backtrace=False,
            diagnose=False,
            encoding="utf-8",
        )

    if intercept_stdlib:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore"):
            stdlib_logger = logging.getLogger(name)
            stdlib_logger.handlers = [InterceptHandler()]
            stdlib_logger.propagate = False

    for name, logger_level in (DEFAULT_QUIET_LOGGERS if quiet is None else quiet).items():
        logging.getLogger(name).setLevel(logger_level)

    logger.bind(service=service)


@contextmanager
def quiet_logger(name: str, level: int = logging.ERROR) -> Iterator[None]:
    """Silence one stdlib logger for the duration of a block.

    Used around tight loops over third-party I/O, where a per-file warning is
    both correct and unreadable at scale.
    """
    target = logging.getLogger(name)
    previous = target.level
    target.setLevel(level)
    try:
        yield
    finally:
        target.setLevel(previous)


def get_logger(service: str, **extra: Any) -> Any:
    """A logger bound to ``service`` plus any extra structured fields."""
    return logger.bind(service=service, **extra)


@contextmanager
def bind_context(**extra: Any) -> Iterator[Any]:
    """Temporarily bind extra fields to every record logged in the block.

    Used to attach ``run_id`` and ``data_source`` to a whole training run
    without threading them through every call::

        with bind_context(run_id=run.id, data_source="real"):
            logger.info("epoch complete")   # carries both fields
    """
    with logger.contextualize(**extra):
        yield logger
