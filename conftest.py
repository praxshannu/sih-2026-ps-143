"""Root pytest configuration for the SENTINEL monorepo.

Two problems are solved here, both of which otherwise make `pytest` fail
before a single test body runs.

1. A temp root the sandbox will not let pytest reuse
   Several SENTINEL tests need `tmp_path`. In a sandboxed environment the
   host-operation broker rejects `mkdir` inside the system temp directory, and
   — worse — it raises `PermissionError: EEXIST` rather than
   `FileExistsError`. That matters because `Path.mkdir(exist_ok=True)` only
   suppresses `FileExistsError`; a `PermissionError` propagates. So once
   `pytest-of-<user>` exists from a previous run, every `tmp_path` test errors
   during setup forever. It looks like a broken test suite; it is a broken
   temp root.

   Fix: give each session a *fresh* temp root that is guaranteed not to exist
   yet, inside the workspace. Nothing is ever reused, so `mkdir` never hits the
   EEXIST path. The root is also purged of previous runs so it cannot grow
   without bound. Override the location with `SENTINEL_PYTEST_TMP`.

2. Duplicate top-level `app` packages
   Every service ships its own `services/<svc>/app/` package, so in a single
   interpreter they all contend for the module name `app`. Per-service test
   modules handle this by purging `app` from `sys.modules` before importing;
   this root conftest deliberately imports no service code so it cannot win the
   race itself.

Nothing here weakens an assertion or skips a test.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import uuid
from pathlib import Path

_KEEP_RUNS = 3


def _prune_old_runs(root: Path) -> None:
    """Remove previous session temp dirs, newest kept, so disk cannot grow."""
    try:
        runs = sorted(
            (p for p in root.iterdir() if p.is_dir() and p.name.startswith("run-")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:  # pragma: no cover - environment dependent
        return
    for stale in runs[_KEEP_RUNS:]:
        shutil.rmtree(stale, ignore_errors=True)


def _install_local_temp_root() -> None:
    root = Path(
        os.environ.get("SENTINEL_PYTEST_TMP") or (Path(__file__).resolve().parent / ".pytest_tmp")
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - environment dependent
        return

    _prune_old_runs(root)

    # Unique per session: pytest's basetemp mkdir must never see an existing dir.
    session_tmp = root / f"run-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        session_tmp.mkdir(parents=True, exist_ok=False)
    except FileExistsError:  # pragma: no cover - uuid collision
        session_tmp = root / f"run-{os.getpid()}-{uuid.uuid4().hex}"
        session_tmp.mkdir(parents=True, exist_ok=False)
    except OSError:  # pragma: no cover - environment dependent
        return

    os.environ["TMPDIR"] = str(session_tmp)
    # tempfile caches its answer in a module global; reset it so the new TMPDIR
    # is honoured by anything already imported.
    tempfile.tempdir = str(session_tmp)
    atexit.register(shutil.rmtree, session_tmp, True)


# Must run at import time: pytest resolves the temp root when the session-scoped
# `tmp_path_factory` fixture is first used, which is after conftest import.
_install_local_temp_root()
