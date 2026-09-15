"""Single source of truth for where detect reads and writes SAR scenes.

Why this module exists
----------------------
The scene directory used to be computed independently in two places:
``main_det.py`` walked four ``parent`` hops up, ``routers/scene.py`` walked
five. Two copies of one rule is two chances to disagree, and they did — both
were at one point hardcoded to ``/app/data/sar``. That path is correct *inside
the container* and wrong everywhere else, which is the worst kind of wrong: a
native run starts cleanly, reports ``scene_count: 0``, accepts uploads, and
writes them to a directory the rest of the process never reads.

Resolution order
----------------
1. ``SENTINEL_SAR_DIR`` — explicit, always wins. The container image sets it
   to ``/app/data/sar`` in its own environment, so there is exactly one rule
   here and no ``if running_in_docker`` branch to get out of sync.
2. ``<repo root>/data/sar`` — the native/development layout.

The repo root is found by walking up for ``pyproject.toml`` rather than by
counting ``parent`` hops. A hop count is a fact about the *current* directory
depth, so moving this file one level silently repoints the data directory;
a marker file is a fact about the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["SAR_DIR", "repo_root", "sar_dir"]

_ENV_VAR = "SENTINEL_SAR_DIR"
_ROOT_MARKER = "pyproject.toml"


def repo_root() -> Path:
    """Repository root, located by marker file, not by hop count."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / _ROOT_MARKER).is_file():
            return candidate
    # No marker found (installed as a package, or a stripped tree). Fall back
    # to the historical depth: services/detect/app/paths.py -> repo root.
    return Path(__file__).resolve().parents[3]


def sar_dir() -> Path:
    """Directory holding SAR scenes, from ``SENTINEL_SAR_DIR`` or the repo."""
    override = os.environ.get(_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return repo_root() / "data" / "sar"


#: Resolved once at import. Call ``sar_dir()`` instead when the environment may
#: have changed since import (tests, or a process that reloads its config).
SAR_DIR: Path = sar_dir()
