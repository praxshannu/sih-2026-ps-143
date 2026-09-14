"""Load the repo-root ``.env`` into ``os.environ`` without a new dependency.

Why this exists
---------------
Every external capability in SENTINEL is gated behind a key (CDSE, CDSAPI,
AISStream, CMEMS). Starting uvicorn by hand — or under `make`, or in a
container that only mounts selected variables — silently drops them, and the
drift service then quietly degrades to synthetic forcing. That is the worst
possible failure mode: the run *succeeds*, returns numbers, and the numbers
are invented.

Importing this module at the top of an entry point makes "no key" loud
instead. Deliberately dependency-free (~25 lines) so it works in any venv,
including the torch-free ones.

Rules:
  * Never overrides an already-set variable — real env (Docker/K8s) wins.
  * Walks up from this file looking for ``.env``, so it works whether the
    service is launched from the repo root or from ``services/drift``.
  * Ignores comments/blank lines and strips one layer of matching quotes.
"""

from __future__ import annotations

import os
from pathlib import Path


def _parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # Tolerate an optional `export ` prefix.
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load_dotenv(start: Path | None = None, max_levels: int = 6) -> Path | None:
    """Populate ``os.environ`` from the nearest ``.env``. Returns its path."""
    here = (start or Path(__file__).resolve()).parent
    for _ in range(max_levels):
        candidate = here / ".env"
        if candidate.is_file():
            try:
                for key, value in _parse(candidate.read_text(encoding="utf-8")).items():
                    os.environ.setdefault(key, value)
            except OSError:
                return None
            return candidate
        if here.parent == here:
            break
        here = here.parent
    return None


def require(*names: str) -> dict[str, str]:
    """Return the named env vars, raising a clear error if any are missing.

    Call this from a service's lifespan so a misconfigured launch fails at
    boot rather than silently serving synthetic data.
    """
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Set them in the repo-root .env or export them before launch."
        )
    return {n: os.environ[n] for n in names}
