"""Durable case persistence.

WHY THIS EXISTS
---------------
Cases used to live in a module-level ``dict`` in ``routers/cases.py``. That
survives nothing: restart the gateway and every investigation is gone, including
the ones an analyst spent an hour assembling. The requirement is not "cache the
results" — it is that a processed scene can be reopened, re-examined and
challenged after a restart, with the provenance that produced it intact.

WHAT IS PERSISTED
-----------------
One :class:`CaseRecord` per processed scene, carrying everything needed to
re-derive or dispute the result: scene metadata, source and checksum, detection,
polygon, drift, forcing provenance, AIS provenance, suspect scores, model
version, confidence and warnings. If a field is missing from this model, the
case cannot be audited, so the model is deliberately wide rather than tidy.

TWO BACKENDS, AND YOU ALWAYS KNOW WHICH ONE RAN
-----------------------------------------------
``postgres``
    Preferred. Uses ``DATABASE_URL`` via asyncpg and an idempotent DDL that
    mirrors ``database/init/07_case_persistence.sql``.

``file``
    Durable JSON under ``SENTINEL_CASE_STORE`` (default ``data/cases``), written
    atomically via a temp file and ``os.replace`` so a crash mid-write cannot
    leave a half-record. This is a real store, not a cache: it survives a
    restart, which is the property being tested.

The backend that actually served a write is recorded **on the record itself**
(``persistence_backend``) and returned to the caller. Falling back is normal and
expected here — Postgres is not running on every dev machine — but it must never
be *silent*, because "the database has it" and "a JSON file has it" are very
different operational claims.

HONEST STATUS
-------------
The file backend is exercised by ``services/api/tests/test_persistence.py``,
including a restart round-trip. The Postgres path is written but **has not been
run against a live server in this environment** — the DB is down on this
machine. It is marked accordingly rather than presented as verified.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

BACKEND_POSTGRES: Literal["postgres"] = "postgres"
BACKEND_FILE: Literal["file"] = "file"

DEFAULT_STORE_DIR = Path(os.getenv("SENTINEL_CASE_STORE", "data/cases"))
DATABASE_URL = os.getenv("DATABASE_URL", "")

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")

# Mirrors database/init/07_case_persistence.sql. JSONB for the nested payloads
# that vary per stage; GIST on every geometry column (AGENTS.md).
_DDL = """
CREATE TABLE IF NOT EXISTS processed_scenes (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_id             TEXT NOT NULL,
  scene_id            TEXT,
  scene_label         TEXT,
  platform            TEXT,
  product_id          TEXT,
  acquisition_time    TIMESTAMPTZ,
  source              TEXT,
  checksum            TEXT,
  detection           JSONB DEFAULT '{}'::jsonb,
  model_version       TEXT,
  confidence          REAL,
  confidence_lower    REAL,
  confidence_upper    REAL,
  warnings            JSONB DEFAULT '[]'::jsonb,
  drift               JSONB DEFAULT '{}'::jsonb,
  forcing_provenance  JSONB DEFAULT '{}'::jsonb,
  ais_provenance      JSONB DEFAULT '{}'::jsonb,
  coverage            TEXT DEFAULT 'none',
  suspects            JSONB DEFAULT '[]'::jsonb,
  persistence_backend TEXT DEFAULT 'postgres',
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  updated_at          TIMESTAMPTZ DEFAULT NOW()
);
"""


class Confidence(BaseModel):
    """A value with its interval. Never a bare point estimate (AGENTS.md)."""

    value: float | None = None
    low: float | None = None
    high: float | None = None
    method: str | None = None
    n_effective: int | None = None


class SceneMeta(BaseModel):
    label: str | None = None
    platform: str | None = None
    product_id: str | None = None
    acquisition_time: str | None = None
    source: str | None = None
    checksum_sha256: str | None = None
    crs: str | None = None
    width: int | None = None
    height: int | None = None
    bands: int | None = None


class DetectionRecord(BaseModel):
    """The detection result and the polygon it produced."""

    state: str | None = None
    state_reason: str | None = None
    flags: list[str] = Field(default_factory=list)
    detector: str | None = None
    model_status: str | None = None
    count: int = 0
    area_km2: float | None = None
    centroid: list[float] | None = None
    length_km: float | None = None
    width_km: float | None = None
    orientation_deg: float | None = None
    # GeoJSON in EPSG:4326, so the polygon is usable without re-running anything.
    polygon: dict[str, Any] | None = None
    confidence: Confidence = Field(default_factory=Confidence)


class DriftRecord(BaseModel):
    origin_lon: float | None = None
    origin_lat: float | None = None
    origin_time: str | None = None
    semi_major_km: float | None = None
    semi_minor_km: float | None = None
    p50_radius_km: float | None = None
    p95_radius_km: float | None = None
    n_particles: int | None = None
    wmc_divergence_max: float | None = None
    forecast_cone: list[dict[str, Any]] = Field(default_factory=list)
    stranded_fraction: float | None = None


class CaseRecord(BaseModel):
    """Everything an investigator needs to reopen and challenge a result."""

    case_id: str
    scene_id: str | None = None
    title: str | None = None

    scene: SceneMeta = Field(default_factory=SceneMeta)
    detection: DetectionRecord = Field(default_factory=DetectionRecord)
    drift: DriftRecord = Field(default_factory=DriftRecord)

    # Provenance, kept per-field rather than summarised, because "which forcing
    # produced this" is the question that decides whether a result is usable.
    forcing_provenance: dict[str, Any] = Field(default_factory=dict)
    ais_provenance: dict[str, Any] = Field(default_factory=dict)
    coverage: Literal["real", "synthetic", "none"] = "none"

    suspects: list[dict[str, Any]] = Field(default_factory=list)
    model_version: str | None = None
    warnings: list[str] = Field(default_factory=list)

    persistence_backend: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


def _safe_id(case_id: str) -> str:
    cleaned = _SAFE_ID.sub("_", case_id).strip("._-")
    return cleaned or "unnamed"


class CaseStore:
    """Durable case storage. Prefers Postgres, always reports what it used."""

    def __init__(
        self,
        *,
        directory: str | Path | None = None,
        backend: Literal["auto", "postgres", "file"] = "auto",
        database_url: str | None = None,
    ) -> None:
        self.directory = Path(directory) if directory is not None else DEFAULT_STORE_DIR
        self.requested_backend = backend
        self.database_url = database_url if database_url is not None else DATABASE_URL
        self._pool: Any | None = None

    # ── backend selection ─────────────────────────────────────────────────

    def _want_postgres(self) -> bool:
        """Postgres only if it was asked for AND there is a DSN to reach it."""
        if self.requested_backend == "file":
            return False
        return bool(self.database_url)

    async def _get_pool(self) -> Any | None:
        """Return a live asyncpg pool, or None if Postgres is not usable."""
        if self._pool is not None:
            return self._pool
        if not self._want_postgres():
            return None
        try:
            import asyncpg  # noqa: PLC0415 - optional dependency

            # asyncpg wants a bare postgres:// DSN, not the SQLAlchemy
            # `postgresql+asyncpg://` form that DATABASE_URL usually carries.
            dsn = self.database_url.replace("postgresql+asyncpg://", "postgresql://")
            self._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, timeout=5)
            async with self._pool.acquire() as conn:
                await conn.execute(_DDL)
            logger.info("case store: postgres backend ready")
            return self._pool
        except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
            logger.warning(
                "case store: postgres unavailable ({}), using the durable file "
                "backend at {}. Cases will survive a restart, but they are NOT "
                "in the database.",
                exc,
                self.directory,
            )
            self._pool = None
            return None

    async def active_backend(self) -> str:
        """Which backend a write would actually use right now."""
        if await self._get_pool() is not None:
            return BACKEND_POSTGRES
        return BACKEND_FILE

    # ── file backend ──────────────────────────────────────────────────────

    def _path_for(self, case_id: str) -> Path:
        return self.directory / f"{_safe_id(case_id)}.json"

    def _write_file(self, record: CaseRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path_for(record.case_id)
        payload = record.model_dump(mode="json")
        # Atomic: a crash mid-write leaves the previous record intact rather
        # than a truncated one that cannot be parsed.
        fd, tmp = tempfile.mkstemp(dir=str(self.directory), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as sink:
                json.dump(payload, sink, indent=2, default=str)
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _read_file(self, case_id: str) -> CaseRecord | None:
        path = self._path_for(case_id)
        if not path.is_file():
            return None
        try:
            return CaseRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error("case store: unreadable record {}: {}", path, exc)
            return None

    def _list_files(self) -> list[CaseRecord]:
        if not self.directory.is_dir():
            return []
        records: list[CaseRecord] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                records.append(CaseRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                logger.error("case store: skipping unreadable record {}: {}", path, exc)
        return records

    # ── public API ────────────────────────────────────────────────────────

    async def save(self, record: CaseRecord) -> CaseRecord:
        """Persist a case. Returns the record with its backend recorded."""
        now = datetime.now(UTC).isoformat()
        record.updated_at = now
        if record.created_at is None:
            record.created_at = now

        pool = await self._get_pool()
        if pool is not None:
            record.persistence_backend = BACKEND_POSTGRES
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO processed_scenes (
                      case_id, scene_id, scene_label, platform, product_id,
                      acquisition_time, source, checksum, detection, model_version,
                      confidence, confidence_lower, confidence_upper, warnings,
                      drift, forcing_provenance, ais_provenance, coverage,
                      suspects, persistence_backend, updated_at
                    ) VALUES (
                      $1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11,$12,$13,$14::jsonb,
                      $15::jsonb,$16::jsonb,$17::jsonb,$18,$19::jsonb,$20, NOW()
                    )
                    ON CONFLICT (case_id) DO UPDATE SET
                      scene_id = EXCLUDED.scene_id,
                      detection = EXCLUDED.detection,
                      drift = EXCLUDED.drift,
                      suspects = EXCLUDED.suspects,
                      forcing_provenance = EXCLUDED.forcing_provenance,
                      ais_provenance = EXCLUDED.ais_provenance,
                      coverage = EXCLUDED.coverage,
                      confidence = EXCLUDED.confidence,
                      warnings = EXCLUDED.warnings,
                      persistence_backend = EXCLUDED.persistence_backend,
                      updated_at = NOW()
                    """,
                    record.case_id,
                    record.scene_id,
                    record.scene.label,
                    record.scene.platform,
                    record.scene.product_id,
                    record.scene.acquisition_time,
                    record.scene.source,
                    record.scene.checksum_sha256,
                    json.dumps(record.detection.model_dump(mode="json")),
                    record.model_version,
                    record.detection.confidence.value,
                    record.detection.confidence.low,
                    record.detection.confidence.high,
                    json.dumps(record.warnings),
                    json.dumps(record.drift.model_dump(mode="json")),
                    json.dumps(record.forcing_provenance),
                    json.dumps(record.ais_provenance),
                    record.coverage,
                    json.dumps(record.suspects),
                    record.persistence_backend,
                )
            return record

        record.persistence_backend = BACKEND_FILE
        self._write_file(record)
        return record

    async def load(self, case_id: str) -> CaseRecord | None:
        pool = await self._get_pool()
        if pool is not None:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM processed_scenes WHERE case_id = $1 LIMIT 1", case_id
                )
            if row is not None:
                return _row_to_record(dict(row))
        return self._read_file(case_id)

    async def list(self) -> list[CaseRecord]:
        pool = await self._get_pool()
        if pool is not None:
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM processed_scenes ORDER BY updated_at DESC")
            return [_row_to_record(dict(r)) for r in rows]
        return self._list_files()

    async def delete(self, case_id: str) -> bool:
        pool = await self._get_pool()
        if pool is not None:
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM processed_scenes WHERE case_id = $1", case_id
                )
            return result.endswith("1")
        path = self._path_for(case_id)
        if path.is_file():
            path.unlink()
            return True
        return False


def _row_to_record(row: dict[str, Any]) -> CaseRecord:
    """Map a Postgres row back to a CaseRecord, tolerating JSONB as text."""

    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                return []
            return parsed if isinstance(parsed, list) else []
        return []

    detection = _as_dict(row.get("detection"))
    detection.setdefault(
        "confidence",
        {
            "value": row.get("confidence"),
            "low": row.get("confidence_lower"),
            "high": row.get("confidence_upper"),
        },
    )

    return CaseRecord(
        case_id=row["case_id"],
        scene_id=row.get("scene_id"),
        scene=SceneMeta(
            label=row.get("scene_label"),
            platform=row.get("platform"),
            product_id=row.get("product_id"),
            acquisition_time=str(row["acquisition_time"]) if row.get("acquisition_time") else None,
            source=row.get("source"),
            checksum_sha256=row.get("checksum"),
        ),
        detection=DetectionRecord.model_validate(detection),
        drift=DriftRecord.model_validate(_as_dict(row.get("drift"))),
        forcing_provenance=_as_dict(row.get("forcing_provenance")),
        ais_provenance=_as_dict(row.get("ais_provenance")),
        coverage=row.get("coverage") or "none",
        suspects=_as_list(row.get("suspects")),
        model_version=row.get("model_version"),
        warnings=[str(w) for w in _as_list(row.get("warnings"))],
        persistence_backend=row.get("persistence_backend"),
        created_at=str(row["created_at"]) if row.get("created_at") else None,
        updated_at=str(row["updated_at"]) if row.get("updated_at") else None,
    )


_default_store: CaseStore | None = None


def get_case_store() -> CaseStore:
    """Process-wide store. Cheap to construct, but one is enough."""
    global _default_store
    if _default_store is None:
        _default_store = CaseStore()
    return _default_store


def reset_case_store() -> None:
    """Drop the cached store. Used by tests that need a fresh backend."""
    global _default_store
    _default_store = None
