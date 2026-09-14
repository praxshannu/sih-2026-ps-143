-- Case persistence for a fully processed scene.
--
-- One row per processed scene holds everything an investigator needs to
-- re-derive or challenge the result: scene metadata, source + checksum,
-- detection, polygon, drift, forcing provenance, AIS provenance, suspect
-- scores, model version, confidence and warnings.
--
-- Conventions (AGENTS.md): UUID primary keys, GIST index on EVERY geometry
-- column, JSONB for the nested payloads that vary per stage.
CREATE TABLE IF NOT EXISTS processed_scenes (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_id             TEXT NOT NULL,
  scene_id            TEXT,

  -- ── Scene metadata + provenance ────────────────────────────────────────
  scene_label         TEXT,
  platform            TEXT,                       -- S1A / S1B / S1C / S1D
  product_id          TEXT,
  acquisition_time    TIMESTAMPTZ,
  source              TEXT,                       -- e.g. cdse_sentinel_hub_process
  checksum            TEXT,                       -- SHA-256 of the ingested raster

  -- ── Detection ──────────────────────────────────────────────────────────
  polygon             GEOMETRY(POLYGON, 4326),
  centroid            GEOMETRY(POINT, 4326),
  detection           JSONB DEFAULT '{}'::jsonb,  -- method, area, per-polygon metrics
  model_version       TEXT,
  confidence          REAL,
  confidence_lower    REAL,                       -- Wilson 95% CI — never a bare point estimate
  confidence_upper    REAL,
  warnings            JSONB DEFAULT '[]'::jsonb,

  -- ── Drift + forcing provenance ─────────────────────────────────────────
  drift               JSONB DEFAULT '{}'::jsonb,
  origin_ellipse      GEOMETRY(POLYGON, 4326),
  forcing_provenance  JSONB DEFAULT '{}'::jsonb,  -- wind_source / current_source / synthetic flags

  -- ── AIS provenance + suspect scores ────────────────────────────────────
  ais_provenance      JSONB DEFAULT '{}'::jsonb,  -- coverage, provenance, reason
  coverage            TEXT DEFAULT 'none',        -- real | synthetic | none
  suspects            JSONB DEFAULT '[]'::jsonb,

  -- ── Bookkeeping ────────────────────────────────────────────────────────
  persistence_backend TEXT DEFAULT 'postgres',    -- postgres | json_fallback
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Every geometry column gets a GIST index (AGENTS.md).
CREATE INDEX IF NOT EXISTS idx_scene_polygon ON processed_scenes USING GIST (polygon);
CREATE INDEX IF NOT EXISTS idx_scene_centroid ON processed_scenes USING GIST (centroid);
CREATE INDEX IF NOT EXISTS idx_scene_origin_ellipse ON processed_scenes USING GIST (origin_ellipse);

-- One row per (case, scene): the upsert target. scene_id is nullable, so the
-- empty string stands in for "no scene id" to keep the index usable.
CREATE UNIQUE INDEX IF NOT EXISTS idx_scene_case_scene
  ON processed_scenes (case_id, COALESCE(scene_id, ''));

-- Lookup paths: by case, by scene, by checksum (chain-of-custody queries).
CREATE INDEX IF NOT EXISTS idx_scene_case ON processed_scenes (case_id);
CREATE INDEX IF NOT EXISTS idx_scene_scene_id ON processed_scenes (scene_id);
CREATE INDEX IF NOT EXISTS idx_scene_checksum ON processed_scenes (checksum);
CREATE INDEX IF NOT EXISTS idx_scene_created ON processed_scenes (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scene_coverage ON processed_scenes (coverage);
