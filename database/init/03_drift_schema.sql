-- Drift Results
CREATE TABLE IF NOT EXISTS drift_results (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  spill_id          UUID REFERENCES spill_detections(id),
  origin_lon        DOUBLE PRECISION,
  origin_lat        DOUBLE PRECISION,
  origin_ellipse    GEOMETRY(POLYGON, 4326),
  t_origin_est      TIMESTAMPTZ,
  t_origin_minus    INTERVAL,
  t_origin_plus     INTERVAL,
  confidence_level  REAL DEFAULT 0.95,
  particle_count    INTEGER,
  k_regime          VARCHAR(20),
  k_tensor_trace    REAL,
  forecast_paths    JSONB,
  shoreline_risk    JSONB,
  computed_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_spill ON drift_results (spill_id);
CREATE INDEX IF NOT EXISTS idx_drift_ellipse ON drift_results USING GIST (origin_ellipse);
CREATE INDEX IF NOT EXISTS idx_drift_computed ON drift_results (computed_at DESC);
