-- Spill Detections
CREATE TABLE IF NOT EXISTS spill_detections (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  detected_at     TIMESTAMPTZ NOT NULL,
  satellite_pass  VARCHAR(50),
  polygon         GEOMETRY(POLYGON, 4326) NOT NULL,
  area_km2        REAL,
  perimeter_km    REAL,
  centroid        GEOMETRY(POINT, 4326),
  orientation_deg REAL,
  age_hours_est   REAL,
  confidence      REAL CHECK (confidence BETWEEN 0 AND 1),
  lookalike_prob  REAL,
  wind_speed_ms   REAL,
  wind_dir_deg    REAL,
  current_speed   REAL,
  current_dir_deg REAL,
  sar_image_path  TEXT,
  metadata        JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_spill_geom ON spill_detections USING GIST (polygon);
CREATE INDEX IF NOT EXISTS idx_spill_centroid ON spill_detections USING GIST (centroid);
CREATE INDEX IF NOT EXISTS idx_spill_detected_at ON spill_detections (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_spill_confidence ON spill_detections (confidence DESC);
