-- Suspect Vessels
CREATE TABLE IF NOT EXISTS suspect_vessels (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_id           UUID,
  mmsi              VARCHAR(9) NOT NULL,
  vessel_name       VARCHAR(100),
  flag_state        CHAR(3),
  vessel_type       SMALLINT,
  imo_number        VARCHAR(10),
  rank              SMALLINT,
  composite_score   REAL CHECK (composite_score BETWEEN 0 AND 1),
  score_proximity   REAL,
  score_temporal    REAL,
  score_trajectory  REAL,
  score_anomaly     REAL,
  score_vessel_type REAL,
  ais_gap_minutes   REAL,
  min_distance_nm   REAL,
  track_segment     GEOMETRY(LINESTRING, 4326),
  anomalies         JSONB DEFAULT '[]',
  is_dark_vessel    BOOLEAN DEFAULT FALSE,
  confidence_lower  REAL,
  confidence_upper  REAL,
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_suspect_case ON suspect_vessels (case_id);
CREATE INDEX IF NOT EXISTS idx_suspect_mmsi ON suspect_vessels (mmsi);
CREATE INDEX IF NOT EXISTS idx_suspect_score ON suspect_vessels (composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_suspect_track ON suspect_vessels USING GIST (track_segment);
