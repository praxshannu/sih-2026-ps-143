-- Investigation Cases
CREATE TABLE IF NOT EXISTS investigation_cases (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_number     VARCHAR(20) UNIQUE NOT NULL,
  status          VARCHAR(20) DEFAULT 'OPEN',
  spill_id        UUID REFERENCES spill_detections(id),
  drift_id        UUID REFERENCES drift_results(id),
  primary_suspect UUID REFERENCES suspect_vessels(id),
  narrative       TEXT,
  evidence_hash   VARCHAR(128),
  case_file_path  TEXT,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW(),
  annotations     JSONB DEFAULT '[]',
  metadata        JSONB DEFAULT '{}'::jsonb,
  alert_sent      BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_case_number ON investigation_cases (case_number);
CREATE INDEX IF NOT EXISTS idx_case_status ON investigation_cases (status);
CREATE INDEX IF NOT EXISTS idx_case_created ON investigation_cases (created_at DESC);
