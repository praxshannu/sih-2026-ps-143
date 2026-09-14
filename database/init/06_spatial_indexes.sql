-- Spatial Indexes for Performance
-- BRIN indexes for time-series data (AIS positions)
CREATE INDEX IF NOT EXISTS idx_ais_brin_time ON ais_positions USING BRIN (timestamp) WITH (pages_per_range = 32);

-- Composite spatial + temporal index for drift queries
CREATE INDEX IF NOT EXISTS idx_spill_spatial_temporal ON spill_detections USING GIST (polygon, detected_at);

-- Covering index for AIS spatiotemporal queries
CREATE INDEX IF NOT EXISTS idx_ais_mmsi_spatial_time ON ais_positions (mmsi, timestamp DESC) INCLUDE (lon, lat, sog, cog, nav_status);

-- ST_DWithin fast proximity query support
CREATE INDEX IF NOT EXISTS idx_ais_geom_gist ON ais_positions USING GIST (geom) WITH (fillfactor = 90);

-- Case status + created composite
CREATE INDEX IF NOT EXISTS idx_case_status_created ON investigation_cases (status, created_at DESC);
