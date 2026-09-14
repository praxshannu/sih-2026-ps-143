-- AIS Positions as TimescaleDB hypertable
CREATE TABLE IF NOT EXISTS ais_positions (
  id          BIGSERIAL,
  mmsi        VARCHAR(9) NOT NULL,
  vessel_name VARCHAR(100),
  vessel_type SMALLINT,
  lon         DOUBLE PRECISION NOT NULL,
  lat         DOUBLE PRECISION NOT NULL,
  sog         REAL,
  cog         REAL,
  heading     SMALLINT,
  nav_status  SMALLINT,
  imo_number  VARCHAR(10),
  flag_state  CHAR(3),
  timestamp   TIMESTAMPTZ NOT NULL,
  geom        GEOMETRY(POINT, 4326),
  source      VARCHAR(20) DEFAULT 'marinecadastre',
  is_interpolated BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_ais_geom ON ais_positions USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_ais_mmsi_time ON ais_positions (mmsi, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ais_timestamp ON ais_positions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ais_vessel_type ON ais_positions (vessel_type);
