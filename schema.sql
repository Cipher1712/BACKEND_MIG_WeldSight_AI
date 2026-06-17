-- MIG-WeldSight AI · PostgreSQL schema
-- Apply once after attaching a Postgres plugin on Railway:
--   psql $DATABASE_URL -f schema.sql

CREATE TABLE IF NOT EXISTS profiles (
    id              SERIAL PRIMARY KEY,
    material        TEXT NOT NULL,
    thickness_mm    NUMERIC(6,2) NOT NULL,
    learned_k       NUMERIC(6,3) NOT NULL,
    mean_score      NUMERIC(8,4) NOT NULL,
    std_score       NUMERIC(8,4) NOT NULL,
    voltage_min     NUMERIC(8,3),
    voltage_max     NUMERIC(8,3),
    rms_min         NUMERIC(8,3),
    rms_max         NUMERIC(8,3),
    trained_windows INTEGER NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (material, thickness_mm)
);

CREATE TABLE IF NOT EXISTS anomaly_events (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    material        TEXT NOT NULL,
    thickness_mm    NUMERIC(6,2) NOT NULL,
    distance_mm     NUMERIC(10,3),
    anomaly_score   NUMERIC(10,4),
    threshold       NUMERIC(10,4),
    physics_label   TEXT,
    severity        TEXT,
    quality_index   INTEGER,
    voltage_features JSONB
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON anomaly_events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_profile ON anomaly_events (material, thickness_mm);

CREATE TABLE IF NOT EXISTS weld_records (
    id              BIGSERIAL PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    material        TEXT,
    thickness_mm    NUMERIC(6,2),
    window_count    INTEGER DEFAULT 0,
    anomaly_count   INTEGER DEFAULT 0,
    avg_quality     INTEGER
);