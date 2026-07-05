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
    event_timestamp_ms BIGINT,
    material        TEXT NOT NULL,
    thickness_mm    NUMERIC(6,2) NOT NULL,
    distance_mm     NUMERIC(10,3),
    distance_source TEXT,
    anomaly_score   NUMERIC(10,4),
    threshold       NUMERIC(10,4),
    physics_label   TEXT,
    severity        TEXT,
    quality_index   INTEGER,
    voltage_features JSONB,
    recording_session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON anomaly_events (ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_profile ON anomaly_events (material, thickness_mm);
CREATE INDEX IF NOT EXISTS idx_events_recording ON anomaly_events (recording_session_id);

CREATE TABLE IF NOT EXISTS recording_sessions (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL UNIQUE,
    start_timestamp     BIGINT NOT NULL,
    end_timestamp       BIGINT,
    duration_ms         BIGINT NOT NULL DEFAULT 0,
    sample_count        INTEGER NOT NULL DEFAULT 0,
    sampling_rate_hz    NUMERIC(10,3),
    distance_mm         NUMERIC(10,3),
    distance_source     TEXT NOT NULL DEFAULT 'Estimated',
    trained             BOOLEAN NOT NULL DEFAULT FALSE,
    healthy_baseline    BOOLEAN NOT NULL DEFAULT FALSE,
    notes               TEXT,
    csv_path            TEXT,
    csv_size_bytes      BIGINT NOT NULL DEFAULT 0,
    model_version_used  TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_recording_sessions_session_id ON recording_sessions (session_id);

CREATE TABLE IF NOT EXISTS telemetry_samples (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES recording_sessions(session_id) ON DELETE CASCADE,
    sample_index    INTEGER NOT NULL,
    timestamp_ms    BIGINT NOT NULL,
    voltage         NUMERIC(12,6) NOT NULL,
    encoder_count   NUMERIC(14,4),
    distance_mm     NUMERIC(10,3) NOT NULL,
    distance_source TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, sample_index)
);
CREATE INDEX IF NOT EXISTS idx_telemetry_samples_session_id ON telemetry_samples (session_id);

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
