-- Identity height profiles (operator-entered and learned), per-observation standing
-- height samples (TimescaleDB hypertable, 14-day retention), and PH height uncertainty.

SET search_path = continuous_tracking, public;

CREATE TABLE IF NOT EXISTS identity_height_profiles (
    identity_id     TEXT PRIMARY KEY REFERENCES identities(identity_id) ON DELETE CASCADE,
    height_m        DOUBLE PRECISION NOT NULL CHECK (height_m BETWEEN 0.9 AND 2.2),
    height_sigma_m  DOUBLE PRECISION NOT NULL CHECK (height_sigma_m > 0),
    source          TEXT NOT NULL CHECK (source IN ('operator', 'learned')),
    samples         INTEGER NOT NULL DEFAULT 0 CHECK (samples >= 0),
    cameras         INTEGER NOT NULL DEFAULT 0 CHECK (cameras >= 0),
    updated_at      TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS ph_height_samples (
    observed_at     TIMESTAMPTZ NOT NULL,
    ph_id           UUID NOT NULL,
    camera_id       TEXT NOT NULL,
    height_m        DOUBLE PRECISION NOT NULL CHECK (height_m BETWEEN 0.9 AND 2.2),
    height_sigma_m  DOUBLE PRECISION NOT NULL CHECK (height_sigma_m > 0)
);

SELECT create_hypertable(
    'continuous_tracking.ph_height_samples',
    'observed_at',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'continuous_tracking.ph_height_samples',
    INTERVAL '14 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ph_height_samples_ph_time
    ON continuous_tracking.ph_height_samples (ph_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_ph_height_samples_camera_time
    ON continuous_tracking.ph_height_samples (camera_id, observed_at DESC);

ALTER TABLE continuous_tracking.person_hypotheses
    ADD COLUMN IF NOT EXISTS height_sigma_m DOUBLE PRECISION;
