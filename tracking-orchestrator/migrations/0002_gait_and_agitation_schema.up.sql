SET search_path = continuous_tracking, public;

-- These objects were originally added to 0001 after deployed databases had
-- already recorded that migration. Keep this migration idempotent so it also
-- repairs databases created from an intermediate version of the baseline.
ALTER TABLE continuous_tracking.person_trajectories
    ADD COLUMN IF NOT EXISTS floor_speed_m_s DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS continuous_tracking.gait_bouts (
    bout_id          UUID PRIMARY KEY,
    identity_id      TEXT NOT NULL REFERENCES continuous_tracking.identities(identity_id)
                        ON DELETE CASCADE,
    started_at       TIMESTAMPTZ NOT NULL,
    ended_at         TIMESTAMPTZ NOT NULL,
    duration_s       DOUBLE PRECISION NOT NULL,
    distance_m       DOUBLE PRECISION NOT NULL,
    median_speed_m_s DOUBLE PRECISION NOT NULL,
    p95_speed_m_s    DOUBLE PRECISION NOT NULL,
    sample_count     INTEGER NOT NULL,
    rooms            TEXT[] NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_gait_bouts_identity
    ON continuous_tracking.gait_bouts (identity_id, started_at DESC);

CREATE TABLE IF NOT EXISTS continuous_tracking.gait_daily (
    identity_id        TEXT NOT NULL REFERENCES continuous_tracking.identities(identity_id)
                          ON DELETE CASCADE,
    local_date         DATE NOT NULL,
    bout_count         INTEGER NOT NULL,
    total_walking_s    DOUBLE PRECISION NOT NULL,
    total_distance_m   DOUBLE PRECISION NOT NULL,
    median_speed_m_s   DOUBLE PRECISION NOT NULL,
    mad_speed_m_s      DOUBLE PRECISION NOT NULL,
    p95_speed_m_s      DOUBLE PRECISION NOT NULL,
    sample_bout_ids    TEXT[] NOT NULL DEFAULT '{}',
    computed_at        TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (identity_id, local_date)
);

CREATE TABLE IF NOT EXISTS continuous_tracking.agitation_windows (
    identity_id  TEXT NOT NULL REFERENCES continuous_tracking.identities(identity_id)
                     ON DELETE CASCADE,
    window_start TIMESTAMPTZ NOT NULL,
    composite    FLOAT8 NOT NULL,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (identity_id, window_start)
);

CREATE INDEX IF NOT EXISTS idx_agitation_windows_identity_start
    ON continuous_tracking.agitation_windows (identity_id, window_start DESC);
