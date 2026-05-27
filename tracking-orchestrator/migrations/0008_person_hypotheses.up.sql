-- 0007_person_hypotheses.up.sql
-- Adds the world-tracker tables for M1.
SET search_path = continuous_tracking, public;

CREATE TABLE person_hypotheses (
    ph_id                        UUID         PRIMARY KEY,
    born_at                      TIMESTAMPTZ  NOT NULL,
    closed_at                    TIMESTAMPTZ,
    last_seen_at                 TIMESTAMPTZ  NOT NULL,
    last_seen_camera             TEXT         NOT NULL,
    observation_count            INTEGER      NOT NULL DEFAULT 0,
    current_identity_id          TEXT,
    current_identity_committed_at TIMESTAMPTZ,
    state_mean                   FLOAT8[]     NOT NULL,
    state_cov                    FLOAT8[]     NOT NULL,
    gallery_mean                 FLOAT4[],
    height_m                     FLOAT8,
    active_cameras               TEXT[]       NOT NULL DEFAULT '{}',
    metadata                     JSONB        NOT NULL DEFAULT '{}',
    CONSTRAINT person_hypotheses_state_mean_size CHECK (array_length(state_mean, 1) = 4),
    CONSTRAINT person_hypotheses_state_cov_size CHECK (array_length(state_cov, 1) = 16)
);

CREATE INDEX idx_ph_last_seen_at_open
    ON person_hypotheses (last_seen_at DESC)
    WHERE closed_at IS NULL;

CREATE INDEX idx_ph_identity
    ON person_hypotheses (current_identity_id)
    WHERE current_identity_id IS NOT NULL;

CREATE INDEX idx_ph_closed_at
    ON person_hypotheses (closed_at DESC)
    WHERE closed_at IS NOT NULL;

-- TimescaleDB requires the partitioning column (captured_at) to be part of
-- every UNIQUE index, including the primary key. PK is composite
-- (observation_id, captured_at); observation_id stays globally unique because
-- UUIDs are unique on their own — the composite only satisfies the hypertable
-- invariant.
CREATE TABLE world_observations (
    observation_id       UUID         NOT NULL,
    ph_id                UUID         NOT NULL REFERENCES person_hypotheses(ph_id) ON DELETE CASCADE,
    camera_id            TEXT         NOT NULL,
    frame_index          BIGINT       NOT NULL,
    captured_at          TIMESTAMPTZ  NOT NULL,
    floor_x_m            FLOAT8       NOT NULL,
    floor_y_m            FLOAT8       NOT NULL,
    detection_confidence FLOAT4       NOT NULL,
    bbox                 JSONB        NOT NULL,
    height_m             FLOAT8,
    metadata             JSONB        NOT NULL DEFAULT '{}',
    PRIMARY KEY (observation_id, captured_at)
);

SELECT create_hypertable(
    'continuous_tracking.world_observations',
    'captured_at',
    chunk_time_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

CREATE INDEX idx_wo_ph_time
    ON world_observations (ph_id, captured_at DESC);

CREATE INDEX idx_wo_camera_time
    ON world_observations (camera_id, captured_at DESC);
