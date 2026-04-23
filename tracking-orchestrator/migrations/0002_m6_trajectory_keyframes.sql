-- M6: Trajectories, room dwells, and tagged keyframes.

-- ---------------------------------------------------------------------------
-- Person trajectories: confirmed ground-plane positions over time
-- ---------------------------------------------------------------------------
CREATE TABLE person_trajectories (
    id                  BIGSERIAL,
    observed_at         TIMESTAMPTZ NOT NULL,
    identity_id         TEXT NOT NULL,
    global_track_id     UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    room_name           TEXT NOT NULL DEFAULT '',
    ground_x            DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- floor-plan meters
    ground_y            DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- floor-plan meters
    posture             TEXT NOT NULL DEFAULT 'unknown'
        CHECK (posture IN ('standing', 'sitting', 'walking', 'lying', 'unknown')),
    identity_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    PRIMARY KEY (id, observed_at)
);

SELECT create_hypertable('person_trajectories', 'observed_at', if_not_exists => TRUE);

CREATE INDEX idx_person_trajectories_identity ON person_trajectories (identity_id, observed_at DESC);
CREATE INDEX idx_person_trajectories_global_track ON person_trajectories (global_track_id);

-- ---------------------------------------------------------------------------
-- Room dwells: contiguous time a person spent in a room
-- ---------------------------------------------------------------------------
CREATE TABLE room_dwells (
    id               BIGSERIAL PRIMARY KEY,
    identity_id      TEXT NOT NULL,
    global_track_id  UUID REFERENCES global_tracks(global_track_id) ON DELETE SET NULL,
    room_name        TEXT NOT NULL,
    entered_at       TIMESTAMPTZ NOT NULL,
    exited_at        TIMESTAMPTZ,                   -- NULL while the dwell is open
    duration_seconds INTEGER,                       -- computed on exit
    entry_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    primary_posture  TEXT NOT NULL DEFAULT 'unknown',
    activity_summary JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_room_dwells_identity ON room_dwells (identity_id, entered_at DESC);
CREATE INDEX idx_room_dwells_global_track ON room_dwells (global_track_id, entered_at DESC);

-- ---------------------------------------------------------------------------
-- Tagged keyframes: periodic + triggered frame samples with annotations
-- ---------------------------------------------------------------------------
CREATE TABLE tagged_keyframes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tracklet_id     UUID REFERENCES tracklets(tracklet_id) ON DELETE SET NULL,
    global_track_id UUID REFERENCES global_tracks(global_track_id) ON DELETE SET NULL,
    camera_id       TEXT NOT NULL,
    minio_key       TEXT NOT NULL,
    captured_at     TIMESTAMPTZ NOT NULL,
    annotations     JSONB NOT NULL DEFAULT '{}',    -- bbox, person_id, posture, activity, confidence
    tag_reason      TEXT NOT NULL
        CHECK (tag_reason IN ('periodic', 'identity_changed', 'hazard', 'dwell_start')),
    expires_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_tagged_keyframes_tracklet ON tagged_keyframes (tracklet_id, captured_at DESC);
CREATE INDEX idx_tagged_keyframes_global_track ON tagged_keyframes (global_track_id, captured_at DESC);
CREATE INDEX idx_tagged_keyframes_expires ON tagged_keyframes (expires_at);
