-- Initial schema for the continuous tracking system.
-- Targets TimescaleDB with pgvectorscale for vector ANN search.
-- Rolled up from 0001–0005 (May 2026) with all corrections applied.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

SET search_path = continuous_tracking, public;

-- ---------------------------------------------------------------------------
-- Cameras: physical camera configuration
-- ---------------------------------------------------------------------------
CREATE TABLE cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    rtsp_url    TEXT NOT NULL,
    location    TEXT NOT NULL DEFAULT '',
    floor_plan  JSONB NOT NULL DEFAULT '{}',
    is_active   BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_cameras_is_active ON cameras(is_active) WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Streams: logical processing streams derived from cameras
-- ---------------------------------------------------------------------------
CREATE TABLE streams (
    stream_id         TEXT PRIMARY KEY,
    camera_id         TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    frame_rate        REAL NOT NULL DEFAULT 5.0,
    resolution_width  INT NOT NULL DEFAULT 640,
    resolution_height INT NOT NULL DEFAULT 480,
    is_active         BOOLEAN NOT NULL DEFAULT true,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_streams_camera_id ON streams(camera_id) WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Tracking events: top-level frame processing results
-- ---------------------------------------------------------------------------
CREATE TABLE tracking_events (
    event_id    UUID NOT NULL,
    event_time  TIMESTAMPTZ NOT NULL,
    camera_id   TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    frame_index BIGINT NOT NULL DEFAULT 0,
    frame_data  JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, event_time)
);

SELECT create_hypertable('tracking_events', 'event_time', if_not_exists => TRUE);

CREATE INDEX idx_tracking_events_camera_time ON tracking_events(camera_id, event_time DESC);

-- ---------------------------------------------------------------------------
-- Detections: individual person detections within a frame
-- ---------------------------------------------------------------------------
CREATE TABLE detections (
    detection_id     UUID PRIMARY KEY,
    event_id         UUID NOT NULL,
    camera_id        TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    bbox             JSONB NOT NULL DEFAULT '{}',
    embedding        vector(768),
    confidence       REAL NOT NULL DEFAULT 1.0,
    tracklet_id      UUID,
    global_track_id  UUID,
    floor_point      JSONB NOT NULL DEFAULT '{}',
    capture_time     TIMESTAMPTZ,
    event_time       TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (event_id, event_time)
        REFERENCES tracking_events(event_id, event_time)
        ON DELETE CASCADE
);

CREATE INDEX idx_detections_embedding ON detections
    USING diskann (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_detections_global_track ON detections(global_track_id)
    WHERE global_track_id IS NOT NULL;

CREATE INDEX idx_detections_event_time ON detections(event_time DESC);

CREATE INDEX idx_detections_event_id ON detections(event_id);

-- ---------------------------------------------------------------------------
-- Tracklets: short-lived trajectories within a single camera
-- ---------------------------------------------------------------------------
CREATE TABLE tracklets (
    tracklet_id   UUID PRIMARY KEY,
    camera_id     TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    detection_ids UUID[] NOT NULL DEFAULT '{}',
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at      TIMESTAMPTZ,
    state         TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'terminated')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tracklets_camera_state ON tracklets(camera_id, state) WHERE state = 'active';

-- ---------------------------------------------------------------------------
-- Tracklet gallery rows
-- ---------------------------------------------------------------------------
CREATE TABLE tracklet_gallery (
    entry_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tracklet_id UUID NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    embedding   vector(768) NOT NULL,
    quality     REAL NOT NULL DEFAULT 1.0,
    seen_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tracklet_gallery_tracklet_id ON tracklet_gallery(tracklet_id, seen_at DESC);

-- ---------------------------------------------------------------------------
-- Global tracks: persistent identity trajectories across cameras
-- ---------------------------------------------------------------------------
CREATE TABLE global_tracks (
    global_track_id UUID PRIMARY KEY,
    camera_ids      TEXT[] NOT NULL DEFAULT '{}',
    tracklet_ids    UUID[] NOT NULL DEFAULT '{}',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    current_identity_id UUID,
    state           TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'closed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_global_tracks_state ON global_tracks(state) WHERE state = 'active';

CREATE INDEX idx_global_tracks_tracklet_ids ON global_tracks USING gin(tracklet_ids);

-- ---------------------------------------------------------------------------
-- Identity revisions: Bayesian posterior updates
-- ---------------------------------------------------------------------------
CREATE TABLE identity_revisions (
    revision_id            UUID NOT NULL DEFAULT gen_random_uuid(),
    revision_time          TIMESTAMPTZ NOT NULL DEFAULT now(),
    global_track_id        UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    tracklet_ids           UUID[] NOT NULL DEFAULT '{}',
    candidates             JSONB NOT NULL DEFAULT '[]',
    map_identity_id        UUID,
    posterior_entropy      REAL NOT NULL,
    previous_identity_id   UUID,
    new_identity_id        UUID,
    reason                 TEXT NOT NULL DEFAULT '',
    evidence               JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (revision_id, revision_time)
);

SELECT create_hypertable('identity_revisions', 'revision_time', if_not_exists => TRUE);

CREATE INDEX idx_identity_revisions_track ON identity_revisions(global_track_id, revision_time DESC);

-- ---------------------------------------------------------------------------
-- Identities and gallery embeddings
-- ---------------------------------------------------------------------------
CREATE TABLE identities (
    identity_id   UUID PRIMARY KEY,
    display_name  TEXT NOT NULL DEFAULT '',
    metadata      JSONB NOT NULL DEFAULT '{}',
    is_active     BOOLEAN NOT NULL DEFAULT true,
    enrolled_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_identities_active ON identities(is_active) WHERE is_active = true;

-- reid_gallery.embedding is nullable: the pipeline uses NULL when ReID is unavailable.
CREATE TABLE reid_gallery (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identity_id       UUID NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
    embedding         vector(768),
    quality           REAL NOT NULL DEFAULT 1.0,
    origin_tracklet_id UUID REFERENCES tracklets(tracklet_id) ON DELETE SET NULL,
    seen_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    face_confirmed    BOOLEAN NOT NULL DEFAULT false,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- pgvectorscale StreamingDiskANN for scalable vector search.
CREATE INDEX idx_reid_gallery_embedding ON reid_gallery
    USING diskann (embedding vector_cosine_ops);

CREATE INDEX idx_reid_gallery_identity_time ON reid_gallery(identity_id, seen_at DESC);

CREATE INDEX idx_reid_gallery_origin_tracklet ON reid_gallery(origin_tracklet_id);

-- ---------------------------------------------------------------------------
-- Person activities: dementia activity layer records
-- ---------------------------------------------------------------------------
CREATE TABLE person_activities (
    activity_id       UUID NOT NULL,
    occurred_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    identity_id       UUID REFERENCES identities(identity_id) ON DELETE SET NULL,
    camera_id         TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    activity_type     TEXT NOT NULL CHECK (activity_type IN (
        'entry',
        'exit',
        'linger',
        'loop',
        'fall_detected',
        'area_entered',
        'area_exited',
        'pacing',
        'sundowning',
        'bathroom_anomaly',
        'stillness',
        'nighttime_movement',
        'absence'
    )),
    metadata          JSONB NOT NULL DEFAULT '{}',
    confidence        REAL NOT NULL DEFAULT 1.0,
    related_event_id  UUID,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (activity_id, occurred_at)
);

SELECT create_hypertable('person_activities', 'occurred_at', if_not_exists => TRUE);

CREATE INDEX idx_person_activities_identity_time ON person_activities(identity_id, occurred_at DESC);
CREATE INDEX idx_person_activities_type ON person_activities(activity_type, occurred_at DESC);

-- ---------------------------------------------------------------------------
-- Stream assignments: room/zone assignments for streams
-- ---------------------------------------------------------------------------
CREATE TABLE stream_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_id     TEXT NOT NULL REFERENCES streams(stream_id) ON DELETE CASCADE,
    room_id       TEXT NOT NULL DEFAULT '',
    zone_id       TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_assignments_stream_id ON stream_assignments(stream_id);

-- ---------------------------------------------------------------------------
-- Person trajectories: confirmed ground-plane positions over time
-- ---------------------------------------------------------------------------
CREATE TABLE person_trajectories (
    id                  BIGSERIAL,
    observed_at         TIMESTAMPTZ NOT NULL,
    identity_id         UUID NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
    global_track_id     UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    room_name           TEXT NOT NULL DEFAULT '',
    ground_x            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ground_y            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
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
    identity_id      UUID NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
    global_track_id  UUID REFERENCES global_tracks(global_track_id) ON DELETE SET NULL,
    room_name        TEXT NOT NULL,
    entered_at       TIMESTAMPTZ NOT NULL,
    exited_at        TIMESTAMPTZ,
    duration_seconds INTEGER,
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
    annotations     JSONB NOT NULL DEFAULT '{}',
    tag_reason      TEXT NOT NULL
        CHECK (tag_reason IN ('periodic', 'identity_changed', 'hazard', 'dwell_start', 'fall', 'dementia_signal')),
    expires_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_tagged_keyframes_tracklet ON tagged_keyframes (tracklet_id, captured_at DESC);
CREATE INDEX idx_tagged_keyframes_global_track ON tagged_keyframes (global_track_id, captured_at DESC);
CREATE INDEX idx_tagged_keyframes_expires ON tagged_keyframes (expires_at);

-- ---------------------------------------------------------------------------
-- Dementia signals: dementia-relevant behavioural patterns
-- ---------------------------------------------------------------------------
CREATE TABLE dementia_signals (
    signal_id        UUID        NOT NULL DEFAULT gen_random_uuid(),
    identity_id      UUID        NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
    signal_kind      VARCHAR(64) NOT NULL,
    severity         VARCHAR(16) NOT NULL CHECK (severity IN ('info', 'warning', 'emergency')),
    value            FLOAT       NOT NULL,
    baseline         FLOAT,
    z_score          FLOAT,
    window_start     TIMESTAMPTZ NOT NULL,
    window_end       TIMESTAMPTZ NOT NULL,
    context_json     JSONB       NOT NULL DEFAULT '{}',
    emitted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (signal_id, emitted_at)
);

SELECT create_hypertable(
    'dementia_signals',
    'emitted_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

CREATE INDEX idx_dementia_signals_identity_kind
    ON dementia_signals (identity_id, signal_kind, emitted_at DESC);

CREATE INDEX idx_dementia_signals_severity
    ON dementia_signals (severity, emitted_at DESC)
    WHERE severity IN ('warning', 'emergency');

-- Retention: keep 365 days of signal history.
SELECT add_retention_policy(
    'dementia_signals',
    INTERVAL '365 days',
    if_not_exists => TRUE
);

-- ---------------------------------------------------------------------------
-- Updated_at trigger (generic)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION _update_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_cameras_updated_at BEFORE UPDATE ON cameras
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
CREATE TRIGGER trg_streams_updated_at BEFORE UPDATE ON streams
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
CREATE TRIGGER trg_tracklets_updated_at BEFORE UPDATE ON tracklets
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
CREATE TRIGGER trg_global_tracks_updated_at BEFORE UPDATE ON global_tracks
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
CREATE TRIGGER trg_identities_updated_at BEFORE UPDATE ON identities
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
CREATE TRIGGER trg_reid_gallery_updated_at BEFORE UPDATE ON reid_gallery
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
CREATE TRIGGER trg_stream_assignments_updated_at BEFORE UPDATE ON stream_assignments
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
