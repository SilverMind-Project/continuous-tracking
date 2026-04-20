-- M1: Initial schema for continuous tracking system
-- Targets TimescaleDB (Postgres extension) with pgvector for ANN search.
--
-- Layering: storage → database (never reverse).
-- All tables use UUID PKs and created_at/updated_at timestamps.

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------------
-- Cameras: physical camera configuration
-- ---------------------------------------------------------------------------
CREATE TABLE cameras (
    camera_id   UUID PRIMARY KEY,
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
    stream_id         UUID PRIMARY KEY,
    camera_id         UUID NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
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
    event_id    UUID PRIMARY KEY,
    camera_id   UUID NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    event_time  TIMESTAMPTZ NOT NULL,
    frame_index BIGINT NOT NULL DEFAULT 0,
    frame_data  JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Hypertable for time-series efficiency
SELECT create_hypertable('tracking_events', 'event_time', if_not_exists => TRUE);

CREATE INDEX idx_tracking_events_camera_time ON tracking_events(camera_id, event_time DESC);

-- ---------------------------------------------------------------------------
-- Detections: individual person detections within a frame
-- ---------------------------------------------------------------------------
CREATE TABLE detections (
    detection_id   UUID PRIMARY KEY,
    event_id       UUID NOT NULL REFERENCES tracking_events(event_id) ON DELETE CASCADE,
    camera_id      UUID NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    bbox           JSONB NOT NULL DEFAULT '{}',
    embedding      vector(512) NOT NULL DEFAULT vector(zeros(512)),
    confidence     REAL NOT NULL DEFAULT 1.0,
    tracklet_id    UUID,
    global_track_id UUID,
    floor_point    JSONB NOT NULL DEFAULT '{}',
    capture_time   TIMESTAMPTZ,
    event_time     TIMESTAMPTZ NOT NULL
);

-- HNSW index for ReID embedding similarity search
CREATE INDEX idx_detections_embedding ON detections
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_detections_global_track ON detections(global_track_id)
    WHERE global_track_id IS NOT NULL;

CREATE INDEX idx_detections_event_time ON detections(event_time DESC);

-- ---------------------------------------------------------------------------
-- Tracklets: short-lived trajectories within a single camera
-- ---------------------------------------------------------------------------
CREATE TABLE tracklets (
    tracklet_id  UUID PRIMARY KEY,
    camera_id    UUID NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    detection_ids UUID[] NOT NULL DEFAULT '{}',
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at     TIMESTAMPTZ,
    state        TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'terminated')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tracklets_camera_state ON tracklets(camera_id, state) WHERE state = 'active';

-- ---------------------------------------------------------------------------
-- Global tracks: persistent identity trajectories across cameras
-- ---------------------------------------------------------------------------
CREATE TABLE global_tracks (
    global_track_id UUID PRIMARY KEY,
    camera_ids      UUID[] NOT NULL DEFAULT '{}',
    tracklet_ids    UUID[] NOT NULL DEFAULT '{}',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    state           TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'closed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_global_tracks_state ON global_tracks(state) WHERE state = 'active';

-- ---------------------------------------------------------------------------
-- Identity revisions: Bayesian posterior updates
-- ---------------------------------------------------------------------------
CREATE TABLE identity_revisions (
    revision_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    global_track_id    UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    candidates         JSONB NOT NULL DEFAULT '[]',
    map_identity_id    UUID,
    posterior_entropy  REAL NOT NULL,
    revision_time      TIMESTAMPTZ NOT NULL DEFAULT now()
);

SELECT create_hypertable('identity_revisions', 'revision_time', if_not_exists => TRUE);

CREATE INDEX idx_identity_revisions_track ON identity_revisions(global_track_id, revision_time DESC);

-- ---------------------------------------------------------------------------
-- Gallery entries: known person records with embeddings
-- ---------------------------------------------------------------------------
CREATE TABLE gallery_entries (
    identity_id   UUID PRIMARY KEY,
    display_name  TEXT NOT NULL DEFAULT '',
    embedding     vector(512) NOT NULL DEFAULT vector(zeros(512)),
    metadata      JSONB NOT NULL DEFAULT '{}',
    enrolled_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active     BOOLEAN NOT NULL DEFAULT true,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for gallery ANN search
CREATE INDEX idx_gallery_embedding ON gallery_entries
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE is_active = true;

CREATE INDEX idx_gallery_active ON gallery_entries(is_active) WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Person activities: dementia activity layer records
-- ---------------------------------------------------------------------------
CREATE TABLE person_activities (
    activity_id     UUID PRIMARY KEY,
    identity_id     UUID REFERENCES gallery_entries(identity_id) ON DELETE SET NULL,
    camera_id       UUID NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    activity_type   TEXT NOT NULL CHECK (activity_type IN (
        'entry', 'exit', 'linger', 'loop', 'fall_detected', 'area_entered', 'area_exited'
    )),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata        JSONB NOT NULL DEFAULT '{}',
    confidence      REAL NOT NULL DEFAULT 1.0,
    related_event_id UUID REFERENCES tracking_events(event_id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

SELECT create_hypertable('person_activities', 'timestamp', if_not_exists => TRUE);

CREATE INDEX idx_person_activities_identity_time ON person_activities(identity_id, timestamp DESC);
CREATE INDEX idx_person_activities_type ON person_activities(activity_type, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Stream assignments: room/zone assignments for streams
-- ---------------------------------------------------------------------------
CREATE TABLE stream_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_id     UUID NOT NULL REFERENCES streams(stream_id) ON DELETE CASCADE,
    room_id       TEXT NOT NULL DEFAULT '',
    zone_id       TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_assignments_stream_id ON stream_assignments(stream_id);

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
CREATE TRIGGER trg_gallery_entries_updated_at BEFORE UPDATE ON gallery_entries
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();
