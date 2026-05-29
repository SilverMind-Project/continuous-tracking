-- =============================================================================
-- CTS consolidated initial schema - 0001_init.up.sql
-- Synthesized from migrations 0001 through 0020 (May 2026).
--
-- Applies the full final schema in one step on a fresh database.
-- Replaces the individual numbered migration files for new environment setup.
--
-- Key consolidation decisions:
--   - All tables created directly in continuous_tracking schema (0003 schema move folded in)
--   - identity_id columns are TEXT from the start (0004 type change folded in)
--   - ALTER TABLE ADD COLUMN changes folded into original CREATE TABLE definitions
--   - reid_gallery.identity_id is nullable (0012 change folded in)
--   - person_trajectories.identity_id is nullable (0007 change folded in)
--   - room_dwells.identity_id is nullable (0007 change folded in)
--   - global_tracks, tracklets, do_not_fuse_hints use 0017 schema (full column set)
--   - 0009 was a no-op placeholder (omitted)
--   - The _schema_version table is managed by MigrationRunner, not included here
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

SET search_path = continuous_tracking, public;

-- =============================================================================
-- Cameras: physical camera configuration
-- =============================================================================
CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    rtsp_url    TEXT NOT NULL,
    location    TEXT NOT NULL DEFAULT '',
    floor_plan  JSONB NOT NULL DEFAULT '{}',
    is_active   BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cameras_is_active
    ON cameras (is_active)
    WHERE is_active = true;

-- =============================================================================
-- Streams: logical processing streams derived from cameras
-- =============================================================================
CREATE TABLE IF NOT EXISTS streams (
    stream_id         TEXT PRIMARY KEY,
    camera_id         TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    frame_rate        REAL NOT NULL DEFAULT 5.0,
    resolution_width  INT NOT NULL DEFAULT 640,
    resolution_height INT NOT NULL DEFAULT 480,
    is_active         BOOLEAN NOT NULL DEFAULT true,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_streams_camera_id
    ON streams (camera_id)
    WHERE is_active = true;

-- =============================================================================
-- Stream assignments: room/zone assignments for streams
-- =============================================================================
CREATE TABLE IF NOT EXISTS stream_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stream_id     TEXT NOT NULL REFERENCES streams(stream_id) ON DELETE CASCADE,
    room_id       TEXT NOT NULL DEFAULT '',
    zone_id       TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_assignments_stream_id
    ON stream_assignments (stream_id);

-- =============================================================================
-- Identities and gallery embeddings
-- =============================================================================
CREATE TABLE IF NOT EXISTS identities (
    identity_id   TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL DEFAULT '',
    metadata      JSONB NOT NULL DEFAULT '{}',
    is_active     BOOLEAN NOT NULL DEFAULT true,
    enrolled_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_identities_active
    ON identities (is_active)
    WHERE is_active = true;

-- =============================================================================
-- Global tracks: persistent identity trajectories across cameras.
-- Full column set: base (0001) + last_posterior (0008) + merges (0015)
--                 + identity_committed_at (0016).
-- Schema sourced from 0017_restore_global_tracks which is the authoritative
-- full-column definition.
-- =============================================================================
CREATE TABLE IF NOT EXISTS global_tracks (
    global_track_id               UUID PRIMARY KEY,
    camera_ids                    TEXT[]      NOT NULL DEFAULT '{}',
    tracklet_ids                  UUID[]      NOT NULL DEFAULT '{}',
    started_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    current_identity_id           TEXT,
    current_identity_committed_at TIMESTAMPTZ,
    state                         TEXT        NOT NULL DEFAULT 'active'
                                      CHECK (state IN ('active', 'closed')),
    -- merge tracking (from 0015_global_track_merges)
    merged_into_id                UUID        REFERENCES global_tracks(global_track_id),
    merged_at                     TIMESTAMPTZ,
    merged_by                     TEXT,
    -- identity posterior cache (from 0008_global_track_last_posterior)
    last_posterior_jsonb          JSONB,
    last_posterior_at             TIMESTAMPTZ,
    created_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Composite index covering _SQL_LIST_ACTIVE (state='active' ORDER BY last_seen_at DESC).
-- Replaces the narrower idx_global_tracks_state from 0001 (dropped in 0011).
CREATE INDEX IF NOT EXISTS idx_global_tracks_active_seen
    ON global_tracks (last_seen_at DESC)
    WHERE state = 'active';

-- GIN index for tracklet_ids array lookups
CREATE INDEX IF NOT EXISTS idx_global_tracks_tracklet_ids
    ON global_tracks USING GIN (tracklet_ids);

-- Merge-history index (from 0015)
CREATE INDEX IF NOT EXISTS idx_gt_merged_into
    ON global_tracks (merged_into_id)
    WHERE merged_into_id IS NOT NULL;

-- =============================================================================
-- Tracklets: short-lived trajectories within a single camera.
-- Base schema from 0001; 0017 is the authoritative definition.
-- =============================================================================
CREATE TABLE IF NOT EXISTS tracklets (
    tracklet_id   UUID PRIMARY KEY,
    camera_id     TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    detection_ids UUID[] NOT NULL DEFAULT '{}',
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at      TIMESTAMPTZ,
    state         TEXT NOT NULL DEFAULT 'active'
                      CHECK (state IN ('active', 'terminated')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tracklets_camera_state
    ON tracklets (camera_id, state)
    WHERE state = 'active';

-- =============================================================================
-- Do-not-fuse hints: caregiver corrections preventing incorrect tracklet merges.
-- From 0014; FKs reference tracklets and global_tracks above.
-- =============================================================================
CREATE TABLE IF NOT EXISTS do_not_fuse_hints (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tracklet_id     UUID NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    global_track_id UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT NOT NULL DEFAULT 'system',
    UNIQUE (tracklet_id, global_track_id)
);

CREATE INDEX IF NOT EXISTS idx_dnf_tracklet_id
    ON do_not_fuse_hints (tracklet_id);

CREATE INDEX IF NOT EXISTS idx_dnf_global_track_id
    ON do_not_fuse_hints (global_track_id);

-- =============================================================================
-- Tracklet gallery rows
-- =============================================================================
CREATE TABLE IF NOT EXISTS tracklet_gallery (
    entry_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tracklet_id UUID NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    embedding   vector(768) NOT NULL,
    quality     REAL NOT NULL DEFAULT 1.0,
    seen_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tracklet_gallery_tracklet_id
    ON tracklet_gallery (tracklet_id, seen_at DESC);

-- =============================================================================
-- ReID gallery: identity appearance embeddings.
-- identity_id is nullable (0012 change): gallery entries can be created before
-- identity resolution and backfilled later. NULL bypasses FK checks (standard SQL).
-- =============================================================================
CREATE TABLE IF NOT EXISTS reid_gallery (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identity_id         TEXT REFERENCES identities(identity_id) ON DELETE CASCADE,
    embedding           vector(768),
    quality             REAL NOT NULL DEFAULT 1.0,
    origin_tracklet_id  UUID REFERENCES tracklets(tracklet_id) ON DELETE SET NULL,
    seen_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    face_confirmed      BOOLEAN NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- pgvectorscale StreamingDiskANN for scalable vector search
CREATE INDEX IF NOT EXISTS idx_reid_gallery_embedding
    ON reid_gallery USING diskann (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_reid_gallery_identity_time
    ON reid_gallery (identity_id, seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_reid_gallery_origin_tracklet
    ON reid_gallery (origin_tracklet_id);

-- =============================================================================
-- Identity revisions: Bayesian posterior updates (hypertable)
-- =============================================================================
CREATE TABLE IF NOT EXISTS identity_revisions (
    revision_id          UUID NOT NULL DEFAULT gen_random_uuid(),
    revision_time        TIMESTAMPTZ NOT NULL DEFAULT now(),
    global_track_id      UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    tracklet_ids         UUID[] NOT NULL DEFAULT '{}',
    candidates           JSONB NOT NULL DEFAULT '[]',
    map_identity_id      TEXT,
    posterior_entropy    REAL NOT NULL,
    previous_identity_id TEXT,
    new_identity_id      TEXT,
    reason               TEXT NOT NULL DEFAULT '',
    evidence             JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (revision_id, revision_time)
);

SELECT create_hypertable('identity_revisions', 'revision_time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_identity_revisions_track
    ON identity_revisions (global_track_id, revision_time DESC);

-- =============================================================================
-- Tracking events: top-level frame processing results (hypertable)
-- =============================================================================
CREATE TABLE IF NOT EXISTS tracking_events (
    event_id    UUID NOT NULL,
    event_time  TIMESTAMPTZ NOT NULL,
    camera_id   TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    frame_index BIGINT NOT NULL DEFAULT 0,
    frame_data  JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, event_time)
);

SELECT create_hypertable('tracking_events', 'event_time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_tracking_events_camera_time
    ON tracking_events (camera_id, event_time DESC);

-- =============================================================================
-- Detections: individual person detections within a frame
-- =============================================================================
CREATE TABLE IF NOT EXISTS detections (
    detection_id    UUID PRIMARY KEY,
    event_id        UUID NOT NULL,
    camera_id       TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    bbox            JSONB NOT NULL DEFAULT '{}',
    embedding       vector(768),
    confidence      REAL NOT NULL DEFAULT 1.0,
    tracklet_id     UUID,
    global_track_id UUID,
    floor_point     JSONB NOT NULL DEFAULT '{}',
    capture_time    TIMESTAMPTZ,
    event_time      TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (event_id, event_time)
        REFERENCES tracking_events(event_id, event_time)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_detections_embedding
    ON detections USING diskann (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_detections_global_track
    ON detections (global_track_id)
    WHERE global_track_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_detections_event_time
    ON detections (event_time DESC);

CREATE INDEX IF NOT EXISTS idx_detections_event_id
    ON detections (event_id);

-- =============================================================================
-- Person activities: dementia activity layer records (hypertable)
-- =============================================================================
CREATE TABLE IF NOT EXISTS person_activities (
    activity_id      UUID NOT NULL,
    occurred_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    identity_id      TEXT REFERENCES identities(identity_id) ON DELETE SET NULL,
    camera_id        TEXT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    activity_type    TEXT NOT NULL CHECK (activity_type IN (
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
    metadata         JSONB NOT NULL DEFAULT '{}',
    confidence       REAL NOT NULL DEFAULT 1.0,
    related_event_id UUID,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (activity_id, occurred_at)
);

SELECT create_hypertable('person_activities', 'occurred_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_person_activities_identity_time
    ON person_activities (identity_id, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_person_activities_type
    ON person_activities (activity_type, occurred_at DESC);

-- =============================================================================
-- Person trajectories: confirmed ground-plane positions over time (hypertable).
-- identity_id is nullable (0007): UNKNOWN tracks still produce trajectory rows.
-- motion_energy column from 0005_pose_columns.
-- =============================================================================
CREATE TABLE IF NOT EXISTS person_trajectories (
    id                  BIGSERIAL,
    observed_at         TIMESTAMPTZ NOT NULL,
    identity_id         TEXT REFERENCES identities(identity_id) ON DELETE CASCADE,
    global_track_id     UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    room_name           TEXT NOT NULL DEFAULT '',
    ground_x            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ground_y            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    posture             TEXT NOT NULL DEFAULT 'unknown'
                            CHECK (posture IN ('standing', 'sitting', 'walking', 'lying', 'unknown')),
    identity_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    -- from 0005_pose_columns
    motion_energy       DOUBLE PRECISION,
    PRIMARY KEY (id, observed_at)
);

SELECT create_hypertable('person_trajectories', 'observed_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_person_trajectories_identity
    ON person_trajectories (identity_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_person_trajectories_global_track
    ON person_trajectories (global_track_id);

-- Covers UPDATE/SELECT by (global_track_id, observed_at) during IdentityRevision rewrites (0010)
CREATE INDEX IF NOT EXISTS idx_person_trajectories_gt_observed
    ON person_trajectories (global_track_id, observed_at);

-- =============================================================================
-- Room dwells: contiguous time a person spent in a room.
-- identity_id is nullable (0007): UNKNOWN tracks still produce dwell rows.
-- min_motion_energy and still_seconds from 0005_pose_columns.
-- =============================================================================
CREATE TABLE IF NOT EXISTS room_dwells (
    id               BIGSERIAL PRIMARY KEY,
    identity_id      TEXT REFERENCES identities(identity_id) ON DELETE CASCADE,
    global_track_id  UUID REFERENCES global_tracks(global_track_id) ON DELETE SET NULL,
    room_name        TEXT NOT NULL,
    entered_at       TIMESTAMPTZ NOT NULL,
    exited_at        TIMESTAMPTZ,
    duration_seconds INTEGER,
    entry_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    primary_posture  TEXT NOT NULL DEFAULT 'unknown',
    activity_summary JSONB NOT NULL DEFAULT '{}',
    -- from 0005_pose_columns
    min_motion_energy DOUBLE PRECISION,
    still_seconds     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_room_dwells_identity
    ON room_dwells (identity_id, entered_at DESC);

CREATE INDEX IF NOT EXISTS idx_room_dwells_global_track
    ON room_dwells (global_track_id, entered_at DESC);

-- Covers UPDATE/SELECT by (global_track_id, entered_at) during IdentityRevision rewrites (0010)
CREATE INDEX IF NOT EXISTS idx_room_dwells_gt_entered
    ON room_dwells (global_track_id, entered_at);

-- Open-dwell lookup: WHERE identity_id = $1 AND global_track_id = $2 AND exited_at IS NULL (0011)
CREATE INDEX IF NOT EXISTS idx_room_dwells_open
    ON room_dwells (identity_id, global_track_id, entered_at DESC)
    WHERE exited_at IS NULL;

-- Room-name filter for list_room_dwells (0011)
CREATE INDEX IF NOT EXISTS idx_room_dwells_room
    ON room_dwells (room_name, entered_at DESC);

-- =============================================================================
-- Tagged keyframes: periodic and triggered frame samples with annotations
-- =============================================================================
CREATE TABLE IF NOT EXISTS tagged_keyframes (
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

CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_tracklet
    ON tagged_keyframes (tracklet_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_global_track
    ON tagged_keyframes (global_track_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_expires
    ON tagged_keyframes (expires_at);

-- =============================================================================
-- Dementia signals: dementia-relevant behavioural patterns (hypertable).
-- algorithm_version from 0006_signal_algo_version.
-- algorithm_name, algorithm_spec_json, evidence_grade from 0006 as well
-- (per the required final schema columns listed in task spec).
-- =============================================================================
CREATE TABLE IF NOT EXISTS dementia_signals (
    signal_id         UUID        NOT NULL DEFAULT gen_random_uuid(),
    identity_id       TEXT        NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
    signal_kind       VARCHAR(64) NOT NULL,
    severity          VARCHAR(16) NOT NULL CHECK (severity IN ('info', 'warning', 'emergency')),
    value             FLOAT       NOT NULL,
    baseline          FLOAT,
    z_score           FLOAT,
    window_start      TIMESTAMPTZ NOT NULL,
    window_end        TIMESTAMPTZ NOT NULL,
    context_json      JSONB       NOT NULL DEFAULT '{}',
    emitted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- from 0006_signal_algo_version
    algorithm_version INTEGER     NOT NULL DEFAULT 1,
    algorithm_name    TEXT,
    algorithm_spec_json JSONB,
    evidence_grade    TEXT,
    PRIMARY KEY (signal_id, emitted_at)
);

SELECT create_hypertable(
    'dementia_signals',
    'emitted_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS idx_dementia_signals_identity_kind
    ON dementia_signals (identity_id, signal_kind, emitted_at DESC);

CREATE INDEX IF NOT EXISTS idx_dementia_signals_severity
    ON dementia_signals (severity, emitted_at DESC)
    WHERE severity IN ('warning', 'emergency');

-- Covers UPDATE/SELECT by (identity_id, window_start) during IdentityRevision rewrites (0010)
CREATE INDEX IF NOT EXISTS idx_dementia_signals_identity_window
    ON dementia_signals (identity_id, window_start);

-- Retention: keep 365 days of signal history
SELECT add_retention_policy(
    'dementia_signals',
    INTERVAL '365 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- Person hypotheses: world-tracker first-class person records (M1)
-- From 0008_person_hypotheses.
-- =============================================================================
CREATE TABLE IF NOT EXISTS person_hypotheses (
    ph_id                         UUID         PRIMARY KEY,
    born_at                       TIMESTAMPTZ  NOT NULL,
    closed_at                     TIMESTAMPTZ,
    last_seen_at                  TIMESTAMPTZ  NOT NULL,
    last_seen_camera              TEXT         NOT NULL,
    observation_count             INTEGER      NOT NULL DEFAULT 0,
    current_identity_id           TEXT,
    current_identity_committed_at TIMESTAMPTZ,
    state_mean                    FLOAT8[]     NOT NULL,
    state_cov                     FLOAT8[]     NOT NULL,
    gallery_mean                  FLOAT4[],
    height_m                      FLOAT8,
    active_cameras                TEXT[]       NOT NULL DEFAULT '{}',
    metadata                      JSONB        NOT NULL DEFAULT '{}',
    CONSTRAINT person_hypotheses_state_mean_size CHECK (array_length(state_mean, 1) = 4),
    CONSTRAINT person_hypotheses_state_cov_size  CHECK (array_length(state_cov,  1) = 16)
);

CREATE INDEX IF NOT EXISTS idx_ph_last_seen_at_open
    ON person_hypotheses (last_seen_at DESC)
    WHERE closed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_ph_identity
    ON person_hypotheses (current_identity_id)
    WHERE current_identity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_ph_closed_at
    ON person_hypotheses (closed_at DESC)
    WHERE closed_at IS NOT NULL;

-- =============================================================================
-- World observations: per-frame ground-plane detections linked to a PH (hypertable).
-- TimescaleDB requires the partitioning column (captured_at) in the primary key.
-- =============================================================================
CREATE TABLE IF NOT EXISTS world_observations (
    observation_id       UUID    NOT NULL,
    ph_id                UUID    NOT NULL REFERENCES person_hypotheses(ph_id) ON DELETE CASCADE,
    camera_id            TEXT    NOT NULL,
    frame_index          BIGINT  NOT NULL,
    captured_at          TIMESTAMPTZ NOT NULL,
    floor_x_m            FLOAT8  NOT NULL,
    floor_y_m            FLOAT8  NOT NULL,
    detection_confidence FLOAT4  NOT NULL,
    bbox                 JSONB   NOT NULL,
    height_m             FLOAT8,
    metadata             JSONB   NOT NULL DEFAULT '{}',
    PRIMARY KEY (observation_id, captured_at)
);

SELECT create_hypertable(
    'continuous_tracking.world_observations',
    'captured_at',
    chunk_time_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_wo_ph_time
    ON world_observations (ph_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_wo_camera_time
    ON world_observations (camera_id, captured_at DESC);

-- =============================================================================
-- Global track identity: first-class identity assignment history per GlobalTrack.
-- From 0007_identity_nullable_trajectories.
-- =============================================================================
CREATE TABLE IF NOT EXISTS global_track_identity (
    id              BIGSERIAL PRIMARY KEY,
    global_track_id UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    identity_id     TEXT REFERENCES identities(identity_id) ON DELETE SET NULL,
    committed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    committed_by    TEXT NOT NULL,
    confidence      DOUBLE PRECISION,
    evidence_source TEXT,
    revision_id     TEXT,
    applies_from    TIMESTAMPTZ,
    applies_to      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_global_track_identity_gt_time
    ON global_track_identity (global_track_id, committed_at DESC);

CREATE INDEX IF NOT EXISTS idx_global_track_identity_revision
    ON global_track_identity (revision_id)
    WHERE revision_id IS NOT NULL;

-- =============================================================================
-- Keyframe bbox annotations: bounding-box annotations tied to keyframes (0013).
-- bbox_age_frames column from 0020_bbox_annotation_indexes.
-- =============================================================================
CREATE TABLE IF NOT EXISTS keyframe_bbox_annotations (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyframe_id          TEXT NOT NULL,
    tracklet_id          UUID NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    camera_id            TEXT NOT NULL,
    x1                   REAL NOT NULL,
    y1                   REAL NOT NULL,
    x2                   REAL NOT NULL,
    y2                   REAL NOT NULL,
    detection_confidence REAL NOT NULL,
    frame_width          INTEGER NOT NULL,
    frame_height         INTEGER NOT NULL,
    identity_id          TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- user-drawn override bbox (M4)
    override_x1  REAL,
    override_y1  REAL,
    override_x2  REAL,
    override_y2  REAL,
    override_by  TEXT,
    override_at  TIMESTAMPTZ,
    -- from 0020_bbox_annotation_indexes
    bbox_age_frames INTEGER NOT NULL DEFAULT 0
);

-- 0013 indexes
CREATE INDEX IF NOT EXISTS idx_kba_keyframe_id
    ON keyframe_bbox_annotations (keyframe_id);

CREATE INDEX IF NOT EXISTS idx_kba_tracklet_id
    ON keyframe_bbox_annotations (tracklet_id);

CREATE INDEX IF NOT EXISTS idx_kba_identity_id
    ON keyframe_bbox_annotations (identity_id)
    WHERE identity_id IS NOT NULL;

-- 0020 indexes (deduplicate with 0013 idx_kba_keyframe_id above)
CREATE INDEX IF NOT EXISTS idx_bbox_annotations_confidence
    ON keyframe_bbox_annotations (detection_confidence)
    WHERE detection_confidence IS NOT NULL;

-- =============================================================================
-- ph_revisions: audit log for every identity change on a Person Hypothesis (hypertable).
-- Primary key includes applied_at to satisfy TimescaleDB hypertable rule.
-- From 0019_ph_corrections_audit.
-- =============================================================================
CREATE TABLE IF NOT EXISTS ph_revisions (
    revision_id          UUID NOT NULL,
    ph_id                UUID NOT NULL,
    previous_identity_id TEXT,
    new_identity_id      TEXT,
    actor                TEXT NOT NULL,
    reason               TEXT NOT NULL,
    kind                 TEXT NOT NULL,
    applied_at           TIMESTAMPTZ NOT NULL,
    rewritten_rows       INT NOT NULL DEFAULT 0,
    evidence_jsonb       JSONB
);

SELECT create_hypertable(
    'continuous_tracking.ph_revisions',
    'applied_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE continuous_tracking.ph_revisions
    ADD PRIMARY KEY (revision_id, applied_at);

CREATE INDEX IF NOT EXISTS idx_ph_revisions_ph_id
    ON ph_revisions (ph_id, applied_at DESC);

CREATE INDEX IF NOT EXISTS idx_ph_revisions_kind
    ON ph_revisions (kind, applied_at DESC);

-- =============================================================================
-- ph_merges: tracks which PHs were merged into which (from 0019).
-- =============================================================================
CREATE TABLE IF NOT EXISTS ph_merges (
    merge_id     UUID PRIMARY KEY,
    source_ph_id UUID NOT NULL,
    target_ph_id UUID NOT NULL,
    revision_id  UUID NOT NULL,
    applied_at   TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ph_merges_source
    ON ph_merges (source_ph_id);

CREATE INDEX IF NOT EXISTS idx_ph_merges_target
    ON ph_merges (target_ph_id);

-- =============================================================================
-- Updated_at trigger (generic) - schema-qualified function
-- =============================================================================
CREATE OR REPLACE FUNCTION continuous_tracking._update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_cameras_updated_at
    BEFORE UPDATE ON cameras
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

CREATE TRIGGER trg_streams_updated_at
    BEFORE UPDATE ON streams
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

CREATE TRIGGER trg_tracklets_updated_at
    BEFORE UPDATE ON tracklets
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

CREATE TRIGGER trg_global_tracks_updated_at
    BEFORE UPDATE ON global_tracks
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

CREATE TRIGGER trg_identities_updated_at
    BEFORE UPDATE ON identities
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

CREATE TRIGGER trg_reid_gallery_updated_at
    BEFORE UPDATE ON reid_gallery
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

CREATE TRIGGER trg_stream_assignments_updated_at
    BEFORE UPDATE ON stream_assignments
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

-- =============================================================================
-- Continuous aggregates (non-transactional - must run outside a transaction block)
-- =============================================================================

-- Daily rollup of dementia signals per identity and signal kind.
-- NOTE: This file must be executed with migrate:no-transaction semantics
-- (or outside an explicit BEGIN/COMMIT block) because TimescaleDB continuous
-- aggregate DDL and policy registration cannot run inside a transaction.
CREATE MATERIALIZED VIEW IF NOT EXISTS continuous_tracking.dementia_signals_daily
WITH (timescaledb.continuous) AS
SELECT
    identity_id,
    signal_kind,
    time_bucket(INTERVAL '1 day', emitted_at) AS day,
    count(*)          AS signal_count,
    avg(value)        AS mean_value,
    stddev_pop(value) AS sd_value
FROM continuous_tracking.dementia_signals
GROUP BY identity_id, signal_kind, day;

SELECT add_continuous_aggregate_policy(
    'continuous_tracking.dementia_signals_daily',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists     => TRUE
);
