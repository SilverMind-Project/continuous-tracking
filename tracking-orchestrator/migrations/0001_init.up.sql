-- migrate:no-transaction
-- =============================================================================
-- CTS consolidated baseline schema - 0001_init.up.sql
-- Synthesized from migrations 0001 through 0020, plus U1 quality capture and
-- PH-native cleanup (previously 0002_quality and 0003_ph_native_purge).
--
-- Applies the full final schema in one step on a fresh database.
-- Drop and recreate the database when migrating from an older chain.
--
-- Key consolidation decisions:
--   - Legacy tracking tables (global_tracks, tracklets, do_not_fuse_hints,
--     tracklet_gallery, global_track_identity, identity_revisions) are not
--     created; they are superseded by the PH-native model.
--   - tracking_events, detections, stream_assignments, and person_activities
--     are not created; they were never written to by any storage repository
--     and have no corresponding protocol. Activity signals flow through
--     dementia_signals; room assignment lives in CC camera configuration.
--   - person_trajectories, room_dwells use ph_id (FK to person_hypotheses)
--     instead of global_track_id.
--   - tagged_keyframes uses ph_id only (no tracklet_id or global_track_id).
--   - keyframe_bbox_annotations uses ph_id (FK to person_hypotheses).
--   - reid_gallery.origin_tracklet_id is a plain UUID (no FK to tracklets).
--   - world_observations.quality and person_hypotheses.mean_quality included.
--   - All identity_id columns are TEXT from the start.
--   - The _schema_version table is managed by MigrationRunner, not here.
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
-- ReID gallery: identity appearance embeddings.
-- identity_id is nullable: gallery entries can be created before identity
-- resolution and backfilled later.
-- origin_tracklet_id is a plain UUID reference (no FK; tracklets table removed).
-- =============================================================================
CREATE TABLE IF NOT EXISTS reid_gallery (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identity_id         TEXT REFERENCES identities(identity_id) ON DELETE CASCADE,
    embedding           vector(768),
    quality             REAL NOT NULL DEFAULT 1.0,
    origin_tracklet_id  UUID,
    seen_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    face_confirmed      BOOLEAN NOT NULL DEFAULT false,
    orientation         SMALLINT NOT NULL DEFAULT 4,  -- OrientationBin value
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT reid_gallery_orientation_range CHECK (orientation >= 0 AND orientation <= 4)
);

-- pgvectorscale StreamingDiskANN for scalable vector search
CREATE INDEX IF NOT EXISTS idx_reid_gallery_embedding
    ON reid_gallery USING diskann (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_reid_gallery_identity_time
    ON reid_gallery (identity_id, seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_reid_gallery_origin_tracklet
    ON reid_gallery (origin_tracklet_id);

-- =============================================================================
-- Person hypotheses: world-tracker first-class person records.
-- mean_quality: exponential moving average of observation quality (alpha=0.1).
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
    mean_quality                  REAL         NOT NULL DEFAULT 0,
    metadata                      JSONB        NOT NULL DEFAULT '{}',
    view_prototypes               BYTEA,        -- serialised ViewPrototype tuples
    CONSTRAINT person_hypotheses_state_mean_size CHECK (array_length(state_mean, 1) = 4),
    CONSTRAINT person_hypotheses_state_cov_size  CHECK (array_length(state_cov,  1) = 16),
    CONSTRAINT person_hypotheses_metadata_object CHECK (jsonb_typeof(metadata) = 'object')
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
-- quality: composite crop quality score [0,1] from CropQuality scorer.
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
    quality              REAL    NOT NULL DEFAULT 0,
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
-- Person trajectories: confirmed ground-plane positions over time (hypertable).
-- ph_id: FK to person_hypotheses (PH-native model; replaces global_track_id).
-- motion_energy: from 0005_pose_columns.
-- =============================================================================
CREATE TABLE IF NOT EXISTS person_trajectories (
    id                  BIGSERIAL,
    observed_at         TIMESTAMPTZ NOT NULL,
    identity_id         TEXT REFERENCES identities(identity_id) ON DELETE CASCADE,
    ph_id               UUID NOT NULL REFERENCES person_hypotheses(ph_id) ON DELETE CASCADE,
    room_name           TEXT NOT NULL DEFAULT '',
    ground_x            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ground_y            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    posture             TEXT NOT NULL DEFAULT 'unknown'
                            CHECK (posture IN ('standing', 'sitting', 'walking', 'lying', 'unknown')),
    identity_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    motion_energy       DOUBLE PRECISION,
    floor_speed_m_s     DOUBLE PRECISION,
    PRIMARY KEY (id, observed_at)
);

SELECT create_hypertable('person_trajectories', 'observed_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_person_trajectories_identity
    ON person_trajectories (identity_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_person_trajectories_ph
    ON person_trajectories (ph_id);

CREATE INDEX IF NOT EXISTS idx_person_trajectories_ph_observed
    ON person_trajectories (ph_id, observed_at);

-- =============================================================================
-- Room dwells: contiguous time a person spent in a room.
-- ph_id: FK to person_hypotheses (PH-native model; replaces global_track_id).
-- min_motion_energy, still_seconds: from 0005_pose_columns.
-- =============================================================================
CREATE TABLE IF NOT EXISTS room_dwells (
    id               BIGSERIAL PRIMARY KEY,
    identity_id      TEXT REFERENCES identities(identity_id) ON DELETE CASCADE,
    ph_id            UUID REFERENCES person_hypotheses(ph_id) ON DELETE SET NULL,
    room_name        TEXT NOT NULL,
    entered_at       TIMESTAMPTZ NOT NULL,
    exited_at        TIMESTAMPTZ,
    duration_seconds INTEGER,
    entry_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    primary_posture  TEXT NOT NULL DEFAULT 'unknown',
    activity_summary JSONB NOT NULL DEFAULT '{}',
    min_motion_energy DOUBLE PRECISION,
    still_seconds     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_room_dwells_identity
    ON room_dwells (identity_id, entered_at DESC);

CREATE INDEX IF NOT EXISTS idx_room_dwells_ph
    ON room_dwells (ph_id, entered_at DESC);

CREATE INDEX IF NOT EXISTS idx_room_dwells_ph_entered
    ON room_dwells (ph_id, entered_at);

CREATE INDEX IF NOT EXISTS idx_room_dwells_open
    ON room_dwells (identity_id, ph_id, entered_at DESC)
    WHERE exited_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_room_dwells_room
    ON room_dwells (room_name, entered_at DESC);

-- =============================================================================
-- Gait bouts: discrete walking episodes persisted by WalkingBoutSegmenter.
-- bout_id is UUID5 over (identity_id, started_at.isoformat()) for idempotent
-- re-processing upserts.
-- =============================================================================
CREATE TABLE IF NOT EXISTS gait_bouts (
    bout_id          UUID PRIMARY KEY,
    identity_id      TEXT NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
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
    ON gait_bouts (identity_id, started_at DESC);

-- =============================================================================
-- Tagged keyframes: periodic and triggered frame samples with annotations.
-- ph_id: FK to person_hypotheses (replaces tracklet_id and global_track_id).
-- =============================================================================
CREATE TABLE IF NOT EXISTS tagged_keyframes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ph_id       UUID REFERENCES person_hypotheses(ph_id) ON DELETE SET NULL,
    camera_id   TEXT NOT NULL,
    minio_key   TEXT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    annotations JSONB NOT NULL DEFAULT '{}',
    tag_reason  TEXT NOT NULL
                    CHECK (tag_reason IN ('periodic', 'identity_changed', 'hazard', 'dwell_start', 'fall', 'dementia_signal')),
    expires_at  TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_ph
    ON tagged_keyframes (ph_id, captured_at DESC);

CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_expires
    ON tagged_keyframes (expires_at);

-- =============================================================================
-- Dementia signals: dementia-relevant behavioural patterns (hypertable).
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

CREATE INDEX IF NOT EXISTS idx_dementia_signals_identity_window
    ON dementia_signals (identity_id, window_start);

SELECT add_retention_policy(
    'dementia_signals',
    INTERVAL '365 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- Keyframe bbox annotations: bounding-box annotations tied to keyframes.
-- ph_id: FK to person_hypotheses (replaces tracklet_id).
-- =============================================================================
CREATE TABLE IF NOT EXISTS keyframe_bbox_annotations (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyframe_id          TEXT NOT NULL,
    ph_id                UUID REFERENCES person_hypotheses(ph_id) ON DELETE CASCADE,
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
    override_x1          REAL,
    override_y1          REAL,
    override_x2          REAL,
    override_y2          REAL,
    override_by          TEXT,
    override_at          TIMESTAMPTZ,
    bbox_age_frames      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_kba_keyframe_id
    ON keyframe_bbox_annotations (keyframe_id);

CREATE INDEX IF NOT EXISTS idx_kba_ph_id
    ON keyframe_bbox_annotations (ph_id);

CREATE INDEX IF NOT EXISTS idx_kba_identity_id
    ON keyframe_bbox_annotations (identity_id)
    WHERE identity_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_bbox_annotations_confidence
    ON keyframe_bbox_annotations (detection_confidence)
    WHERE detection_confidence IS NOT NULL;

-- =============================================================================
-- ph_revisions: audit log for every identity change on a Person Hypothesis (hypertable).
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
-- ph_merges: tracks which PHs were merged into which.
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

CREATE TRIGGER trg_identities_updated_at
    BEFORE UPDATE ON identities
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

CREATE TRIGGER trg_reid_gallery_updated_at
    BEFORE UPDATE ON reid_gallery
    FOR EACH ROW EXECUTE FUNCTION _update_updated_at();

-- =============================================================================
-- Camera adjacency topology (M5)
-- =============================================================================

-- Learned transit-time distributions between camera pairs from observed handoffs.
-- Updated online via the Welford algorithm; the topology model service computes
-- a plausibility score from the stored count, mean, and variance.
CREATE TABLE IF NOT EXISTS camera_topology_edges (
    from_camera        TEXT        NOT NULL,
    to_camera          TEXT        NOT NULL,
    observation_count  INTEGER     NOT NULL DEFAULT 0,
    mean_transit_s     FLOAT8      NOT NULL DEFAULT 0.0,
    variance_transit_s2 FLOAT8     NOT NULL DEFAULT 0.0,
    last_updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (from_camera, to_camera)
);

-- Identity-level co-presence links for overlapping cameras.
-- Written when two open PHs in the same overlap group share a committed identity.
-- The CHECK constraint prevents duplicate directional pairs.
CREATE TABLE IF NOT EXISTS co_presence_links (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id            TEXT        NOT NULL,
    ph_id_a             UUID        NOT NULL REFERENCES person_hypotheses(ph_id) ON DELETE CASCADE,
    ph_id_b             UUID        NOT NULL REFERENCES person_hypotheses(ph_id) ON DELETE CASCADE,
    identity_id         TEXT        NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
    first_observed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_observed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    observation_count   INTEGER     NOT NULL DEFAULT 1,
    CONSTRAINT uq_copresence_ph_pair UNIQUE (ph_id_a, ph_id_b),
    CONSTRAINT chk_ph_a_lt_ph_b CHECK (ph_id_a < ph_id_b)
);

-- =============================================================================
-- Continuous aggregates (non-transactional)
-- =============================================================================
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
