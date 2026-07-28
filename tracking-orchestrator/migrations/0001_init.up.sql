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
    position_sigma_m    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    primary_camera_id   TEXT NOT NULL DEFAULT '',
    contributing_camera_count INTEGER NOT NULL DEFAULT 1,
    footpoint_reliable  BOOLEAN NOT NULL DEFAULT TRUE,
    motion_energy       DOUBLE PRECISION,
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


-- ============================================================================
-- Folded from 0002_gait_and_agitation_schema
-- ============================================================================
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


-- ============================================================================
-- Folded from 0003_identity_evidence_clock
-- ============================================================================
SET search_path = continuous_tracking, public;

-- M02: Separate independent-evidence time from the identity label write time.
-- Prior-only maintenance must not advance this clock; only direct ArcFace,
-- verified ReID, and operator corrections may refresh it.
ALTER TABLE continuous_tracking.person_hypotheses
    ADD COLUMN IF NOT EXISTS last_independent_identity_evidence_at TIMESTAMPTZ NULL;


-- ============================================================================
-- Folded from 0004_identity_provenance
-- ============================================================================
SET search_path = continuous_tracking, public;

CREATE TABLE continuous_tracking.identity_decisions (
    decision_id UUID PRIMARY KEY,
    ph_id UUID NOT NULL,
    observation_id UUID NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    inferred_identity_id TEXT NULL,
    effective_identity_id TEXT NULL,
    authority TEXT NOT NULL,
    decision_source TEXT NOT NULL,
    conflict_kind TEXT NULL,
    top_probability REAL NULL,
    second_probability REAL NULL,
    posterior_entropy REAL NULL,
    last_independent_evidence_at TIMESTAMPTZ NULL,
    config_hash TEXT NULL,
    resolver_version TEXT NULL,
    model_set_version TEXT NULL,
    diagnostics_schema_version TEXT NULL,
    diagnostics JSONB NOT NULL DEFAULT '{}',
    CONSTRAINT unique_decision_round UNIQUE (ph_id, observation_id, resolver_version)
);

CREATE INDEX idx_identity_decisions_ph_id_time ON continuous_tracking.identity_decisions (ph_id, captured_at DESC);
CREATE INDEX idx_identity_decisions_observation ON continuous_tracking.identity_decisions (observation_id);
CREATE INDEX idx_identity_decisions_effective_id ON continuous_tracking.identity_decisions (effective_identity_id);
CREATE INDEX idx_identity_decisions_conflict ON continuous_tracking.identity_decisions (conflict_kind) WHERE conflict_kind IS NOT NULL;
CREATE INDEX idx_identity_decisions_authority ON continuous_tracking.identity_decisions (authority);
CREATE INDEX idx_identity_decisions_source ON continuous_tracking.identity_decisions (decision_source);

CREATE TABLE continuous_tracking.identity_evidence_items (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID NOT NULL REFERENCES continuous_tracking.identity_decisions(decision_id) ON DELETE CASCADE,
    source_identity_id TEXT NULL,
    score_type TEXT NOT NULL,
    score_value REAL NOT NULL,
    quality REAL NULL,
    camera_id TEXT NULL,
    timestamp TIMESTAMPTZ NULL,
    model_version TEXT NULL,
    preprocessing_version TEXT NULL,
    calibration_version TEXT NULL,
    directness TEXT NULL,
    authoritative_eligibility BOOLEAN NULL
);

CREATE INDEX idx_identity_evidence_decision ON continuous_tracking.identity_evidence_items (decision_id);

CREATE TABLE continuous_tracking.identity_decision_gallery_hits (
    hit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID NOT NULL REFERENCES continuous_tracking.identity_decisions(decision_id) ON DELETE CASCADE,
    entry_id UUID NOT NULL,
    identity_id TEXT NOT NULL,
    raw_similarity REAL NOT NULL,
    trust_multiplier REAL NOT NULL,
    recency_factor REAL NOT NULL,
    source_episode_group TEXT NULL,
    orientation TEXT NULL,
    rank INTEGER NOT NULL,
    weighted_contribution REAL NOT NULL
);

CREATE INDEX idx_identity_gallery_hits_decision ON continuous_tracking.identity_decision_gallery_hits (decision_id);


-- ============================================================================
-- Folded from 0005_governed_reid_gallery
-- ============================================================================
SET search_path = continuous_tracking, public;

-- 'auto_verified' (Identity Continuity M02 D3) is declared here rather than
-- appended by a later ALTER TYPE; it stays last so the enum sort order matches.
CREATE TYPE continuous_tracking.gallery_entry_state AS ENUM ('pending_review', 'operator_verified', 'rejected', 'auto_verified');

-- Add new columns to reid_gallery
ALTER TABLE continuous_tracking.reid_gallery 
    ADD COLUMN state continuous_tracking.gallery_entry_state NOT NULL DEFAULT 'pending_review',
    ADD COLUMN proposed_identity_id TEXT NULL,
    ADD COLUMN effective_identity_id TEXT NULL,
    ADD COLUMN label_source TEXT NULL,
    ADD COLUMN model_version TEXT NULL,
    ADD COLUMN preprocessing_version TEXT NULL,
    ADD COLUMN dimension INTEGER NULL,
    ADD COLUMN source_frame_key TEXT NULL,
    ADD COLUMN crop_key TEXT NULL,
    ADD COLUMN frame_hash TEXT NULL,
    ADD COLUMN crop_hash TEXT NULL,
    ADD COLUMN bbox JSONB NULL,
    ADD COLUMN crop_width INTEGER NULL,
    ADD COLUMN crop_height INTEGER NULL,
    ADD COLUMN ph_id UUID NULL,
    ADD COLUMN observation_id UUID NULL,
    ADD COLUMN keyframe_id UUID NULL,
    ADD COLUMN camera_id TEXT NULL,
    ADD COLUMN capture_time TIMESTAMPTZ NULL,
    ADD COLUMN confidence REAL NULL,
    ADD COLUMN is_truncated BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN is_occluded BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN candidate_reason TEXT NULL,
    ADD COLUMN source_episode_id UUID NULL,
    ADD COLUMN created_actor TEXT NULL,
    ADD COLUMN reviewed_actor TEXT NULL,
    ADD COLUMN reviewed_time TIMESTAMPTZ NULL,
    ADD COLUMN review_reason TEXT NULL,
    ADD COLUMN review_note TEXT NULL,
    ADD COLUMN supersedes_id UUID NULL REFERENCES continuous_tracking.reid_gallery(id),
    ADD COLUMN superseded_by_id UUID NULL REFERENCES continuous_tracking.reid_gallery(id),
    ADD COLUMN audit_version INTEGER NOT NULL DEFAULT 1;

-- Backfill every existing row to pending_review.
-- The default is already pending_review, but we ensure any row is pending_review
UPDATE continuous_tracking.reid_gallery SET state = 'pending_review';

-- Create gallery_review_events table
CREATE TABLE continuous_tracking.gallery_review_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_id UUID NOT NULL REFERENCES continuous_tracking.reid_gallery(id),
    previous_state continuous_tracking.gallery_entry_state NOT NULL,
    new_state continuous_tracking.gallery_entry_state NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NULL,
    note TEXT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT now(),
    audit_version INTEGER NOT NULL
);

CREATE INDEX idx_gallery_review_events_entry ON continuous_tracking.gallery_review_events(entry_id);


-- ============================================================================
-- Folded from 0006_identity_corrections
-- ============================================================================
SET search_path = continuous_tracking, public;

-- =============================================================================
-- Milestone 06: Segment correction, revision ranges, jobs, and effective
-- projections.
--
-- These tables layer onto the existing ``ph_revisions`` (operator overrides) and
-- ``identity_decisions`` (raw inference) tables; they do not replace either.
--   * identity_corrections  -- the authoritative, append-only operator record
--   * identity_revision_ranges -- effective identity over an explicit time range
--   * identity_revision_jobs -- projection lifecycle (pending/applying/...)
--   * identity_projection_acks -- per-consumer acknowledgement of one revision
--
-- Raw ``identity_decisions.inferred_identity_id`` never changes. Effective reads
-- apply operator revision ranges on top of inference.
-- =============================================================================

CREATE TYPE continuous_tracking.correction_reason_code AS ENUM (
    'wrong_person',
    'identity_uncertain',
    'track_handoff',
    'duplicate_hypothesis',
    'bad_bbox',
    'other'
);

CREATE TYPE continuous_tracking.correction_kind AS ENUM (
    'label',          -- ordinary bounded/frame-only identity correction
    'frame_only',     -- single reviewed frame
    'handoff_split',  -- track-handoff correction that composed a PH split
    'geometry',       -- bbox/geometry correction sharing the audit envelope
    'compensation'    -- undo of a prior correction
);

CREATE TYPE continuous_tracking.revision_authority AS ENUM ('operator', 'inferred');

CREATE TYPE continuous_tracking.revision_job_status AS ENUM (
    'pending',
    'applying',
    'completed',
    'failed'
);

CREATE TYPE continuous_tracking.projection_ack_status AS ENUM ('acked', 'failed');

-- -----------------------------------------------------------------------------
-- identity_corrections: the immutable operator record. One row per operator
-- action. Raw inference is never mutated; this drives revision ranges.
-- -----------------------------------------------------------------------------
CREATE TABLE continuous_tracking.identity_corrections (
    correction_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ph_id                  UUID NOT NULL,
    actor                  TEXT NOT NULL,
    reason_code            continuous_tracking.correction_reason_code NOT NULL,
    note                   TEXT NULL,
    source_view            TEXT NULL,
    -- target_identity_id NULL with set_unknown=true means "Set to Unknown".
    target_identity_id     TEXT NULL,
    set_unknown            BOOLEAN NOT NULL DEFAULT false,
    correction_kind        continuous_tracking.correction_kind NOT NULL DEFAULT 'label',
    frame_only             BOOLEAN NOT NULL DEFAULT false,
    reviewed_frame_id      TEXT NULL,
    reviewed_bbox          JSONB NULL,
    observation_start      TIMESTAMPTZ NOT NULL,
    observation_end        TIMESTAMPTZ NOT NULL,
    -- Optimistic version token captured from the PH at proposal time.
    base_ph_version        BIGINT NOT NULL,
    base_revision_id       UUID NULL,
    revision_id            UUID NOT NULL,
    -- For compensation rows: the original correction being undone.
    compensates_correction_id UUID NULL
        REFERENCES continuous_tracking.identity_corrections(correction_id),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT identity_corrections_reviewed_bbox_object
        CHECK (reviewed_bbox IS NULL OR jsonb_typeof(reviewed_bbox) = 'object'),
    CONSTRAINT identity_corrections_range_order
        CHECK (observation_end >= observation_start),
    -- Either a concrete identity target or an explicit Unknown; never both empty.
    CONSTRAINT identity_corrections_target_present
        CHECK (set_unknown OR target_identity_id IS NOT NULL)
);

CREATE INDEX idx_corrections_ph_time
    ON continuous_tracking.identity_corrections (ph_id, observation_start);
CREATE INDEX idx_corrections_revision
    ON continuous_tracking.identity_corrections (revision_id);

-- -----------------------------------------------------------------------------
-- identity_revision_ranges: effective-identity projection. Operator ranges are
-- authoritative inside their bounds and cannot be superseded by inferred ranges.
-- -----------------------------------------------------------------------------
CREATE TABLE continuous_tracking.identity_revision_ranges (
    range_id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    revision_id            UUID NOT NULL,
    correction_id          UUID NULL
        REFERENCES continuous_tracking.identity_corrections(correction_id),
    ph_id                  UUID NOT NULL,
    effective_identity_id  TEXT NULL,  -- NULL == Unknown
    authority              continuous_tracking.revision_authority NOT NULL,
    range_start            TIMESTAMPTZ NOT NULL,
    range_end              TIMESTAMPTZ NOT NULL,
    supersedes_range_id    UUID NULL
        REFERENCES continuous_tracking.identity_revision_ranges(range_id),
    superseded_by_range_id UUID NULL
        REFERENCES continuous_tracking.identity_revision_ranges(range_id),
    compensated_by_revision_id UUID NULL,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT identity_revision_ranges_order CHECK (range_end >= range_start)
);

CREATE INDEX idx_revision_ranges_ph_time
    ON continuous_tracking.identity_revision_ranges (ph_id, range_start, range_end);
CREATE INDEX idx_revision_ranges_revision
    ON continuous_tracking.identity_revision_ranges (revision_id);
-- Effective lookups read only live (non-superseded) ranges.
CREATE INDEX idx_revision_ranges_live
    ON continuous_tracking.identity_revision_ranges (ph_id, authority)
    WHERE superseded_by_range_id IS NULL;

-- -----------------------------------------------------------------------------
-- identity_revision_jobs: a correction is complete only after every required
-- projection acknowledges the same revision_id. Failures retry idempotently;
-- an accepted correction is never rolled back.
-- -----------------------------------------------------------------------------
CREATE TABLE continuous_tracking.identity_revision_jobs (
    job_id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    revision_id            UUID NOT NULL UNIQUE,
    correction_id          UUID NULL
        REFERENCES continuous_tracking.identity_corrections(correction_id),
    status                 continuous_tracking.revision_job_status NOT NULL DEFAULT 'pending',
    required_projections   TEXT[] NOT NULL DEFAULT '{}',
    attempts               INTEGER NOT NULL DEFAULT 0,
    last_error             TEXT NULL,
    row_counts             JSONB NOT NULL DEFAULT '{}',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT identity_revision_jobs_row_counts_object
        CHECK (jsonb_typeof(row_counts) = 'object')
);

CREATE INDEX idx_revision_jobs_status
    ON continuous_tracking.identity_revision_jobs (status);

-- -----------------------------------------------------------------------------
-- identity_projection_acks: one row per (revision, consumer). Idempotent on
-- replay via the unique key.
-- -----------------------------------------------------------------------------
CREATE TABLE continuous_tracking.identity_projection_acks (
    ack_id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    revision_id            UUID NOT NULL,
    consumer               TEXT NOT NULL,
    schema_version         TEXT NOT NULL,
    status                 continuous_tracking.projection_ack_status NOT NULL,
    counts                 JSONB NOT NULL DEFAULT '{}',
    applied_at             TIMESTAMPTZ NULL,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT identity_projection_acks_counts_object
        CHECK (jsonb_typeof(counts) = 'object'),
    CONSTRAINT identity_projection_acks_unique UNIQUE (revision_id, consumer)
);

CREATE INDEX idx_projection_acks_revision
    ON continuous_tracking.identity_projection_acks (revision_id);


-- ============================================================================
-- Folded from 0007_keyframe_read_indexes
-- ============================================================================
-- 0007_keyframe_read_indexes
--
-- M07 keyframe read model: indexes for grouping trigger rows into physical-frame
-- cards and resolving per-bbox provenance in bounded queries.
--
-- physical_frame_id is a read-time uuid5 over (camera_id, minio_key,
-- captured_at) and is never stored, so it cannot be indexed directly. The
-- composite over those source columns serves both the grouping key and the
-- camera-scoped recency scan that feeds one page.
--
-- Already covered by earlier migrations (no duplicate here):
--   identity_decisions (ph_id, captured_at DESC)  -- 0004 latest-per-PH lookup
--   identity_decisions conflict/authority/source   -- 0004 provenance filters
--   keyframe_bbox_annotations (keyframe_id, ph_id, identity_id) -- 0001 joins
--   identity_revision_ranges (ph_id, authority) WHERE live -- 0006 effective read

SET search_path = continuous_tracking, public;

-- Physical-frame grouping key + camera/time scoped window scan.
CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_physical
    ON tagged_keyframes (camera_id, minio_key, captured_at);

-- Recency-first window scan when no camera filter is applied.
CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_captured
    ON tagged_keyframes (captured_at DESC);

-- Pending-ReID indicator: which PHs have a candidate awaiting review.
CREATE INDEX IF NOT EXISTS idx_reid_gallery_ph_pending
    ON reid_gallery (ph_id)
    WHERE state = 'pending_review';


-- ============================================================================
-- Folded from 0008_auto_verified_gallery_state
-- ============================================================================
-- 0008_auto_verified_gallery_state
--
-- Identity Continuity M02 (decision D3): a fourth reid_gallery lifecycle
-- state, 'auto_verified', minted at candidate-creation time for calibrated
-- high-confidence face matches. Only operator_verified and auto_verified
-- rows vote in identity resolution.
--
-- ALTER TYPE ... ADD VALUE runs fine inside the MigrationRunner's default
-- transactional path on PG18: the restriction on using ADD VALUE inside the
-- same transaction that added it does not apply here because this migration
-- does not reference the new value anywhere else in the same statement batch
-- (verified against the target timescale/timescaledb-ha:pg18 image at
-- implementation time). No `-- migrate:no-transaction` pragma is needed.

SET search_path = continuous_tracking, public;

-- (folded into the CREATE TYPE above; no ALTER needed in a baseline)


-- ============================================================================
-- Folded from 0009_daily_appearance_profiles
-- ============================================================================
SET search_path = continuous_tracking, public;

-- DL-M07: daily quality-weighted appearance centroid per identity per local day,
-- feeding the same_clothes_suspected evaluator. centroid mirrors
-- person_hypotheses.gallery_mean's storage type exactly (FLOAT4[], L2-normalised
-- SOLIDER embedding).
CREATE TABLE IF NOT EXISTS continuous_tracking.daily_appearance_profiles (
    identity_id          TEXT NOT NULL REFERENCES continuous_tracking.identities(identity_id)
                             ON DELETE CASCADE,
    day                  DATE NOT NULL,
    centroid             FLOAT4[] NOT NULL,
    sample_count         INTEGER NOT NULL,
    mean_quality         REAL NOT NULL,
    best_keyframe_objects TEXT[] NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (identity_id, day)
);

CREATE INDEX IF NOT EXISTS idx_daily_appearance_profiles_identity
    ON continuous_tracking.daily_appearance_profiles (identity_id, day DESC);
