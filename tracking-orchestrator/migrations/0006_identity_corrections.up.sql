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
