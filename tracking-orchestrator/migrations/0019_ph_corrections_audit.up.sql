-- 0019_ph_corrections_audit.up.sql
--
-- Adds audit tables for N1 Person Hypothesis corrections: ph_revisions
-- (identity change history) and ph_merges (PH merge tracking).
--
-- ph_revisions is a hypertable; the primary key includes applied_at
-- to comply with TimescaleDB hypertable rule (universal rule 15).

SET search_path = continuous_tracking, public;

-- ---------------------------------------------------------------------------
-- ph_revisions: audit log for every identity change on a Person Hypothesis
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS continuous_tracking.ph_revisions (
    revision_id         UUID NOT NULL,
    ph_id               UUID NOT NULL,
    previous_identity_id TEXT,
    new_identity_id     TEXT,
    actor               TEXT NOT NULL,
    reason              TEXT NOT NULL,
    kind                TEXT NOT NULL,
    applied_at          TIMESTAMPTZ NOT NULL,
    rewritten_rows      INT NOT NULL DEFAULT 0,
    evidence_jsonb      JSONB
);

SELECT create_hypertable(
    'continuous_tracking.ph_revisions', 'applied_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE continuous_tracking.ph_revisions
    ADD PRIMARY KEY (revision_id, applied_at);

CREATE INDEX IF NOT EXISTS idx_ph_revisions_ph_id
    ON continuous_tracking.ph_revisions (ph_id, applied_at DESC);

CREATE INDEX IF NOT EXISTS idx_ph_revisions_kind
    ON continuous_tracking.ph_revisions (kind, applied_at DESC);

-- ---------------------------------------------------------------------------
-- ph_merges: tracks which PHs were merged into which
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS continuous_tracking.ph_merges (
    merge_id     UUID PRIMARY KEY,
    source_ph_id UUID NOT NULL,
    target_ph_id UUID NOT NULL,
    revision_id  UUID NOT NULL,
    applied_at   TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ph_merges_source
    ON continuous_tracking.ph_merges (source_ph_id);

CREATE INDEX IF NOT EXISTS idx_ph_merges_target
    ON continuous_tracking.ph_merges (target_ph_id);
