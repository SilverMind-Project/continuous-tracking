-- Phase 5: Allow identity_id to be NULL on person_trajectories and
-- room_dwells so UNKNOWN tracks produce trajectory/dwell rows.
-- CR-12: Identity-bearing rows must never be skipped on UNKNOWN.
SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.person_trajectories
    ALTER COLUMN identity_id DROP NOT NULL;

ALTER TABLE continuous_tracking.room_dwells
    ALTER COLUMN identity_id DROP NOT NULL;

-- New table: first-class identity assignment history for each GlobalTrack.
-- One row per identity decision (initial commit, change, demotion, revision).
CREATE TABLE continuous_tracking.global_track_identity (
    id              BIGSERIAL PRIMARY KEY,
    global_track_id UUID NOT NULL REFERENCES continuous_tracking.global_tracks(global_track_id) ON DELETE CASCADE,
    identity_id     TEXT REFERENCES continuous_tracking.identities(identity_id) ON DELETE SET NULL,
    committed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    committed_by    TEXT NOT NULL,  -- 'auto' | 'manual'
    confidence      DOUBLE PRECISION,
    evidence_source TEXT,            -- 'face_high_confidence' | 'bayesian_posterior' | 'manual'
    revision_id     TEXT,
    applies_from    TIMESTAMPTZ,
    applies_to      TIMESTAMPTZ
);

CREATE INDEX idx_global_track_identity_gt_time
    ON continuous_tracking.global_track_identity (global_track_id, committed_at DESC);
CREATE INDEX idx_global_track_identity_revision
    ON continuous_tracking.global_track_identity (revision_id)
    WHERE revision_id IS NOT NULL;
