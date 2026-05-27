-- Phase 1: support fast retroactive rewrites by the OrchestratorIdentityRewriter.
-- These indexes cover the WHERE clauses in the UPDATE / SELECT statements that
-- scan rows by (global_track_id, time) when an IdentityRevision fires.

SET search_path = continuous_tracking, public;

CREATE INDEX IF NOT EXISTS idx_person_trajectories_gt_observed
    ON continuous_tracking.person_trajectories (global_track_id, observed_at);

CREATE INDEX IF NOT EXISTS idx_room_dwells_gt_entered
    ON continuous_tracking.room_dwells (global_track_id, entered_at);

CREATE INDEX IF NOT EXISTS idx_dementia_signals_identity_window
    ON continuous_tracking.dementia_signals (identity_id, window_start);
