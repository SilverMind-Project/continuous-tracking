-- Phase 1: persist the latest per-frame Bayesian posterior on the global track.
-- The CC-side inspector drawer reads this column directly so it can show a
-- posterior bar even for tracks that have no committed revision yet.
-- Written by _emit_commit (or the per-frame identity loop) on every frame;
-- the actual commit gate is separate (identity_committer_enabled flag).

ALTER TABLE continuous_tracking.global_tracks
    ADD COLUMN IF NOT EXISTS last_posterior_jsonb JSONB,
    ADD COLUMN IF NOT EXISTS last_posterior_at TIMESTAMPTZ;
