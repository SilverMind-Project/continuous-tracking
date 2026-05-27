SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.global_tracks
DROP COLUMN IF EXISTS current_identity_committed_at;
