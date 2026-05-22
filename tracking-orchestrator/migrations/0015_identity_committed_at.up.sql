SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.global_tracks
ADD COLUMN IF NOT EXISTS current_identity_committed_at timestamptz;
