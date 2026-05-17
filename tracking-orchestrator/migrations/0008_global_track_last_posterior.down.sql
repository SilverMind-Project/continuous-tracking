ALTER TABLE continuous_tracking.global_tracks
    DROP COLUMN IF EXISTS last_posterior_jsonb,
    DROP COLUMN IF EXISTS last_posterior_at;
