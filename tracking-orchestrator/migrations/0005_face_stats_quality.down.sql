SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP CONSTRAINT IF EXISTS face_observation_stats_quality_pair_check,
    DROP CONSTRAINT IF EXISTS face_observation_stats_quality_score_check;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP COLUMN IF EXISTS quality_version,
    DROP COLUMN IF EXISTS quality_score;
