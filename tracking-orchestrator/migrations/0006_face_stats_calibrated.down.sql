SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP CONSTRAINT IF EXISTS face_observation_stats_calibrated_confidence_check;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP COLUMN IF EXISTS calibrated_confidence;
