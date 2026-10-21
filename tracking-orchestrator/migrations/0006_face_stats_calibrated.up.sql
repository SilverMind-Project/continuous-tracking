-- The calibrated confidence the face stage would trust for each face (null whenever it would
-- not), so the camera report can show how often recognized faces reach a usable probability.
SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    ADD COLUMN IF NOT EXISTS calibrated_confidence DOUBLE PRECISION;

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_calibrated_confidence_check
    CHECK (calibrated_confidence IS NULL OR calibrated_confidence BETWEEN 0 AND 1);
