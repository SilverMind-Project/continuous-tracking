-- Face image quality as reported by person-identification-service, so camera reports can break
-- face traffic down by quality band. Both columns are null for rows written before the service
-- scored quality, and they are null or set together.
SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS quality_version TEXT;

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_quality_score_check
    CHECK (quality_score IS NULL OR quality_score BETWEEN 0 AND 1);

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_quality_pair_check
    CHECK ((quality_score IS NULL) = (quality_version IS NULL));
