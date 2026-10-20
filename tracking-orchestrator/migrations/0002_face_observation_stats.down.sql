-- migrate:no-transaction
SET search_path = continuous_tracking, public;

SELECT remove_retention_policy(
    'continuous_tracking.face_observation_stats', if_exists => TRUE
);

DROP TABLE IF EXISTS continuous_tracking.face_observation_stats;
