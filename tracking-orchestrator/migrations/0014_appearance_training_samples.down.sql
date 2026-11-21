-- migrate:no-transaction
SET search_path = continuous_tracking, public;

SELECT remove_retention_policy(
    'continuous_tracking.appearance_training_samples',
    if_exists => TRUE
);

DROP TABLE IF EXISTS continuous_tracking.appearance_training_samples;
