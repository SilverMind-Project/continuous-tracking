-- Revert identity height profiles, ph_height_samples hypertable, and PH height_sigma_m.

SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.person_hypotheses
    DROP COLUMN IF EXISTS height_sigma_m;

SELECT remove_retention_policy(
    'continuous_tracking.ph_height_samples',
    if_exists => TRUE
);

DROP TABLE IF EXISTS continuous_tracking.ph_height_samples;
DROP TABLE IF EXISTS continuous_tracking.identity_height_profiles;
