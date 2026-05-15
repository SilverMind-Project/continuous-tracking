SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.dementia_signals
    DROP COLUMN algorithm_version;
