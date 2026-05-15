SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.dementia_signals
    ADD COLUMN algorithm_version INTEGER NOT NULL DEFAULT 1;
