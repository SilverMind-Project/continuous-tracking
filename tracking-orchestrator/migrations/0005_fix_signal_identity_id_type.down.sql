-- Rollback 0005: Revert dementia_signals.identity_id from UUID back to VARCHAR(255).
ALTER TABLE continuous_tracking.dementia_signals
    ALTER COLUMN identity_id TYPE VARCHAR(255)
    USING identity_id::varchar;
