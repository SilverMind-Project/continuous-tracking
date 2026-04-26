-- Migration 0005: Fix dementia_signals.identity_id VARCHAR(255) → UUID
-- Mismatches identities.identity_id which is UUID.

ALTER TABLE continuous_tracking.dementia_signals
    ALTER COLUMN identity_id TYPE UUID
    USING identity_id::uuid;
