-- migrate:no-transaction
-- Revert identity_id columns from TEXT back to UUID.
-- WARNING: This will fail if any non-UUID identity_id values exist
-- (e.g. household_member string IDs from cognitive-companion).

SET search_path = continuous_tracking, public;

-- Step 1: Drop the continuous aggregate and its policy
DO $$
BEGIN
    BEGIN
        PERFORM remove_continuous_aggregate_policy(
            'continuous_tracking.dementia_signals_daily',
            if_not_exists => TRUE
        );
    EXCEPTION WHEN OTHERS THEN NULL;
    END;
    DROP MATERIALIZED VIEW IF EXISTS continuous_tracking.dementia_signals_daily;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

-- Step 2: Drop FK constraints
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT conname, conrelid::regclass::text AS tbl
        FROM pg_constraint
        WHERE confrelid = 'continuous_tracking.identities'::regclass
          AND contype = 'f'
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', r.tbl, r.conname);
    END LOOP;
END $$;

-- Step 3: Convert back to UUID (fails if non-UUID strings present)
ALTER TABLE continuous_tracking.identities ALTER COLUMN identity_id TYPE UUID USING identity_id::UUID;

ALTER TABLE continuous_tracking.reid_gallery ALTER COLUMN identity_id TYPE UUID USING identity_id::UUID;
ALTER TABLE continuous_tracking.person_activities ALTER COLUMN identity_id TYPE UUID USING identity_id::UUID;
ALTER TABLE continuous_tracking.person_trajectories ALTER COLUMN identity_id TYPE UUID USING identity_id::UUID;
ALTER TABLE continuous_tracking.room_dwells ALTER COLUMN identity_id TYPE UUID USING identity_id::UUID;
ALTER TABLE continuous_tracking.dementia_signals ALTER COLUMN identity_id TYPE UUID USING identity_id::UUID;

ALTER TABLE continuous_tracking.global_tracks ALTER COLUMN current_identity_id TYPE UUID USING current_identity_id::UUID;

ALTER TABLE continuous_tracking.identity_revisions ALTER COLUMN map_identity_id TYPE UUID USING map_identity_id::UUID;
ALTER TABLE continuous_tracking.identity_revisions ALTER COLUMN previous_identity_id TYPE UUID USING previous_identity_id::UUID;
ALTER TABLE continuous_tracking.identity_revisions ALTER COLUMN new_identity_id TYPE UUID USING new_identity_id::UUID;

-- Step 4: Re-add FK constraints
ALTER TABLE continuous_tracking.reid_gallery
    ADD FOREIGN KEY (identity_id) REFERENCES continuous_tracking.identities(identity_id) ON DELETE CASCADE;

ALTER TABLE continuous_tracking.person_activities
    ADD FOREIGN KEY (identity_id) REFERENCES continuous_tracking.identities(identity_id) ON DELETE SET NULL;

ALTER TABLE continuous_tracking.person_trajectories
    ADD FOREIGN KEY (identity_id) REFERENCES continuous_tracking.identities(identity_id) ON DELETE CASCADE;

ALTER TABLE continuous_tracking.room_dwells
    ADD FOREIGN KEY (identity_id) REFERENCES continuous_tracking.identities(identity_id) ON DELETE CASCADE;

ALTER TABLE continuous_tracking.dementia_signals
    ADD FOREIGN KEY (identity_id) REFERENCES continuous_tracking.identities(identity_id) ON DELETE CASCADE;

-- Step 5: Recreate the continuous aggregate
CREATE MATERIALIZED VIEW IF NOT EXISTS continuous_tracking.dementia_signals_daily
WITH (timescaledb.continuous) AS
SELECT
    identity_id,
    signal_kind,
    time_bucket(INTERVAL '1 day', emitted_at) AS day,
    count(*)            AS signal_count,
    avg(value)          AS mean_value,
    stddev_pop(value)   AS sd_value
FROM continuous_tracking.dementia_signals
GROUP BY identity_id, signal_kind, day;

SELECT add_continuous_aggregate_policy(
    'continuous_tracking.dementia_signals_daily',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists     => TRUE
);
