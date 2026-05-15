-- migrate:no-transaction
-- Convert identity_id columns from UUID to TEXT to accept
-- cognitive-companion household_member IDs (String(64)) and
-- person-identification-service person_ids.
--
-- Non-transactional because TimescaleDB continuous aggregate operations
-- (drop/recreate materialized view) cannot run inside a transaction block.

SET search_path = continuous_tracking, public;

-- Step 1: Drop the continuous aggregate and its policy so we can alter
-- the source column type on dementia_signals.  Wrapped in a DO block so
-- the migration proceeds cleanly on databases where 0002 was never applied.
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

-- Step 2: Drop all FK constraints referencing identities(identity_id)
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

-- Step 3: Alter all identity_id columns from UUID to TEXT
ALTER TABLE continuous_tracking.identities ALTER COLUMN identity_id TYPE TEXT;

ALTER TABLE continuous_tracking.reid_gallery ALTER COLUMN identity_id TYPE TEXT;
ALTER TABLE continuous_tracking.person_activities ALTER COLUMN identity_id TYPE TEXT;
ALTER TABLE continuous_tracking.person_trajectories ALTER COLUMN identity_id TYPE TEXT;
ALTER TABLE continuous_tracking.room_dwells ALTER COLUMN identity_id TYPE TEXT;
ALTER TABLE continuous_tracking.dementia_signals ALTER COLUMN identity_id TYPE TEXT;

ALTER TABLE continuous_tracking.global_tracks ALTER COLUMN current_identity_id TYPE TEXT;

ALTER TABLE continuous_tracking.identity_revisions ALTER COLUMN map_identity_id TYPE TEXT;
ALTER TABLE continuous_tracking.identity_revisions ALTER COLUMN previous_identity_id TYPE TEXT;
ALTER TABLE continuous_tracking.identity_revisions ALTER COLUMN new_identity_id TYPE TEXT;

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
