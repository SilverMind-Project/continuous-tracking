-- migrate:no-transaction
-- Rollback 0005: Revert dementia_signals.identity_id from UUID back to VARCHAR(255).

SELECT remove_continuous_aggregate_policy(
    'continuous_tracking.dementia_signals_daily',
    if_not_exists => TRUE
);

DROP MATERIALIZED VIEW IF EXISTS continuous_tracking.dementia_signals_daily;

ALTER TABLE continuous_tracking.dementia_signals
    ALTER COLUMN identity_id TYPE VARCHAR(255)
    USING identity_id::varchar;

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
