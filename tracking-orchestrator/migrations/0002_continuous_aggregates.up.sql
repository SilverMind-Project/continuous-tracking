-- migrate:no-transaction
-- CREATE MATERIALIZED VIEW WITH (timescaledb.continuous) cannot run inside
-- a transaction block, and add_continuous_aggregate_policy must execute
-- outside a transaction as well.

CREATE MATERIALIZED VIEW IF NOT EXISTS dementia_signals_daily
WITH (timescaledb.continuous) AS
SELECT
    identity_id,
    signal_kind,
    time_bucket(INTERVAL '1 day', emitted_at) AS day,
    count(*)            AS signal_count,
    avg(value)          AS mean_value,
    stddev_pop(value)   AS sd_value
FROM dementia_signals
GROUP BY identity_id, signal_kind, day;

SELECT add_continuous_aggregate_policy(
    'dementia_signals_daily',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists     => TRUE
);
