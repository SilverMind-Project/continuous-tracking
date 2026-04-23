-- Migration 0003: M8 dementia signals table
-- Requires TimescaleDB (already enabled in 0001_init.sql).

CREATE TABLE IF NOT EXISTS continuous_tracking.dementia_signals (
    signal_id        UUID        NOT NULL DEFAULT uuid_generate_v4(),
    identity_id      VARCHAR(255) NOT NULL,
    signal_kind      VARCHAR(64)  NOT NULL,
    severity         VARCHAR(16)  NOT NULL CHECK (severity IN ('info', 'warning', 'emergency')),
    value            FLOAT        NOT NULL,
    baseline         FLOAT,
    z_score          FLOAT,
    window_start     TIMESTAMPTZ  NOT NULL,
    window_end       TIMESTAMPTZ  NOT NULL,
    context_json     JSONB        NOT NULL DEFAULT '{}',
    emitted_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (signal_id, emitted_at)
);

SELECT create_hypertable(
    'continuous_tracking.dementia_signals',
    'emitted_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS dementia_signals_identity_kind_idx
    ON continuous_tracking.dementia_signals (identity_id, signal_kind, emitted_at DESC);

CREATE INDEX IF NOT EXISTS dementia_signals_severity_idx
    ON continuous_tracking.dementia_signals (severity, emitted_at DESC)
    WHERE severity IN ('warning', 'emergency');

-- Retention: keep 365 days of signal history.
SELECT add_retention_policy(
    'continuous_tracking.dementia_signals',
    INTERVAL '365 days',
    if_not_exists => TRUE
);

-- Daily baseline aggregate: mean and stddev per identity per signal kind.
-- Used by the DementiaSignalWorker to compute z-scores.
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
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists     => TRUE
);
