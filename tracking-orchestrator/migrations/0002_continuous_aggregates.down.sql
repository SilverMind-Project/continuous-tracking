-- Rollback: remove the continuous aggregate and its refresh policy.

SELECT remove_continuous_aggregate_policy('dementia_signals_daily', if_not_exists => TRUE);
DROP MATERIALIZED VIEW IF EXISTS dementia_signals_daily;
