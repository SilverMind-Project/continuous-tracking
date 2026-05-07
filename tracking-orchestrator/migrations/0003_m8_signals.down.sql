-- Rollback 0003: Remove dementia signals schema, table, and continuous aggregate.
DROP MATERIALIZED VIEW IF EXISTS continuous_tracking.dementia_signals_daily;
DROP TABLE IF EXISTS continuous_tracking.dementia_signals;
DROP SCHEMA IF EXISTS continuous_tracking;
