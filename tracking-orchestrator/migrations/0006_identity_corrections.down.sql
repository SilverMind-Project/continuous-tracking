SET search_path = continuous_tracking, public;

DROP TABLE IF EXISTS continuous_tracking.identity_projection_acks;
DROP TABLE IF EXISTS continuous_tracking.identity_revision_jobs;
DROP TABLE IF EXISTS continuous_tracking.identity_revision_ranges;
DROP TABLE IF EXISTS continuous_tracking.identity_corrections;

DROP TYPE IF EXISTS continuous_tracking.projection_ack_status;
DROP TYPE IF EXISTS continuous_tracking.revision_job_status;
DROP TYPE IF EXISTS continuous_tracking.revision_authority;
DROP TYPE IF EXISTS continuous_tracking.correction_kind;
DROP TYPE IF EXISTS continuous_tracking.correction_reason_code;
