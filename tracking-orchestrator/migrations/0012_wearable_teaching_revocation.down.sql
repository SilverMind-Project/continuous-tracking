SET search_path = continuous_tracking, public;

DROP INDEX IF EXISTS continuous_tracking.idx_identity_revision_ranges_source_assoc;

ALTER TABLE continuous_tracking.identity_revision_ranges
    DROP COLUMN IF EXISTS source_association_id;

DROP TABLE IF EXISTS continuous_tracking.wearable_teaching_holds;

DROP INDEX IF EXISTS continuous_tracking.idx_wearable_associations_revoked_at;

ALTER TABLE continuous_tracking.wearable_associations
    DROP CONSTRAINT IF EXISTS wearable_associations_revocation_complete;

ALTER TABLE continuous_tracking.wearable_associations
    DROP COLUMN IF EXISTS revoke_reason,
    DROP COLUMN IF EXISTS revoked_by,
    DROP COLUMN IF EXISTS revoked_at;
