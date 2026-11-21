-- Wearable teaching revocation: stamp revoked association segments, track per-tag teaching
-- holds until an operator resumes teaching, and link inferred backfill ranges back to the
-- wearable association segment that triggered them.
SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.wearable_associations
    ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS revoked_by TEXT,
    ADD COLUMN IF NOT EXISTS revoke_reason TEXT
        CHECK (revoke_reason IN ('swap_suspected', 'operator'));

ALTER TABLE continuous_tracking.wearable_associations
    ADD CONSTRAINT wearable_associations_revocation_complete CHECK (
        (revoked_at IS NULL) = (revoked_by IS NULL)
        AND (revoked_at IS NULL) = (revoke_reason IS NULL)
    );

CREATE INDEX IF NOT EXISTS idx_wearable_associations_revoked_at
    ON continuous_tracking.wearable_associations (revoked_at DESC)
    WHERE revoked_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS continuous_tracking.wearable_teaching_holds (
    hold_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tag_id     TEXT NOT NULL,
    held_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason     TEXT NOT NULL,
    cleared_at TIMESTAMPTZ,
    cleared_by TEXT,
    CHECK ((cleared_at IS NULL) = (cleared_by IS NULL))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_wearable_teaching_holds_open_tag
    ON continuous_tracking.wearable_teaching_holds (tag_id)
    WHERE cleared_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_wearable_teaching_holds_tag_held
    ON continuous_tracking.wearable_teaching_holds (tag_id, held_at DESC);

ALTER TABLE continuous_tracking.identity_revision_ranges
    ADD COLUMN IF NOT EXISTS source_association_id UUID NULL;

CREATE INDEX IF NOT EXISTS idx_identity_revision_ranges_source_assoc
    ON continuous_tracking.identity_revision_ranges (source_association_id)
    WHERE source_association_id IS NOT NULL;
