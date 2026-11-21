-- migrate:no-transaction
-- Adds the 'expired' lifecycle state for bounded body-sample retention and an index
-- supporting retention sweeps by (state, seen_at). Expired rows never vote and never
-- count toward the working-set cap; their embeddings are nulled and crops deleted
-- while the row and its gallery_review_events stay as an audit tombstone.

ALTER TYPE continuous_tracking.gallery_entry_state ADD VALUE IF NOT EXISTS 'expired';

CREATE INDEX IF NOT EXISTS idx_reid_gallery_state_seen_at
    ON continuous_tracking.reid_gallery (state, seen_at);
