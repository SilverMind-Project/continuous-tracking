SET search_path = continuous_tracking, public;

-- Where a gallery row's identity label came from. 'face' is every row minted before this
-- migration and every face-derived candidate; 'wearable_attested' rows come from a
-- trajectory-attested wearable association and vote at their own, lower trust;
-- 'wearable_room_attested' rows come from a room-attested association and are always minted
-- pending review; 'operator' is reserved for rows an operator adds directly.
ALTER TABLE reid_gallery
    ADD COLUMN IF NOT EXISTS seed_source TEXT NOT NULL DEFAULT 'face';

ALTER TABLE reid_gallery
    ADD CONSTRAINT reid_gallery_seed_source_check
    CHECK (seed_source IN ('face', 'wearable_attested', 'wearable_room_attested', 'operator'));

-- The wearable association segment that seeded the row, so a wrong binding can be demoted in
-- one operation. No foreign key: segments of a purged PH cascade away, and the gallery row keeps
-- its audit trail.
ALTER TABLE reid_gallery
    ADD COLUMN IF NOT EXISTS seed_association_id UUID;

CREATE INDEX IF NOT EXISTS idx_reid_gallery_seed_source
    ON reid_gallery (seed_source, seen_at DESC);
