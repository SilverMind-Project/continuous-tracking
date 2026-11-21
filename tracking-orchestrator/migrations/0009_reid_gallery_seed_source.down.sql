SET search_path = continuous_tracking, public;

DROP INDEX IF EXISTS continuous_tracking.idx_reid_gallery_seed_source;
ALTER TABLE reid_gallery DROP COLUMN IF EXISTS seed_association_id;
ALTER TABLE reid_gallery DROP CONSTRAINT IF EXISTS reid_gallery_seed_source_check;
ALTER TABLE reid_gallery DROP COLUMN IF EXISTS seed_source;
