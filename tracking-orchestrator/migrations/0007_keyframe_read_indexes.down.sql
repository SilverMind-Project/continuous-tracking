-- 0007_keyframe_read_indexes (rollback)

SET search_path = continuous_tracking, public;

DROP INDEX IF EXISTS continuous_tracking.idx_reid_gallery_ph_pending;
DROP INDEX IF EXISTS continuous_tracking.idx_tagged_keyframes_captured;
DROP INDEX IF EXISTS continuous_tracking.idx_tagged_keyframes_physical;
