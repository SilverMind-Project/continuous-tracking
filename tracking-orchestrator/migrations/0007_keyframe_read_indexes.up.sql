-- 0007_keyframe_read_indexes
--
-- M07 keyframe read model: indexes for grouping trigger rows into physical-frame
-- cards and resolving per-bbox provenance in bounded queries.
--
-- physical_frame_id is a read-time uuid5 over (camera_id, minio_key,
-- captured_at) and is never stored, so it cannot be indexed directly. The
-- composite over those source columns serves both the grouping key and the
-- camera-scoped recency scan that feeds one page.
--
-- Already covered by earlier migrations (no duplicate here):
--   identity_decisions (ph_id, captured_at DESC)  -- 0004 latest-per-PH lookup
--   identity_decisions conflict/authority/source   -- 0004 provenance filters
--   keyframe_bbox_annotations (keyframe_id, ph_id, identity_id) -- 0001 joins
--   identity_revision_ranges (ph_id, authority) WHERE live -- 0006 effective read

SET search_path = continuous_tracking, public;

-- Physical-frame grouping key + camera/time scoped window scan.
CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_physical
    ON tagged_keyframes (camera_id, minio_key, captured_at);

-- Recency-first window scan when no camera filter is applied.
CREATE INDEX IF NOT EXISTS idx_tagged_keyframes_captured
    ON tagged_keyframes (captured_at DESC);

-- Pending-ReID indicator: which PHs have a candidate awaiting review.
CREATE INDEX IF NOT EXISTS idx_reid_gallery_ph_pending
    ON reid_gallery (ph_id)
    WHERE state = 'pending_review';
