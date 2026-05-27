-- 0020_bbox_annotation_indexes.up.sql
-- Indexes for batch bbox annotation operations + bbox_age_frames column (M3).
SET search_path = continuous_tracking, public;

ALTER TABLE keyframe_bbox_annotations
    ADD COLUMN IF NOT EXISTS bbox_age_frames INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_bbox_annotations_keyframe
    ON keyframe_bbox_annotations (keyframe_id);

CREATE INDEX IF NOT EXISTS idx_bbox_annotations_confidence
    ON keyframe_bbox_annotations (detection_confidence)
    WHERE detection_confidence IS NOT NULL;
