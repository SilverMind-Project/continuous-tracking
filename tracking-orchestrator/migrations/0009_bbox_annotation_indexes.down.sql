-- 0009_bbox_annotation_indexes.down.sql
SET search_path = continuous_tracking, public;

DROP INDEX IF EXISTS idx_bbox_annotations_keyframe;
DROP INDEX IF EXISTS idx_bbox_annotations_confidence;
ALTER TABLE keyframe_bbox_annotations DROP COLUMN IF EXISTS bbox_age_frames;
