SET search_path = continuous_tracking, public;

DROP INDEX IF EXISTS idx_gt_merged_into;

ALTER TABLE global_tracks
    DROP COLUMN IF EXISTS merged_into_id,
    DROP COLUMN IF EXISTS merged_at,
    DROP COLUMN IF EXISTS merged_by;
