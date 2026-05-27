SET search_path = continuous_tracking, public;

-- Records merge history. Source tracks are soft-deleted (merged_into_id stamped).
ALTER TABLE global_tracks
    ADD COLUMN IF NOT EXISTS merged_into_id UUID REFERENCES global_tracks(global_track_id),
    ADD COLUMN IF NOT EXISTS merged_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS merged_by TEXT;

CREATE INDEX IF NOT EXISTS idx_gt_merged_into
    ON global_tracks (merged_into_id)
    WHERE merged_into_id IS NOT NULL;
