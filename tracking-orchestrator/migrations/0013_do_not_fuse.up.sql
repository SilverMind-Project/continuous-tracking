SET search_path = continuous_tracking, public;

-- Persists explicit "do not fuse these two" hints from caregiver corrections.
-- The cross-camera associator checks this table before merging tracklets.
CREATE TABLE IF NOT EXISTS do_not_fuse_hints (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tracklet_id     UUID NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    global_track_id UUID NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT NOT NULL DEFAULT 'system',
    UNIQUE (tracklet_id, global_track_id)
);

CREATE INDEX IF NOT EXISTS idx_dnf_tracklet_id
    ON do_not_fuse_hints (tracklet_id);

CREATE INDEX IF NOT EXISTS idx_dnf_global_track_id
    ON do_not_fuse_hints (global_track_id);
