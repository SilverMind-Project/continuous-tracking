SET search_path = continuous_tracking, public;

CREATE TABLE IF NOT EXISTS keyframe_bbox_annotations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyframe_id     TEXT NOT NULL,
    tracklet_id     UUID NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    camera_id       TEXT NOT NULL,
    x1              REAL NOT NULL,
    y1              REAL NOT NULL,
    x2              REAL NOT NULL,
    y2              REAL NOT NULL,
    detection_confidence REAL NOT NULL,
    frame_width     INTEGER NOT NULL,
    frame_height    INTEGER NOT NULL,
    identity_id     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Allow user-drawn override bbox (M4 will write to these columns)
    override_x1     REAL,
    override_y1     REAL,
    override_x2     REAL,
    override_y2     REAL,
    override_by     TEXT,        -- user who drew the override
    override_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_kba_keyframe_id
    ON keyframe_bbox_annotations (keyframe_id);

CREATE INDEX IF NOT EXISTS idx_kba_tracklet_id
    ON keyframe_bbox_annotations (tracklet_id);

CREATE INDEX IF NOT EXISTS idx_kba_identity_id
    ON keyframe_bbox_annotations (identity_id)
    WHERE identity_id IS NOT NULL;
