-- 0008_drop_legacy_tracking.down.sql
-- Rollback recreates empty stubs only. Data is not recoverable; the user
-- has accepted this in M1.
SET search_path = continuous_tracking, public;

CREATE TABLE IF NOT EXISTS tracklets (
    tracklet_id TEXT PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS global_tracks (
    global_track_id TEXT PRIMARY KEY,
    metadata        JSONB NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS cross_camera_links (
    id        UUID PRIMARY KEY,
    metadata  JSONB NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS do_not_fuse_hints (
    id        UUID PRIMARY KEY,
    metadata  JSONB NOT NULL DEFAULT '{}'
);
