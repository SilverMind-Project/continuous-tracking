-- 0008_drop_legacy_tracking.up.sql
-- Run AFTER 0007 is verified and the world tracker is writing to the new tables.
SET search_path = continuous_tracking, public;

DROP TABLE IF EXISTS cross_camera_links CASCADE;
DROP TABLE IF EXISTS do_not_fuse_hints  CASCADE;
DROP TABLE IF EXISTS global_tracks       CASCADE;
DROP TABLE IF EXISTS tracklets           CASCADE;
DROP TABLE IF EXISTS local_tracks        CASCADE;
