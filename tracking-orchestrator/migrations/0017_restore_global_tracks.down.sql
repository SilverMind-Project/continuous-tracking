-- 0016_restore_global_tracks.down.sql
-- Reverses the restore by dropping the re-created tables.
-- Only safe to run if you intend to tear down the legacy tracking stack.

SET search_path = continuous_tracking, public;

DROP TABLE IF EXISTS do_not_fuse_hints  CASCADE;
DROP TABLE IF EXISTS tracklets           CASCADE;
DROP TABLE IF EXISTS global_tracks       CASCADE;
