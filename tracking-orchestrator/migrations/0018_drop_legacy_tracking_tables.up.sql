-- 0018_drop_legacy_tracking_tables.up.sql
--
-- Drops tables from the legacy BoT-SORT + GlobalTrack pipeline that are no
-- longer used by the world-coordinate tracker (M1).  The world tracker uses
-- person_hypotheses and world_observations instead.
--
-- Tables dropped:
--   do_not_fuse_hints  (depends on both tracklets and global_tracks)
--   tracklets           (per-camera short-lived trajectories)
--   global_tracks       (cross-camera persistent tracks)
--
-- Drop order respects FK dependencies.
--
-- NO DATA MIGRATION: the legacy table contents are stale and were never
-- authoritative after the world tracker deployment.  The down migration
-- recreates empty tables.

SET search_path = continuous_tracking, public;

DROP TABLE IF EXISTS continuous_tracking.do_not_fuse_hints CASCADE;
DROP TABLE IF EXISTS continuous_tracking.tracklets CASCADE;
DROP TABLE IF EXISTS continuous_tracking.global_tracks CASCADE;
