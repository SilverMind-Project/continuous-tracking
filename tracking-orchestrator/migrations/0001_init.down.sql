-- Rollback 0001: Drop all initial-schema objects in reverse dependency order.
-- This is a destructive operation that removes all tracking data.

DROP TRIGGER IF EXISTS trg_stream_assignments_updated_at ON stream_assignments;
DROP TRIGGER IF EXISTS trg_identities_updated_at ON identities;
DROP TRIGGER IF EXISTS trg_global_tracks_updated_at ON global_tracks;
DROP TRIGGER IF EXISTS trg_tracklets_updated_at ON tracklets;
DROP TRIGGER IF EXISTS trg_streams_updated_at ON streams;
DROP TRIGGER IF EXISTS trg_cameras_updated_at ON cameras;
DROP FUNCTION IF EXISTS _update_updated_at;

DROP TABLE IF EXISTS stream_assignments;
DROP TABLE IF EXISTS person_activities;
DROP TABLE IF EXISTS reid_gallery;
DROP TABLE IF EXISTS identities;
DROP TABLE IF EXISTS identity_revisions;
DROP TABLE IF EXISTS global_tracks;
DROP TABLE IF EXISTS tracklet_gallery;
DROP TABLE IF EXISTS tracklets;
DROP TABLE IF EXISTS detections;
DROP TABLE IF EXISTS tracking_events;
DROP TABLE IF EXISTS streams;
DROP TABLE IF EXISTS cameras;
