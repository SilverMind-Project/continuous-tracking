-- Rollback: drop all continuous-tracking objects in reverse dependency order.

DROP TRIGGER IF EXISTS trg_stream_assignments_updated_at ON continuous_tracking.stream_assignments;
DROP TRIGGER IF EXISTS trg_reid_gallery_updated_at ON continuous_tracking.reid_gallery;
DROP TRIGGER IF EXISTS trg_identities_updated_at ON continuous_tracking.identities;
DROP TRIGGER IF EXISTS trg_global_tracks_updated_at ON continuous_tracking.global_tracks;
DROP TRIGGER IF EXISTS trg_tracklets_updated_at ON continuous_tracking.tracklets;
DROP TRIGGER IF EXISTS trg_streams_updated_at ON continuous_tracking.streams;
DROP TRIGGER IF EXISTS trg_cameras_updated_at ON continuous_tracking.cameras;
DROP FUNCTION IF EXISTS continuous_tracking._update_updated_at;

DROP TABLE IF EXISTS continuous_tracking.dementia_signals;
DROP TABLE IF EXISTS continuous_tracking.tagged_keyframes;
DROP TABLE IF EXISTS continuous_tracking.room_dwells;
DROP TABLE IF EXISTS continuous_tracking.person_trajectories;
DROP TABLE IF EXISTS continuous_tracking.stream_assignments;
DROP TABLE IF EXISTS continuous_tracking.person_activities;
DROP TABLE IF EXISTS continuous_tracking.reid_gallery;
DROP TABLE IF EXISTS continuous_tracking.identities;
DROP TABLE IF EXISTS continuous_tracking.identity_revisions;
DROP TABLE IF EXISTS continuous_tracking.global_tracks;
DROP TABLE IF EXISTS continuous_tracking.tracklet_gallery;
DROP TABLE IF EXISTS continuous_tracking.tracklets;
DROP TABLE IF EXISTS continuous_tracking.detections;
DROP TABLE IF EXISTS continuous_tracking.tracking_events;
DROP TABLE IF EXISTS continuous_tracking.streams;
DROP TABLE IF EXISTS continuous_tracking.cameras;
