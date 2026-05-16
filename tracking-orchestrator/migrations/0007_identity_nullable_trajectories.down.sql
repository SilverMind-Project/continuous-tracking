SET search_path = continuous_tracking, public;

DROP TABLE IF EXISTS continuous_tracking.global_track_identity;

ALTER TABLE continuous_tracking.person_trajectories
    ALTER COLUMN identity_id SET NOT NULL;

ALTER TABLE continuous_tracking.room_dwells
    ALTER COLUMN identity_id SET NOT NULL;
