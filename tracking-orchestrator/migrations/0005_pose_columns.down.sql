SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.person_trajectories
    DROP COLUMN motion_energy;

ALTER TABLE continuous_tracking.room_dwells
    DROP COLUMN min_motion_energy,
    DROP COLUMN still_seconds;
