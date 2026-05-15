SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.person_trajectories
    ADD COLUMN motion_energy DOUBLE PRECISION;

ALTER TABLE continuous_tracking.room_dwells
    ADD COLUMN min_motion_energy DOUBLE PRECISION,
    ADD COLUMN still_seconds INTEGER NOT NULL DEFAULT 0;
