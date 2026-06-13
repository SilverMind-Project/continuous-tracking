SET search_path = continuous_tracking, public;

DROP TABLE IF EXISTS continuous_tracking.agitation_windows;
DROP TABLE IF EXISTS continuous_tracking.gait_daily;
DROP TABLE IF EXISTS continuous_tracking.gait_bouts;

ALTER TABLE continuous_tracking.person_trajectories
    DROP COLUMN IF EXISTS floor_speed_m_s;
