-- Rollback 0002: Remove trajectory, room dwell, and keyframe tables.
DROP TABLE IF EXISTS tagged_keyframes;
DROP TABLE IF EXISTS room_dwells;
DROP TABLE IF EXISTS person_trajectories;
