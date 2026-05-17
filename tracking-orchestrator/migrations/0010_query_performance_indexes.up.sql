-- Add indexes covering hot-path queries that current indexes miss.
--
-- 1. global_tracks: _SQL_LIST_ACTIVE filters on state='active' + last_seen_at
-- 2. room_dwells:  _SQL_GET_OPEN_DWELL filters on identity_id + global_track_id
-- 3. room_dwells:  list_room_dwells can filter by room_name

SET search_path = continuous_tracking, public;

-- Replace single-column partial index with a composite that covers both
-- the filter and the ORDER BY in _SQL_LIST_ACTIVE:
--   WHERE state = 'active' AND last_seen_at > now() - interval '5 minutes'
--   ORDER BY last_seen_at DESC
DROP INDEX IF EXISTS continuous_tracking.idx_global_tracks_state;
CREATE INDEX idx_global_tracks_active_seen
    ON continuous_tracking.global_tracks (last_seen_at DESC)
    WHERE state = 'active';

-- Open-dwell lookup: WHERE identity_id = $1 AND global_track_id = $2::uuid
-- AND exited_at IS NULL ORDER BY entered_at DESC LIMIT 1
CREATE INDEX idx_room_dwells_open
    ON continuous_tracking.room_dwells (identity_id, global_track_id, entered_at DESC)
    WHERE exited_at IS NULL;

-- Room-name filter for list_room_dwells: WHERE room_name = $2 ORDER BY entered_at DESC
CREATE INDEX idx_room_dwells_room
    ON continuous_tracking.room_dwells (room_name, entered_at DESC);
