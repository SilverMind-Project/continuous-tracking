SET search_path = continuous_tracking, public;

DROP INDEX IF EXISTS continuous_tracking.idx_room_dwells_room;
DROP INDEX IF EXISTS continuous_tracking.idx_room_dwells_open;
DROP INDEX IF EXISTS continuous_tracking.idx_global_tracks_active_seen;

-- Restore the original single-column partial index that was replaced
CREATE INDEX idx_global_tracks_state
    ON continuous_tracking.global_tracks (state)
    WHERE state = 'active';
