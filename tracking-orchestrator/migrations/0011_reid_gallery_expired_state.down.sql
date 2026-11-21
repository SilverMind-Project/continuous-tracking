-- Refuse rollback while expired rows exist; Postgres cannot drop an enum value
-- without recreating the type. Drop only the retention index once no expired row remains.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM continuous_tracking.reid_gallery
        WHERE state = 'expired'
    ) THEN
        RAISE EXCEPTION 'Cannot roll back 0011_reid_gallery_expired_state while expired rows exist';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM continuous_tracking.gallery_review_events
        WHERE previous_state = 'expired' OR new_state = 'expired'
    ) THEN
        RAISE EXCEPTION 'Cannot roll back 0011_reid_gallery_expired_state while expired review events exist';
    END IF;
END $$;

DROP INDEX IF EXISTS continuous_tracking.idx_reid_gallery_state_seen_at;
