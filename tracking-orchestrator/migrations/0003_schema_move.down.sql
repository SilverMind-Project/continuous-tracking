-- Rollback: move tables back from continuous_tracking to public schema.
-- Skips tables already in public.

DO $$
DECLARE
    tbl TEXT;
    tables TEXT[] := ARRAY[
        'cameras',
        'streams',
        'tracking_events',
        'detections',
        'tracklets',
        'tracklet_gallery',
        'global_tracks',
        'identity_revisions',
        'identities',
        'reid_gallery',
        'person_activities',
        'stream_assignments',
        'person_trajectories',
        'room_dwells',
        'tagged_keyframes',
        'dementia_signals'
    ];
BEGIN
    FOREACH tbl IN ARRAY tables LOOP
        IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'continuous_tracking' AND tablename = tbl) THEN
            IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = tbl) THEN
                RAISE NOTICE '% already in public, skipping move', tbl;
            ELSE
                EXECUTE format('ALTER TABLE continuous_tracking.%I SET SCHEMA public', tbl);
            END IF;
        END IF;
    END LOOP;
END $$;

DROP FUNCTION IF EXISTS continuous_tracking._update_updated_at();
