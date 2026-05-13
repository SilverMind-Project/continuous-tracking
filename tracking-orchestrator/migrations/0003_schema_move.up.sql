-- Move tracking tables from public to continuous_tracking schema.
-- Skips tables already in continuous_tracking (e.g. after a fresh 0001_init
-- that creates them in the correct schema directly).

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
        IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = tbl) THEN
            IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'continuous_tracking' AND tablename = tbl) THEN
                RAISE NOTICE '% already in continuous_tracking, skipping move', tbl;
            ELSE
                EXECUTE format('ALTER TABLE public.%I SET SCHEMA continuous_tracking', tbl);
            END IF;
        END IF;
    END LOOP;
END $$;

-- Re-create the _update_updated_at function in continuous_tracking schema
-- so it resolves without search_path dependency.
CREATE OR REPLACE FUNCTION continuous_tracking._update_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
