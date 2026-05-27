-- Make reid_gallery.identity_id nullable so gallery entries can be created
-- before identity resolution (M5 phase) and backfilled later.
-- The FK reference to identities is preserved; NULL values bypass FK checks
-- (standard SQL behaviour), which is the desired semantics for unassigned
-- gallery entries.
ALTER TABLE continuous_tracking.reid_gallery
    ALTER COLUMN identity_id DROP NOT NULL;
