-- 0008_auto_verified_gallery_state (rollback)
--
-- PostgreSQL cannot drop a value from an enum type. This rollback demotes
-- every 'auto_verified' row back to 'pending_review' (safe: it is the same
-- pre-mint state those rows would have landed in without this migration) and
-- leaves the 'auto_verified' enum label in place as a harmless orphan value.
-- This is the standard, documented PostgreSQL limitation, not an oversight.

SET search_path = continuous_tracking, public;

UPDATE continuous_tracking.reid_gallery
SET state = 'pending_review'
WHERE state = 'auto_verified';
