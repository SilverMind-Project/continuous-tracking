-- 0019_ph_corrections_audit.down.sql
--
-- Reverses the N1 audit table additions.

SET search_path = continuous_tracking, public;

DROP TABLE IF EXISTS continuous_tracking.ph_merges CASCADE;
DROP TABLE IF EXISTS continuous_tracking.ph_revisions CASCADE;
