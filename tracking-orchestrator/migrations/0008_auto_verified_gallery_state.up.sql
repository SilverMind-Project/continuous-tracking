-- 0008_auto_verified_gallery_state
--
-- Identity Continuity M02 (decision D3): a fourth reid_gallery lifecycle
-- state, 'auto_verified', minted at candidate-creation time for calibrated
-- high-confidence face matches. Only operator_verified and auto_verified
-- rows vote in identity resolution.
--
-- ALTER TYPE ... ADD VALUE runs fine inside the MigrationRunner's default
-- transactional path on PG18: the restriction on using ADD VALUE inside the
-- same transaction that added it does not apply here because this migration
-- does not reference the new value anywhere else in the same statement batch
-- (verified against the target timescale/timescaledb-ha:pg18 image at
-- implementation time). No `-- migrate:no-transaction` pragma is needed.

SET search_path = continuous_tracking, public;

ALTER TYPE continuous_tracking.gallery_entry_state ADD VALUE IF NOT EXISTS 'auto_verified';
