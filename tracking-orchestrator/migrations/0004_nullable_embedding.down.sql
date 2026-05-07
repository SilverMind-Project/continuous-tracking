-- Rollback 0004: Revert reid_gallery.embedding to NOT NULL.
-- NOTE: fails if any NULL embeddings exist in the table.
ALTER TABLE reid_gallery ALTER COLUMN embedding SET NOT NULL;
