-- Migration 0004: Make reid_gallery.embedding nullable.
--
-- Issue #13: The pipeline writes all-zeros placeholder embeddings when
-- ReID inference is unavailable. With the zero-embedding fix (Issue #14),
-- embeddings are now None/NULL instead. This migration allows NULL values.

ALTER TABLE reid_gallery ALTER COLUMN embedding DROP NOT NULL;
