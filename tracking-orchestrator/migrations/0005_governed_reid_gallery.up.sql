SET search_path = continuous_tracking, public;

CREATE TYPE continuous_tracking.gallery_entry_state AS ENUM ('pending_review', 'operator_verified', 'rejected');

-- Add new columns to reid_gallery
ALTER TABLE continuous_tracking.reid_gallery 
    ADD COLUMN state continuous_tracking.gallery_entry_state NOT NULL DEFAULT 'pending_review',
    ADD COLUMN proposed_identity_id TEXT NULL,
    ADD COLUMN effective_identity_id TEXT NULL,
    ADD COLUMN label_source TEXT NULL,
    ADD COLUMN model_version TEXT NULL,
    ADD COLUMN preprocessing_version TEXT NULL,
    ADD COLUMN dimension INTEGER NULL,
    ADD COLUMN source_frame_key TEXT NULL,
    ADD COLUMN crop_key TEXT NULL,
    ADD COLUMN frame_hash TEXT NULL,
    ADD COLUMN crop_hash TEXT NULL,
    ADD COLUMN bbox JSONB NULL,
    ADD COLUMN crop_width INTEGER NULL,
    ADD COLUMN crop_height INTEGER NULL,
    ADD COLUMN ph_id UUID NULL,
    ADD COLUMN observation_id UUID NULL,
    ADD COLUMN keyframe_id UUID NULL,
    ADD COLUMN camera_id TEXT NULL,
    ADD COLUMN capture_time TIMESTAMPTZ NULL,
    ADD COLUMN confidence REAL NULL,
    ADD COLUMN is_truncated BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN is_occluded BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN candidate_reason TEXT NULL,
    ADD COLUMN source_episode_id UUID NULL,
    ADD COLUMN created_actor TEXT NULL,
    ADD COLUMN reviewed_actor TEXT NULL,
    ADD COLUMN reviewed_time TIMESTAMPTZ NULL,
    ADD COLUMN review_reason TEXT NULL,
    ADD COLUMN review_note TEXT NULL,
    ADD COLUMN supersedes_id UUID NULL REFERENCES continuous_tracking.reid_gallery(id),
    ADD COLUMN superseded_by_id UUID NULL REFERENCES continuous_tracking.reid_gallery(id),
    ADD COLUMN audit_version INTEGER NOT NULL DEFAULT 1;

-- Backfill every existing row to pending_review.
-- The default is already pending_review, but we ensure any row is pending_review
UPDATE continuous_tracking.reid_gallery SET state = 'pending_review';

-- Create gallery_review_events table
CREATE TABLE continuous_tracking.gallery_review_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_id UUID NOT NULL REFERENCES continuous_tracking.reid_gallery(id),
    previous_state continuous_tracking.gallery_entry_state NOT NULL,
    new_state continuous_tracking.gallery_entry_state NOT NULL,
    actor TEXT NOT NULL,
    reason TEXT NULL,
    note TEXT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT now(),
    audit_version INTEGER NOT NULL
);

CREATE INDEX idx_gallery_review_events_entry ON continuous_tracking.gallery_review_events(entry_id);
