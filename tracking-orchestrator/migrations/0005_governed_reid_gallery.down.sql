SET search_path = continuous_tracking, public;

DROP TABLE IF EXISTS continuous_tracking.gallery_review_events;

ALTER TABLE continuous_tracking.reid_gallery 
    DROP COLUMN IF EXISTS state,
    DROP COLUMN IF EXISTS proposed_identity_id,
    DROP COLUMN IF EXISTS effective_identity_id,
    DROP COLUMN IF EXISTS label_source,
    DROP COLUMN IF EXISTS model_version,
    DROP COLUMN IF EXISTS preprocessing_version,
    DROP COLUMN IF EXISTS dimension,
    DROP COLUMN IF EXISTS source_frame_key,
    DROP COLUMN IF EXISTS crop_key,
    DROP COLUMN IF EXISTS frame_hash,
    DROP COLUMN IF EXISTS crop_hash,
    DROP COLUMN IF EXISTS bbox,
    DROP COLUMN IF EXISTS crop_width,
    DROP COLUMN IF EXISTS crop_height,
    DROP COLUMN IF EXISTS ph_id,
    DROP COLUMN IF EXISTS observation_id,
    DROP COLUMN IF EXISTS keyframe_id,
    DROP COLUMN IF EXISTS camera_id,
    DROP COLUMN IF EXISTS capture_time,
    DROP COLUMN IF EXISTS confidence,
    DROP COLUMN IF EXISTS is_truncated,
    DROP COLUMN IF EXISTS is_occluded,
    DROP COLUMN IF EXISTS candidate_reason,
    DROP COLUMN IF EXISTS source_episode_id,
    DROP COLUMN IF EXISTS created_actor,
    DROP COLUMN IF EXISTS reviewed_actor,
    DROP COLUMN IF EXISTS reviewed_time,
    DROP COLUMN IF EXISTS review_reason,
    DROP COLUMN IF EXISTS review_note,
    DROP COLUMN IF EXISTS supersedes_id,
    DROP COLUMN IF EXISTS superseded_by_id,
    DROP COLUMN IF EXISTS audit_version;

DROP TYPE IF EXISTS continuous_tracking.gallery_entry_state;
