-- Development data repair: remove PH rows with invalid metadata shape.
--
-- The domain contract is strict: person_hypotheses.metadata is a JSON object.
-- This system is still under development, so historical rows that violate
-- the contract are discarded instead of carrying compatibility decoding.

WITH invalid_ph AS (
    SELECT ph_id
    FROM continuous_tracking.person_hypotheses
    WHERE jsonb_typeof(metadata) <> 'object'
),
deleted_revisions AS (
    DELETE FROM continuous_tracking.ph_revisions r
    USING invalid_ph
    WHERE r.ph_id = invalid_ph.ph_id
),
deleted_merges AS (
    DELETE FROM continuous_tracking.ph_merges m
    USING invalid_ph
    WHERE m.source_ph_id = invalid_ph.ph_id
       OR m.target_ph_id = invalid_ph.ph_id
)
DELETE FROM continuous_tracking.person_hypotheses ph
USING invalid_ph
WHERE ph.ph_id = invalid_ph.ph_id;

ALTER TABLE continuous_tracking.person_hypotheses
    ADD CONSTRAINT person_hypotheses_metadata_object
    CHECK (jsonb_typeof(metadata) = 'object');
