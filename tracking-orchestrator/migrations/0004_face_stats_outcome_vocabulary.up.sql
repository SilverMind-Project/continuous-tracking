-- The recognized-face floor is a raw ArcFace cosine similarity (face_id.min_raw_similarity), not a
-- confidence, so the outcome that records it is renamed. Existing rows are rewritten; the value
-- list is otherwise unchanged.
SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP CONSTRAINT face_observation_stats_gate_outcome_check;

UPDATE continuous_tracking.face_observation_stats
SET gate_outcome = 'dropped_min_similarity'
WHERE gate_outcome = 'dropped_min_confidence';

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_gate_outcome_check
    CHECK (gate_outcome IN ('emitted', 'dropped_min_similarity', 'dropped_no_candidate',
                            'dropped_no_detection', 'dropped_not_in_head_region',
                            'dropped_ambiguous'));
