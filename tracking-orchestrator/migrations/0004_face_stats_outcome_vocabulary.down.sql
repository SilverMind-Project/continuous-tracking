SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP CONSTRAINT face_observation_stats_gate_outcome_check;

UPDATE continuous_tracking.face_observation_stats
SET gate_outcome = 'dropped_min_confidence'
WHERE gate_outcome = 'dropped_min_similarity';

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_gate_outcome_check
    CHECK (gate_outcome IN ('emitted', 'dropped_min_confidence', 'dropped_no_candidate',
                            'dropped_no_detection', 'dropped_not_in_head_region',
                            'dropped_ambiguous'));
