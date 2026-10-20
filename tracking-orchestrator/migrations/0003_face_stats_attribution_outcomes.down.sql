SET search_path = continuous_tracking, public;

DELETE FROM continuous_tracking.face_observation_stats
WHERE gate_outcome IN ('dropped_not_in_head_region', 'dropped_ambiguous');

ALTER TABLE continuous_tracking.face_observation_stats
    DROP CONSTRAINT face_observation_stats_gate_outcome_check;

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_gate_outcome_check
    CHECK (gate_outcome IN ('emitted', 'dropped_min_confidence', 'dropped_no_candidate',
                            'dropped_no_detection'));
