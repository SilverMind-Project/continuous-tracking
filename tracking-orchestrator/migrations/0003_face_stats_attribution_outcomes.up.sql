-- Face observation outcomes for head-region attribution: a face outside the person's head
-- region belongs to someone else, and every face of a crop with two faces in the head region is
-- ambiguous. Both are recorded so attribution can be measured per camera.
SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP CONSTRAINT face_observation_stats_gate_outcome_check;

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_gate_outcome_check
    CHECK (gate_outcome IN ('emitted', 'dropped_min_confidence', 'dropped_no_candidate',
                            'dropped_no_detection', 'dropped_not_in_head_region',
                            'dropped_ambiguous'));
