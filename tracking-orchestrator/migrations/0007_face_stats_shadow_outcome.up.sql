-- The "shadow" face evidence policy emits recognized faces below the legacy raw-similarity floor
-- to the resolver's shadow evaluation only; the stage records them with their own outcome so the
-- face resolution report can count what the "calibrated" policy would add.
SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP CONSTRAINT face_observation_stats_gate_outcome_check;

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_gate_outcome_check
    CHECK (gate_outcome IN ('emitted', 'emitted_shadow_only', 'dropped_min_similarity',
                            'dropped_no_candidate', 'dropped_no_detection',
                            'dropped_not_in_head_region', 'dropped_ambiguous'));
