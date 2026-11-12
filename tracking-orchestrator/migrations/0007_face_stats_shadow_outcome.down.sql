-- Without the shadow policy a face recorded as emitted_shadow_only would have been dropped below
-- the raw-similarity floor, so that is what those rows become before the narrower check returns.
SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.face_observation_stats
    DROP CONSTRAINT face_observation_stats_gate_outcome_check;

UPDATE continuous_tracking.face_observation_stats
SET gate_outcome = 'dropped_min_similarity'
WHERE gate_outcome = 'emitted_shadow_only';

ALTER TABLE continuous_tracking.face_observation_stats
    ADD CONSTRAINT face_observation_stats_gate_outcome_check
    CHECK (gate_outcome IN ('emitted', 'dropped_min_similarity', 'dropped_no_candidate',
                            'dropped_no_detection', 'dropped_not_in_head_region',
                            'dropped_ambiguous'));
