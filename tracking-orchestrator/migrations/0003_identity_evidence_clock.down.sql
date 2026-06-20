SET search_path = continuous_tracking, public;

ALTER TABLE continuous_tracking.person_hypotheses
    DROP COLUMN IF EXISTS last_independent_identity_evidence_at;
