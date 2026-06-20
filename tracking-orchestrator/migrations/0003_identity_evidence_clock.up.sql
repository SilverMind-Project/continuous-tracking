SET search_path = continuous_tracking, public;

-- M02: Separate independent-evidence time from the identity label write time.
-- Prior-only maintenance must not advance this clock; only direct ArcFace,
-- verified ReID, and operator corrections may refresh it.
ALTER TABLE continuous_tracking.person_hypotheses
    ADD COLUMN IF NOT EXISTS last_independent_identity_evidence_at TIMESTAMPTZ NULL;
