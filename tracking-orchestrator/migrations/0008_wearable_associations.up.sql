SET search_path = continuous_tracking, public;

CREATE TABLE IF NOT EXISTS continuous_tracking.wearable_associations (
    segment_id        UUID PRIMARY KEY,
    ph_id             UUID NOT NULL
                      REFERENCES continuous_tracking.person_hypotheses(ph_id)
                      ON DELETE CASCADE,
    tag_id            TEXT NOT NULL,
    identity_id       TEXT NOT NULL
                      REFERENCES continuous_tracking.identities(identity_id)
                      ON DELETE CASCADE,
    started_at        TIMESTAMPTZ NOT NULL,
    ended_at          TIMESTAMPTZ,
    attestation_grade TEXT NOT NULL DEFAULT 'none'
                      CHECK (attestation_grade IN ('none', 'room', 'trajectory')),
    window_llr        DOUBLE PRECISION NOT NULL,
    margin_llr        DOUBLE PRECISION NOT NULL,
    matched_fixes     INTEGER NOT NULL
                      CHECK (matched_fixes >= 0),
    mean_residual_m   DOUBLE PRECISION
                      CHECK (mean_residual_m IS NULL OR mean_residual_m >= 0.0),
    swap_suspected    BOOLEAN NOT NULL DEFAULT FALSE,
    close_reason      TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

CREATE INDEX IF NOT EXISTS idx_wearable_associations_ph_started
    ON continuous_tracking.wearable_associations (ph_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_wearable_associations_tag_started
    ON continuous_tracking.wearable_associations (tag_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_wearable_associations_identity_started
    ON continuous_tracking.wearable_associations (identity_id, started_at DESC);
