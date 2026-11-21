SET search_path = continuous_tracking, public;

-- Seconds of image-only identity scored against wearable truth, per identity, UTC day,
-- evaluation arm, and attestation grade. control: decided without any wearable-taught gallery
-- row; treatment: with them, minus rows seen in the exclusion window. Aggregate seconds only:
-- no positions, crops, or embeddings. Purged with the identity.
CREATE TABLE IF NOT EXISTS image_only_evaluation_daily (
    day               DATE NOT NULL,
    identity_id       TEXT NOT NULL
                      REFERENCES continuous_tracking.identities(identity_id)
                      ON DELETE CASCADE,
    arm               TEXT NOT NULL CHECK (arm IN ('control', 'treatment')),
    attestation_grade TEXT NOT NULL CHECK (attestation_grade IN ('room', 'trajectory')),
    correct_s         DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (correct_s >= 0),
    wrong_s           DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (wrong_s >= 0),
    unknown_s         DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (unknown_s >= 0),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (day, identity_id, arm, attestation_grade)
);

CREATE INDEX IF NOT EXISTS idx_image_only_evaluation_daily_identity_day
    ON image_only_evaluation_daily (identity_id, day DESC);
