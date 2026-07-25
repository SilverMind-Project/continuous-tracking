SET search_path = continuous_tracking, public;

-- DL-M07: daily quality-weighted appearance centroid per identity per local day,
-- feeding the same_clothes_suspected evaluator. centroid mirrors
-- person_hypotheses.gallery_mean's storage type exactly (FLOAT4[], L2-normalised
-- SOLIDER embedding).
CREATE TABLE IF NOT EXISTS continuous_tracking.daily_appearance_profiles (
    identity_id          TEXT NOT NULL REFERENCES continuous_tracking.identities(identity_id)
                             ON DELETE CASCADE,
    day                  DATE NOT NULL,
    centroid             FLOAT4[] NOT NULL,
    sample_count         INTEGER NOT NULL,
    mean_quality         REAL NOT NULL,
    best_keyframe_objects TEXT[] NOT NULL DEFAULT '{}',
    created_at           TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (identity_id, day)
);

CREATE INDEX IF NOT EXISTS idx_daily_appearance_profiles_identity
    ON continuous_tracking.daily_appearance_profiles (identity_id, day DESC);
