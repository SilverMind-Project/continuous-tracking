-- migrate:no-transaction
-- Multi-day body-appearance training samples for the household appearance classifier.
-- Stores frozen body embeddings labeled only from trajectory-attested wearable intervals,
-- calibrated direct-face authority, explicit operator verification, or open-set negatives
-- (never posterior-only inferred labels). Hypertable partitioned on observed_at with 60-day
-- retention; identity rows cascade on member deletion.
SET search_path = continuous_tracking, public;

CREATE TABLE IF NOT EXISTS continuous_tracking.appearance_training_samples (
    sample_id           UUID NOT NULL,
    observed_at         TIMESTAMPTZ NOT NULL,
    day                 DATE NOT NULL,
    identity_id         TEXT NULL
                        REFERENCES continuous_tracking.identities(identity_id)
                        ON DELETE CASCADE,
    label_source        TEXT NOT NULL
                        CHECK (label_source IN (
                            'wearable_attested',
                            'direct_face',
                            'operator_verified',
                            'negative'
                        )),
    camera_id           TEXT NOT NULL,
    orientation         SMALLINT NOT NULL
                        CHECK (orientation BETWEEN 0 AND 4),
    quality             DOUBLE PRECISION NOT NULL
                        CHECK (quality >= 0.0 AND quality <= 1.0),
    feature_version     TEXT NOT NULL,
    embedding           vector(768) NOT NULL,
    seed_association_id UUID NULL
                        REFERENCES continuous_tracking.wearable_associations(segment_id)
                        ON DELETE SET NULL,
    revoked_at          TIMESTAMPTZ NULL,
    PRIMARY KEY (sample_id, observed_at),
    CHECK (
        (label_source = 'negative' AND identity_id IS NULL)
        OR (label_source <> 'negative' AND identity_id IS NOT NULL)
    )
);

SELECT create_hypertable(
    'continuous_tracking.appearance_training_samples',
    'observed_at',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'continuous_tracking.appearance_training_samples',
    INTERVAL '60 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_appearance_training_samples_identity_day
    ON continuous_tracking.appearance_training_samples (identity_id, day DESC, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_appearance_training_samples_feature_day
    ON continuous_tracking.appearance_training_samples (feature_version, day DESC, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_appearance_training_samples_seed_assoc
    ON continuous_tracking.appearance_training_samples (seed_association_id)
    WHERE seed_association_id IS NOT NULL;
