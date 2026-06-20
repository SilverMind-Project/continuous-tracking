SET search_path = continuous_tracking, public;

CREATE TABLE continuous_tracking.identity_decisions (
    decision_id UUID PRIMARY KEY,
    ph_id UUID NOT NULL,
    observation_id UUID NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    inferred_identity_id TEXT NULL,
    effective_identity_id TEXT NULL,
    authority TEXT NOT NULL,
    decision_source TEXT NOT NULL,
    conflict_kind TEXT NULL,
    top_probability REAL NULL,
    second_probability REAL NULL,
    posterior_entropy REAL NULL,
    last_independent_evidence_at TIMESTAMPTZ NULL,
    config_hash TEXT NULL,
    resolver_version TEXT NULL,
    model_set_version TEXT NULL,
    diagnostics_schema_version TEXT NULL,
    diagnostics JSONB NOT NULL DEFAULT '{}',
    CONSTRAINT unique_decision_round UNIQUE (ph_id, observation_id, resolver_version)
);

CREATE INDEX idx_identity_decisions_ph_id_time ON continuous_tracking.identity_decisions (ph_id, captured_at DESC);
CREATE INDEX idx_identity_decisions_observation ON continuous_tracking.identity_decisions (observation_id);
CREATE INDEX idx_identity_decisions_effective_id ON continuous_tracking.identity_decisions (effective_identity_id);
CREATE INDEX idx_identity_decisions_conflict ON continuous_tracking.identity_decisions (conflict_kind) WHERE conflict_kind IS NOT NULL;
CREATE INDEX idx_identity_decisions_authority ON continuous_tracking.identity_decisions (authority);
CREATE INDEX idx_identity_decisions_source ON continuous_tracking.identity_decisions (decision_source);

CREATE TABLE continuous_tracking.identity_evidence_items (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID NOT NULL REFERENCES continuous_tracking.identity_decisions(decision_id) ON DELETE CASCADE,
    source_identity_id TEXT NULL,
    score_type TEXT NOT NULL,
    score_value REAL NOT NULL,
    quality REAL NULL,
    camera_id TEXT NULL,
    timestamp TIMESTAMPTZ NULL,
    model_version TEXT NULL,
    preprocessing_version TEXT NULL,
    calibration_version TEXT NULL,
    directness TEXT NULL,
    authoritative_eligibility BOOLEAN NULL
);

CREATE INDEX idx_identity_evidence_decision ON continuous_tracking.identity_evidence_items (decision_id);

CREATE TABLE continuous_tracking.identity_decision_gallery_hits (
    hit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id UUID NOT NULL REFERENCES continuous_tracking.identity_decisions(decision_id) ON DELETE CASCADE,
    entry_id UUID NOT NULL,
    identity_id TEXT NOT NULL,
    raw_similarity REAL NOT NULL,
    trust_multiplier REAL NOT NULL,
    recency_factor REAL NOT NULL,
    source_episode_group TEXT NULL,
    orientation TEXT NULL,
    rank INTEGER NOT NULL,
    weighted_contribution REAL NOT NULL
);

CREATE INDEX idx_identity_gallery_hits_decision ON continuous_tracking.identity_decision_gallery_hits (decision_id);
