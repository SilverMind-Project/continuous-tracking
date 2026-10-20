-- migrate:no-transaction
-- Face observation statistics: one non-biometric row per face returned by
-- person-identification-service, so camera placement and face resolution can be
-- measured from live data. No embeddings, crops, or names are stored; rows expire
-- after 14 days.
SET search_path = continuous_tracking, public;

CREATE TABLE IF NOT EXISTS continuous_tracking.face_observation_stats (
    observed_at        TIMESTAMPTZ NOT NULL,
    camera_id          TEXT NOT NULL,
    detection_id       TEXT NOT NULL,
    crop_width_px      INTEGER NOT NULL CHECK (crop_width_px > 0),
    crop_height_px     INTEGER NOT NULL CHECK (crop_height_px > 0),
    face_width_px      DOUBLE PRECISION NOT NULL CHECK (face_width_px >= 0),
    face_height_px     DOUBLE PRECISION NOT NULL CHECK (face_height_px >= 0),
    interocular_px     DOUBLE PRECISION CHECK (interocular_px IS NULL OR interocular_px >= 0),
    det_score          DOUBLE PRECISION NOT NULL,
    yaw_deg            DOUBLE PRECISION NOT NULL,
    pitch_deg          DOUBLE PRECISION NOT NULL,
    raw_similarity     DOUBLE PRECISION NOT NULL,
    recognition_state  TEXT NOT NULL
        CHECK (recognition_state IN ('recognized', 'candidate', 'unrecognized')),
    best_candidate_id  TEXT,
    gate_outcome       TEXT NOT NULL
        CHECK (gate_outcome IN ('emitted', 'dropped_min_confidence',
                                'dropped_no_candidate', 'dropped_no_detection'))
);

SELECT create_hypertable(
    'continuous_tracking.face_observation_stats', 'observed_at', if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_face_obs_stats_camera_time
    ON continuous_tracking.face_observation_stats (camera_id, observed_at DESC);

SELECT add_retention_policy(
    'continuous_tracking.face_observation_stats', INTERVAL '14 days', if_not_exists => TRUE
);
