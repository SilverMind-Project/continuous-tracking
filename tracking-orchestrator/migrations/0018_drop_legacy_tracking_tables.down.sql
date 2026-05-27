-- 0018_drop_legacy_tracking_tables.down.sql
--
-- Recreates the legacy tracking tables as empty shells.
-- Data IS NOT restored — the world tracker pipeline populates
-- person_hypotheses and world_observations, not these tables.
-- This exists solely for reversibility (rule 11).

SET search_path = continuous_tracking, public;

-- ---------------------------------------------------------------------------
-- global_tracks
-- Full column set: base (0001) + last_posterior (was 0008) + merges (0015)
--                 + identity_committed_at (0016)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS global_tracks (
    global_track_id                 UUID PRIMARY KEY,
    camera_ids                      TEXT[]      NOT NULL DEFAULT '{}',
    tracklet_ids                    UUID[]      NOT NULL DEFAULT '{}',
    started_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    current_identity_id             TEXT,
    current_identity_committed_at   TIMESTAMPTZ,
    state                           TEXT        NOT NULL DEFAULT 'active'
                                        CHECK (state IN ('active', 'closed')),
    merged_into_id                  UUID        REFERENCES global_tracks(global_track_id),
    merged_at                       TIMESTAMPTZ,
    merged_by                       TEXT,
    last_posterior_jsonb            JSONB,
    last_posterior_at               TIMESTAMPTZ,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_global_tracks_active_seen
    ON global_tracks (last_seen_at DESC)
    WHERE state = 'active';

CREATE INDEX IF NOT EXISTS idx_global_tracks_tracklet_ids
    ON global_tracks USING GIN (tracklet_ids);

CREATE INDEX IF NOT EXISTS idx_gt_merged_into
    ON global_tracks (merged_into_id)
    WHERE merged_into_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- tracklets
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tracklets (
    tracklet_id     UUID        PRIMARY KEY,
    camera_id       TEXT        NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    detection_ids   UUID[]      NOT NULL DEFAULT '{}',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    state           TEXT        NOT NULL DEFAULT 'active'
                        CHECK (state IN ('active', 'terminated')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tracklets_camera_state
    ON tracklets (camera_id, state)
    WHERE state = 'active';

-- ---------------------------------------------------------------------------
-- do_not_fuse_hints
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS do_not_fuse_hints (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tracklet_id     UUID        NOT NULL REFERENCES tracklets(tracklet_id)     ON DELETE CASCADE,
    global_track_id UUID        NOT NULL REFERENCES global_tracks(global_track_id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      TEXT        NOT NULL DEFAULT 'system',
    UNIQUE (tracklet_id, global_track_id)
);

CREATE INDEX IF NOT EXISTS idx_dnf_tracklet_id
    ON do_not_fuse_hints (tracklet_id);

CREATE INDEX IF NOT EXISTS idx_dnf_global_track_id
    ON do_not_fuse_hints (global_track_id);
