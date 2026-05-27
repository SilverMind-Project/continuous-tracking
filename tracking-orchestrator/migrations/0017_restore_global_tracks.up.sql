-- 0016_restore_global_tracks.up.sql
--
-- Restores the tables that were prematurely dropped by 0008_drop_legacy_tracking.
-- The world-tracker migration (0008_drop_legacy_tracking) was applied before the
-- code finished migrating away from these tables.  The current code still uses
-- global_tracks, tracklets, and do_not_fuse_hints.
--
-- Re-creates each table with its full schema, incorporating all column additions
-- from the migrations that had already run (0008_global_track_last_posterior,
-- 0014_global_track_merges, 0015_identity_committed_at).
--
-- Existing data was lost when the tables were dropped; this migration creates
-- empty tables so the service can resume normal operation.  Existing FK constraints
-- that were silently removed by the CASCADE drop are re-added here.

SET search_path = continuous_tracking, public;

-- ---------------------------------------------------------------------------
-- global_tracks
-- Full column set: base (0001) + last_posterior (0008) + merges (0014)
--                 + identity_committed_at (0015)
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
    -- merge tracking (from 0014)
    merged_into_id                  UUID        REFERENCES global_tracks(global_track_id),
    merged_at                       TIMESTAMPTZ,
    merged_by                       TEXT,
    -- identity posterior (from 0008_global_track_last_posterior)
    last_posterior_jsonb            JSONB,
    last_posterior_at               TIMESTAMPTZ,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Composite index covering _SQL_LIST_ACTIVE (state='active' ORDER BY last_seen_at DESC)
CREATE INDEX IF NOT EXISTS idx_global_tracks_active_seen
    ON global_tracks (last_seen_at DESC)
    WHERE state = 'active';

-- GIN index for tracklet_ids array look-ups
CREATE INDEX IF NOT EXISTS idx_global_tracks_tracklet_ids
    ON global_tracks USING GIN (tracklet_ids);

-- Merge-history index (from 0014)
CREATE INDEX IF NOT EXISTS idx_gt_merged_into
    ON global_tracks (merged_into_id)
    WHERE merged_into_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- tracklets
-- Base schema from 0001_init; no subsequent column additions found.
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
-- From 0013_do_not_fuse; FKs reference the two tables above.
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
