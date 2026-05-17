"""Postgres implementation of GlobalTrackRepository."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]

from ...domain import GlobalTrack, IdentityCandidate
from ..base import GlobalTrackRepository

_SQL_SAVE = """
INSERT INTO continuous_tracking.global_tracks (
    global_track_id, camera_ids, tracklet_ids, started_at,
    last_seen_at, current_identity_id, state
)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (global_track_id) DO UPDATE SET
    camera_ids = (
        SELECT array_agg(DISTINCT v)
        FROM (
            SELECT unnest(EXCLUDED.camera_ids || continuous_tracking.global_tracks.camera_ids) AS v
        ) sub
        WHERE v <> ''
    ),
    tracklet_ids = (
        SELECT array_agg(DISTINCT v)
        FROM (
            SELECT unnest(
                EXCLUDED.tracklet_ids || continuous_tracking.global_tracks.tracklet_ids
            ) AS v
        ) sub
    ),
    started_at = LEAST(EXCLUDED.started_at, continuous_tracking.global_tracks.started_at),
    last_seen_at = GREATEST(EXCLUDED.last_seen_at, continuous_tracking.global_tracks.last_seen_at),
    current_identity_id = COALESCE(
        EXCLUDED.current_identity_id,
        continuous_tracking.global_tracks.current_identity_id
    ),
    state = EXCLUDED.state,
    updated_at = now()
"""

_SQL_GET = """
SELECT global_track_id, camera_ids, tracklet_ids, started_at,
       last_seen_at, current_identity_id, state, last_posterior_jsonb
FROM continuous_tracking.global_tracks
WHERE global_track_id = $1
"""

_SQL_LIST_ACTIVE = """
SELECT global_track_id, camera_ids, tracklet_ids, started_at,
       last_seen_at, current_identity_id, state, last_posterior_jsonb
FROM continuous_tracking.global_tracks
WHERE state = 'active'
ORDER BY last_seen_at DESC
"""

_SQL_GET_BY_TRACKLET = """
SELECT global_track_id, camera_ids, tracklet_ids, started_at,
       last_seen_at, current_identity_id, state, last_posterior_jsonb
FROM continuous_tracking.global_tracks
WHERE $1::uuid = ANY(tracklet_ids)
ORDER BY state = 'active' DESC, last_seen_at DESC
LIMIT 1
"""

_SQL_ASSIGN_IDENTITY = """
UPDATE continuous_tracking.global_tracks
SET current_identity_id = $2, updated_at = now()
WHERE global_track_id = $1
"""


class PostgresGlobalTrackRepository(GlobalTrackRepository):
    """Postgres-backed global-track repository."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, track: GlobalTrack) -> None:
        identity_id = track.current_identity_id if track.current_identity_id else None
        clean_tracklet_ids = [tid for tid in track.tracklet_ids if tid]
        import structlog

        _log = structlog.get_logger(__name__)
        _log.debug(
            "save_global_track",
            global_track_id=repr(track.global_track_id),
            camera_ids=track.camera_ids,
            tracklet_ids=clean_tracklet_ids,
            identity_id=repr(identity_id),
            state=track.state,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_SAVE,
                track.global_track_id,
                track.camera_ids,
                clean_tracklet_ids,
                track.started_at,
                track.last_seen_at,
                identity_id,
                track.state,
            )

    async def get(self, global_track_id: str) -> GlobalTrack | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET, global_track_id)
        return _row_to_global_track(row) if row is not None else None

    async def list_active(self) -> list[GlobalTrack]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_LIST_ACTIVE)
        return [_row_to_global_track(row) for row in rows]

    async def merge_tracklets(
        self,
        tracklet_ids: list[str],
        camera_ids: list[str],
        existing: GlobalTrack | None = None,
    ) -> GlobalTrack:
        now = datetime.now(UTC)
        track = (
            GlobalTrack(
                global_track_id=existing.global_track_id,
                camera_ids=list(dict.fromkeys(existing.camera_ids + camera_ids)),
                tracklet_ids=list(dict.fromkeys(existing.tracklet_ids + tracklet_ids)),
                started_at=existing.started_at,
                last_seen_at=now,
                current_identity_id=existing.current_identity_id,
                state="active",
            )
            if existing is not None
            else GlobalTrack(
                global_track_id=str(uuid.uuid4()),
                camera_ids=list(dict.fromkeys(camera_ids)),
                tracklet_ids=list(dict.fromkeys(tracklet_ids)),
                started_at=now,
                last_seen_at=now,
                current_identity_id=None,
                state="active",
            )
        )
        await self.save(track)
        return track

    async def merge_global_tracks(self, into_id: str, from_id: str) -> GlobalTrack | None:
        if into_id == from_id:
            return await self.get(into_id)

        async with self._pool.acquire() as conn, conn.transaction():
            into_row = await conn.fetchrow(_SQL_GET, into_id)
            from_row = await conn.fetchrow(_SQL_GET, from_id)
            if into_row is None or from_row is None:
                return _row_to_global_track(into_row) if into_row is not None else None

            into = _row_to_global_track(into_row)
            from_track = _row_to_global_track(from_row)
            merged = GlobalTrack(
                global_track_id=into.global_track_id,
                camera_ids=list(dict.fromkeys(into.camera_ids + from_track.camera_ids)),
                tracklet_ids=list(dict.fromkeys(into.tracklet_ids + from_track.tracklet_ids)),
                started_at=min(into.started_at, from_track.started_at),
                last_seen_at=max(into.last_seen_at, from_track.last_seen_at),
                current_identity_id=into.current_identity_id or from_track.current_identity_id,
                state="active",
            )
            await conn.execute(
                _SQL_SAVE,
                merged.global_track_id,
                merged.camera_ids,
                merged.tracklet_ids,
                merged.started_at,
                merged.last_seen_at,
                merged.current_identity_id,
                merged.state,
            )
            await conn.execute(
                "UPDATE continuous_tracking.global_tracks SET state = 'closed', updated_at = now() "
                "WHERE global_track_id = $1",
                from_id,
            )
            return merged

    async def assign_identity(
        self,
        global_track_id: str,
        identity_id: str | None,
        candidates: list[IdentityCandidate] | None = None,
    ) -> None:
        del candidates
        async with self._pool.acquire() as conn:
            await conn.execute(_SQL_ASSIGN_IDENTITY, global_track_id, identity_id)

    async def get_by_tracklet_id(self, tracklet_id: str) -> GlobalTrack | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_BY_TRACKLET, tracklet_id)
        return _row_to_global_track(row) if row is not None else None

    async def update_last_posterior(
        self,
        global_track_id: str,
        posterior_json: dict[str, float],
        at: datetime,
    ) -> None:
        import json

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE continuous_tracking.global_tracks
                SET last_posterior_jsonb = $2::jsonb,
                    last_posterior_at    = $3
                WHERE global_track_id = $1
                """,
                global_track_id,
                json.dumps(posterior_json),
                at,
            )


def _row_to_global_track(row: Any) -> GlobalTrack:
    raw_posterior = row["last_posterior_jsonb"]
    posterior: dict[str, Any] | None = (
        json.loads(raw_posterior) if isinstance(raw_posterior, str) else raw_posterior
    )
    return GlobalTrack(
        global_track_id=str(row["global_track_id"]),
        camera_ids=list(row["camera_ids"] or []),
        tracklet_ids=[str(tid) for tid in row["tracklet_ids"] or []],
        started_at=row["started_at"],
        last_seen_at=row["last_seen_at"],
        current_identity_id=(
            str(row["current_identity_id"]) if row["current_identity_id"] else None
        ),
        state=row["state"],
        last_posterior_jsonb=posterior,
    )
