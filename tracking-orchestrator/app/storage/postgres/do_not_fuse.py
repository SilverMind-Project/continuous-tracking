"""Postgres implementation of DoNotFuseRepository."""

from __future__ import annotations

import asyncpg  # type: ignore[import-untyped]


class PostgresDoNotFuseRepository:
    """Postgres-backed do-not-fuse hints repository."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add_hint(
        self, tracklet_id: str, global_track_id: str, created_by: str = "system"
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.do_not_fuse_hints
                    (tracklet_id, global_track_id, created_by)
                VALUES ($1::uuid, $2::uuid, $3)
                ON CONFLICT (tracklet_id, global_track_id) DO NOTHING
                """,
                tracklet_id,
                global_track_id,
                created_by,
            )

    async def is_blocked(self, tracklet_id: str, global_track_id: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM continuous_tracking.do_not_fuse_hints
                WHERE tracklet_id = $1::uuid AND global_track_id = $2::uuid
                """,
                tracklet_id,
                global_track_id,
            )
        return row is not None

    async def get_hints_for_tracklet(self, tracklet_id: str) -> list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT global_track_id::text
                FROM continuous_tracking.do_not_fuse_hints
                WHERE tracklet_id = $1::uuid
                """,
                tracklet_id,
            )
        return [r["global_track_id"] for r in rows]

    async def remove_hint(self, tracklet_id: str, global_track_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM continuous_tracking.do_not_fuse_hints
                WHERE tracklet_id = $1::uuid AND global_track_id = $2::uuid
                """,
                tracklet_id,
                global_track_id,
            )
