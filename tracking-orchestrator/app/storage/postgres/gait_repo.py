"""Postgres implementation of GaitBoutRepository."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg

from ...trajectory.gait import WalkingBout
from ..gait import GaitBoutRepository

_SQL_UPSERT_BOUT = """
INSERT INTO continuous_tracking.gait_bouts
    (bout_id, identity_id, started_at, ended_at,
     duration_s, distance_m, median_speed_m_s, p95_speed_m_s,
     sample_count, rooms)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (bout_id) DO UPDATE SET
    ended_at         = EXCLUDED.ended_at,
    duration_s       = EXCLUDED.duration_s,
    distance_m       = EXCLUDED.distance_m,
    median_speed_m_s = EXCLUDED.median_speed_m_s,
    p95_speed_m_s    = EXCLUDED.p95_speed_m_s,
    sample_count     = EXCLUDED.sample_count,
    rooms            = EXCLUDED.rooms
"""

_SQL_LIST_BOUTS = """
SELECT bout_id, identity_id, started_at, ended_at,
       duration_s, distance_m, median_speed_m_s, p95_speed_m_s,
       sample_count, rooms
FROM continuous_tracking.gait_bouts
WHERE TRUE
"""


class PostgresGaitBoutRepository(GaitBoutRepository):
    """Postgres-backed GaitBoutRepository."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_bout(self, bout: WalkingBout) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_BOUT,
                bout.bout_id,
                bout.identity_id,
                bout.started_at,
                bout.ended_at,
                bout.duration_s,
                bout.distance_m,
                bout.median_speed_m_s,
                bout.p95_speed_m_s,
                bout.sample_count,
                bout.rooms,
            )

    async def list_bouts(
        self,
        identity_id: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 200,
    ) -> list[WalkingBout]:
        sql = _SQL_LIST_BOUTS
        args: list[Any] = []
        n = 1
        if identity_id is not None:
            sql += f" AND identity_id = ${n}"
            args.append(identity_id)
            n += 1
        if after is not None:
            sql += f" AND started_at >= ${n}"
            args.append(after)
            n += 1
        if before is not None:
            sql += f" AND started_at <= ${n}"
            args.append(before)
            n += 1
        sql += f" ORDER BY started_at DESC LIMIT ${n}"
        args.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_bout(r) for r in rows]


def _row_to_bout(row: Any) -> WalkingBout:
    return WalkingBout(
        identity_id=row["identity_id"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        duration_s=float(row["duration_s"]),
        distance_m=float(row["distance_m"]),
        median_speed_m_s=float(row["median_speed_m_s"]),
        p95_speed_m_s=float(row["p95_speed_m_s"]),
        sample_count=int(row["sample_count"]),
        rooms=list(row["rooms"]) if row["rooms"] else [],
    )
