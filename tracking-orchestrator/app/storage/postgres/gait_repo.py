"""Postgres implementations of GaitBoutRepository and GaitDailyRepository."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]

from ...trajectory.gait import GaitDailyRecord, WalkingBout
from ..gait import GaitBoutRepository, GaitDailyRepository

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


_SQL_UPSERT_DAILY = """
INSERT INTO continuous_tracking.gait_daily
    (identity_id, local_date, bout_count, total_walking_s, total_distance_m,
     median_speed_m_s, mad_speed_m_s, p95_speed_m_s, sample_bout_ids, computed_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (identity_id, local_date) DO UPDATE SET
    bout_count        = EXCLUDED.bout_count,
    total_walking_s   = EXCLUDED.total_walking_s,
    total_distance_m  = EXCLUDED.total_distance_m,
    median_speed_m_s  = EXCLUDED.median_speed_m_s,
    mad_speed_m_s     = EXCLUDED.mad_speed_m_s,
    p95_speed_m_s     = EXCLUDED.p95_speed_m_s,
    sample_bout_ids   = EXCLUDED.sample_bout_ids,
    computed_at       = EXCLUDED.computed_at
"""

_SQL_LIST_DAILY = """
SELECT identity_id, local_date, bout_count, total_walking_s, total_distance_m,
       median_speed_m_s, mad_speed_m_s, p95_speed_m_s, sample_bout_ids, computed_at
FROM continuous_tracking.gait_daily
WHERE identity_id = $1
"""


class PostgresGaitDailyRepository(GaitDailyRepository):
    """Postgres-backed GaitDailyRepository."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_day(self, record: GaitDailyRecord) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_DAILY,
                record.identity_id,
                record.local_date,
                record.bout_count,
                record.total_walking_s,
                record.total_distance_m,
                record.median_speed_m_s,
                record.mad_speed_m_s,
                record.p95_speed_m_s,
                [str(bid) for bid in record.sample_bout_ids],
                record.computed_at,
            )

    async def list_days(
        self,
        identity_id: str,
        since: date | None = None,
        until: date | None = None,
    ) -> list[GaitDailyRecord]:
        sql = _SQL_LIST_DAILY
        args: list[Any] = [identity_id]
        n = 2
        if since is not None:
            sql += f" AND local_date >= ${n}"
            args.append(since)
            n += 1
        if until is not None:
            sql += f" AND local_date <= ${n}"
            args.append(until)
            n += 1
        sql += " ORDER BY local_date ASC"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_daily(r) for r in rows]


def _row_to_daily(row: Any) -> GaitDailyRecord:
    raw_ids = row["sample_bout_ids"]
    bout_ids: list[str] = list(raw_ids) if raw_ids else []
    return GaitDailyRecord(
        identity_id=row["identity_id"],
        local_date=row["local_date"],
        bout_count=int(row["bout_count"]),
        total_walking_s=float(row["total_walking_s"]),
        total_distance_m=float(row["total_distance_m"]),
        median_speed_m_s=float(row["median_speed_m_s"]),
        mad_speed_m_s=float(row["mad_speed_m_s"]),
        p95_speed_m_s=float(row["p95_speed_m_s"]),
        sample_bout_ids=bout_ids,
        computed_at=row["computed_at"],
    )
