"""Postgres implementation of DailyAppearanceRepo (DL-M07)."""

from __future__ import annotations

from datetime import date
from typing import Any

import asyncpg

from ...trajectory.appearance_profile import DailyAppearanceProfile
from ..appearance import DailyAppearanceRepo

_SQL_UPSERT = """
INSERT INTO continuous_tracking.daily_appearance_profiles
    (identity_id, day, centroid, sample_count, mean_quality, best_keyframe_objects, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (identity_id, day) DO UPDATE SET
    centroid              = EXCLUDED.centroid,
    sample_count          = EXCLUDED.sample_count,
    mean_quality          = EXCLUDED.mean_quality,
    best_keyframe_objects = EXCLUDED.best_keyframe_objects,
    created_at            = EXCLUDED.created_at
"""

_SQL_GET = """
SELECT identity_id, day, centroid, sample_count, mean_quality, best_keyframe_objects, created_at
FROM continuous_tracking.daily_appearance_profiles
WHERE identity_id = $1 AND day = $2
"""

_SQL_LIST_DAYS = """
SELECT identity_id, day, centroid, sample_count, mean_quality, best_keyframe_objects, created_at
FROM continuous_tracking.daily_appearance_profiles
WHERE identity_id = $1
"""


class PostgresDailyAppearanceRepo(DailyAppearanceRepo):
    """Postgres-backed DailyAppearanceRepo."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert_profile(self, profile: DailyAppearanceProfile) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT,
                profile.identity_id,
                profile.day,
                list(profile.centroid),
                profile.sample_count,
                profile.mean_quality,
                list(profile.best_keyframe_objects),
                profile.created_at,
            )

    async def get_profile(self, identity_id: str, day: date) -> DailyAppearanceProfile | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET, identity_id, day)
        return _row_to_profile(row) if row else None

    async def list_days(
        self,
        identity_id: str,
        since_day: date | None = None,
    ) -> list[DailyAppearanceProfile]:
        sql = _SQL_LIST_DAYS
        args: list[Any] = [identity_id]
        if since_day is not None:
            sql += " AND day >= $2"
            args.append(since_day)
        sql += " ORDER BY day ASC"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_profile(r) for r in rows]


def _row_to_profile(row: Any) -> DailyAppearanceProfile:
    raw_centroid = row["centroid"]
    raw_objects = row["best_keyframe_objects"]
    return DailyAppearanceProfile(
        identity_id=row["identity_id"],
        day=row["day"],
        centroid=tuple(float(v) for v in raw_centroid) if raw_centroid else (),
        sample_count=int(row["sample_count"]),
        mean_quality=float(row["mean_quality"]),
        best_keyframe_objects=tuple(raw_objects) if raw_objects else (),
        created_at=row["created_at"],
    )
