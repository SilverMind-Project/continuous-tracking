"""Postgres implementation of BehaviorBaselineRepository.

Derives baselines from continuous_tracking.room_dwells and
continuous_tracking.person_trajectories using asyncpg with $N positional
placeholders throughout.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from structlog import get_logger

from ..signals import BehaviorBaselineRepository, HourlyActivitySummary, StillnessEpisode

logger = get_logger(__name__)

_SQL_DWELL_DURATIONS = """
SELECT duration_seconds::float8
FROM continuous_tracking.room_dwells
WHERE identity_id = $1
  AND exited_at IS NOT NULL
  AND duration_seconds IS NOT NULL
  AND ($2::text IS NULL OR lower(room_name) LIKE '%' || lower($2) || '%')
  AND ($3::timestamptz IS NULL OR entered_at >= $3)
  AND ($4::timestamptz IS NULL OR entered_at <= $4)
ORDER BY entered_at
"""

_SQL_HOURLY_ACTIVITY = """
WITH pts AS (
  SELECT observed_at,
         room_name,
         LAG(room_name) OVER (ORDER BY observed_at) AS prev_room
  FROM continuous_tracking.person_trajectories
  WHERE identity_id = $1
    AND ($2::timestamptz IS NULL OR observed_at >= $2)
    AND ($3::timestamptz IS NULL OR observed_at <= $3)
)
SELECT EXTRACT(HOUR FROM observed_at)::int AS hour,
       COUNT(*) FILTER (WHERE prev_room IS NOT NULL AND room_name <> prev_room) AS transition_count,
       COUNT(*) AS observed_minutes
FROM pts
GROUP BY 1
"""

_SQL_STILLNESS_EPISODES = """
SELECT room_name, primary_posture, duration_seconds, min_motion_energy, entered_at
FROM continuous_tracking.room_dwells
WHERE identity_id = $1
  AND exited_at IS NOT NULL
  AND (min_motion_energy IS NOT NULL OR still_seconds > 0)
  AND ($2::timestamptz IS NULL OR entered_at >= $2)
  AND ($3::timestamptz IS NULL OR entered_at <= $3)
ORDER BY entered_at
"""


class PostgresBehaviorBaselineRepository(BehaviorBaselineRepository):
    """Postgres-backed BehaviorBaselineRepository.

    Requires a connected asyncpg.Pool injected at construction time.
    Derives baselines from room_dwells and person_trajectories; never
    reads previously emitted signals.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def dwell_durations(
        self,
        identity_id: str,
        room_predicate: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[float]:
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    _SQL_DWELL_DURATIONS,
                    identity_id,
                    room_predicate,
                    since,
                    until,
                )
            return [float(row[0]) for row in rows]
        except Exception:
            logger.exception(
                "baseline_repo.dwell_durations_failed",
                identity_id=identity_id,
            )
            return []

    async def hourly_activity(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[int, HourlyActivitySummary]:
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    _SQL_HOURLY_ACTIVITY,
                    identity_id,
                    since,
                    until,
                )
            return {
                int(row["hour"]): HourlyActivitySummary(
                    transition_count=int(row["transition_count"]),
                    observed_minutes=int(row["observed_minutes"]),
                )
                for row in rows
            }
        except Exception:
            logger.exception(
                "baseline_repo.hourly_activity_failed",
                identity_id=identity_id,
            )
            return {}

    async def stillness_episodes(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[StillnessEpisode]:
        try:
            async with self._pool.acquire() as conn:
                rows: list[Any] = await conn.fetch(
                    _SQL_STILLNESS_EPISODES,
                    identity_id,
                    since,
                    until,
                )
            return [
                StillnessEpisode(
                    room_name=row["room_name"],
                    posture=row["primary_posture"],
                    duration_seconds=int(row["duration_seconds"] or 0),
                    min_motion_energy=float(row["min_motion_energy"] or 0.0),
                    occurred_at=row["entered_at"],
                )
                for row in rows
            ]
        except Exception:
            logger.exception(
                "baseline_repo.stillness_episodes_failed",
                identity_id=identity_id,
            )
            return []
