"""Postgres implementation of BehaviorBaselineRepository.

Derives baselines from continuous_tracking.room_dwells and
continuous_tracking.person_trajectories using asyncpg with $N positional
placeholders throughout.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import asyncpg
from structlog import get_logger

from ..signals import (
    BehaviorBaselineRepository,
    DailyWindowSample,
    HourlyActivitySummary,
    StillnessEpisode,
)

_SQL_AGITATION_WINDOW_SAMPLES = """
SELECT composite
FROM continuous_tracking.agitation_windows
WHERE identity_id = $1
  AND window_start >= $2
  AND window_start < $3
ORDER BY window_start
"""

_SQL_SAVE_AGITATION_WINDOW = """
INSERT INTO continuous_tracking.agitation_windows
    (identity_id, window_start, composite, computed_at)
VALUES ($1, $2, $3, $4)
ON CONFLICT (identity_id, window_start) DO UPDATE
    SET composite = EXCLUDED.composite,
        computed_at = EXCLUDED.computed_at
"""

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

_SQL_DAILY_WINDOW_RATES = """
WITH local_pts AS (
    SELECT
        observed_at AT TIME ZONE $4 AS local_time,
        room_name,
        CASE
            WHEN $2::int > $3::int
                 AND EXTRACT(HOUR FROM observed_at AT TIME ZONE $4)::int < $3::int
            THEN ((observed_at AT TIME ZONE $4)::date - INTERVAL '1 day')::date
            ELSE (observed_at AT TIME ZONE $4)::date
        END AS window_date,
        CASE
            WHEN $2::int > $3::int THEN
                EXTRACT(HOUR FROM observed_at AT TIME ZONE $4)::int >= $2::int
                OR EXTRACT(HOUR FROM observed_at AT TIME ZONE $4)::int < $3::int
            ELSE
                EXTRACT(HOUR FROM observed_at AT TIME ZONE $4)::int >= $2::int
                AND EXTRACT(HOUR FROM observed_at AT TIME ZONE $4)::int < $3::int
        END AS in_window
    FROM continuous_tracking.person_trajectories
    WHERE identity_id = $1
      AND observed_at >= $5
      AND observed_at <= $6
),
filtered AS (
    SELECT local_time, room_name, window_date
    FROM local_pts
    WHERE in_window
),
with_lag AS (
    SELECT
        local_time,
        room_name,
        window_date,
        LAG(room_name) OVER (PARTITION BY window_date ORDER BY local_time) AS prev_room
    FROM filtered
)
SELECT
    window_date AS local_date,
    COUNT(*) FILTER (WHERE prev_room IS NOT NULL AND room_name <> prev_room)::int
        AS transition_count,
    COUNT(*)::int AS observed_points
FROM with_lag
GROUP BY window_date
ORDER BY window_date
"""

_SQL_PACING_WINDOW_RATES = """
WITH pts AS (
    SELECT
        date_bin($2, observed_at, $3::timestamptz) AS bucket,
        room_name,
        observed_at
    FROM continuous_tracking.person_trajectories
    WHERE identity_id = $1
      AND observed_at >= $3
      AND observed_at < $4
),
with_lag AS (
    SELECT
        bucket,
        room_name,
        LAG(room_name) OVER (PARTITION BY bucket ORDER BY observed_at) AS prev_room
    FROM pts
),
buckets AS (
    SELECT
        bucket,
        COUNT(*) AS point_count,
        COUNT(*) FILTER (
            WHERE prev_room IS NOT NULL AND room_name <> prev_room
        ) AS transitions
    FROM with_lag
    GROUP BY bucket
    HAVING COUNT(*) >= $5
)
SELECT transitions::float / $6::float AS rate_per_minute
FROM buckets
ORDER BY bucket
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

    async def daily_window_rates(
        self,
        identity_id: str,
        local_hour_start: int,
        local_hour_end: int,
        tz_name: str,
        since: datetime,
        until: datetime,
    ) -> list[DailyWindowSample]:
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    _SQL_DAILY_WINDOW_RATES,
                    identity_id,
                    local_hour_start,
                    local_hour_end,
                    tz_name,
                    since,
                    until,
                )
            return [
                DailyWindowSample(
                    local_date=row["local_date"],
                    transition_count=int(row["transition_count"]),
                    observed_points=int(row["observed_points"]),
                )
                for row in rows
            ]
        except Exception:
            logger.exception(
                "baseline_repo.daily_window_rates_failed",
                identity_id=identity_id,
                local_hour_start=local_hour_start,
                local_hour_end=local_hour_end,
            )
            return []

    async def pacing_window_rates(
        self,
        identity_id: str,
        window_minutes: int,
        since: datetime,
        until: datetime,
    ) -> list[float]:
        stride = timedelta(minutes=window_minutes)
        min_points = int(window_minutes * 0.5)
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    _SQL_PACING_WINDOW_RATES,
                    identity_id,
                    stride,
                    since,
                    until,
                    min_points,
                    float(window_minutes),
                )
            return [float(row["rate_per_minute"]) for row in rows]
        except Exception:
            logger.exception(
                "baseline_repo.pacing_window_rates_failed",
                identity_id=identity_id,
                window_minutes=window_minutes,
            )
            return []

    async def agitation_window_samples(
        self,
        identity_id: str,
        since: datetime,
        until: datetime,
    ) -> list[float]:
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    _SQL_AGITATION_WINDOW_SAMPLES,
                    identity_id,
                    since,
                    until,
                )
            return [float(row["composite"]) for row in rows]
        except Exception:
            logger.exception(
                "baseline_repo.agitation_window_samples_failed",
                identity_id=identity_id,
            )
            return []

    async def save_agitation_window(
        self,
        identity_id: str,
        window_start: datetime,
        composite: float,
    ) -> None:
        from datetime import UTC

        computed_at = datetime.now(UTC)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _SQL_SAVE_AGITATION_WINDOW,
                    identity_id,
                    window_start,
                    composite,
                    computed_at,
                )
        except Exception:
            logger.exception(
                "baseline_repo.save_agitation_window_failed",
                identity_id=identity_id,
            )
