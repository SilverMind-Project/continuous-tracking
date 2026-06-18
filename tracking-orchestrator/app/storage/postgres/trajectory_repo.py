"""Postgres implementation of TrajectoryRepository.

Uses asyncpg with $N positional placeholders and datetime.now(UTC)
throughout, consistent with the rest of the Postgres storage layer.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg  # type: ignore[import-untyped]
from structlog import get_logger

from ...domain import PersonTrajectoryPoint, RoomDwell
from ..base import TrajectoryRepository

logger = get_logger(__name__)

_SQL_INSERT_TRAJECTORY = """
INSERT INTO continuous_tracking.person_trajectories
    (observed_at, identity_id, ph_id, room_name,
     ground_x, ground_y, posture, identity_confidence,
     position_sigma_m, primary_camera_id, contributing_camera_count, footpoint_reliable,
     motion_energy, floor_speed_m_s)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
"""

_SQL_INSERT_DWELL = """
INSERT INTO continuous_tracking.room_dwells
    (identity_id, ph_id, room_name, entered_at,
     entry_confidence, primary_posture, activity_summary,
     min_motion_energy, still_seconds)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
RETURNING id
"""

_SQL_UPDATE_DWELL = """
UPDATE continuous_tracking.room_dwells
SET exited_at         = $1,
    duration_seconds  = $2,
    primary_posture   = $3,
    activity_summary  = $4,
    min_motion_energy = $6,
    still_seconds     = $7
WHERE id = $5
"""

_SQL_CLOSE_DANGLING_DWELLS = """
UPDATE continuous_tracking.room_dwells d
SET exited_at        = ex.exit_at,
    duration_seconds = GREATEST(0, EXTRACT(EPOCH FROM (ex.exit_at - d.entered_at)))::int
FROM (
    SELECT d2.id,
           LEAST(
               $1::timestamptz,
               GREATEST(
                   d2.entered_at,
                   COALESCE(
                       (SELECT max(p.observed_at)
                        FROM continuous_tracking.person_trajectories p
                        WHERE p.ph_id = d2.ph_id),
                       d2.entered_at
                   )
               )
           ) AS exit_at
    FROM continuous_tracking.room_dwells d2
    WHERE d2.exited_at IS NULL
) ex
WHERE d.id = ex.id
"""

_SQL_GET_OPEN_DWELL = """
SELECT id, identity_id, ph_id, room_name,
       entered_at, entry_confidence, primary_posture, activity_summary
FROM continuous_tracking.room_dwells
WHERE identity_id = $1
  AND ph_id = $2::uuid
  AND exited_at IS NULL
ORDER BY entered_at DESC
LIMIT 1
"""

_SQL_LIST_TRAJECTORY = """
SELECT identity_id, ph_id, observed_at, room_name,
       ground_x, ground_y, posture, identity_confidence,
       position_sigma_m, primary_camera_id, contributing_camera_count, footpoint_reliable,
       motion_energy, floor_speed_m_s
FROM continuous_tracking.person_trajectories
WHERE TRUE
"""

_SQL_LIST_DWELLS = """
SELECT id, identity_id, ph_id, room_name,
       entered_at, exited_at, duration_seconds,
       entry_confidence, primary_posture, activity_summary,
       min_motion_energy, still_seconds
FROM continuous_tracking.room_dwells
WHERE TRUE
"""


class PostgresTrajectoryRepository(TrajectoryRepository):
    """Postgres-backed TrajectoryRepository.

    Requires a connected asyncpg.Pool injected at construction time.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        # In-memory cache of open dwell row IDs, keyed by (identity_id, ph_id).
        # Used to find the DB row id for update_room_dwell without an extra SELECT.
        self._open_dwell_db_id: dict[tuple[str | None, str], int] = {}

    async def save_trajectory_point(self, point: PersonTrajectoryPoint) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_INSERT_TRAJECTORY,
                point.observed_at,
                point.identity_id,
                point.ph_id,
                point.room_name,
                point.ground_x,
                point.ground_y,
                point.posture,
                point.identity_confidence,
                point.position_sigma_m,
                point.primary_camera_id,
                point.contributing_camera_count,
                point.footpoint_reliable,
                point.motion_energy,
                point.floor_speed_m_s,
            )

    async def save_room_dwell(self, dwell: RoomDwell) -> None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                _SQL_INSERT_DWELL,
                dwell.identity_id,
                dwell.ph_id,
                dwell.room_name,
                dwell.entered_at,
                dwell.entry_confidence,
                dwell.primary_posture,
                json.dumps(dwell.activity_summary),
                dwell.min_motion_energy,
                dwell.still_seconds,
            )
            if row is not None:
                db_id: int = row["id"]
                self._open_dwell_db_id[(dwell.identity_id, dwell.ph_id)] = db_id

    async def update_room_dwell(self, dwell: RoomDwell) -> None:
        key = (dwell.identity_id, dwell.ph_id)
        db_id = self._open_dwell_db_id.pop(key, None)
        if db_id is None:
            logger.warning(
                "update_room_dwell: no DB row id for dwell",
                identity_id=dwell.identity_id,
                ph_id=dwell.ph_id,
            )
            return

        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPDATE_DWELL,
                dwell.exited_at,
                dwell.duration_seconds,
                dwell.primary_posture,
                json.dumps(dwell.activity_summary),
                db_id,
                dwell.min_motion_energy,
                dwell.still_seconds,
            )

    async def get_open_dwell(self, identity_id: str, ph_id: str) -> RoomDwell | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_GET_OPEN_DWELL, identity_id, ph_id)
        if row is None:
            return None
        return _row_to_dwell(row)

    async def close_dangling_open_dwells(self, closed_at: datetime) -> int:
        async with self._pool.acquire() as conn:
            status = await conn.execute(_SQL_CLOSE_DANGLING_DWELLS, closed_at)
        # asyncpg returns a command tag like "UPDATE 15933"; the last token is
        # the affected row count.
        try:
            return int(status.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def list_trajectory_points(
        self,
        identity_id: str | None = None,
        ph_id: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[PersonTrajectoryPoint]:
        sql = _SQL_LIST_TRAJECTORY
        args: list[Any] = []
        n = 1
        if identity_id is not None:
            sql += f" AND identity_id = ${n}"
            args.append(identity_id)
            n += 1
        if ph_id is not None:
            sql += f" AND ph_id = ${n}::uuid"
            args.append(ph_id)
            n += 1
        if after is not None:
            sql += f" AND observed_at >= ${n}"
            args.append(after)
            n += 1
        sql += f" ORDER BY observed_at DESC LIMIT ${n}"
        args.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_point(r) for r in rows]

    async def list_room_dwells(
        self,
        identity_id: str | None = None,
        room_name: str | None = None,
        after: datetime | None = None,
        limit: int = 100,
    ) -> list[RoomDwell]:
        sql = _SQL_LIST_DWELLS
        args: list[Any] = []
        n = 1
        if identity_id is not None:
            sql += f" AND identity_id = ${n}"
            args.append(identity_id)
            n += 1
        if room_name is not None:
            sql += f" AND room_name = ${n}"
            args.append(room_name)
            n += 1
        if after is not None:
            sql += f" AND entered_at >= ${n}"
            args.append(after)
            n += 1
        sql += f" ORDER BY entered_at DESC LIMIT ${n}"
        args.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [_row_to_dwell(r) for r in rows]


def _row_to_point(row: Any) -> PersonTrajectoryPoint:
    return PersonTrajectoryPoint(
        identity_id=row["identity_id"],
        ph_id=str(row["ph_id"]),
        observed_at=row["observed_at"],
        room_name=row["room_name"],
        ground_x=float(row["ground_x"]),
        ground_y=float(row["ground_y"]),
        posture=row["posture"],
        identity_confidence=float(row["identity_confidence"]),
        position_sigma_m=float(row["position_sigma_m"]),
        primary_camera_id=row["primary_camera_id"],
        contributing_camera_count=int(row["contributing_camera_count"]),
        footpoint_reliable=bool(row["footpoint_reliable"]),
        motion_energy=row.get("motion_energy"),
        floor_speed_m_s=row.get("floor_speed_m_s"),
    )


def _row_to_dwell(row: Any) -> RoomDwell:
    summary_raw = row["activity_summary"]
    summary: dict[str, Any] = (
        json.loads(summary_raw) if isinstance(summary_raw, str) else dict(summary_raw or {})
    )
    db_id = row["id"]
    return RoomDwell(
        dwell_id=str(db_id),
        identity_id=row["identity_id"],
        ph_id=str(row["ph_id"]),
        room_name=row["room_name"],
        entered_at=row["entered_at"],
        exited_at=row.get("exited_at"),
        duration_seconds=row.get("duration_seconds"),
        entry_confidence=float(row["entry_confidence"]),
        primary_posture=row["primary_posture"],
        min_motion_energy=row.get("min_motion_energy"),
        still_seconds=int(row.get("still_seconds", 0)),
        activity_summary=summary,
    )
