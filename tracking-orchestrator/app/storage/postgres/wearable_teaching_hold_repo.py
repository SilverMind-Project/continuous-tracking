"""Postgres implementation of :class:`WearableTeachingHoldRepository`."""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg

from app.domain import WearableTeachingHold

_COLUMNS = "hold_id, tag_id, held_at, reason, cleared_at, cleared_by"


def _row_to_hold(row: asyncpg.Record) -> WearableTeachingHold:
    return WearableTeachingHold(
        hold_id=str(row["hold_id"]),
        tag_id=str(row["tag_id"]),
        held_at=row["held_at"],
        reason=str(row["reason"]),
        cleared_at=row["cleared_at"],
        cleared_by=str(row["cleared_by"]) if row["cleared_by"] is not None else None,
    )


class PostgresWearableTeachingHoldRepository:
    """Asyncpg-backed repository for ``continuous_tracking.wearable_teaching_holds``."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_hold(
        self,
        *,
        tag_id: str,
        reason: str,
        held_at: datetime | None = None,
    ) -> WearableTeachingHold:
        at = held_at or datetime.now(UTC)
        async with self._pool.acquire() as conn, conn.transaction():
            # _COLUMNS is a module-level constant; values are bound via $1..$3.
            row = await conn.fetchrow(
                f"""
                INSERT INTO continuous_tracking.wearable_teaching_holds
                    (tag_id, held_at, reason)
                VALUES ($1, $2, $3)
                ON CONFLICT (tag_id) WHERE cleared_at IS NULL DO NOTHING
                RETURNING {_COLUMNS}
                """,  # noqa: S608  # module constant _COLUMNS
                tag_id,
                at,
                reason,
            )
            if row is None:
                row = await conn.fetchrow(
                    f"""
                    SELECT {_COLUMNS}
                    FROM continuous_tracking.wearable_teaching_holds
                    WHERE tag_id = $1 AND cleared_at IS NULL
                    """,  # noqa: S608  # module constant _COLUMNS
                    tag_id,
                )
        assert row is not None
        return _row_to_hold(row)

    async def get_open_hold(self, tag_id: str) -> WearableTeachingHold | None:
        async with self._pool.acquire() as conn:
            # _COLUMNS is a module-level constant; tag_id is bound as $1.
            row = await conn.fetchrow(
                f"""
                SELECT {_COLUMNS}
                FROM continuous_tracking.wearable_teaching_holds
                WHERE tag_id = $1 AND cleared_at IS NULL
                """,  # noqa: S608  # module constant _COLUMNS
                tag_id,
            )
        return _row_to_hold(row) if row is not None else None

    async def is_held(self, tag_id: str) -> bool:
        return (await self.get_open_hold(tag_id)) is not None

    async def list_open_holds(self) -> list[WearableTeachingHold]:
        async with self._pool.acquire() as conn:
            # _COLUMNS is a module-level constant.
            rows = await conn.fetch(
                f"""
                SELECT {_COLUMNS}
                FROM continuous_tracking.wearable_teaching_holds
                WHERE cleared_at IS NULL
                ORDER BY held_at DESC, hold_id DESC
                """  # noqa: S608  # module constant _COLUMNS
            )
        return [_row_to_hold(r) for r in rows]

    async def clear_hold(
        self,
        tag_id: str,
        *,
        cleared_by: str,
        cleared_at: datetime | None = None,
    ) -> WearableTeachingHold | None:
        at = cleared_at or datetime.now(UTC)
        async with self._pool.acquire() as conn:
            # _COLUMNS is a module-level constant; values are bound via $1..$3.
            row = await conn.fetchrow(
                f"""
                UPDATE continuous_tracking.wearable_teaching_holds
                SET cleared_at = $2, cleared_by = $3
                WHERE tag_id = $1 AND cleared_at IS NULL
                RETURNING {_COLUMNS}
                """,  # noqa: S608  # module constant _COLUMNS
                tag_id,
                at,
                cleared_by,
            )
        return _row_to_hold(row) if row is not None else None
