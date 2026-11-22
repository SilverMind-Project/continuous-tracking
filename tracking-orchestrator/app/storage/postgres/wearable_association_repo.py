"""Postgres implementation of :class:`WearableAssociationRepository`."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import asyncpg

from app.domain import WearableAssociationSegment, WearableAttestationGrade

_SQL_UPSERT_SEGMENT = """
    INSERT INTO continuous_tracking.wearable_associations (
        segment_id, ph_id, tag_id, identity_id, started_at, ended_at,
        attestation_grade, window_llr, margin_llr, matched_fixes,
        mean_residual_m, swap_suspected, close_reason
    ) VALUES (
        $1::uuid, $2::uuid, $3, $4, $5, $6,
        $7, $8, $9, $10,
        $11, $12, $13
    )
    ON CONFLICT (segment_id) DO UPDATE SET
        ended_at = EXCLUDED.ended_at,
        attestation_grade = EXCLUDED.attestation_grade,
        window_llr = EXCLUDED.window_llr,
        margin_llr = EXCLUDED.margin_llr,
        matched_fixes = EXCLUDED.matched_fixes,
        mean_residual_m = EXCLUDED.mean_residual_m,
        swap_suspected = EXCLUDED.swap_suspected,
        close_reason = EXCLUDED.close_reason,
        updated_at = now()
"""

_SQL_SELECT_COLUMNS = """
    segment_id, ph_id, tag_id, identity_id, started_at, ended_at,
    attestation_grade, window_llr, margin_llr, matched_fixes,
    mean_residual_m, swap_suspected, close_reason,
    revoked_at, revoked_by, revoke_reason
"""


def _row_to_segment(row: asyncpg.Record) -> WearableAssociationSegment:
    return WearableAssociationSegment(
        segment_id=str(row["segment_id"]),
        ph_id=str(row["ph_id"]),
        tag_id=str(row["tag_id"]),
        identity_id=str(row["identity_id"]),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        attestation_grade=WearableAttestationGrade(str(row["attestation_grade"])),
        window_llr=float(row["window_llr"]),
        margin_llr=float(row["margin_llr"]),
        matched_fixes=int(row["matched_fixes"]),
        mean_residual_m=(
            float(row["mean_residual_m"]) if row["mean_residual_m"] is not None else None
        ),
        swap_suspected=bool(row["swap_suspected"]),
        close_reason=str(row["close_reason"]) if row["close_reason"] is not None else None,
        revoked_at=row["revoked_at"],
        revoked_by=str(row["revoked_by"]) if row["revoked_by"] is not None else None,
        revoke_reason=str(row["revoke_reason"]) if row["revoke_reason"] is not None else None,
    )


class PostgresWearableAssociationRepository:
    """Asyncpg-backed repository for ``continuous_tracking.wearable_associations``."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_segment(self, segment: WearableAssociationSegment) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_UPSERT_SEGMENT,
                segment.segment_id,
                segment.ph_id,
                segment.tag_id,
                segment.identity_id,
                segment.started_at,
                segment.ended_at,
                str(segment.attestation_grade),
                segment.window_llr,
                segment.margin_llr,
                segment.matched_fixes,
                segment.mean_residual_m,
                segment.swap_suspected,
                segment.close_reason,
            )

    async def get_segment(self, segment_id: str) -> WearableAssociationSegment | None:
        async with self._pool.acquire() as conn:
            # _SQL_SELECT_COLUMNS is a module-level constant; segment_id is bound as $1.
            row = await conn.fetchrow(
                f"SELECT {_SQL_SELECT_COLUMNS} "  # noqa: S608
                "FROM continuous_tracking.wearable_associations "
                "WHERE segment_id = $1::uuid",
                segment_id,
            )
        return _row_to_segment(row) if row is not None else None

    async def list_by_ph(self, ph_id: str, *, limit: int = 50) -> list[WearableAssociationSegment]:
        return await self.list_recent(ph_id=ph_id, limit=limit)

    async def list_active(self, *, limit: int = 50) -> list[WearableAssociationSegment]:
        return await self.list_recent(active_only=True, limit=limit)

    async def list_recent(
        self,
        *,
        tag_id: str | None = None,
        ph_id: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[WearableAssociationSegment]:
        predicates: list[str] = []
        params: list[Any] = []
        idx = 1

        if tag_id is not None:
            predicates.append(f"tag_id = ${idx}")
            params.append(tag_id)
            idx += 1
        if ph_id is not None:
            predicates.append(f"ph_id = ${idx}::uuid")
            params.append(ph_id)
            idx += 1
        if active_only:
            predicates.append("ended_at IS NULL")

        where = ("WHERE " + " AND ".join(predicates)) if predicates else ""
        params.append(limit)
        async with self._pool.acquire() as conn:
            # _SQL_SELECT_COLUMNS is a module constant; where holds hardcoded $N predicates.
            rows = await conn.fetch(
                f"SELECT {_SQL_SELECT_COLUMNS} "  # noqa: S608
                f"FROM continuous_tracking.wearable_associations {where} "
                f"ORDER BY started_at DESC, segment_id DESC LIMIT ${idx}",
                *params,
            )
        return [_row_to_segment(r) for r in rows]

    async def mark_revoked(
        self,
        segment_id: str,
        *,
        revoked_at: datetime,
        revoked_by: str,
        revoke_reason: str,
    ) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE continuous_tracking.wearable_associations
                SET revoked_at = $2,
                    revoked_by = $3,
                    revoke_reason = $4,
                    updated_at = now()
                WHERE segment_id = $1::uuid
                  AND revoked_at IS NULL
                RETURNING segment_id
                """,
                segment_id,
                revoked_at,
                revoked_by,
                revoke_reason,
            )
        return row is not None

    async def list_unrevoked_overlapping(
        self,
        tag_id: str,
        *,
        since: datetime,
        until: datetime,
    ) -> list[WearableAssociationSegment]:
        async with self._pool.acquire() as conn:
            # _SQL_SELECT_COLUMNS is a module-level constant; filter values are $1..$3 params.
            rows = await conn.fetch(
                f"""
                SELECT {_SQL_SELECT_COLUMNS}
                FROM continuous_tracking.wearable_associations
                WHERE tag_id = $1
                  AND revoked_at IS NULL
                  AND started_at <= $3
                  AND (ended_at IS NULL OR ended_at >= $2)
                ORDER BY started_at ASC, segment_id ASC
                """,  # noqa: S608  # module constant _SQL_SELECT_COLUMNS
                tag_id,
                since,
                until,
            )
        return [_row_to_segment(r) for r in rows]

    async def count_revoked_since(self, since: datetime) -> int:
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT count(*)
                FROM continuous_tracking.wearable_associations
                WHERE revoked_at IS NOT NULL
                  AND revoked_at >= $1
                """,
                since,
            )
        return int(count or 0)

    async def purge_identity(self, identity_id: str) -> int:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "DELETE FROM continuous_tracking.wearable_associations "
                "WHERE identity_id = $1 RETURNING segment_id",
                identity_id,
            )
        return len(rows)

    async def purge_ph(self, ph_id: str) -> int:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "DELETE FROM continuous_tracking.wearable_associations "
                "WHERE ph_id = $1::uuid RETURNING segment_id",
                ph_id,
            )
        return len(rows)
