"""Postgres-backed identity correction / revision-range / job / ack repository.

Implements :class:`IdentityCorrectionRepositoryProtocol` (M06) using asyncpg
against the ``continuous_tracking`` schema. Receives only an ``asyncpg.Pool``.

The effective-identity read (:meth:`effective_identity`) applies live revision
ranges on top of raw inference; operator ranges win inside their bounds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import asyncpg
from structlog import get_logger

from ...domain import (
    IdentityRevisionJob,
    IdentityRevisionRange,
    IdentitySegmentCorrection,
    ProjectionAck,
    RevisionAuthority,
    RevisionJobStatus,
)

logger = get_logger(__name__)


def _loads_obj(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(value, dict):
        return value
    return None


class PostgresIdentityCorrectionRepository:
    """asyncpg implementation of the M06 correction repository."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # -- corrections ---------------------------------------------------------

    async def save_correction(self, correction: IdentitySegmentCorrection) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.identity_corrections (
                    correction_id, ph_id, actor, reason_code, note, source_view,
                    target_identity_id, set_unknown, correction_kind, frame_only,
                    reviewed_frame_id, reviewed_bbox, observation_start,
                    observation_end, base_ph_version, base_revision_id,
                    revision_id, compensates_correction_id, created_at
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$14,$15,$16,
                    $17,$18,$19
                )
                ON CONFLICT (correction_id) DO NOTHING
                """,
                correction.correction_id,
                correction.ph_id,
                correction.actor,
                correction.reason_code,
                correction.note,
                correction.source_view,
                correction.target_identity_id,
                correction.set_unknown,
                correction.kind,
                correction.frame_only,
                correction.reviewed_frame_id,
                json.dumps(correction.reviewed_bbox)
                if correction.reviewed_bbox is not None
                else None,
                correction.observation_start,
                correction.observation_end,
                correction.base_ph_version,
                correction.base_revision_id,
                correction.revision_id,
                correction.compensates_correction_id,
                correction.created_at,
            )

    async def get_correction(self, correction_id: str) -> IdentitySegmentCorrection | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.identity_corrections WHERE correction_id = $1",
                correction_id,
            )
        return _correction_from_row(row) if row else None

    async def list_corrections(self, ph_id: str) -> list[IdentitySegmentCorrection]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM continuous_tracking.identity_corrections "
                "WHERE ph_id = $1 ORDER BY observation_start ASC",
                ph_id,
            )
        return [_correction_from_row(r) for r in rows]

    # -- ranges --------------------------------------------------------------

    async def save_range(self, revision_range: IdentityRevisionRange) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.identity_revision_ranges (
                    range_id, revision_id, correction_id, ph_id,
                    effective_identity_id, authority, range_start, range_end,
                    supersedes_range_id, superseded_by_range_id,
                    compensated_by_revision_id, created_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (range_id) DO NOTHING
                """,
                revision_range.range_id,
                revision_range.revision_id,
                revision_range.correction_id,
                revision_range.ph_id,
                revision_range.effective_identity_id,
                revision_range.authority,
                revision_range.range_start,
                revision_range.range_end,
                revision_range.supersedes_range_id,
                revision_range.superseded_by_range_id,
                revision_range.compensated_by_revision_id,
                revision_range.created_at,
            )

    async def supersede_range(self, range_id: str, *, by_range_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE continuous_tracking.identity_revision_ranges "
                "SET superseded_by_range_id = $2 WHERE range_id = $1",
                range_id,
                by_range_id,
            )

    async def list_ranges(
        self, ph_id: str, *, live_only: bool = True
    ) -> list[IdentityRevisionRange]:
        sql = "SELECT * FROM continuous_tracking.identity_revision_ranges WHERE ph_id = $1"
        if live_only:
            sql += " AND superseded_by_range_id IS NULL"
        sql += " ORDER BY created_at ASC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, ph_id)
        return [_range_from_row(r) for r in rows]

    async def live_ranges_for_phs(
        self, ph_ids: list[str]
    ) -> dict[str, list[IdentityRevisionRange]]:
        if not ph_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM continuous_tracking.identity_revision_ranges
                WHERE ph_id = ANY($1::uuid[])
                  AND superseded_by_range_id IS NULL
                ORDER BY ph_id, created_at ASC
                """,
                ph_ids,
            )
        result: dict[str, list[IdentityRevisionRange]] = {}
        for row in rows:
            rng = _range_from_row(row)
            result.setdefault(rng.ph_id, []).append(rng)
        return result

    async def effective_identity(
        self, ph_id: str, at: datetime
    ) -> tuple[str | None, RevisionAuthority | None]:
        # Operator authority wins; among same authority the newest range wins.
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT effective_identity_id, authority
                FROM continuous_tracking.identity_revision_ranges
                WHERE ph_id = $1
                  AND superseded_by_range_id IS NULL
                  AND range_start <= $2
                  AND range_end >= $2
                ORDER BY (authority = 'operator') DESC, created_at DESC
                LIMIT 1
                """,
                ph_id,
                at,
            )
        if row is None:
            return (None, None)
        return (row["effective_identity_id"], row["authority"])

    async def operator_ranges_overlapping(
        self, ph_id: str, start: datetime, end: datetime
    ) -> list[IdentityRevisionRange]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM continuous_tracking.identity_revision_ranges
                WHERE ph_id = $1
                  AND superseded_by_range_id IS NULL
                  AND authority = 'operator'
                  AND range_start <= $3
                  AND range_end >= $2
                ORDER BY created_at ASC
                """,
                ph_id,
                start,
                end,
            )
        return [_range_from_row(r) for r in rows]

    # -- jobs ----------------------------------------------------------------

    async def save_job(self, job: IdentityRevisionJob) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.identity_revision_jobs (
                    job_id, revision_id, correction_id, status,
                    required_projections, attempts, last_error, row_counts,
                    created_at, updated_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10)
                ON CONFLICT (revision_id) DO NOTHING
                """,
                job.job_id,
                job.revision_id,
                job.correction_id,
                job.status,
                list(job.required_projections),
                job.attempts,
                job.last_error,
                json.dumps(job.row_counts),
                job.created_at,
                job.updated_at,
            )

    async def get_job(self, revision_id: str) -> IdentityRevisionJob | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.identity_revision_jobs WHERE revision_id = $1",
                revision_id,
            )
        return _job_from_row(row) if row else None

    async def update_job(
        self,
        revision_id: str,
        *,
        status: RevisionJobStatus | None = None,
        last_error: str | None = None,
        increment_attempts: bool = False,
        row_counts: dict[str, int] | None = None,
    ) -> IdentityRevisionJob | None:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM continuous_tracking.identity_revision_jobs "
                "WHERE revision_id = $1 FOR UPDATE",
                revision_id,
            )
            if row is None:
                return None
            job = _job_from_row(row)
            merged = dict(job.row_counts)
            if row_counts:
                merged.update(row_counts)
            new_status = status if status is not None else job.status
            new_error = last_error if last_error is not None else job.last_error
            new_attempts = job.attempts + 1 if increment_attempts else job.attempts
            now = datetime.now(UTC)
            await conn.execute(
                """
                UPDATE continuous_tracking.identity_revision_jobs
                SET status = $2, last_error = $3, attempts = $4,
                    row_counts = $5::jsonb, updated_at = $6
                WHERE revision_id = $1
                """,
                revision_id,
                new_status,
                new_error,
                new_attempts,
                json.dumps(merged),
                now,
            )
            return _job_from_row(
                {
                    **dict(row),
                    "status": new_status,
                    "last_error": new_error,
                    "attempts": new_attempts,
                    "row_counts": json.dumps(merged),
                    "updated_at": now,
                }
            )

    # -- acks ----------------------------------------------------------------

    async def record_ack(self, ack: ProjectionAck) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO continuous_tracking.identity_projection_acks (
                    revision_id, consumer, schema_version, status, counts,
                    applied_at
                ) VALUES ($1,$2,$3,$4,$5::jsonb,$6)
                ON CONFLICT (revision_id, consumer) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    status = EXCLUDED.status,
                    counts = EXCLUDED.counts,
                    applied_at = EXCLUDED.applied_at
                """,
                ack.revision_id,
                ack.consumer,
                ack.schema_version,
                ack.status,
                json.dumps(ack.counts),
                ack.applied_at,
            )

    async def list_acks(self, revision_id: str) -> list[ProjectionAck]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM continuous_tracking.identity_projection_acks WHERE revision_id = $1",
                revision_id,
            )
        return [_ack_from_row(r) for r in rows]

    async def complete_job_if_ready(self, revision_id: str) -> bool:
        job = await self.get_job(revision_id)
        if job is None or job.status == "completed":
            return False
        acks = await self.list_acks(revision_id)
        ack_by_consumer = {a.consumer: a for a in acks}
        if any(a.status == "failed" for a in acks):
            await self.update_job(revision_id, status="failed")
            return False
        ready = all(
            ack_by_consumer.get(p) is not None and ack_by_consumer[p].status == "acked"
            for p in job.required_projections
        )
        if ready:
            await self.update_job(revision_id, status="completed")
            return True
        return False


# ---------------------------------------------------------------------------
# row -> domain helpers
# ---------------------------------------------------------------------------


def _correction_from_row(row: Any) -> IdentitySegmentCorrection:
    return IdentitySegmentCorrection(
        correction_id=str(row["correction_id"]),
        ph_id=str(row["ph_id"]),
        actor=row["actor"],
        reason_code=row["reason_code"],
        observation_start=row["observation_start"],
        observation_end=row["observation_end"],
        base_ph_version=row["base_ph_version"],
        revision_id=str(row["revision_id"]),
        target_identity_id=row["target_identity_id"],
        set_unknown=row["set_unknown"],
        kind=row["correction_kind"],
        frame_only=row["frame_only"],
        note=row["note"],
        source_view=row["source_view"],
        reviewed_frame_id=row["reviewed_frame_id"],
        reviewed_bbox=_loads_obj(row["reviewed_bbox"]),
        base_revision_id=str(row["base_revision_id"]) if row["base_revision_id"] else None,
        compensates_correction_id=str(row["compensates_correction_id"])
        if row["compensates_correction_id"]
        else None,
        created_at=row["created_at"],
    )


def _range_from_row(row: Any) -> IdentityRevisionRange:
    return IdentityRevisionRange(
        range_id=str(row["range_id"]),
        revision_id=str(row["revision_id"]),
        ph_id=str(row["ph_id"]),
        authority=row["authority"],
        range_start=row["range_start"],
        range_end=row["range_end"],
        effective_identity_id=row["effective_identity_id"],
        correction_id=str(row["correction_id"]) if row["correction_id"] else None,
        supersedes_range_id=str(row["supersedes_range_id"]) if row["supersedes_range_id"] else None,
        superseded_by_range_id=str(row["superseded_by_range_id"])
        if row["superseded_by_range_id"]
        else None,
        compensated_by_revision_id=str(row["compensated_by_revision_id"])
        if row["compensated_by_revision_id"]
        else None,
        created_at=row["created_at"],
    )


def _job_from_row(row: Any) -> IdentityRevisionJob:
    counts = _loads_obj(row["row_counts"]) or {}
    return IdentityRevisionJob(
        job_id=str(row["job_id"]),
        revision_id=str(row["revision_id"]),
        status=row["status"],
        required_projections=tuple(row["required_projections"] or ()),
        correction_id=str(row["correction_id"]) if row["correction_id"] else None,
        attempts=row["attempts"],
        last_error=row["last_error"],
        row_counts={k: int(v) for k, v in counts.items()},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _ack_from_row(row: Any) -> ProjectionAck:
    counts = _loads_obj(row["counts"]) or {}
    return ProjectionAck(
        revision_id=str(row["revision_id"]),
        consumer=row["consumer"],
        schema_version=row["schema_version"],
        status=row["status"],
        counts={k: int(v) for k, v in counts.items()},
        applied_at=row["applied_at"],
    )
