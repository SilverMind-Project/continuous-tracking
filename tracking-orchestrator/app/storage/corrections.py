"""Repository for M06 segment corrections, revision ranges, jobs, and acks.

This module owns the persistence contract for operator identity corrections and
the effective-identity projection that layers on top of raw inference. It is the
storage half of :class:`app.services.identity_correction_service`.

Three artifacts per the project storage convention:

* ``IdentityCorrectionRepositoryProtocol`` -- the structural contract.
* ``InMemoryIdentityCorrectionRepository`` -- zero-I/O implementation for tests.
* ``PostgresIdentityCorrectionRepository`` -- asyncpg implementation
  (``storage/postgres/correction_repo.py``).

The effective-identity read (``effective_identity``) is the core invariant:
``identity_decisions.inferred_identity_id`` never changes; the effective label is
inference with live revision ranges applied, operator authority winning inside
its bounds.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime
from typing import Protocol

from ..domain import (
    IdentityRevisionJob,
    IdentityRevisionRange,
    IdentitySegmentCorrection,
    ProjectionAck,
    RevisionAuthority,
    RevisionJobStatus,
)


class IdentityCorrectionRepositoryProtocol(Protocol):
    """Persist operator corrections and the effective-identity projection."""

    # -- corrections --
    async def save_correction(self, correction: IdentitySegmentCorrection) -> None: ...
    async def get_correction(self, correction_id: str) -> IdentitySegmentCorrection | None: ...
    async def list_corrections(self, ph_id: str) -> list[IdentitySegmentCorrection]: ...

    # -- revision ranges (effective projection) --
    async def save_range(self, revision_range: IdentityRevisionRange) -> None: ...
    async def supersede_range(self, range_id: str, *, by_range_id: str) -> None: ...
    async def list_ranges(
        self, ph_id: str, *, live_only: bool = True
    ) -> list[IdentityRevisionRange]: ...
    async def live_ranges_for_phs(
        self, ph_ids: list[str]
    ) -> dict[str, list[IdentityRevisionRange]]:
        """Batch live revision ranges for many PHs (M07 read model).

        Returns a mapping ``ph_id -> live ranges``; PHs with no live range are
        omitted. Lets the keyframe read model resolve effective identity per
        bbox in memory without an effective-identity query per box.
        """
        ...

    async def effective_identity(
        self, ph_id: str, at: datetime
    ) -> tuple[str | None, RevisionAuthority | None]:
        """Return ``(effective_identity_id, authority)`` for ``ph_id`` at ``at``.

        Returns ``(None, None)`` when no range covers ``at`` (caller falls back
        to the raw inferred decision). An operator range covering ``at`` always
        wins over an inferred range.
        """
        ...

    async def operator_ranges_overlapping(
        self, ph_id: str, start: datetime, end: datetime
    ) -> list[IdentityRevisionRange]:
        """Live operator ranges intersecting ``[start, end]`` (conflict check)."""
        ...

    # -- jobs --
    async def save_job(self, job: IdentityRevisionJob) -> None: ...
    async def get_job(self, revision_id: str) -> IdentityRevisionJob | None: ...
    async def update_job(
        self,
        revision_id: str,
        *,
        status: RevisionJobStatus | None = None,
        last_error: str | None = None,
        increment_attempts: bool = False,
        row_counts: dict[str, int] | None = None,
    ) -> IdentityRevisionJob | None: ...

    # -- projection acks --
    async def record_ack(self, ack: ProjectionAck) -> None: ...
    async def list_acks(self, revision_id: str) -> list[ProjectionAck]: ...
    async def complete_job_if_ready(self, revision_id: str) -> bool:
        """Flip job to ``completed`` once every required projection acked.

        Returns ``True`` if the job transitioned to ``completed`` on this call.
        A failed ack flips the job to ``failed`` instead and returns ``False``.
        """
        ...


class InMemoryIdentityCorrectionRepository:
    """In-memory store -- dict/list only, zero I/O."""

    def __init__(self) -> None:
        self._corrections: dict[str, IdentitySegmentCorrection] = {}
        self._ranges: dict[str, IdentityRevisionRange] = {}
        self._jobs: dict[str, IdentityRevisionJob] = {}  # keyed by revision_id
        self._acks: dict[tuple[str, str], ProjectionAck] = {}
        self._lock = asyncio.Lock()

    # -- corrections --

    async def save_correction(self, correction: IdentitySegmentCorrection) -> None:
        async with self._lock:
            self._corrections[correction.correction_id] = correction

    async def get_correction(self, correction_id: str) -> IdentitySegmentCorrection | None:
        return self._corrections.get(correction_id)

    async def list_corrections(self, ph_id: str) -> list[IdentitySegmentCorrection]:
        rows = [c for c in self._corrections.values() if c.ph_id == ph_id]
        rows.sort(key=lambda c: c.observation_start)
        return rows

    # -- ranges --

    async def save_range(self, revision_range: IdentityRevisionRange) -> None:
        async with self._lock:
            self._ranges[revision_range.range_id] = revision_range

    async def supersede_range(self, range_id: str, *, by_range_id: str) -> None:
        async with self._lock:
            existing = self._ranges.get(range_id)
            if existing is not None:
                self._ranges[range_id] = dataclasses.replace(
                    existing, superseded_by_range_id=by_range_id
                )

    async def list_ranges(
        self, ph_id: str, *, live_only: bool = True
    ) -> list[IdentityRevisionRange]:
        rows = [
            r
            for r in self._ranges.values()
            if r.ph_id == ph_id and (not live_only or r.superseded_by_range_id is None)
        ]
        rows.sort(key=lambda r: r.created_at)
        return rows

    async def live_ranges_for_phs(
        self, ph_ids: list[str]
    ) -> dict[str, list[IdentityRevisionRange]]:
        result: dict[str, list[IdentityRevisionRange]] = {}
        for ph_id in set(ph_ids):
            ranges = await self.list_ranges(ph_id, live_only=True)
            if ranges:
                result[ph_id] = ranges
        return result

    async def effective_identity(
        self, ph_id: str, at: datetime
    ) -> tuple[str | None, RevisionAuthority | None]:
        live = await self.list_ranges(ph_id, live_only=True)
        covering = [r for r in live if r.range_start <= at <= r.range_end]
        if not covering:
            return (None, None)
        operator = [r for r in covering if r.authority == "operator"]
        pool = operator if operator else covering
        winner = max(pool, key=lambda r: r.created_at)
        return (winner.effective_identity_id, winner.authority)

    async def operator_ranges_overlapping(
        self, ph_id: str, start: datetime, end: datetime
    ) -> list[IdentityRevisionRange]:
        live = await self.list_ranges(ph_id, live_only=True)
        return [
            r
            for r in live
            if r.authority == "operator" and r.range_start <= end and r.range_end >= start
        ]

    # -- jobs --

    async def save_job(self, job: IdentityRevisionJob) -> None:
        async with self._lock:
            self._jobs[job.revision_id] = job

    async def get_job(self, revision_id: str) -> IdentityRevisionJob | None:
        return self._jobs.get(revision_id)

    async def update_job(
        self,
        revision_id: str,
        *,
        status: RevisionJobStatus | None = None,
        last_error: str | None = None,
        increment_attempts: bool = False,
        row_counts: dict[str, int] | None = None,
    ) -> IdentityRevisionJob | None:
        async with self._lock:
            job = self._jobs.get(revision_id)
            if job is None:
                return None
            merged_counts = dict(job.row_counts)
            if row_counts:
                merged_counts.update(row_counts)
            updated = dataclasses.replace(
                job,
                status=status if status is not None else job.status,
                last_error=last_error if last_error is not None else job.last_error,
                attempts=job.attempts + 1 if increment_attempts else job.attempts,
                row_counts=merged_counts,
                updated_at=datetime.now(UTC),
            )
            self._jobs[revision_id] = updated
            return updated

    # -- acks --

    async def record_ack(self, ack: ProjectionAck) -> None:
        async with self._lock:
            self._acks[(ack.revision_id, ack.consumer)] = ack

    async def list_acks(self, revision_id: str) -> list[ProjectionAck]:
        return [a for k, a in self._acks.items() if k[0] == revision_id]

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
