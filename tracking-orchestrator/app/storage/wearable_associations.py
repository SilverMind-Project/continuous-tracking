"""Storage protocol and InMemory implementation for wearable association segments."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from typing import Protocol

from app.domain import WearableAssociationSegment


class WearableAssociationRepository(Protocol):
    """Durable store for wearable-to-PH association segments."""

    async def save_segment(self, segment: WearableAssociationSegment) -> None: ...

    async def get_segment(self, segment_id: str) -> WearableAssociationSegment | None: ...

    async def list_by_ph(
        self, ph_id: str, *, limit: int = 50
    ) -> list[WearableAssociationSegment]: ...

    async def list_active(self, *, limit: int = 50) -> list[WearableAssociationSegment]: ...

    async def list_recent(
        self,
        *,
        tag_id: str | None = None,
        ph_id: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[WearableAssociationSegment]: ...

    async def mark_revoked(
        self,
        segment_id: str,
        *,
        revoked_at: datetime,
        revoked_by: str,
        revoke_reason: str,
    ) -> bool:
        """Stamp *segment_id* revoked when not already revoked; returns True if newly stamped."""
        ...

    async def list_unrevoked_overlapping(
        self,
        tag_id: str,
        *,
        since: datetime,
        until: datetime,
    ) -> list[WearableAssociationSegment]:
        """Return unrevoked segments for *tag_id* overlapping ``[since, until]``."""
        ...

    async def count_revoked_since(self, since: datetime) -> int:
        """Count segments stamped revoked at or after *since*."""
        ...

    async def purge_identity(self, identity_id: str) -> int: ...

    async def purge_ph(self, ph_id: str) -> int: ...


class InMemoryWearableAssociationRepository:
    """In-memory peer of :class:`PostgresWearableAssociationRepository`."""

    def __init__(self) -> None:
        self._segments: dict[str, WearableAssociationSegment] = {}
        self._lock = asyncio.Lock()

    async def save_segment(self, segment: WearableAssociationSegment) -> None:
        async with self._lock:
            existing = self._segments.get(segment.segment_id)
            if (
                existing is not None
                and existing.revoked_at is not None
                and segment.revoked_at is None
            ):
                segment = replace(
                    segment,
                    revoked_at=existing.revoked_at,
                    revoked_by=existing.revoked_by,
                    revoke_reason=existing.revoke_reason,
                )
            self._segments[segment.segment_id] = segment

    async def get_segment(self, segment_id: str) -> WearableAssociationSegment | None:
        return self._segments.get(segment_id)

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
        rows = [
            seg
            for seg in self._segments.values()
            if (tag_id is None or seg.tag_id == tag_id)
            and (ph_id is None or seg.ph_id == ph_id)
            and (not active_only or seg.ended_at is None)
        ]
        rows.sort(key=lambda s: (s.started_at, s.segment_id), reverse=True)
        return rows[:limit]

    async def mark_revoked(
        self,
        segment_id: str,
        *,
        revoked_at: datetime,
        revoked_by: str,
        revoke_reason: str,
    ) -> bool:
        async with self._lock:
            existing = self._segments.get(segment_id)
            if existing is None or existing.revoked_at is not None:
                return False
            self._segments[segment_id] = replace(
                existing,
                revoked_at=revoked_at,
                revoked_by=revoked_by,
                revoke_reason=revoke_reason,
            )
            return True

    async def list_unrevoked_overlapping(
        self,
        tag_id: str,
        *,
        since: datetime,
        until: datetime,
    ) -> list[WearableAssociationSegment]:
        rows = [
            seg
            for seg in self._segments.values()
            if seg.tag_id == tag_id
            and seg.revoked_at is None
            and seg.started_at <= until
            and (seg.ended_at is None or seg.ended_at >= since)
        ]
        rows.sort(key=lambda s: (s.started_at, s.segment_id))
        return rows

    async def count_revoked_since(self, since: datetime) -> int:
        return sum(
            1
            for seg in self._segments.values()
            if seg.revoked_at is not None and seg.revoked_at >= since
        )

    async def purge_identity(self, identity_id: str) -> int:
        async with self._lock:
            to_delete = [
                sid for sid, seg in self._segments.items() if seg.identity_id == identity_id
            ]
            for sid in to_delete:
                del self._segments[sid]
            return len(to_delete)

    async def purge_ph(self, ph_id: str) -> int:
        async with self._lock:
            to_delete = [sid for sid, seg in self._segments.items() if seg.ph_id == ph_id]
            for sid in to_delete:
                del self._segments[sid]
            return len(to_delete)
