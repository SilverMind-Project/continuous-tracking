"""Storage protocol and InMemory implementation for wearable association segments."""

from __future__ import annotations

import asyncio
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

    async def purge_identity(self, identity_id: str) -> int: ...

    async def purge_ph(self, ph_id: str) -> int: ...


class InMemoryWearableAssociationRepository:
    """In-memory peer of :class:`PostgresWearableAssociationRepository`."""

    def __init__(self) -> None:
        self._segments: dict[str, WearableAssociationSegment] = {}
        self._lock = asyncio.Lock()

    async def save_segment(self, segment: WearableAssociationSegment) -> None:
        async with self._lock:
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
