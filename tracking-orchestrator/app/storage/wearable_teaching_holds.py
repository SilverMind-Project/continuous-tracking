"""Storage protocol and InMemory implementation for wearable teaching holds."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol

from app.domain import WearableTeachingHold


class WearableTeachingHoldRepository(Protocol):
    """Durable store for per-tag wearable teaching holds."""

    async def create_hold(
        self,
        *,
        tag_id: str,
        reason: str,
        held_at: datetime | None = None,
    ) -> WearableTeachingHold:
        """Return the active open hold for *tag_id*, creating one if none is open (idempotent)."""
        ...

    async def get_open_hold(self, tag_id: str) -> WearableTeachingHold | None:
        """Return the currently open hold for *tag_id*, or None."""
        ...

    async def is_held(self, tag_id: str) -> bool:
        """Return True when *tag_id* currently has an open teaching hold."""
        ...

    async def list_open_holds(self) -> list[WearableTeachingHold]:
        """Return all currently open teaching holds, newest first."""
        ...

    async def clear_hold(
        self,
        tag_id: str,
        *,
        cleared_by: str,
        cleared_at: datetime | None = None,
    ) -> WearableTeachingHold | None:
        """Clear the open hold for *tag_id* and return it, or None when not held."""
        ...


class InMemoryWearableTeachingHoldRepository:
    """In-memory peer of :class:`PostgresWearableTeachingHoldRepository`."""

    def __init__(self) -> None:
        self._holds: dict[str, WearableTeachingHold] = {}
        self._lock = asyncio.Lock()

    async def create_hold(
        self,
        *,
        tag_id: str,
        reason: str,
        held_at: datetime | None = None,
    ) -> WearableTeachingHold:
        async with self._lock:
            for existing in self._holds.values():
                if existing.tag_id == tag_id and existing.cleared_at is None:
                    return existing
            hold = WearableTeachingHold(
                hold_id=str(uuid.uuid4()),
                tag_id=tag_id,
                held_at=held_at or datetime.now(UTC),
                reason=reason,
            )
            self._holds[hold.hold_id] = hold
            return hold

    async def get_open_hold(self, tag_id: str) -> WearableTeachingHold | None:
        for hold in self._holds.values():
            if hold.tag_id == tag_id and hold.cleared_at is None:
                return hold
        return None

    async def is_held(self, tag_id: str) -> bool:
        return (await self.get_open_hold(tag_id)) is not None

    async def list_open_holds(self) -> list[WearableTeachingHold]:
        rows = [h for h in self._holds.values() if h.cleared_at is None]
        rows.sort(key=lambda h: (h.held_at, h.hold_id), reverse=True)
        return rows

    async def clear_hold(
        self,
        tag_id: str,
        *,
        cleared_by: str,
        cleared_at: datetime | None = None,
    ) -> WearableTeachingHold | None:
        async with self._lock:
            for hold_id, hold in self._holds.items():
                if hold.tag_id == tag_id and hold.cleared_at is None:
                    updated = replace(
                        hold,
                        cleared_at=cleared_at or datetime.now(UTC),
                        cleared_by=cleared_by,
                    )
                    self._holds[hold_id] = updated
                    return updated
            return None
