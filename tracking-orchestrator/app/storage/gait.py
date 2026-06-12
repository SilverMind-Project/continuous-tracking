"""Gait bout storage: Protocol + InMemory implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..trajectory.gait import WalkingBout


class GaitBoutRepository(ABC):
    """Persist walking bouts from WalkingBoutSegmenter."""

    @abstractmethod
    async def upsert_bout(self, bout: WalkingBout) -> None:
        """Store or update a walking bout (idempotent via stable bout_id)."""

    @abstractmethod
    async def list_bouts(
        self,
        identity_id: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 200,
    ) -> list[WalkingBout]:
        """List walking bouts with optional filters, newest first."""


class InMemoryGaitBoutRepository(GaitBoutRepository):
    """In-memory store for walking bouts."""

    def __init__(self) -> None:
        self._bouts: dict[str, WalkingBout] = {}

    async def upsert_bout(self, bout: WalkingBout) -> None:
        self._bouts[bout.bout_id] = bout

    async def list_bouts(
        self,
        identity_id: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 200,
    ) -> list[WalkingBout]:
        results = list(self._bouts.values())
        if identity_id is not None:
            results = [b for b in results if b.identity_id == identity_id]
        if after is not None:
            results = [b for b in results if b.started_at >= after]
        if before is not None:
            results = [b for b in results if b.started_at <= before]
        results.sort(key=lambda b: b.started_at, reverse=True)
        return results[:limit]
