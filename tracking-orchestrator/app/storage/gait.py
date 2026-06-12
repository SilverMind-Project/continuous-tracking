"""Gait bout and daily aggregate storage: Protocol + InMemory implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime

from ..trajectory.gait import GaitDailyRecord, WalkingBout


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


class GaitDailyRepository(ABC):
    """Persist per-resident per-day gait aggregates from GaitAggregator."""

    @abstractmethod
    async def upsert_day(self, record: GaitDailyRecord) -> None:
        """Store or overwrite a daily aggregate row (keyed by identity_id + local_date)."""

    @abstractmethod
    async def list_days(
        self,
        identity_id: str,
        since: date | None = None,
        until: date | None = None,
    ) -> list[GaitDailyRecord]:
        """List daily records for one resident, oldest first."""


class InMemoryGaitDailyRepository(GaitDailyRepository):
    """In-memory store for daily gait aggregates."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, date], GaitDailyRecord] = {}

    async def upsert_day(self, record: GaitDailyRecord) -> None:
        self._rows[(record.identity_id, record.local_date)] = record

    async def list_days(
        self,
        identity_id: str,
        since: date | None = None,
        until: date | None = None,
    ) -> list[GaitDailyRecord]:
        results = [r for r in self._rows.values() if r.identity_id == identity_id]
        if since is not None:
            results = [r for r in results if r.local_date >= since]
        if until is not None:
            results = [r for r in results if r.local_date <= until]
        results.sort(key=lambda r: r.local_date)
        return results
