"""Daily appearance profile storage: Protocol + InMemory implementation (DL-M07)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from ..trajectory.appearance_profile import DailyAppearanceProfile


class DailyAppearanceRepo(ABC):
    """Persist per-identity per-day appearance centroids from AppearanceEvaluator."""

    @abstractmethod
    async def upsert_profile(self, profile: DailyAppearanceProfile) -> None:
        """Store or overwrite a daily profile (keyed by identity_id + day)."""

    @abstractmethod
    async def get_profile(self, identity_id: str, day: date) -> DailyAppearanceProfile | None:
        """Fetch one identity's profile for one local day, or None if absent."""

    @abstractmethod
    async def list_days(
        self,
        identity_id: str,
        since_day: date | None = None,
    ) -> list[DailyAppearanceProfile]:
        """List an identity's profiles oldest first, optionally from ``since_day``."""


class InMemoryDailyAppearanceRepo(DailyAppearanceRepo):
    """In-memory store for daily appearance profiles."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, date], DailyAppearanceProfile] = {}

    async def upsert_profile(self, profile: DailyAppearanceProfile) -> None:
        self._rows[(profile.identity_id, profile.day)] = profile

    async def get_profile(self, identity_id: str, day: date) -> DailyAppearanceProfile | None:
        return self._rows.get((identity_id, day))

    async def list_days(
        self,
        identity_id: str,
        since_day: date | None = None,
    ) -> list[DailyAppearanceProfile]:
        results = [p for p in self._rows.values() if p.identity_id == identity_id]
        if since_day is not None:
            results = [p for p in results if p.day >= since_day]
        results.sort(key=lambda p: p.day)
        return results
