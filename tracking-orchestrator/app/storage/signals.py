"""Dementia signal and behavior baseline storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from ..domain import DementiaSignal, PersonTrajectoryPoint, RoomDwell


class DementiaSignalRepository(ABC):
    """Persist dementia signals."""

    @abstractmethod
    async def upsert_signal(self, signal: DementiaSignal) -> None:
        """Store or update a dementia signal."""

    @abstractmethod
    async def list_signals(
        self,
        identity_id: str | None = None,
        signal_kind: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 200,
    ) -> list[DementiaSignal]:
        """List dementia signals with optional filters."""


@dataclass(frozen=True)
class HourlyActivitySummary:
    """Per-hour activity summary for a resident."""

    transition_count: int
    observed_minutes: int


@dataclass(frozen=True)
class StillnessEpisode:
    """A historical contiguous low-motion episode."""

    room_name: str
    posture: str
    duration_seconds: int
    min_motion_energy: float
    occurred_at: datetime


class BehaviorBaselineRepository(ABC):
    """Summarise raw trajectory/dwell history for robust signal baselines.

    All methods are independent of the signal repository — baselines are
    derived from raw behaviour data, never from previously emitted signals.
    """

    @abstractmethod
    async def dwell_durations(
        self,
        identity_id: str,
        room_predicate: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[float]:
        """Return closed-dwell durations (seconds) for rooms matching *room_predicate*."""

    @abstractmethod
    async def hourly_activity(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[int, HourlyActivitySummary]:
        """Return per-hour-of-day room-transition counts and observed-minutes."""

    @abstractmethod
    async def stillness_episodes(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[StillnessEpisode]:
        """Return historical low-motion episodes for baseline comparison."""


class InMemoryBehaviorBaselineRepository(BehaviorBaselineRepository):
    """In-memory baseline repository backed by trajectory/dwell lists."""

    def __init__(
        self,
        points: list[PersonTrajectoryPoint] | None = None,
        dwells: list[RoomDwell] | None = None,
    ) -> None:
        self.points: list[PersonTrajectoryPoint] = points or []
        self.dwells: list[RoomDwell] = dwells or []

    async def dwell_durations(
        self,
        identity_id: str,
        room_predicate: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[float]:
        results: list[float] = []
        for d in self.dwells:
            if d.identity_id != identity_id:
                continue
            if d.exited_at is None or d.duration_seconds is None:
                continue
            if room_predicate and room_predicate not in d.room_name.lower():
                continue
            if since is not None and d.entered_at < since:
                continue
            if until is not None and d.entered_at > until:
                continue
            results.append(float(d.duration_seconds))
        return results

    async def hourly_activity(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> dict[int, HourlyActivitySummary]:
        buckets: dict[int, dict[str, int]] = {}
        sorted_pts = sorted(
            [p for p in self.points if p.identity_id == identity_id],
            key=lambda p: p.observed_at,
        )
        prev_room: str | None = None
        for p in sorted_pts:
            if since and p.observed_at < since:
                continue
            if until and p.observed_at > until:
                continue
            hour = p.observed_at.hour
            b = buckets.setdefault(hour, {"transitions": 0, "minutes": 0})
            b["minutes"] += 1
            if prev_room is not None and p.room_name != prev_room:
                b["transitions"] += 1
            prev_room = p.room_name
        return {
            h: HourlyActivitySummary(
                transition_count=v["transitions"],
                observed_minutes=v["minutes"],
            )
            for h, v in buckets.items()
        }

    async def stillness_episodes(
        self,
        identity_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[StillnessEpisode]:
        episodes: list[StillnessEpisode] = []
        for d in self.dwells:
            if d.identity_id != identity_id:
                continue
            if d.exited_at is None:
                continue
            if since and d.entered_at < since:
                continue
            if until and d.entered_at > until:
                continue
            if d.min_motion_energy is not None or d.still_seconds > 0:
                episodes.append(
                    StillnessEpisode(
                        room_name=d.room_name,
                        posture=d.primary_posture,
                        duration_seconds=d.duration_seconds or 0,
                        min_motion_energy=d.min_motion_energy or 0.0,
                        occurred_at=d.entered_at,
                    )
                )
        return episodes


class InMemoryDementiaSignalRepository(DementiaSignalRepository):
    """In-memory store for dementia signals."""

    def __init__(self) -> None:
        self._signals: dict[str, DementiaSignal] = {}

    async def upsert_signal(self, signal: DementiaSignal) -> None:
        self._signals[signal.signal_id] = signal

    async def list_signals(
        self,
        identity_id: str | None = None,
        signal_kind: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int = 200,
    ) -> list[DementiaSignal]:
        results = list(self._signals.values())
        if identity_id is not None:
            results = [s for s in results if s.identity_id == identity_id]
        if signal_kind is not None:
            results = [s for s in results if s.signal_kind == signal_kind]
        if after is not None:
            results = [s for s in results if s.emitted_at >= after]
        if before is not None:
            results = [s for s in results if s.emitted_at <= before]
        results.sort(key=lambda s: s.emitted_at, reverse=True)
        return results[:limit]
