"""Dementia signal and behavior baseline storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

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
class DailyWindowSample:
    """One local-calendar-day slice of a timed window (e.g. 17:00-22:00).

    Used for sundowning (evening window) and nighttime movement (night window)
    baselines where each calendar day is one independent sample.

    For windows that wrap midnight (e.g. 22:00-06:00) points before the end
    hour (06:00) are bucketed to the *previous* local date so that one night
    produces exactly one sample.
    """

    local_date: date
    transition_count: int
    observed_points: int


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


@dataclass(frozen=True)
class AgitationWindowRecord:
    """One 30-minute agitation composite index sample.

    Stored in ``agitation_windows`` and used as the personal baseline for
    robust_z comparison.  The composite is the raw [0, 1] heuristic index
    (not an emitted signal value) so the baseline-from-raw-behavior rule
    is satisfied.
    """

    identity_id: str
    window_start: datetime
    composite: float
    computed_at: datetime


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

    @abstractmethod
    async def daily_window_rates(
        self,
        identity_id: str,
        local_hour_start: int,
        local_hour_end: int,
        tz_name: str,
        since: datetime,
        until: datetime,
    ) -> list[DailyWindowSample]:
        """Return one sample per local calendar day with observations in the given hour window.

        For windows that wrap midnight (local_hour_start > local_hour_end, e.g. 22-06)
        points whose local hour is less than local_hour_end are assigned to the previous
        local date so one night produces one sample.  Only days with at least one
        observation in the window are returned.
        """

    @abstractmethod
    async def pacing_window_rates(
        self,
        identity_id: str,
        window_minutes: int,
        since: datetime,
        until: datetime,
    ) -> list[float]:
        """Return transitions-per-minute for each dense tumbling window.

        Fixed tumbling windows of window_minutes are aligned to since.
        Windows with fewer than window_minutes * 0.5 points are excluded
        so that sparse coverage periods do not produce spurious zero-rate samples.
        """

    @abstractmethod
    async def agitation_window_samples(
        self,
        identity_id: str,
        since: datetime,
        until: datetime,
    ) -> list[float]:
        """Return historical 30-minute agitation composite index values.

        Each value is the raw [0, 1] composite from a prior evaluation window,
        stored by ``save_agitation_window``.  The caller must exclude the
        current window from the query range (pass ``until`` before the current
        window start) so the baseline self-exclusion rule holds.
        """

    @abstractmethod
    async def save_agitation_window(
        self,
        identity_id: str,
        window_start: datetime,
        composite: float,
    ) -> None:
        """Persist one 30-minute agitation composite sample for future baselining.

        Called after each ``_compute_agitation`` evaluation regardless of
        whether a signal is emitted.  The raw composite (not the signal) is
        stored so baselines are derived from raw behaviour, not emitted signals.
        Rows older than 90 days are candidates for retention-policy cleanup.
        """


class InMemoryBehaviorBaselineRepository(BehaviorBaselineRepository):
    """In-memory baseline repository backed by trajectory/dwell lists."""

    def __init__(
        self,
        points: list[PersonTrajectoryPoint] | None = None,
        dwells: list[RoomDwell] | None = None,
    ) -> None:
        self.points: list[PersonTrajectoryPoint] = points or []
        self.dwells: list[RoomDwell] = dwells or []
        self._agitation_windows: list[AgitationWindowRecord] = []

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

    async def daily_window_rates(
        self,
        identity_id: str,
        local_hour_start: int,
        local_hour_end: int,
        tz_name: str,
        since: datetime,
        until: datetime,
    ) -> list[DailyWindowSample]:
        tz = ZoneInfo(tz_name)
        wraps_midnight = local_hour_start > local_hour_end

        def in_window(local_hour: int) -> bool:
            if wraps_midnight:
                return local_hour >= local_hour_start or local_hour < local_hour_end
            return local_hour_start <= local_hour < local_hour_end

        def bucket_date(local_dt: datetime) -> date:
            if wraps_midnight and local_dt.hour < local_hour_end:
                return (local_dt - timedelta(days=1)).date()
            return local_dt.date()

        day_points: dict[date, list[PersonTrajectoryPoint]] = {}
        for p in self.points:
            if p.identity_id != identity_id:
                continue
            if p.observed_at < since or p.observed_at > until:
                continue
            local_dt = p.observed_at.astimezone(tz)
            if not in_window(local_dt.hour):
                continue
            d = bucket_date(local_dt)
            day_points.setdefault(d, []).append(p)

        results: list[DailyWindowSample] = []
        for d, pts in sorted(day_points.items()):
            pts_sorted = sorted(pts, key=lambda p: p.observed_at)
            transitions = sum(
                1
                for i in range(1, len(pts_sorted))
                if pts_sorted[i].room_name != pts_sorted[i - 1].room_name
            )
            results.append(
                DailyWindowSample(
                    local_date=d,
                    transition_count=transitions,
                    observed_points=len(pts_sorted),
                )
            )
        return results

    async def pacing_window_rates(
        self,
        identity_id: str,
        window_minutes: int,
        since: datetime,
        until: datetime,
    ) -> list[float]:
        min_points = window_minutes * 0.5

        filtered = sorted(
            [
                p
                for p in self.points
                if p.identity_id == identity_id
                and p.observed_at >= since
                and p.observed_at <= until
            ],
            key=lambda p: p.observed_at,
        )
        if not filtered:
            return []

        rates: list[float] = []
        window_dur = timedelta(minutes=window_minutes)
        bucket_start = since
        while bucket_start < until:
            bucket_end = bucket_start + window_dur
            bucket_pts = [p for p in filtered if bucket_start <= p.observed_at < bucket_end]
            if len(bucket_pts) >= min_points:
                transitions = sum(
                    1
                    for i in range(1, len(bucket_pts))
                    if bucket_pts[i].room_name != bucket_pts[i - 1].room_name
                )
                rates.append(transitions / window_minutes)
            bucket_start = bucket_end
        return rates

    async def agitation_window_samples(
        self,
        identity_id: str,
        since: datetime,
        until: datetime,
    ) -> list[float]:
        return [
            r.composite
            for r in self._agitation_windows
            if r.identity_id == identity_id and r.window_start >= since and r.window_start < until
        ]

    async def save_agitation_window(
        self,
        identity_id: str,
        window_start: datetime,
        composite: float,
    ) -> None:
        from datetime import UTC

        self._agitation_windows.append(
            AgitationWindowRecord(
                identity_id=identity_id,
                window_start=window_start,
                composite=composite,
                computed_at=datetime.now(UTC),
            )
        )


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
