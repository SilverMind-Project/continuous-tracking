"""DementiaSignalWorker: periodic computation of dementia signals.

This worker runs on a scheduler (e.g. APScheduler or a simple asyncio
loop) and computes dementia-relevant signals from trajectory and dwell
data.  Each signal covers a time window and carries a severity and
z-score relative to the person's historical baseline.

Signal types
------------
- **pacing**: >N room entries in 30 min traversing 2-3 unique rooms
- **bathroom_dwell_anomaly**: current dwell > 2 std of 30-day mean
- **sundowning_index**: activity 17:00-22:00 vs 12:00-17:00 baseline
- **nighttime_movement**: room transitions 01:00-05:00
- **stillness_anomaly**: no pose movement > N min in non-bed room
- **absence**: no detection > threshold when historically present

The worker imports nothing from transport or pipeline — it only reads
trajectory/dwell data via the repository protocol and publishes via
the :class:`EventPublisher` protocol.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from ..domain import DementiaSignal, DementiaSignalSeverity, PersonTrajectoryPoint, RoomDwell
from ..storage.base import DementiaSignalRepository, TrajectoryRepository

_SIGNAL_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL


def _stillness_severity(duration_seconds: int) -> DementiaSignalSeverity:
    """Escalate stillness severity with duration.

    30-59 min → info (unusual but could be reading/watching TV)
    60-119 min → warning (concerning; may indicate a fall or confusion)
    120+ min → emergency (prolonged immobility; fall or incapacitation likely)
    """
    minutes = duration_seconds // 60
    if minutes >= 120:
        return "emergency"
    if minutes >= 60:
        return "warning"
    return "info"


def _stable_signal_id(
    identity_id: str, signal_kind: str, window_start: datetime, window_end: datetime | str
) -> str:
    """Return a deterministic UUID5 for (identity, kind, window) so upserts are idempotent.

    ``window_end`` may be a datetime (closed window) or the string ``"open"``
    for signals that track an ongoing state.
    """
    end_str = window_end.isoformat() if isinstance(window_end, datetime) else str(window_end)
    key = f"{identity_id}\x00{signal_kind}\x00{window_start.isoformat()}\x00{end_str}"
    return str(uuid.uuid5(_SIGNAL_NS, key))


class DementiaSignalWorker:
    """Periodically compute dementia signals from trajectory/dwell data.

    Usage::

        worker = DementiaSignalWorker(
            trajectory_repo=trajectory_repo,
            signal_repo=signal_repo,
            cfg=SignalConfig(),
        )

        # Called periodically (e.g. every 5 minutes).
        await worker.run_once(now=datetime.now(UTC))
    """

    def __init__(
        self,
        trajectory_repo: TrajectoryRepository,
        signal_repo: DementiaSignalRepository,
        cfg: SignalConfig | None = None,
    ) -> None:
        self._trajectory_repo = trajectory_repo
        self._signal_repo = signal_repo
        self._cfg = cfg or SignalConfig()

    async def run_once(self, now: datetime | None = None) -> list[DementiaSignal]:
        """Run one computation cycle.

        For each tracked identity, compute all signal types and persist
        any that exceed the severity threshold.

        Args:
            now: current wall-clock time (defaults to ``datetime.now(UTC)``).

        Returns:
            List of signals emitted during this cycle.
        """
        if now is None:
            now = datetime.now(UTC)

        # Gather identities from recent trajectory data.
        identities = await self._get_tracked_identities(now)

        all_signals: list[DementiaSignal] = []
        for identity_id in identities:
            window = await self._trajectory_repo.list_trajectory_points(
                identity_id=identity_id,
                after=now - self._cfg.window,
            )
            dwells = await self._trajectory_repo.list_room_dwells(
                identity_id=identity_id,
                after=now - self._cfg.window,
            )

            for signal in await self._compute_signals(identity_id, window, dwells, now):
                await self._signal_repo.upsert_signal(signal)
                all_signals.append(signal)

        return all_signals

    async def _get_tracked_identities(self, now: datetime) -> list[str]:
        """Return identity IDs that have been active in the observation window."""
        points = await self._trajectory_repo.list_trajectory_points(
            after=now - self._cfg.window,
            limit=10000,
        )
        return list({p.identity_id for p in points})

    async def _compute_signals(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        dwells: list[RoomDwell],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Compute all signal types for one identity."""
        signals: list[DementiaSignal] = []

        signals.extend(await self._compute_pacing(identity_id, window, now))
        signals.extend(await self._compute_sundowning(identity_id, window, now))
        signals.extend(self._compute_bathroom_dwell_anomaly(identity_id, dwells, now))
        signals.extend(self._compute_nighttime_movement(identity_id, window, now))
        signals.extend(self._compute_stillness_anomaly(identity_id, dwells, now))
        signals.extend(self._compute_absence(identity_id, window, now))

        return signals

    # -- Signal computations ------------------------------------------------

    async def _compute_pacing(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect pacing: >N room entries traversing 2-3 unique rooms.

        Pacing is identified when a person enters a new room more than
        ``self._cfg.pacing_room_threshold`` times within the pacing window
        (default 30 minutes), visiting at least 2 unique rooms.
        """
        # Narrow to the pacing analysis window (e.g. last 30 min) rather
        # than using the full 24 h observation window, so the count resets
        # as the window slides forward.
        pacing_cutoff = now - self._cfg.pacing_window
        pacing_points = [p for p in window if p.observed_at >= pacing_cutoff]

        if len(pacing_points) < 3:
            return []

        # Sort ascending so consecutive comparisons are meaningful.
        sorted_window = sorted(pacing_points, key=lambda p: p.observed_at)

        # Count room transitions and track unique rooms visited.
        room_changes = 0
        unique_rooms: set[str] = {sorted_window[0].room_name}
        for i in range(1, len(sorted_window)):
            if sorted_window[i].room_name != sorted_window[i - 1].room_name:
                room_changes += 1
                unique_rooms.add(sorted_window[i].room_name)

        if room_changes < self._cfg.pacing_room_threshold:
            return []

        # Compute severity based on room transition rate.
        window_duration = (
            sorted_window[-1].observed_at - sorted_window[0].observed_at
        ).total_seconds()
        if window_duration <= 0:
            return []

        rate = room_changes / (window_duration / 60.0)  # transitions per minute
        if rate > 0.3:
            severity: DementiaSignalSeverity = "emergency"
        elif rate > 0.15:
            severity = "warning"
        else:
            severity = "info"

        z_score = await self._compute_z_score(
            value=room_changes,
            signal_kind="pacing",
            identity_id=identity_id,
        )

        w_start = sorted_window[0].observed_at
        w_end = sorted_window[-1].observed_at
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, "pacing", w_start, w_end),
                identity_id=identity_id,
                signal_kind="pacing",
                severity=severity,
                value=float(room_changes),
                baseline=z_score.baseline,
                z_score=z_score.z_score,
                window_start=w_start,
                window_end=w_end,
                emitted_at=w_end,
                context={
                    "room_transitions": room_changes,
                    "unique_rooms": len(unique_rooms),
                    "rooms": sorted(unique_rooms),
                    "rate_per_minute": round(rate, 3),
                },
            )
        ]

    async def _compute_sundowning(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect sundowning: increased activity 17:00-22:00 vs 12:00-17:00.

        Compares the number of room transitions in the late-afternoon/
        evening window against the afternoon baseline.
        """
        if len(window) < 2:
            return []

        # Sort ascending by time so consecutive comparisons are meaningful.
        sorted_window = sorted(window, key=lambda p: p.observed_at)

        afternoon_transitions = 0
        evening_transitions = 0
        afternoon_points = 0
        evening_points = 0

        for i in range(1, len(sorted_window)):
            p_prev = sorted_window[i - 1]
            p_curr = sorted_window[i]

            # Convert UTC timestamps to local time before comparing hours.
            curr_hour = p_curr.observed_at.astimezone(self._cfg.tz).hour
            if 12 <= curr_hour < 17:
                afternoon_points += 1
            elif 17 <= curr_hour < 22:
                evening_points += 1

            if p_curr.room_name != p_prev.room_name:
                # Classify the transition by its midpoint so a change near
                # the 17:00 boundary is not misattributed to the wrong period.
                midpoint = p_prev.observed_at + (p_curr.observed_at - p_prev.observed_at) / 2
                mid_hour = midpoint.astimezone(self._cfg.tz).hour
                if 12 <= mid_hour < 17:
                    afternoon_transitions += 1
                elif 17 <= mid_hour < 22:
                    evening_transitions += 1

        if afternoon_points < 2 or evening_points < 2:
            return []

        # Sundowning index: ratio of evening to afternoon activity.
        baseline_rate = afternoon_transitions / afternoon_points
        current_rate = evening_transitions / evening_points

        if baseline_rate == 0:
            return []

        index = current_rate / baseline_rate

        if index < 1.5:
            return []  # Not significant enough

        if index > 3.0:
            severity: DementiaSignalSeverity = "emergency"
        elif index > 2.0:
            severity = "warning"
        else:
            severity = "info"

        z_score = await self._compute_z_score(
            value=index,
            signal_kind="sundowning_index",
            identity_id=identity_id,
        )

        w_start = sorted_window[0].observed_at
        w_end = sorted_window[-1].observed_at
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, "sundowning_index", w_start, w_end),
                identity_id=identity_id,
                signal_kind="sundowning_index",
                severity=severity,
                value=round(index, 3),
                baseline=z_score.baseline,
                z_score=z_score.z_score,
                window_start=w_start,
                window_end=w_end,
                emitted_at=w_end,
                context={
                    "afternoon_transitions": afternoon_transitions,
                    "evening_transitions": evening_transitions,
                    "baseline_rate": round(baseline_rate, 3),
                    "current_rate": round(current_rate, 3),
                },
            )
        ]

    def _compute_bathroom_dwell_anomaly(
        self,
        identity_id: str,
        dwells: list[RoomDwell],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect bathroom dwell anomaly: current dwell > 2 std of 30-day mean.

        Checks if any current (open) or recent bathroom dwell exceeds
        two standard deviations above the historical mean.
        """
        bathroom_dwells = [d for d in dwells if "bath" in d.room_name.lower()]

        if len(bathroom_dwells) < 2:
            return []

        # Calculate mean and std of all bathroom dwells.
        durations = [d.duration_seconds or 0 for d in bathroom_dwells]
        mean_dur = sum(durations) / len(durations)
        variance = sum((d - mean_dur) ** 2 for d in durations) / (len(durations) - 1)
        std_dur = math.sqrt(variance) if variance > 0 else 0

        # Check for open (current) bathroom dwell.
        open_bathroom = [d for d in bathroom_dwells if d.exited_at is None]
        if not open_bathroom:
            # Check the most recent closed dwell.
            recent = bathroom_dwells[-1]
            if recent.duration_seconds is None:
                return []
            current_dur = recent.duration_seconds
        else:
            current = open_bathroom[-1]
            current_dur = int((now - current.entered_at).total_seconds())

        if std_dur == 0 or current_dur <= mean_dur + 2 * std_dur:
            return []

        z_score_val = (current_dur - mean_dur) / std_dur if std_dur > 0 else 0

        if z_score_val > 4:
            severity: DementiaSignalSeverity = "emergency"
        elif z_score_val > 3:
            severity = "warning"
        else:
            severity = "info"

        w_start = open_bathroom[-1].entered_at if open_bathroom else bathroom_dwells[-1].entered_at
        w_end = now
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, "bathroom_dwell_anomaly", w_start, "open"),
                identity_id=identity_id,
                signal_kind="bathroom_dwell_anomaly",
                severity=severity,
                value=float(current_dur),
                baseline=float(mean_dur),
                z_score=round(z_score_val, 2),
                window_start=w_start,
                window_end=w_end,
                emitted_at=w_end,
                context={
                    "current_duration_seconds": current_dur,
                    "mean_duration_seconds": round(mean_dur, 1),
                    "std_duration_seconds": round(std_dur, 1),
                    "z_score": round(z_score_val, 2),
                    "dwells_analyzed": len(bathroom_dwells),
                },
            )
        ]

    def _compute_nighttime_movement(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect nighttime movement: room transitions 01:00-05:00.

        Counts room transitions during the nighttime window and flags
        if the count exceeds the configured threshold.
        """
        night_transitions = 0
        # Nighttime window: 22:00–06:00 local time (spans midnight).
        # Convert each point to local time before comparing hours.
        night_points = sorted(
            [
                p for p in window
                if (h := p.observed_at.astimezone(self._cfg.tz).hour) >= 22 or h < 6
            ],
            key=lambda p: p.observed_at,
        )

        for i in range(1, len(night_points)):
            if night_points[i].room_name != night_points[i - 1].room_name:
                night_transitions += 1

        if night_transitions < self._cfg.nighttime_transition_threshold:
            return []

        if night_transitions >= 5:
            severity: DementiaSignalSeverity = "emergency"
        elif night_transitions >= 3:
            severity = "warning"
        else:
            severity = "info"

        w_start = night_points[0].observed_at if night_points else now
        w_end = night_points[-1].observed_at if night_points else now
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, "nighttime_movement", w_start, w_end),
                identity_id=identity_id,
                signal_kind="nighttime_movement",
                severity=severity,
                value=float(night_transitions),
                window_start=w_start,
                window_end=w_end,
                emitted_at=w_end,
                context={
                    "room_transitions": night_transitions,
                    "nighttime_points": len(night_points),
                },
            )
        ]

    def _compute_stillness_anomaly(
        self,
        identity_id: str,
        dwells: list[RoomDwell],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect stillness anomaly: no pose movement > N min in non-bed room.

        Flags when a person has been in a non-bed room for longer than
        the configured stillness threshold without activity.
        """
        still_signals: list[DementiaSignal] = []

        for dwell in dwells:
            # Skip bed room.
            if "bed" in dwell.room_name.lower() or "bedroom" in dwell.room_name.lower():
                continue

            if dwell.exited_at is not None:
                # Closed dwell — check if it was unusually long/still.
                duration = dwell.duration_seconds or 0
                if duration >= self._cfg.stillness_threshold_minutes * 60:
                    w_end = dwell.exited_at
                    still_signals.append(
                        DementiaSignal(
                            signal_id=_stable_signal_id(
                                identity_id, "stillness_anomaly", dwell.entered_at, w_end
                            ),
                            identity_id=identity_id,
                            signal_kind="stillness_anomaly",
                            severity=_stillness_severity(duration),
                            value=float(duration),
                            window_start=dwell.entered_at,
                            window_end=w_end,
                            emitted_at=w_end,
                            context={
                                "room_name": dwell.room_name,
                                "duration_seconds": duration,
                            },
                        )
                    )
            else:
                # Open dwell — check current duration.
                duration = int((now - dwell.entered_at).total_seconds())
                if duration >= self._cfg.stillness_threshold_minutes * 60:
                    still_signals.append(
                        DementiaSignal(
                            signal_id=_stable_signal_id(
                                identity_id, "stillness_anomaly", dwell.entered_at, "open"
                            ),
                            identity_id=identity_id,
                            signal_kind="stillness_anomaly",
                            severity=_stillness_severity(duration),
                            value=float(duration),
                            window_start=dwell.entered_at,
                            window_end=now,
                            emitted_at=now,
                            context={
                                "room_name": dwell.room_name,
                                "duration_seconds": duration,
                                "is_open": True,
                                "window_end_marker": "open",
                            },
                        )
                    )

        return still_signals

    def _compute_absence(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect absence: no detections for a configurable window.

        Flags when the most recent trajectory point is older than
        ``self._cfg.absence_threshold_minutes`` minutes, indicating the
        person has not been detected across any camera for that duration.

        Uses a fixed ``window_end`` marker ("open") so repeated runs
        produce the same signal_id and upsert is idempotent.
        """
        if not window:
            # No data at all in the observation window — cannot distinguish
            # "never tracked" from "absent"; skip.
            return []

        most_recent = max(p.observed_at for p in window)
        gap_minutes = (now - most_recent).total_seconds() / 60.0

        if gap_minutes < self._cfg.absence_threshold_minutes:
            return []

        if gap_minutes >= 120:
            severity: DementiaSignalSeverity = "emergency"
        elif gap_minutes >= 60:
            severity = "warning"
        else:
            severity = "info"

        # Use a fixed sentinel for window_end so the signal_id is stable
        # across runs, making upsert idempotent.
        open_marker = "open"
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, "absence", most_recent, open_marker),
                identity_id=identity_id,
                signal_kind="absence",
                severity=severity,
                value=round(gap_minutes, 1),
                window_start=most_recent,
                window_end=now,
                emitted_at=now,
                context={
                    "last_seen_at": most_recent.isoformat(),
                    "gap_minutes": round(gap_minutes, 1),
                    "window_end_marker": open_marker,
                },
            )
        ]

    async def _compute_z_score(
        self,
        value: float,
        signal_kind: str,
        identity_id: str,
    ) -> ZScoreResult:
        """Compute a z-score relative to a historical baseline.

        Queries the signal repo for historical signals of the same kind
        and computes mean/std from the stored values.  If no baseline
        exists (fewer than 2 prior signals), both are ``None``.
        """
        try:
            historical = await self._signal_repo.list_signals(
                identity_id=identity_id,
                signal_kind=signal_kind,
                limit=100,
            )
        except Exception:
            return ZScoreResult(baseline=None, z_score=None)

        if len(historical) < 2:
            return ZScoreResult(baseline=None, z_score=None)

        values = [s.value for s in historical]
        mean_val = sum(values) / len(values)
        # Sample variance (N-1 denominator) so the z-score is not inflated
        # for small baselines (the <2 guard ensures len(values) >= 2).
        variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
        std_val = math.sqrt(variance) if variance > 0 else 0.0

        if std_val == 0:
            return ZScoreResult(baseline=mean_val, z_score=0.0)

        z = (value - mean_val) / std_val
        return ZScoreResult(baseline=round(mean_val, 4), z_score=round(z, 4))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ZScoreResult:
    """Container for a z-score computation result."""

    def __init__(self, baseline: float | None, z_score: float | None) -> None:
        self.baseline = baseline
        self.z_score = z_score


class SignalConfig:
    """Configuration for dementia signal computation."""

    def __init__(
        self,
        window_hours: int = 24,
        pacing_room_threshold: int = 5,
        pacing_window_minutes: int = 30,
        nighttime_transition_threshold: int = 2,
        stillness_threshold_minutes: int = 30,
        absence_threshold_minutes: int = 30,
        tz_name: str = "UTC",
    ) -> None:
        self.window = timedelta(hours=window_hours)
        self.pacing_room_threshold = pacing_room_threshold
        self.pacing_window = timedelta(minutes=pacing_window_minutes)
        self.nighttime_transition_threshold = nighttime_transition_threshold
        self.stillness_threshold_minutes = stillness_threshold_minutes
        self.absence_threshold_minutes = absence_threshold_minutes
        self.tz = ZoneInfo(tz_name)
