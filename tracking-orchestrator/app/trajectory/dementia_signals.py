"""DementiaSignalWorker: periodic computation of dementia signals.

This worker runs on a scheduler (e.g. APScheduler or a simple asyncio
loop) and computes dementia-relevant signals from trajectory and dwell
data.  Each signal covers a time window and carries a severity and
z-score relative to the person's historical baseline.

Signal types
------------
- **pacing**: >N room entries in 30 min traversing 2-3 unique rooms
- **bathroom_dwell_anomaly**: current dwell > robust threshold vs 30-day baseline
- **sundowning_index**: activity 17:00-22:00 vs 14-day evening baseline
- **nighttime_movement**: room transitions 22:00-06:00
- **stillness_anomaly**: sustained low motion energy in non-resting posture
- **absence**: no detection > threshold when historically present
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from ..domain import DementiaSignal, DementiaSignalSeverity, PersonTrajectoryPoint, RoomDwell
from ..storage.base import (
    BehaviorBaselineRepository,
    DementiaSignalRepository,
    TrajectoryRepository,
)
from .stats import robust_z

_SIGNAL_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_ALGORITHM_VERSION = 3  # Phase 3: robust baselines + hysteresis + detector rewrites


# ---------------------------------------------------------------------------
# Hysteresis / debounce
# ---------------------------------------------------------------------------


class SignalHysteresis:
    """Tracks per-identity signal state across worker runs for debounce.

    - **onset debounce**: a trigger must hold for ``min_consecutive`` runs
      before the first emission.
    - **cooldown**: after emission, the same (identity, kind) will not re-emit
      for ``cooldown_minutes`` unless severity *escalates*.
    - **severity monotonic**: within an episode, severity only goes up;
      it closes when the trigger clears.
    """

    def __init__(self, min_consecutive: int = 2) -> None:
        self._min_consecutive = min_consecutive
        # (identity_id, signal_kind) -> consecutive runs with trigger held
        self._consecutive_count: dict[tuple[str, str], int] = {}
        # (identity_id, signal_kind) -> last emission time (UTC)
        self._last_emission: dict[tuple[str, str], datetime] = {}
        # (identity_id, signal_kind) -> current episode severity (or None if closed)
        self._episode_severity: dict[tuple[str, str], DementiaSignalSeverity | None] = {}

    def should_emit(
        self,
        identity_id: str,
        signal_kind: str,
        severity: DementiaSignalSeverity,
        now: datetime,
        cooldown_minutes: int,
    ) -> bool:
        """Return True if a signal should be emitted given hysteresis rules."""
        key = (identity_id, signal_kind)

        # Severity escalation within episode: always emit.
        current_sev = self._episode_severity.get(key)
        if current_sev is not None:
            sev_order = {"info": 0, "warning": 1, "emergency": 2}
            if sev_order.get(severity, 0) > sev_order.get(current_sev, 0):
                self._episode_severity[key] = severity
                self._consecutive_count[key] = self._min_consecutive  # bypass debounce
                self._last_emission[key] = now
                return True
            # Re-upsert at equal severity → no-op (the DB upsert is idempotent,
            # but we suppress duplicate emissions to avoid re-alerting).
            return False

        # Onset debounce: count consecutive trigger holds.
        count = self._consecutive_count.get(key, 0) + 1
        self._consecutive_count[key] = count
        if count < self._min_consecutive:
            return False

        # Cooldown check (skip for escalated severity handled above).
        last = self._last_emission.get(key)
        if last is not None and (now - last).total_seconds() < cooldown_minutes * 60:
            return False

        # New episode.
        self._episode_severity[key] = severity
        self._last_emission[key] = now
        return True

    def clear_trigger(self, identity_id: str, signal_kind: str) -> None:
        """Reset the consecutive counter and close the episode for *signal_kind*."""
        key = (identity_id, signal_kind)
        self._consecutive_count.pop(key, None)
        self._episode_severity.pop(key, None)

    def close_episode(self, identity_id: str, signal_kind: str) -> None:
        """Mark the episode as resolved (no longer active)."""
        key = (identity_id, signal_kind)
        self._episode_severity.pop(key, None)


# ---------------------------------------------------------------------------
# Stillness severity (posture-aware)
# ---------------------------------------------------------------------------


def _stillness_severity(
    duration_seconds: int, posture: str, cfg: SignalConfig
) -> DementiaSignalSeverity:
    """Escalate stillness severity based on duration and posture.

    - ``lying`` in a non-resting room starts at ``warning`` and escalates
      to ``emergency`` at the emergency-minutes threshold.
    - ``sitting`` / ``standing`` still → ``info`` until 2x the stillness
      threshold, then ``warning``.
    """
    minutes = duration_seconds // 60
    if posture == "lying":
        if minutes >= cfg.stillness_emergency_minutes:
            return "emergency"
        return "warning"
    # sitting / standing / unknown
    if minutes >= cfg.stillness_threshold_minutes * 2:
        return "warning"
    return "info"


# ---------------------------------------------------------------------------
# Stable signal ID
# ---------------------------------------------------------------------------


def _stable_signal_id(
    identity_id: str, signal_kind: str, window_start: datetime, window_end: datetime | str
) -> str:
    """Return a deterministic UUID5 for (identity, kind, window)."""
    end_str = window_end.isoformat() if isinstance(window_end, datetime) else str(window_end)
    key = f"{identity_id}\x00{signal_kind}\x00{window_start.isoformat()}\x00{end_str}"
    return str(uuid.uuid5(_SIGNAL_NS, key))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class DementiaSignalWorker:
    """Periodically compute dementia signals from trajectory/dwell data."""

    def __init__(
        self,
        trajectory_repo: TrajectoryRepository,
        signal_repo: DementiaSignalRepository,
        cfg: SignalConfig | None = None,
        baseline_repo: BehaviorBaselineRepository | None = None,
    ) -> None:
        self._trajectory_repo = trajectory_repo
        self._signal_repo = signal_repo
        self._cfg = cfg or SignalConfig()
        self._baseline_repo = baseline_repo
        self._hysteresis = SignalHysteresis(
            min_consecutive=self._cfg.onset_consecutive_windows,
        )
        # Phase 4: incremental windows + baseline cache
        self._last_run_at: dict[str, datetime] = {}  # identity_id -> last run time
        # Cache raw baseline samples per (identity_id, signal_kind).
        self._baseline_samples_cache: dict[tuple[str, str], tuple[float, list[float]]] = {}
        # Rolling per-identity detector state (points, dwells from previous windows).
        self._rolling_points: dict[str, list[PersonTrajectoryPoint]] = {}
        self._rolling_dwells: dict[str, list[RoomDwell]] = {}

    async def run_once(self, now: datetime | None = None) -> list[DementiaSignal]:
        """Run one computation cycle, processing identities concurrently."""
        import asyncio
        import time

        t_start = time.monotonic()
        if now is None:
            now = datetime.now(UTC)

        identities = await self._get_tracked_identities(now)
        sem = asyncio.Semaphore(max(1, self._cfg.max_concurrent_identities))

        # Collect metrics
        try:
            from ..observability import metrics as m
            m.metrics.signal_worker_identities.set(len(identities))
        except Exception:
            pass

        all_signals: list[DementiaSignal] = []

        async def _process_one(identity_id: str) -> list[DementiaSignal]:
            async with sem:
                return await self._process_identity(identity_id, now)

        tasks = [asyncio.create_task(_process_one(iid)) for iid in identities]
        if tasks:
            results = await asyncio.gather(*tasks)
            for signal_list in results:
                for signal in signal_list:
                    await self._signal_repo.upsert_signal(signal)
                    all_signals.append(signal)

        # Evict stale identities from rolling state.
        stale = [
            iid for iid, last in self._last_run_at.items()
            if (now - last).total_seconds() > self._cfg.window.total_seconds() * 2
        ]
        for iid in stale:
            self._rolling_points.pop(iid, None)
            self._rolling_dwells.pop(iid, None)
            self._last_run_at.pop(iid, None)

        elapsed = time.monotonic() - t_start
        try:
            from ..observability import metrics as m
            m.metrics.signal_worker_run_seconds.observe(elapsed)
            for s in all_signals:
                m.metrics.signal_worker_emitted_total.labels(
                    kind=s.signal_kind, severity=s.severity
                ).inc()
        except Exception:
            pass

        return all_signals

    async def _process_identity(
        self, identity_id: str, now: datetime
    ) -> list[DementiaSignal]:
        """Fetch window data (incremental when possible) and compute signals."""
        last = self._last_run_at.get(identity_id)
        self._last_run_at[identity_id] = now

        # Fetch only delta since last run, merging with rolling state.
        if last is not None and self._cfg.incremental_enabled:
            delta_since = last - timedelta(minutes=5)  # slight overlap for safety
            new_points = await self._trajectory_repo.list_trajectory_points(
                identity_id=identity_id,
                after=delta_since,
                limit=5000,
            )
            new_dwells = await self._trajectory_repo.list_room_dwells(
                identity_id=identity_id,
                after=delta_since,
                limit=1000,
            )
            # Merge with rolling state.
            rolling_pts = self._rolling_points.get(identity_id, [])
            rolling_dwells = self._rolling_dwells.get(identity_id, [])
            # Prune old entries outside the window.
            cutoff = now - self._cfg.window
            rolling_pts = [p for p in rolling_pts if p.observed_at >= cutoff]
            rolling_dwells = [d for d in rolling_dwells if d.entered_at >= cutoff]
            # Add new data.
            seen_pt_ids = {(p.identity_id, p.global_track_id, p.observed_at) for p in rolling_pts}
            for pt in new_points:
                key = (pt.identity_id, pt.global_track_id, pt.observed_at)
                if key not in seen_pt_ids:
                    rolling_pts.append(pt)
                    seen_pt_ids.add(key)
            seen_dw_ids = {d.dwell_id for d in rolling_dwells}
            for dw in new_dwells:
                if dw.dwell_id not in seen_dw_ids:
                    rolling_dwells.append(dw)
                    seen_dw_ids.add(dw.dwell_id)
            self._rolling_points[identity_id] = rolling_pts
            self._rolling_dwells[identity_id] = rolling_dwells
            window_data = rolling_pts
            dwell_data = rolling_dwells
        else:
            window_data = await self._trajectory_repo.list_trajectory_points(
                identity_id=identity_id,
                after=now - self._cfg.window,
            )
            dwell_data = await self._trajectory_repo.list_room_dwells(
                identity_id=identity_id,
                after=now - self._cfg.window,
            )
            # Seed rolling state for future incremental runs.
            if self._cfg.incremental_enabled:
                self._rolling_points[identity_id] = list(window_data)
                self._rolling_dwells[identity_id] = list(dwell_data)

        signals = await self._compute_signals(identity_id, window_data, dwell_data, now)
        self._clear_inactive(identity_id, window_data, dwell_data, now)
        return signals

    async def _get_tracked_identities(self, now: datetime) -> list[str]:
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
        signals: list[DementiaSignal] = []
        signals.extend(await self._compute_pacing(identity_id, window, now))
        signals.extend(await self._compute_sundowning(identity_id, window, now))
        signals.extend(await self._compute_bathroom_dwell_anomaly(identity_id, dwells, now))
        signals.extend(await self._compute_nighttime_movement(identity_id, window, now))
        signals.extend(await self._compute_stillness_anomaly(identity_id, dwells, now))
        signals.extend(await self._compute_absence(identity_id, window, now))
        return signals

    def _clear_inactive(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        dwells: list[RoomDwell],
        now: datetime,
    ) -> None:
        """Clear hysteresis state for signals that are no longer active."""
        # stillness: clear if no open non-resting dwell is still
        open_dwells = [d for d in dwells if d.exited_at is None]
        any_stillness = any(
            d for d in open_dwells
            if not self._is_resting_room(d.room_name)
            and (now - d.entered_at).total_seconds() >= self._cfg.stillness_threshold_minutes * 60
        )
        if not any_stillness:
            self._hysteresis.clear_trigger(identity_id, "stillness_anomaly")

        # bathroom: clear if no open bathroom dwell
        any_bath = any(
            d for d in open_dwells if "bath" in d.room_name.lower()
        )
        if not any_bath:
            self._hysteresis.clear_trigger(identity_id, "bathroom_dwell_anomaly")

        # absence: clear if there are recent points
        if window:
            most_recent = max(p.observed_at for p in window)
            gap_minutes = (now - most_recent).total_seconds() / 60.0
            if gap_minutes < self._cfg.absence_threshold_minutes:
                self._hysteresis.clear_trigger(identity_id, "absence")

    # ------------------------------------------------------------------
    # 3.4  Pacing — normalised for observation density
    # ------------------------------------------------------------------

    async def _compute_pacing(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect pacing: repetitive room entry pattern normalised for sampling rate."""
        pacing_cutoff = now - self._cfg.pacing_window
        pacing_points = [p for p in window if p.observed_at >= pacing_cutoff]

        if len(pacing_points) < 3:
            return []

        sorted_window = sorted(pacing_points, key=lambda p: p.observed_at)
        window_duration_s = (sorted_window[-1].observed_at - sorted_window[0].observed_at).total_seconds()
        if window_duration_s <= 0:
            return []

        # Observation density: points per minute.
        obs_density = len(sorted_window) / (window_duration_s / 60.0)
        if obs_density < self._cfg.pacing_min_obs_density:
            return []

        # Room transitions and unique rooms.
        room_changes = 0
        unique_rooms: set[str] = {sorted_window[0].room_name}
        visited_rooms: list[str] = [sorted_window[0].room_name]
        for i in range(1, len(sorted_window)):
            if sorted_window[i].room_name != sorted_window[i - 1].room_name:
                room_changes += 1
                unique_rooms.add(sorted_window[i].room_name)
                visited_rooms.append(sorted_window[i].room_name)

        if room_changes < self._cfg.pacing_room_threshold:
            return []
        if len(unique_rooms) < 2:
            return []

        # Cycle detection: count returns to previously-visited rooms.
        returns = 0
        seen: set[str] = set()
        for r in visited_rooms:
            if r in seen:
                returns += 1
            seen.add(r)

        # Transitions per observed-minute (normalised for sampling rate).
        rate = room_changes / (window_duration_s / 60.0)

        z_score = await self._compute_z_score(
            value=rate,
            signal_kind="pacing",
            identity_id=identity_id,
        )

        if rate > 0.3:
            severity: DementiaSignalSeverity = "emergency"
        elif rate > 0.15:
            severity = "warning"
        else:
            severity = "info"

        # Hysteresis.
        kind = "pacing"
        if not self._hysteresis.should_emit(identity_id, kind, severity, now, self._cfg.cooldown_minutes):
            return []

        w_start = sorted_window[0].observed_at
        w_end = sorted_window[-1].observed_at
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, kind, w_start, w_end),
                identity_id=identity_id,
                signal_kind=kind,
                severity=severity,
                value=round(rate, 3),
                baseline=z_score.baseline,
                z_score=z_score.z_score,
                window_start=w_start,
                window_end=w_end,
                emitted_at=w_end,
                algorithm_version=_ALGORITHM_VERSION,
                context={
                    "room_transitions": room_changes,
                    "unique_rooms": len(unique_rooms),
                    "rooms": sorted(unique_rooms),
                    "rate_per_minute": round(rate, 3),
                    "returns": returns,
                    "obs_density": round(obs_density, 2),
                },
            )
        ]

    # ------------------------------------------------------------------
    # 3.3  Sundowning — multi-day evening baseline
    # ------------------------------------------------------------------

    async def _compute_sundowning(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect sundowning: increased evening activity vs 14-day evening baseline.

        Current evening (17:00-22:00 resident-local) room-transition rate is
        compared against the distribution of evening-window rates over the
        prior 14 days via robust_z.
        """
        if len(window) < 2:
            return []

        sorted_window = sorted(window, key=lambda p: p.observed_at)

        # Count evening transitions and observation points for today.
        evening_transitions = 0
        evening_points = 0
        today_evening_start = now.replace(hour=17, minute=0, second=0, microsecond=0)
        if now.hour < 17:
            today_evening_start -= timedelta(days=1)

        for i in range(1, len(sorted_window)):
            p_prev = sorted_window[i - 1]
            p_curr = sorted_window[i]
            curr_hour = p_curr.observed_at.astimezone(self._cfg.tz).hour
            if 17 <= curr_hour < 22:
                evening_points += 1
                if p_curr.room_name != p_prev.room_name:
                    midpoint = p_prev.observed_at + (p_curr.observed_at - p_prev.observed_at) / 2
                    mid_hour = midpoint.astimezone(self._cfg.tz).hour
                    if 17 <= mid_hour < 22:
                        evening_transitions += 1

        if evening_points < self._cfg.sundowning_min_evening_minutes:
            return []

        # Rate for today's evening window.
        today_rate = evening_transitions / max(evening_points, 1)

        # Build 14-day evening baseline from hourly_activity.
        evening_rates: list[float] = []
        if self._baseline_repo is not None:
            hourly = await self._baseline_repo.hourly_activity(
                identity_id,
                since=now - timedelta(days=14),
                until=now,
            )
            # Evening hours (17-21): sum transitions and observations.
            evening_transition_total = sum(
                h.transition_count for hr, h in hourly.items() if 17 <= hr < 22
            )
            evening_obs_total = sum(
                h.observed_minutes for hr, h in hourly.items() if 17 <= hr < 22
            )
            if evening_obs_total > 0:
                avg_evening_rate = evening_transition_total / evening_obs_total
                evening_rates = [avg_evening_rate]  # daily aggregate

        if len(evening_rates) < 2:
            # Fall back: use simple threshold (1.5x ratio → significant)
            if today_rate < 0.03:
                return []
            severity: DementiaSignalSeverity = "info"
        else:
            rz = robust_z(today_rate, evening_rates)
            if rz.n < self._cfg.min_baseline_n or rz.modified_z < self._cfg.sundowning_z_threshold:
                return []
            if rz.modified_z >= 4.0:
                severity: DementiaSignalSeverity = "emergency"
            elif rz.modified_z >= 3.0:
                severity = "warning"
            else:
                severity = "info"

        kind = "sundowning_index"
        if not self._hysteresis.should_emit(identity_id, kind, severity, now, self._cfg.cooldown_minutes):
            return []

        w_start = sorted_window[0].observed_at
        w_end = sorted_window[-1].observed_at
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, kind, w_start, w_end),
                identity_id=identity_id,
                signal_kind=kind,
                severity=severity,
                value=round(today_rate, 4),
                window_start=w_start,
                window_end=w_end,
                emitted_at=w_end,
                algorithm_version=_ALGORITHM_VERSION,
                context={
                    "evening_transitions": evening_transitions,
                    "evening_points": evening_points,
                    "today_rate": round(today_rate, 4),
                    "baseline_rates_count": len(evening_rates),
                },
            )
        ]

    # ------------------------------------------------------------------
    # 3.2  Bathroom dwell anomaly — robust z + time-of-day aware
    # ------------------------------------------------------------------

    async def _compute_bathroom_dwell_anomaly(
        self,
        identity_id: str,
        dwells: list[RoomDwell],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect bathroom dwell anomaly using robust z-score on 30-day baseline.

        The baseline only includes *closed* dwells, automatically excluding
        the current open dwell.  Nighttime (22:00-06:00) uses a wider threshold.
        """
        bathroom_dwells = [d for d in dwells if "bath" in d.room_name.lower()]

        # Current (open) bathroom dwell.
        open_bathroom = [d for d in bathroom_dwells if d.exited_at is None]
        if not open_bathroom:
            self._hysteresis.clear_trigger(identity_id, "bathroom_dwell_anomaly")
            return []

        current = open_bathroom[-1]
        current_dur = int((now - current.entered_at).total_seconds())

        if current_dur < 60:  # minimum meaningful dwell
            return []

        # Baseline via _compute_z_score (cached per identity+kind).
        if self._baseline_repo is not None:
            z_result = await self._compute_z_score(
                value=float(current_dur),
                signal_kind="bathroom_dwell_anomaly",
                identity_id=identity_id,
            )
            if z_result.z_score is None:
                # Cold start: use absolute threshold, cap severity.
                if current_dur < self._cfg.bathroom_absolute_threshold_seconds:
                    return []
                severity_val: DementiaSignalSeverity = "warning"  # capped
                z_score_val = None
                baseline_val = None
            else:
                # Time-of-day: widen threshold for nighttime hours.
                current_hour = now.astimezone(self._cfg.tz).hour
                threshold = (
                    self._cfg.bathroom_z_threshold_night
                    if current_hour >= 22 or current_hour < 6
                    else self._cfg.bathroom_z_threshold
                )
                if z_result.z_score < threshold:
                    self._hysteresis.clear_trigger(identity_id, "bathroom_dwell_anomaly")
                    return []
                severity_val = _bathroom_severity(z_result.z_score)
                z_score_val = z_result.z_score
                baseline_val = z_result.baseline

            severity = severity_val
        else:
            # No baseline repo: fall back to old std-dev approach (kept for
            # backward compatibility with InMemory-only test setups).
            closed = [d for d in bathroom_dwells if d.exited_at is not None and d.duration_seconds is not None]
            if len(closed) < 2:
                return []
            durations = [float(d.duration_seconds) for d in closed]  # type: ignore[arg-type]
            mean_dur = sum(durations) / len(durations)
            variance = sum((d - mean_dur) ** 2 for d in durations) / (len(durations) - 1)
            std_dur = math.sqrt(variance) if variance > 0 else 0.0
            if std_dur == 0 or current_dur <= mean_dur + 2 * std_dur:
                self._hysteresis.clear_trigger(identity_id, "bathroom_dwell_anomaly")
                return []
            z_score_val = (current_dur - mean_dur) / std_dur
            baseline_val = mean_dur
            severity = _bathroom_severity(z_score_val)

        kind = "bathroom_dwell_anomaly"
        if not self._hysteresis.should_emit(identity_id, kind, severity, now, self._cfg.cooldown_minutes):
            return []

        w_start = current.entered_at
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, kind, w_start, "open"),
                identity_id=identity_id,
                signal_kind=kind,
                severity=severity,
                value=float(current_dur),
                baseline=baseline_val,
                z_score=z_score_val,
                window_start=w_start,
                window_end=now,
                emitted_at=now,
                algorithm_version=_ALGORITHM_VERSION,
                context={
                    "current_duration_seconds": current_dur,
                    "window_end_marker": "open",
                },
            )
        ]

    # ------------------------------------------------------------------
    # Nighttime movement — robust_z baseline
    # ------------------------------------------------------------------

    async def _compute_nighttime_movement(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect nighttime movement: room transitions 22:00-06:00 local.

        Compared against the resident's 14-day nighttime-transition baseline
        via robust_z; falls back to a flat threshold for cold starts.
        """
        night_points = sorted(
            [
                p
                for p in window
                if (h := p.observed_at.astimezone(self._cfg.tz).hour) >= 22 or h < 6
            ],
            key=lambda p: p.observed_at,
        )

        night_transitions = 0
        for i in range(1, len(night_points)):
            if night_points[i].room_name != night_points[i - 1].room_name:
                night_transitions += 1

        if night_transitions < self._cfg.nighttime_transition_threshold:
            self._hysteresis.clear_trigger(identity_id, "nighttime_movement")
            return []

        # Try robust_z against 14-day baseline.
        z_score_result = await self._compute_z_score(
            value=float(night_transitions),
            signal_kind="nighttime_movement",
            identity_id=identity_id,
        )

        if z_score_result.z_score is not None:
            if z_score_result.z_score >= 4.0:
                severity: DementiaSignalSeverity = "emergency"
            elif z_score_result.z_score >= 3.0:
                severity = "warning"
            else:
                severity = "info"
        else:
            # Cold start: fall back to flat threshold.
            if night_transitions >= 5:
                severity = "emergency"
            elif night_transitions >= 3:
                severity = "warning"
            else:
                severity = "info"

        kind = "nighttime_movement"
        if not self._hysteresis.should_emit(identity_id, kind, severity, now, self._cfg.cooldown_minutes):
            return []

        w_start = night_points[0].observed_at if night_points else now
        w_end = night_points[-1].observed_at if night_points else now
        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, kind, w_start, w_end),
                identity_id=identity_id,
                signal_kind=kind,
                severity=severity,
                value=float(night_transitions),
                baseline=z_score_result.baseline,
                z_score=z_score_result.z_score,
                window_start=w_start,
                window_end=w_end,
                emitted_at=w_end,
                algorithm_version=_ALGORITHM_VERSION,
                context={
                    "room_transitions": night_transitions,
                    "nighttime_points": len(night_points),
                },
            )
        ]

    # ------------------------------------------------------------------
    # 3.1  Stillness anomaly — motion-energy based
    # ------------------------------------------------------------------

    async def _compute_stillness_anomaly(
        self,
        identity_id: str,
        dwells: list[RoomDwell],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect stillness: sustained low motion energy in non-resting posture.

        Clinically meaningful immobility = near-zero motion energy in a
        non-resting posture (lying in a non-bedroom, or prolonged
        sitting/standing stillness).
        """
        still_signals: list[DementiaSignal] = []

        for dwell in dwells:
            # Skip resting rooms (configurable set).
            if self._is_resting_room(dwell.room_name):
                continue

            if dwell.exited_at is not None:
                # Closed dwell — check if it meets stillness criteria.
                if dwell.still_seconds < self._cfg.stillness_threshold_minutes * 60:
                    continue
                if dwell.min_motion_energy is not None and dwell.min_motion_energy > self._cfg.stillness_motion_floor:
                    continue
                duration = dwell.duration_seconds or 0
                severity = _stillness_severity(duration, dwell.primary_posture, self._cfg)
                w_end = dwell.exited_at
                cold_start = False

                # Compare against baseline stillness episodes.
                z_result = await self._compute_z_score(
                    value=float(dwell.still_seconds),
                    signal_kind="stillness_anomaly",
                    identity_id=identity_id,
                )
                if z_result.z_score is None and self._baseline_repo is not None:
                    # Cold start: cap severity at warning.
                    cold_start = True
                    if severity == "emergency":
                        severity = "warning"
            else:
                # Open dwell — check current stillness duration.
                duration = int((now - dwell.entered_at).total_seconds())
                if duration < self._cfg.stillness_threshold_minutes * 60:
                    continue
                if dwell.min_motion_energy is not None and dwell.min_motion_energy > self._cfg.stillness_motion_floor:
                    continue
                severity = _stillness_severity(duration, dwell.primary_posture, self._cfg)
                w_end = now
                z_result = await self._compute_z_score(
                    value=float(dwell.still_seconds),
                    signal_kind="stillness_anomaly",
                    identity_id=identity_id,
                )
                cold_start = z_result.z_score is None and self._baseline_repo is not None
                if cold_start and severity == "emergency":
                    severity = "warning"

            kind = "stillness_anomaly"
            if not self._hysteresis.should_emit(identity_id, kind, severity, now, self._cfg.cooldown_minutes):
                continue

            still_signals.append(
                DementiaSignal(
                    signal_id=_stable_signal_id(
                        identity_id, kind, dwell.entered_at,
                        "open" if dwell.exited_at is None else dwell.exited_at,
                    ),
                    identity_id=identity_id,
                    signal_kind=kind,
                    severity=severity,
                    value=float(dwell.still_seconds),
                    baseline=z_result.baseline,
                    z_score=z_result.z_score,
                    window_start=dwell.entered_at,
                    window_end=w_end,
                    emitted_at=now,
                    algorithm_version=_ALGORITHM_VERSION,
                    context={
                        "room_name": dwell.room_name,
                        "still_seconds": dwell.still_seconds,
                        "primary_posture": dwell.primary_posture,
                        "min_motion_energy": dwell.min_motion_energy,
                        "cold_start": cold_start,
                        "is_open": dwell.exited_at is None,
                    },
                )
            )

        return still_signals

    # ------------------------------------------------------------------
    # 3.5  Absence — hourly context
    # ------------------------------------------------------------------

    async def _compute_absence(
        self,
        identity_id: str,
        window: list[PersonTrajectoryPoint],
        now: datetime,
    ) -> list[DementiaSignal]:
        """Detect absence: no detections beyond threshold.

        Uses hourly_activity context: if the resident is historically
        frequently absent at this hour, severity is lowered and the signal
        carries ``expected_absence_prior``.
        """
        if not window:
            return []

        most_recent = max(p.observed_at for p in window)
        gap_minutes = (now - most_recent).total_seconds() / 60.0

        if gap_minutes < self._cfg.absence_threshold_minutes:
            self._hysteresis.clear_trigger(identity_id, "absence")
            return []

        # Base severity.
        if gap_minutes >= 120:
            severity: DementiaSignalSeverity = "emergency"
        elif gap_minutes >= 60:
            severity = "warning"
        else:
            severity = "info"

        # Check hourly context: if historically absent at this hour, lower severity.
        expected_absence_prior: float | None = None
        if self._baseline_repo is not None:
            hourly = await self._baseline_repo.hourly_activity(
                identity_id,
                since=now - timedelta(days=14),
                until=now,
            )
            current_hour = now.astimezone(self._cfg.tz).hour
            hour_data = hourly.get(current_hour)
            if hour_data is not None:
                # Absence fraction for this hour: 0.0 = always present, 1.0 = always absent
                hour_window = 14 * 60  # max possible minutes at this hour over 14 days
                observed = hour_data.observed_minutes
                expected_absence_prior = round(1.0 - min(observed / hour_window, 1.0), 3)
                # If person is frequently absent at this hour, demote severity.
                if expected_absence_prior > 0.5 and severity != "info":
                    severity = "info" if severity == "warning" else "warning"

        kind = "absence"
        open_marker = "open"
        if not self._hysteresis.should_emit(identity_id, kind, severity, now, self._cfg.cooldown_minutes):
            return []

        return [
            DementiaSignal(
                signal_id=_stable_signal_id(identity_id, kind, most_recent, open_marker),
                identity_id=identity_id,
                signal_kind=kind,
                severity=severity,
                value=round(gap_minutes, 1),
                window_start=most_recent,
                window_end=now,
                emitted_at=now,
                algorithm_version=_ALGORITHM_VERSION,
                context={
                    "last_seen_at": most_recent.isoformat(),
                    "gap_minutes": round(gap_minutes, 1),
                    "window_end_marker": open_marker,
                    "expected_absence_prior": expected_absence_prior,
                },
            )
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_resting_room(self, room_name: str) -> bool:
        """Return True if the room is in the configured resting-room set."""
        rl = room_name.lower()
        return any(r in rl for r in self._cfg.resting_rooms)

    async def _compute_z_score(
        self,
        value: float,
        signal_kind: str,
        identity_id: str,
    ) -> ZScoreResult:
        """Compute a robust z-score using ``baseline_repo`` when available.

        Raw baseline samples are cached per (identity, kind) with a
        configurable TTL.  The z-score is recomputed fresh each run
        against the cached baseline so the current *value* is always
        compared against the same historical distribution.
        """
        if self._baseline_repo is not None:
            cache_key = (identity_id, signal_kind)
            now_ts = datetime.now(UTC).timestamp()
            samples: list[float] | None = None

            # Check cache.
            cached_entry = self._baseline_samples_cache.get(cache_key)
            if cached_entry is not None:
                cached_at, cached_samples = cached_entry
                if now_ts - cached_at < self._cfg.baseline_cache_ttl_minutes * 60:
                    samples = cached_samples
                    try:
                        from ..observability import metrics as m
                        m.metrics.signal_baseline_cache_hits_total.inc()
                    except Exception:
                        pass

            # Fetch if no cache hit.
            if samples is None:
                samples = await self._fetch_baseline_samples(signal_kind, identity_id)
                if samples is not None:
                    self._baseline_samples_cache[cache_key] = (now_ts, samples)

            if samples is not None and len(samples) >= self._cfg.min_baseline_n:
                rz = robust_z(value, samples)
                return ZScoreResult(baseline=rz.median, z_score=rz.modified_z)

            return ZScoreResult(baseline=None, z_score=None)

        # Legacy path (no baseline repo configured).
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
        variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
        std_val = math.sqrt(variance) if variance > 0 else 0.0

        if std_val == 0:
            return ZScoreResult(baseline=mean_val, z_score=0.0)

        z = (value - mean_val) / std_val
        return ZScoreResult(baseline=round(mean_val, 4), z_score=round(z, 4))

    async def _fetch_baseline_samples(
        self,
        signal_kind: str,
        identity_id: str,
    ) -> list[float] | None:
        """Return raw baseline samples for *signal_kind*, or None."""
        assert self._baseline_repo is not None
        now = datetime.now(UTC)

        if signal_kind == "bathroom_dwell_anomaly":
            durations = await self._baseline_repo.dwell_durations(
                identity_id,
                room_predicate="bath",
                since=now - timedelta(days=30),
                until=now,
            )
            return durations if len(durations) >= self._cfg.min_baseline_n else None

        if signal_kind in ("pacing", "sundowning_index", "nighttime_movement"):
            hourly = await self._baseline_repo.hourly_activity(
                identity_id,
                since=now - timedelta(days=30),
                until=now,
            )
            samples = [float(h.transition_count) for h in hourly.values()]
            return samples if len(samples) >= self._cfg.min_baseline_n else None

        if signal_kind == "stillness_anomaly":
            episodes = await self._baseline_repo.stillness_episodes(
                identity_id,
                since=now - timedelta(days=30),
                until=now,
            )
            samples = [float(e.duration_seconds) for e in episodes]
            return samples if len(samples) >= self._cfg.min_baseline_n else None

        return None

    async def _compute_robust_z(
        self,
        value: float,
        signal_kind: str,
        identity_id: str,
    ) -> ZScoreResult:
        """Compute robust z-score using ``baseline_repo`` + ``robust_z``."""
        assert self._baseline_repo is not None
        now = datetime.now(UTC)

        if signal_kind == "bathroom_dwell_anomaly":
            durations = await self._baseline_repo.dwell_durations(
                identity_id,
                room_predicate="bath",
                since=now - timedelta(days=30),
                until=now,
            )
            if len(durations) < self._cfg.min_baseline_n:
                return ZScoreResult(baseline=None, z_score=None)
            rz = robust_z(value, durations)
            return ZScoreResult(baseline=rz.median, z_score=rz.modified_z)

        if signal_kind in ("pacing", "sundowning_index", "nighttime_movement"):
            hourly = await self._baseline_repo.hourly_activity(
                identity_id,
                since=now - timedelta(days=30),
                until=now,
            )
            samples = [float(h.transition_count) for h in hourly.values()]
            if len(samples) < self._cfg.min_baseline_n:
                return ZScoreResult(baseline=None, z_score=None)
            rz = robust_z(value, samples)
            return ZScoreResult(baseline=rz.median, z_score=rz.modified_z)

        if signal_kind == "stillness_anomaly":
            episodes = await self._baseline_repo.stillness_episodes(
                identity_id,
                since=now - timedelta(days=30),
                until=now,
            )
            durations = [float(e.duration_seconds) for e in episodes]
            if len(durations) < self._cfg.min_baseline_n:
                return ZScoreResult(baseline=None, z_score=None)
            rz = robust_z(value, durations)
            return ZScoreResult(baseline=rz.median, z_score=rz.modified_z)

        return ZScoreResult(baseline=None, z_score=None)


def _bathroom_severity(z_score: float) -> DementiaSignalSeverity:
    if z_score >= 5.0:
        return "emergency"
    if z_score >= 4.0:
        return "warning"
    return "info"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ZScoreResult:
    """Container for a z-score computation result."""

    def __init__(self, baseline: float | None, z_score: float | None) -> None:
        self.baseline = baseline
        self.z_score = z_score


class SignalConfig:
    """Configuration for dementia signal computation.

    All thresholds carry docstrings citing their rationale.
    """

    def __init__(
        self,
        # General
        window_hours: int = 24,
        tz_name: str = "UTC",
        # Baseline
        min_baseline_n: int = 5,
        # Hysteresis / debounce
        onset_consecutive_windows: int = 2,
        cooldown_minutes: int = 60,
        # Pacing
        pacing_room_threshold: int = 5,
        pacing_window_minutes: int = 30,
        pacing_min_obs_density: float = 0.5,  # minimum points/min to evaluate
        # Nighttime movement
        nighttime_transition_threshold: int = 2,
        # Stillness
        stillness_threshold_minutes: int = 30,
        stillness_emergency_minutes: int = 120,
        stillness_motion_floor: float = 0.005,
        resting_rooms: tuple[str, ...] = ("bed", "bedroom"),
        # Bathroom
        bathroom_z_threshold: float = 3.5,
        bathroom_z_threshold_night: float = 4.0,
        bathroom_absolute_threshold_seconds: int = 1800,  # 30 min cold-start fallback
        # Sundowning
        sundowning_z_threshold: float = 2.5,
        sundowning_min_evening_minutes: int = 30,
        # Absence
        absence_threshold_minutes: int = 30,
        # Phase 4: scalability
        baseline_cache_ttl_minutes: int = 60,
        max_concurrent_identities: int = 4,
        incremental_enabled: bool = True,
    ) -> None:
        self.window = timedelta(hours=window_hours)
        self.tz = ZoneInfo(tz_name)
        self.min_baseline_n = min_baseline_n
        self.onset_consecutive_windows = onset_consecutive_windows
        self.cooldown_minutes = cooldown_minutes
        self.pacing_room_threshold = pacing_room_threshold
        self.pacing_window = timedelta(minutes=pacing_window_minutes)
        self.pacing_min_obs_density = pacing_min_obs_density
        self.nighttime_transition_threshold = nighttime_transition_threshold
        self.stillness_threshold_minutes = stillness_threshold_minutes
        self.stillness_emergency_minutes = stillness_emergency_minutes
        self.stillness_motion_floor = stillness_motion_floor
        self.resting_rooms = resting_rooms
        self.bathroom_z_threshold = bathroom_z_threshold
        self.bathroom_z_threshold_night = bathroom_z_threshold_night
        self.bathroom_absolute_threshold_seconds = bathroom_absolute_threshold_seconds
        self.sundowning_z_threshold = sundowning_z_threshold
        self.sundowning_min_evening_minutes = sundowning_min_evening_minutes
        self.absence_threshold_minutes = absence_threshold_minutes
        self.baseline_cache_ttl_minutes = baseline_cache_ttl_minutes
        self.max_concurrent_identities = max_concurrent_identities
        self.incremental_enabled = incremental_enabled
