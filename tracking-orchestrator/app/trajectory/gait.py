"""Walking bout segmentation from per-frame floor speeds.

WalkingBoutSegmenter consumes calibrated floor-speed observations (one per
frame per PersonHypothesis) and emits discrete WalkingBout records when a
sustained walking episode ends.  All arithmetic is pure; no I/O.

Clinical rationale for defaults
--------------------------------
Comfortable indoor walking for cognitively impaired seniors is roughly
0.4-1.0 m/s.  The 0.3 m/s open threshold captures the slow tail without
triggering on postural sway.  The 0.2 m/s median discard gate eliminates
standing-drift artefacts.  2.5 m/s is an absolute projection-glitch ceiling
(no ambulatory person moves faster indoors).

Reference: Hegde et al., Alzheimer's & Dementia 2025 (passive home camera
gait validation); PMC8968722 (gait speed and cognitive decline).
"""

from __future__ import annotations

import math
import uuid
import zoneinfo
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..storage.gait import GaitBoutRepository, GaitDailyRepository

# UUID namespace for stable bout IDs — arbitrary fixed UUID.
_BOUT_NAMESPACE = uuid.UUID("a3e4f8c2-1b5d-4e7a-9f6c-2d0b3e8a1c5f")


@dataclass(frozen=True)
class WalkingBout:
    """A completed walking episode for one resident."""

    identity_id: str
    started_at: datetime
    ended_at: datetime
    duration_s: float
    distance_m: float
    median_speed_m_s: float
    p95_speed_m_s: float
    sample_count: int
    rooms: list[str]

    @property
    def bout_id(self) -> str:
        """UUID5 over (identity_id, started_at ISO) for idempotent upserts."""
        key = f"{self.identity_id}\x00{self.started_at.isoformat()}"
        return str(uuid.uuid5(_BOUT_NAMESPACE, key))


@dataclass(frozen=True)
class GaitDailyRecord:
    """One day's aggregated gait statistics for one resident."""

    identity_id: str
    local_date: date
    bout_count: int
    total_walking_s: float
    total_distance_m: float
    median_speed_m_s: float
    mad_speed_m_s: float
    p95_speed_m_s: float
    sample_bout_ids: list[str]
    computed_at: datetime


@dataclass
class GaitConfig:
    """Tunable thresholds for the WalkingBoutSegmenter and GaitAggregator."""

    bout_min_speed_m_s: float = 0.3
    bout_min_duration_s: float = 3.0
    bout_close_grace_s: float = 2.0
    max_plausible_speed_m_s: float = 2.5
    # Bouts with median below this are discarded as standing-drift.
    min_median_speed_m_s: float = 0.2
    # Aggregator: how often to recompute daily summaries (seconds).
    aggregate_interval_s: int = 3600
    # Aggregator: minimum bouts / walking seconds to treat a day as data-rich.
    # A day below these thresholds still gets a row; the trend detector gates on it.
    min_daily_bouts: int = 3
    min_daily_walking_s: float = 60.0
    # Aggregator: deployment timezone for local-date bucketing.
    tz_name: str = "UTC"


@dataclass
class _BoutState:
    """Per-PH in-progress bout accumulator."""

    identity_id: str
    ph_id: str
    open: bool = False
    # Time of the last above-threshold sample.
    last_above_at: datetime | None = None
    # Time of the first above-threshold sample in the current bout.
    started_at: datetime | None = None
    # Preceding calibrated position (metres) for displacement integration.
    prev_x_m: float | None = None
    prev_y_m: float | None = None
    prev_captured_at: datetime | None = None
    # Accumulated samples (speed values and positions) in the current bout.
    speed_samples: list[float] = field(default_factory=list)
    accumulated_distance_m: float = 0.0
    rooms_seen: list[str] = field(default_factory=list)
    # Samples since opening that have been above-threshold (for min-duration gate).
    above_threshold_duration_s: float = 0.0


class WalkingBoutSegmenter:
    """Convert per-frame floor speeds into discrete WalkingBout records.

    Call ``ingest()`` once per frame per PH with calibrated observations.
    Call ``flush_ph()`` when a PH closes to emit any open bout.
    Call ``flush_all()`` at pipeline shutdown.
    """

    def __init__(self, config: GaitConfig | None = None) -> None:
        self._cfg = config or GaitConfig()
        self._states: dict[str, _BoutState] = {}

    def ingest(
        self,
        *,
        ph_id: str,
        identity_id: str,
        captured_at: datetime,
        floor_speed_m_s: float | None,
        floor_x_m: float,
        floor_y_m: float,
        room_name: str,
    ) -> WalkingBout | None:
        """Process one calibrated frame observation.

        Returns a completed WalkingBout if this observation closed one, else
        None.  Uncalibrated observations (floor_speed_m_s=None) are ignored.
        """
        if floor_speed_m_s is None:
            return None

        cfg = self._cfg
        state = self._states.get(ph_id)
        if state is None or state.identity_id != identity_id:
            state = _BoutState(identity_id=identity_id, ph_id=ph_id)
            self._states[ph_id] = state

        # Glitch sample: update position but do not contribute to bout.
        if floor_speed_m_s > cfg.max_plausible_speed_m_s:
            state.prev_x_m = floor_x_m
            state.prev_y_m = floor_y_m
            state.prev_captured_at = captured_at
            return None

        above = floor_speed_m_s >= cfg.bout_min_speed_m_s
        closed_bout: WalkingBout | None = None

        if not state.open:
            if above:
                state.open = True
                state.started_at = captured_at
                state.last_above_at = captured_at
                state.speed_samples = [floor_speed_m_s]
                state.accumulated_distance_m = 0.0
                state.rooms_seen = [room_name] if room_name else []
                state.above_threshold_duration_s = 0.0
        else:
            # Bout is open — integrate displacement and accumulate.
            if state.prev_x_m is not None and state.prev_y_m is not None:
                dx = floor_x_m - state.prev_x_m
                dy = floor_y_m - state.prev_y_m
                state.accumulated_distance_m += math.sqrt(dx * dx + dy * dy)

            if above:
                if state.prev_captured_at is not None:
                    state.above_threshold_duration_s += (
                        captured_at - state.prev_captured_at
                    ).total_seconds()
                state.last_above_at = captured_at
                state.speed_samples.append(floor_speed_m_s)
                if room_name and (not state.rooms_seen or state.rooms_seen[-1] != room_name):
                    state.rooms_seen.append(room_name)
            else:
                # Below threshold — close if grace window has expired.
                gap_s = (
                    (captured_at - state.last_above_at).total_seconds()
                    if state.last_above_at
                    else 0.0
                )
                if gap_s > cfg.bout_close_grace_s:
                    closed_bout = self._close_bout(state, captured_at)
                    state.open = False
                    state.speed_samples = []
                    state.accumulated_distance_m = 0.0
                    state.rooms_seen = []
                    state.started_at = None
                    state.last_above_at = None
                    state.above_threshold_duration_s = 0.0

        state.prev_x_m = floor_x_m
        state.prev_y_m = floor_y_m
        state.prev_captured_at = captured_at
        return closed_bout

    def flush_ph(self, ph_id: str, closed_at: datetime) -> WalkingBout | None:
        """Flush any open bout for a closing PH."""
        state = self._states.pop(ph_id, None)
        if state is None or not state.open:
            return None
        return self._close_bout(state, closed_at)

    def flush_all(self, closed_at: datetime) -> list[WalkingBout]:
        """Flush all open bouts. Called at pipeline shutdown."""
        bouts: list[WalkingBout] = []
        for ph_id in list(self._states):
            b = self.flush_ph(ph_id, closed_at)
            if b is not None:
                bouts.append(b)
        return bouts

    def _close_bout(self, state: _BoutState, ended_at: datetime) -> WalkingBout | None:
        """Validate and build a WalkingBout; returns None if it should be discarded.

        ``ended_at`` is the observer time (grace-window expiry or PH close).
        The bout's duration is measured to ``last_above_at`` (the last walking
        frame), not to the grace-window expiry, so grace time does not inflate
        the clinical duration metric.
        """
        cfg = self._cfg
        if not state.started_at or not state.speed_samples:
            return None

        # Use last_above_at as the walking endpoint; fall back to ended_at.
        bout_end = state.last_above_at if state.last_above_at is not None else ended_at
        duration_s = (bout_end - state.started_at).total_seconds()

        # Discard bouts shorter than the minimum duration or with low median.
        if duration_s < cfg.bout_min_duration_s:
            return None
        median_speed = _percentile(state.speed_samples, 50)
        if median_speed < cfg.min_median_speed_m_s:
            return None

        p95_speed = _percentile(state.speed_samples, 95)

        return WalkingBout(
            identity_id=state.identity_id,
            started_at=state.started_at,
            ended_at=bout_end,
            duration_s=duration_s,
            distance_m=state.accumulated_distance_m,
            median_speed_m_s=median_speed,
            p95_speed_m_s=p95_speed,
            sample_count=len(state.speed_samples),
            rooms=list(dict.fromkeys(state.rooms_seen)),  # deduplicated, order-preserving
        )


def _percentile(values: list[float], pct: int) -> float:
    """Nearest-rank percentile over a non-empty list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Nearest-rank: index = ceil(pct/100 * n) - 1, clamped.
    idx = max(0, min(n - 1, math.ceil(pct / 100.0 * n) - 1))
    return sorted_vals[idx]


def _mad(values: list[float]) -> float:
    """Median absolute deviation of a non-empty list."""
    if not values:
        return 0.0
    med = _percentile(sorted(values), 50)
    return _percentile([abs(v - med) for v in values], 50)


class GaitAggregator:
    """Recompute gait_daily rows for today and yesterday (local time).

    Called from the existing _signal_loop in frame_pipeline.py at
    ``gait.aggregate_interval_s`` cadence.  Uses the last-run timestamp tracked
    internally so the same scheduler loop drives both signal computation and gait
    aggregation without spawning a second asyncio task.
    """

    def __init__(
        self,
        bout_repo: GaitBoutRepository,
        daily_repo: GaitDailyRepository,
        config: GaitConfig | None = None,
    ) -> None:
        self._bout_repo = bout_repo
        self._daily_repo = daily_repo
        self._cfg = config or GaitConfig()
        self._tz = zoneinfo.ZoneInfo(self._cfg.tz_name)
        self._last_run_at: datetime | None = None

    def due(self, now: datetime) -> bool:
        """Return True if the aggregator interval has elapsed since last run."""
        if self._last_run_at is None:
            return True
        elapsed = (now - self._last_run_at).total_seconds()
        return elapsed >= self._cfg.aggregate_interval_s

    async def run_once(self, now: datetime) -> None:
        """Recompute today and yesterday for every identity with bouts in range.

        ``now`` is a UTC datetime.  Local dates are derived from the configured
        timezone so that a midnight boundary in America/New_York does not split
        an evening session across two UTC days.
        """
        from datetime import UTC, timedelta

        self._last_run_at = now

        local_now = now.astimezone(self._tz)
        today = local_now.date()
        yesterday = today - timedelta(days=1)

        # Window: midnight of yesterday (local) to now.
        yesterday_start_local = datetime(
            yesterday.year, yesterday.month, yesterday.day, tzinfo=self._tz
        )
        since = yesterday_start_local.astimezone(UTC)
        until = now

        bouts = await self._bout_repo.list_bouts(after=since, before=until, limit=50_000)

        # Group by (identity_id, local_date).
        grouped: dict[tuple[str, date], list[WalkingBout]] = {}
        for bout in bouts:
            local_date = bout.started_at.astimezone(self._tz).date()
            if local_date not in (today, yesterday):
                continue
            grouped.setdefault((bout.identity_id, local_date), []).append(bout)

        for (identity_id, local_date), day_bouts in grouped.items():
            record = self._aggregate(identity_id, local_date, day_bouts, now)
            await self._daily_repo.upsert_day(record)

    def _aggregate(
        self,
        identity_id: str,
        local_date: date,
        bouts: list[WalkingBout],
        computed_at: datetime,
    ) -> GaitDailyRecord:
        from .stats import weighted_median

        speeds = [b.median_speed_m_s for b in bouts]
        weights = [b.duration_s for b in bouts]

        med_speed = weighted_median(speeds, weights) if speeds else 0.0
        mad_speed = _mad(speeds)
        p95_speed = _percentile(speeds, 95)

        return GaitDailyRecord(
            identity_id=identity_id,
            local_date=local_date,
            bout_count=len(bouts),
            total_walking_s=sum(b.duration_s for b in bouts),
            total_distance_m=sum(b.distance_m for b in bouts),
            median_speed_m_s=med_speed,
            mad_speed_m_s=mad_speed,
            p95_speed_m_s=p95_speed,
            sample_bout_ids=[b.bout_id for b in bouts],
            computed_at=computed_at,
        )
