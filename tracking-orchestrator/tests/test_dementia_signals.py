"""Unit tests for DementiaSignalWorker.

All tests use InMemory repositories — no DB required.
Each test seeds fixture trajectories/dwells and asserts the expected
signal kind and severity.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

from app.domain import PersonTrajectoryPoint, RoomDwell
from app.storage.base import (
    InMemoryBehaviorBaselineRepository,
    InMemoryDementiaSignalRepository,
    InMemoryTrajectoryRepository,
)
from app.trajectory.dementia_signals import DementiaSignalWorker, SignalConfig, SignalHysteresis
from app.trajectory.stats import robust_z as _robust_z

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC)
_IDENTITY = "grandma"
_GT = "gt-001"


def _point(
    room: str,
    offset_minutes: float,
    posture: str = "walking",
    now: datetime = _NOW,
) -> PersonTrajectoryPoint:
    return PersonTrajectoryPoint(
        identity_id=_IDENTITY,
        ph_id=_GT,
        observed_at=now - timedelta(minutes=offset_minutes),
        room_name=room,
        posture=posture,  # type: ignore[arg-type]
        identity_confidence=0.9,
    )


def _dwell(
    room: str,
    entered_offset_minutes: float,
    duration_seconds: int | None = None,
    exited: bool = True,
    now: datetime = _NOW,
) -> RoomDwell:
    entered_at = now - timedelta(minutes=entered_offset_minutes)
    exited_at = None
    if exited and duration_seconds is not None:
        exited_at = entered_at + timedelta(seconds=duration_seconds)
    return RoomDwell(
        dwell_id=str(uuid.uuid4()),
        identity_id=_IDENTITY,
        ph_id=_GT,
        room_name=room,
        entered_at=entered_at,
        exited_at=exited_at,
        duration_seconds=duration_seconds if exited else None,
        entry_confidence=0.9,
    )


def _make_worker(
    cfg: SignalConfig | None = None,
) -> tuple[
    DementiaSignalWorker,
    InMemoryTrajectoryRepository,
    InMemoryDementiaSignalRepository,
]:
    traj_repo = InMemoryTrajectoryRepository()
    sig_repo = InMemoryDementiaSignalRepository()
    worker = DementiaSignalWorker(
        trajectory_repo=traj_repo,
        signal_repo=sig_repo,
        cfg=cfg,
    )
    return worker, traj_repo, sig_repo


# ---------------------------------------------------------------------------
# Pacing detector
# ---------------------------------------------------------------------------


class TestPacingDetector:
    @pytest.mark.asyncio
    async def test_pacing_detected_above_threshold(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                pacing_room_threshold=4,
                onset_consecutive_windows=1,
            )
        )
        # 10 alternating room transitions in 20 minutes.
        rooms = ["kitchen", "hallway"] * 5
        for i, room in enumerate(rooms):
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=20 - i * 2))

        signals = await worker.run_once(now=_NOW)
        pacing = [s for s in signals if s.signal_kind == "pacing"]
        assert len(pacing) == 1
        assert pacing[0].identity_id == _IDENTITY
        assert pacing[0].value >= 0.15  # rate per minute (was >= 4 room changes)

    @pytest.mark.asyncio
    async def test_pacing_not_detected_below_threshold(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                pacing_room_threshold=10,
                onset_consecutive_windows=1,
            )
        )
        # Only 3 transitions.
        for room in ["kitchen", "hallway", "kitchen", "hallway"]:
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=5))

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "pacing" for s in signals)

    @pytest.mark.asyncio
    async def test_pacing_severity_emergency_at_high_rate(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                pacing_room_threshold=3,
                onset_consecutive_windows=1,
            )
        )
        # 12 transitions in 2 minutes = very high rate.
        rooms = ["kitchen", "hallway"] * 6
        for i, room in enumerate(rooms):
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=2 - i * 0.15))

        signals = await worker.run_once(now=_NOW)
        pacing = [s for s in signals if s.signal_kind == "pacing"]
        assert pacing
        assert pacing[0].severity in ("warning", "emergency")


# ---------------------------------------------------------------------------
# Sundowning detector
# ---------------------------------------------------------------------------


class TestSundowningDetector:
    @pytest.mark.asyncio
    async def test_sundowning_detected_high_evening_activity(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(sundowning_min_evening_minutes=5, onset_consecutive_windows=1)
        )
        now = datetime(2026, 4, 23, 20, 0, 0, tzinfo=UTC)  # 20:00

        # Afternoon (12-17): 2 transitions across 6 points (low rate).
        afternoon_rooms = ["kitchen", "kitchen", "living_room", "living_room", "kitchen", "kitchen"]
        for i, room in enumerate(afternoon_rooms):
            await traj_repo.save_trajectory_point(
                _point(room, offset_minutes=0, now=datetime(2026, 4, 23, 14, i * 5, 0, tzinfo=UTC))
            )

        # Evening (17-22): 8 transitions across 10 points (high rate).
        evening_rooms = [
            "kitchen",
            "hallway",
            "kitchen",
            "hallway",
            "kitchen",
            "hallway",
            "kitchen",
            "hallway",
            "kitchen",
            "hallway",
        ]
        for i, room in enumerate(evening_rooms):
            await traj_repo.save_trajectory_point(
                _point(room, offset_minutes=0, now=datetime(2026, 4, 23, 18, i, 0, tzinfo=UTC))
            )

        signals = await worker.run_once(now=now)
        sundowning = [s for s in signals if s.signal_kind == "sundowning_index"]
        assert len(sundowning) == 1
        assert sundowning[0].value >= 0.03  # today_rate threshold

    @pytest.mark.asyncio
    async def test_sundowning_not_detected_equal_activity(self):
        worker, traj_repo, _ = _make_worker()
        now = datetime(2026, 4, 23, 20, 0, 0, tzinfo=UTC)

        # Equal transitions in afternoon and evening.
        for i in range(6):
            room = "kitchen" if i % 2 == 0 else "living_room"
            await traj_repo.save_trajectory_point(
                _point(room, offset_minutes=0, now=datetime(2026, 4, 23, 14, i * 5, 0, tzinfo=UTC))
            )
        for i in range(6):
            room = "kitchen" if i % 2 == 0 else "living_room"
            await traj_repo.save_trajectory_point(
                _point(room, offset_minutes=0, now=datetime(2026, 4, 23, 18, i * 5, 0, tzinfo=UTC))
            )

        signals = await worker.run_once(now=now)
        assert not any(s.signal_kind == "sundowning_index" for s in signals)


# ---------------------------------------------------------------------------
# Bathroom dwell anomaly detector
# ---------------------------------------------------------------------------


class TestBathroomDwellAnomalyDetector:
    @pytest.mark.asyncio
    async def test_anomaly_detected_when_current_dwell_exceeds_2std(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(onset_consecutive_windows=1))

        # Need at least one trajectory point so the identity is discovered.
        await traj_repo.save_trajectory_point(_point("bathroom", offset_minutes=25))

        # Historical dwells: mean ~5 min, std ~1 min.
        for dur in [240, 300, 360, 300, 280]:
            d = _dwell("bathroom", 120, duration_seconds=dur, exited=True)
            await traj_repo.save_room_dwell(d)
            await traj_repo.update_room_dwell(d)

        # Current open dwell: 20 minutes (far above mean + 2*std).
        open_dwell = _dwell("bathroom", 20, duration_seconds=None, exited=False)
        await traj_repo.save_room_dwell(open_dwell)

        signals = await worker.run_once(now=_NOW)
        anomaly = [s for s in signals if s.signal_kind == "bathroom_dwell_anomaly"]
        assert len(anomaly) == 1
        assert anomaly[0].z_score is not None
        assert anomaly[0].z_score > 2.0

    @pytest.mark.asyncio
    async def test_no_anomaly_when_dwell_is_normal(self):
        worker, traj_repo, _ = _make_worker()

        await traj_repo.save_trajectory_point(_point("bathroom", offset_minutes=7))

        for dur in [280, 300, 320, 290, 310]:
            d = _dwell("bathroom", 120, duration_seconds=dur, exited=True)
            await traj_repo.save_room_dwell(d)
            await traj_repo.update_room_dwell(d)

        # Current dwell: 6 minutes — within normal range.
        open_dwell = _dwell("bathroom", 6, duration_seconds=None, exited=False)
        await traj_repo.save_room_dwell(open_dwell)

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "bathroom_dwell_anomaly" for s in signals)


# ---------------------------------------------------------------------------
# Nighttime movement detector
# ---------------------------------------------------------------------------


class TestNighttimeMovementDetector:
    @pytest.mark.asyncio
    async def test_nighttime_movement_detected(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                nighttime_transition_threshold=2,
                onset_consecutive_windows=1,
            )
        )
        now = datetime(2026, 4, 23, 3, 0, 0, tzinfo=UTC)  # 03:00

        # 4 room transitions between 01:00 and 05:00.
        rooms = ["bedroom", "kitchen", "hallway", "bedroom", "kitchen"]
        for i, room in enumerate(rooms):
            t = datetime(2026, 4, 23, 2, i * 10, 0, tzinfo=UTC)
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=0, now=t))

        signals = await worker.run_once(now=now)
        night = [s for s in signals if s.signal_kind == "nighttime_movement"]
        assert len(night) == 1
        assert night[0].value >= 2

    @pytest.mark.asyncio
    async def test_no_nighttime_signal_during_day(self):
        worker, traj_repo, _ = _make_worker()
        # Daytime transitions — should not trigger nighttime signal.
        for room in ["kitchen", "living_room", "kitchen", "hallway"]:
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=5))

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "nighttime_movement" for s in signals)


# ---------------------------------------------------------------------------
# Stillness anomaly detector
# ---------------------------------------------------------------------------


class TestStillnessAnomalyDetector:
    @pytest.mark.asyncio
    async def test_stillness_detected_in_non_bed_room(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                stillness_threshold_minutes=20,
                onset_consecutive_windows=1,
            )
        )
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=41))
        # Open dwell in kitchen for 40 minutes with high still_seconds.
        entered_at = _NOW - timedelta(minutes=40)
        open_dwell = RoomDwell(
            dwell_id=str(uuid.uuid4()),
            identity_id=_IDENTITY,
            ph_id=_GT,
            room_name="kitchen",
            entered_at=entered_at,
            still_seconds=30 * 60,  # 30 minutes of actual stillness
            min_motion_energy=0.001,  # below motion floor
        )
        await traj_repo.save_room_dwell(open_dwell)

        signals = await worker.run_once(now=_NOW)
        stillness = [s for s in signals if s.signal_kind == "stillness_anomaly"]
        assert len(stillness) == 1
        assert stillness[0].value >= 20 * 60

    @pytest.mark.asyncio
    async def test_stillness_not_detected_in_bedroom(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                stillness_threshold_minutes=20,
                onset_consecutive_windows=1,
            )
        )
        await traj_repo.save_trajectory_point(_point("bedroom", offset_minutes=121))
        # Long dwell in bedroom — should be ignored.
        open_dwell = _dwell("bedroom", 120, duration_seconds=None, exited=False)
        await traj_repo.save_room_dwell(open_dwell)

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "stillness_anomaly" for s in signals)

    @pytest.mark.asyncio
    async def test_stillness_not_detected_below_threshold(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                stillness_threshold_minutes=30,
                onset_consecutive_windows=1,
            )
        )
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=11))
        # Only 10 minutes in kitchen.
        open_dwell = _dwell("kitchen", 10, duration_seconds=None, exited=False)
        await traj_repo.save_room_dwell(open_dwell)

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "stillness_anomaly" for s in signals)


# ---------------------------------------------------------------------------
# Absence detector
# ---------------------------------------------------------------------------


class TestAbsenceDetector:
    @pytest.mark.asyncio
    async def test_absence_detected_when_gap_exceeds_threshold(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                absence_threshold_minutes=30,
                onset_consecutive_windows=1,
            )
        )
        # Last seen 90 minutes ago.
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=90))

        signals = await worker.run_once(now=_NOW)
        absence = [s for s in signals if s.signal_kind == "absence"]
        assert len(absence) == 1
        assert absence[0].value >= 90.0

    @pytest.mark.asyncio
    async def test_absence_severity_emergency_at_2h(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                absence_threshold_minutes=30,
                onset_consecutive_windows=1,
            )
        )
        # Seed enough points to pass data-quality coverage gate.
        # 5 points from 134 min ago to 130 min ago (1/min), last seen 130 min ago.
        for i in range(5):
            await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=130 + i))

        signals = await worker.run_once(now=_NOW)
        absence = [s for s in signals if s.signal_kind == "absence"]
        assert absence
        assert absence[0].severity == "emergency"

    @pytest.mark.asyncio
    async def test_no_absence_when_recently_seen(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                absence_threshold_minutes=30,
                onset_consecutive_windows=1,
            )
        )
        # Seed enough points for coverage; last seen 5 minutes ago.
        for i in range(5):
            await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=5 + i))

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "absence" for s in signals)

    @pytest.mark.asyncio
    async def test_no_absence_when_no_data(self):
        worker, _, _ = _make_worker(
            SignalConfig(
                absence_threshold_minutes=30,
                onset_consecutive_windows=1,
            )
        )
        # No trajectory data at all — should not emit absence.
        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "absence" for s in signals)


# ---------------------------------------------------------------------------
# run_once: persistence and multi-identity
# ---------------------------------------------------------------------------


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_signals_persisted_to_repo(self):
        worker, traj_repo, sig_repo = _make_worker(
            SignalConfig(
                pacing_room_threshold=3,
                onset_consecutive_windows=1,
            )
        )
        rooms = ["kitchen", "hallway"] * 4
        for i, room in enumerate(rooms):
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=20 - i * 2))

        await worker.run_once(now=_NOW)
        stored = await sig_repo.list_signals()
        assert len(stored) > 0

    @pytest.mark.asyncio
    async def test_multiple_identities_processed_independently(self):
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                pacing_room_threshold=3,
                onset_consecutive_windows=1,
            )
        )
        # Identity A: pacing.
        rooms = ["kitchen", "hallway"] * 4
        for i, room in enumerate(rooms):
            pt = PersonTrajectoryPoint(
                identity_id="alice",
                ph_id="gt-a",
                observed_at=_NOW - timedelta(minutes=20 - i * 2),
                room_name=room,
                identity_confidence=0.9,
            )
            await traj_repo.save_trajectory_point(pt)

        # Identity B: no pacing.
        pt_b = PersonTrajectoryPoint(
            identity_id="bob",
            ph_id="gt-b",
            observed_at=_NOW - timedelta(minutes=5),
            room_name="bedroom",
            identity_confidence=0.9,
        )
        await traj_repo.save_trajectory_point(pt_b)

        signals = await worker.run_once(now=_NOW)
        alice_signals = [s for s in signals if s.identity_id == "alice"]
        bob_signals = [s for s in signals if s.identity_id == "bob" and s.signal_kind == "pacing"]
        assert alice_signals
        assert not bob_signals


# ---------------------------------------------------------------------------
# Phase 4 — Incremental windows + baseline cache
# ---------------------------------------------------------------------------


class _SpyBaselineRepository:
    """Wraps InMemoryBehaviorBaselineRepository and counts method calls."""

    def __init__(self, points: list, dwells: list) -> None:
        self._inner = InMemoryBehaviorBaselineRepository(points, dwells)
        self.dwell_durations_calls = 0
        self.hourly_activity_calls = 0
        self.stillness_episodes_calls = 0
        self.daily_window_rates_calls = 0
        self.pacing_window_rates_calls = 0

    async def dwell_durations(self, *args, **kwargs):
        self.dwell_durations_calls += 1
        return await self._inner.dwell_durations(*args, **kwargs)

    async def hourly_activity(self, *args, **kwargs):
        self.hourly_activity_calls += 1
        return await self._inner.hourly_activity(*args, **kwargs)

    async def stillness_episodes(self, *args, **kwargs):
        self.stillness_episodes_calls += 1
        return await self._inner.stillness_episodes(*args, **kwargs)

    async def daily_window_rates(self, *args, **kwargs):
        self.daily_window_rates_calls += 1
        return await self._inner.daily_window_rates(*args, **kwargs)

    async def pacing_window_rates(self, *args, **kwargs):
        self.pacing_window_rates_calls += 1
        return await self._inner.pacing_window_rates(*args, **kwargs)


class TestIncrementalWindows:
    @pytest.mark.asyncio
    async def test_incremental_matches_full_run(self):
        """Incremental run produces identical signals to a cold full run."""
        traj_repo = InMemoryTrajectoryRepository()
        sig_repo_full = InMemoryDementiaSignalRepository()
        sig_repo_incr = InMemoryDementiaSignalRepository()

        # Seed data for one identity with pacing pattern.
        rooms = ["kitchen", "hallway"] * 5
        for i, room in enumerate(rooms):
            pt = _point(room, offset_minutes=20 - i * 2)
            await traj_repo.save_trajectory_point(pt)

        cfg = SignalConfig(
            pacing_room_threshold=4,
            onset_consecutive_windows=1,
            incremental_enabled=False,
        )
        worker_full = DementiaSignalWorker(traj_repo, sig_repo_full, cfg=cfg)
        signals_full = await worker_full.run_once(now=_NOW)

        cfg_incr = SignalConfig(
            pacing_room_threshold=4,
            onset_consecutive_windows=1,
            incremental_enabled=True,
        )
        worker_incr = DementiaSignalWorker(traj_repo, sig_repo_incr, cfg=cfg_incr)
        signals_incr = await worker_incr.run_once(now=_NOW)

        assert len(signals_full) == len(signals_incr)
        for sf, si in zip(
            sorted(signals_full, key=lambda s: s.signal_id),
            sorted(signals_incr, key=lambda s: s.signal_id),
            strict=True,
        ):
            assert sf.signal_id == si.signal_id
            assert sf.signal_kind == si.signal_kind
            assert sf.severity == si.severity

    @pytest.mark.asyncio
    async def test_incremental_rolling_state_retains_points(self):
        """Rolling state accumulates points across runs."""
        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()

        # Seed one trajectory point.
        pt = _point("kitchen", offset_minutes=5)
        await traj_repo.save_trajectory_point(pt)

        cfg = SignalConfig(
            onset_consecutive_windows=1,
            incremental_enabled=True,
        )
        worker = DementiaSignalWorker(traj_repo, sig_repo, cfg=cfg)
        # First run — full fetch.
        await worker.run_once(now=_NOW)

        # Rolling state should have the point.
        rolling = worker._rolling_points.get(_IDENTITY, [])
        assert len(rolling) == 1

        # Seed another point at a different time.
        pt2 = _point("hallway", offset_minutes=3)
        await traj_repo.save_trajectory_point(pt2)

        # Second run — delta fetch picks up new point.
        await worker.run_once(now=_NOW + timedelta(minutes=1))
        rolling = worker._rolling_points.get(_IDENTITY, [])
        # Should have both points (old from rolling state, new from delta).
        assert len(rolling) >= 2


class TestBaselineCache:
    @pytest.mark.asyncio
    async def test_cache_hit_avoids_requery(self):
        """Second run within TTL does not re-query baseline repo."""
        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()

        # Seed bathroom dwells so we get a valid baseline.
        for dur in [240, 300, 360, 300, 280]:
            d = _dwell("bathroom", 120, duration_seconds=dur, exited=True)
            await traj_repo.save_room_dwell(d)
            await traj_repo.update_room_dwell(d)
        # Open bathroom dwell.
        await traj_repo.save_trajectory_point(_point("bathroom", offset_minutes=25))
        open_dwell = _dwell("bathroom", 20, duration_seconds=None, exited=False)
        await traj_repo.save_room_dwell(open_dwell)

        # Gather all points and dwells for the spy baseline repo.
        points = await traj_repo.list_trajectory_points(limit=1000)
        dwells_list = await traj_repo.list_room_dwells(limit=1000)
        spy = _SpyBaselineRepository(points, dwells_list)

        cfg = SignalConfig(
            onset_consecutive_windows=1,
            baseline_cache_ttl_minutes=60,
            min_baseline_n=3,
        )
        worker = DementiaSignalWorker(traj_repo, sig_repo, cfg=cfg, baseline_repo=spy)

        # First run populates cache.
        await worker.run_once(now=_NOW)
        calls_after_first = spy.dwell_durations_calls

        # Second run within TTL — should use cache.
        await worker.run_once(now=_NOW + timedelta(minutes=1))
        assert spy.dwell_durations_calls == calls_after_first  # no new calls


# ---------------------------------------------------------------------------
# Task 1.2 — Baseline semantics: sundowning, nighttime movement, pacing
# ---------------------------------------------------------------------------


# Helper: build one historical evening's points (19 pts, last one switches room).
# 19 points at 15-min intervals span 17:00-21:30 (all inside the 17-22 evening window).
def _mk_hist_evening(
    identity_id: str,
    ph_id: str,
    day_offset: int,
    ref_now: datetime,
) -> list[PersonTrajectoryPoint]:
    base = (ref_now - timedelta(days=day_offset)).replace(
        hour=17, minute=0, second=0, microsecond=0
    )
    return [
        PersonTrajectoryPoint(
            identity_id=identity_id,
            ph_id=ph_id,
            observed_at=base + timedelta(minutes=i * 15),
            room_name="kitchen" if i < 18 else "hallway",
            identity_confidence=0.9,
        )
        for i in range(19)
    ]


class TestSundowningWithBaseline:
    """Task 1.2: sundowning uses daily_window_rates for per-evening z-score."""

    @pytest.mark.asyncio
    async def test_escalation_emits_warning_or_emergency(self) -> None:
        """13 historical evenings at rate 1/19; today at rate 10/10 → z=inf → emergency."""
        now = datetime(2026, 4, 23, 20, 0, 0, tzinfo=UTC)

        # Historical: 13 prior evenings (days 1-13 before now), each 19 points / 1 transition.
        hist_pts = []
        for day in range(1, 14):
            hist_pts.extend(_mk_hist_evening(_IDENTITY, _GT, day, now))

        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()
        baseline_repo = InMemoryBehaviorBaselineRepository(points=hist_pts)
        cfg = SignalConfig(
            tz_name="UTC",
            onset_consecutive_windows=1,
            sundowning_min_evening_minutes=5,
        )
        worker = DementiaSignalWorker(traj_repo, sig_repo, cfg=cfg, baseline_repo=baseline_repo)

        # Today: 11 alternating points at 18:xx UTC → 10 pairs, all transitions → rate = 1.0.
        for i in range(11):
            room = "kitchen" if i % 2 == 0 else "hallway"
            await traj_repo.save_trajectory_point(
                PersonTrajectoryPoint(
                    identity_id=_IDENTITY,
                    ph_id=_GT,
                    observed_at=now.replace(hour=18, minute=0, second=0) + timedelta(minutes=i * 5),
                    room_name=room,
                    identity_confidence=0.9,
                )
            )

        signals = await worker.run_once(now=now)
        sundowning = [s for s in signals if s.signal_kind == "sundowning_index"]
        assert len(sundowning) == 1
        assert sundowning[0].severity in ("warning", "emergency")
        assert sundowning[0].z_score is not None

    @pytest.mark.asyncio
    async def test_quiet_rate_equals_median_no_signal(self) -> None:
        """Today's rate = historical median (1/19) → z = 0 < threshold → no signal."""
        now = datetime(2026, 4, 23, 20, 0, 0, tzinfo=UTC)

        hist_pts = []
        for day in range(1, 14):
            hist_pts.extend(_mk_hist_evening(_IDENTITY, _GT, day, now))

        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()
        baseline_repo = InMemoryBehaviorBaselineRepository(points=hist_pts)
        cfg = SignalConfig(
            tz_name="UTC",
            onset_consecutive_windows=1,
            sundowning_min_evening_minutes=5,
        )
        worker = DementiaSignalWorker(traj_repo, sig_repo, cfg=cfg, baseline_repo=baseline_repo)

        # Today: 20 points, only last switches room → 19 pairs, 1 transition.
        # today_rate = 1/19 == historical median → z = 0, below threshold.
        for i in range(20):
            room = "kitchen" if i < 19 else "hallway"
            await traj_repo.save_trajectory_point(
                PersonTrajectoryPoint(
                    identity_id=_IDENTITY,
                    ph_id=_GT,
                    observed_at=now.replace(hour=18, minute=0, second=0) + timedelta(minutes=i * 3),
                    room_name=room,
                    identity_confidence=0.9,
                )
            )

        signals = await worker.run_once(now=now)
        assert not any(s.signal_kind == "sundowning_index" for s in signals), (
            "Sundowning must not fire when today_rate equals historical median"
        )


class TestNighttimeMovementWithBaseline:
    """Task 1.2: nighttime movement uses daily_window_rates(22, 6) for nightly z-score."""

    @pytest.mark.asyncio
    async def test_elevated_tonight_fires(self) -> None:
        """13 prior nights (2 transitions each); tonight 8 transitions → z=inf → emergency."""
        # 12:30 AM UTC on April 24 — inside the April 23 night window.
        now = datetime(2026, 4, 24, 0, 30, 0, tzinfo=UTC)

        # Historical: nights of April 10-22 (days 2-14 ago from now).
        # Each: bedroom@22:00, kitchen@23:00, bedroom@23:30 → 2 transitions.
        hist_pts = []
        for day in range(2, 15):
            night_start = (now - timedelta(days=day)).replace(
                hour=22, minute=0, second=0, microsecond=0
            )
            for offset, room in [(0, "bedroom"), (60, "kitchen"), (90, "bedroom")]:
                hist_pts.append(
                    PersonTrajectoryPoint(
                        identity_id=_IDENTITY,
                        ph_id=_GT,
                        observed_at=night_start + timedelta(minutes=offset),
                        room_name=room,
                        identity_confidence=0.9,
                    )
                )

        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()
        baseline_repo = InMemoryBehaviorBaselineRepository(points=hist_pts)
        cfg = SignalConfig(
            tz_name="UTC",
            onset_consecutive_windows=1,
            nighttime_transition_threshold=3,
        )
        worker = DementiaSignalWorker(traj_repo, sig_repo, cfg=cfg, baseline_repo=baseline_repo)

        # Tonight: 9 points with 8 transitions at April 23 22:xx.
        tonight_base = datetime(2026, 4, 23, 22, 0, 0, tzinfo=UTC)
        rooms = [
            "bedroom",
            "kitchen",
            "bathroom",
            "bedroom",
            "kitchen",
            "bathroom",
            "bedroom",
            "kitchen",
            "bedroom",
        ]
        for i, room in enumerate(rooms):
            await traj_repo.save_trajectory_point(
                PersonTrajectoryPoint(
                    identity_id=_IDENTITY,
                    ph_id=_GT,
                    observed_at=tonight_base + timedelta(minutes=i * 10),
                    room_name=room,
                    identity_confidence=0.9,
                )
            )

        signals = await worker.run_once(now=now)
        night_sigs = [s for s in signals if s.signal_kind == "nighttime_movement"]
        assert night_sigs, "expected nighttime_movement signal"
        assert night_sigs[0].severity in ("warning", "emergency")

    @pytest.mark.asyncio
    async def test_below_flat_threshold_no_signal(self) -> None:
        """2 transitions tonight < nighttime_transition_threshold=3 → no signal."""
        now = datetime(2026, 4, 24, 0, 30, 0, tzinfo=UTC)
        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()
        cfg = SignalConfig(
            tz_name="UTC",
            onset_consecutive_windows=1,
            nighttime_transition_threshold=3,
        )
        worker = DementiaSignalWorker(traj_repo, sig_repo, cfg=cfg)

        for offset, room in [(0, "bedroom"), (30, "kitchen"), (60, "bedroom")]:
            await traj_repo.save_trajectory_point(
                PersonTrajectoryPoint(
                    identity_id=_IDENTITY,
                    ph_id=_GT,
                    observed_at=(
                        datetime(2026, 4, 23, 23, 0, 0, tzinfo=UTC) + timedelta(minutes=offset)
                    ),
                    room_name=room,
                    identity_confidence=0.9,
                )
            )

        signals = await worker.run_once(now=now)
        assert not any(s.signal_kind == "nighttime_movement" for s in signals)

    @pytest.mark.asyncio
    async def test_window_wraps_midnight_single_bucket(self) -> None:
        """Points at 23:30 and 01:30 UTC both land in the same April 23 night bucket."""
        p1 = PersonTrajectoryPoint(
            identity_id=_IDENTITY,
            ph_id=_GT,
            observed_at=datetime(2026, 4, 23, 23, 30, 0, tzinfo=UTC),
            room_name="bedroom",
            identity_confidence=0.9,
        )
        p2 = PersonTrajectoryPoint(
            identity_id=_IDENTITY,
            ph_id=_GT,
            observed_at=datetime(2026, 4, 24, 1, 30, 0, tzinfo=UTC),
            room_name="kitchen",
            identity_confidence=0.9,
        )
        repo = InMemoryBehaviorBaselineRepository(points=[p1, p2])
        since = datetime(2026, 4, 20, 0, 0, 0, tzinfo=UTC)
        until = datetime(2026, 4, 24, 6, 0, 0, tzinfo=UTC)

        samples = await repo.daily_window_rates(_IDENTITY, 22, 6, "UTC", since=since, until=until)

        assert len(samples) == 1, f"expected 1 night bucket, got {len(samples)}: {samples}"
        assert samples[0].local_date == date(2026, 4, 23)
        assert samples[0].transition_count == 1
        assert samples[0].observed_points == 2


class TestPacingWithBaseline:
    """Task 1.2: pacing uses pacing_window_rates for per-window-rate z-score."""

    @pytest.mark.asyncio
    async def test_z_score_non_null_and_consistent_with_robust_z(self) -> None:
        """Historical 30-min windows at 2/30 tpm; today at 0.5 tpm → z=inf, non-null."""
        now = datetime(2026, 4, 23, 14, 0, 0, tzinfo=UTC)
        since_30d = now - timedelta(days=30)

        # Historical: 10 dense 30-min windows aligned to since_30d.
        # Each: 15 points at 2-min intervals; rooms kitchen(13) + hallway + bedroom = 2 transitions.
        hist_pts = []
        for w in range(10):
            window_base = since_30d + timedelta(minutes=w * 30)
            for pt_idx in range(15):
                room = "kitchen" if pt_idx < 13 else "hallway" if pt_idx == 13 else "bedroom"
                hist_pts.append(
                    PersonTrajectoryPoint(
                        identity_id=_IDENTITY,
                        ph_id=_GT,
                        observed_at=window_base + timedelta(minutes=pt_idx * 2),
                        room_name=room,
                        identity_confidence=0.9,
                    )
                )

        traj_repo = InMemoryTrajectoryRepository()
        sig_repo = InMemoryDementiaSignalRepository()
        baseline_repo = InMemoryBehaviorBaselineRepository(points=hist_pts)
        cfg = SignalConfig(
            tz_name="UTC",
            onset_consecutive_windows=1,
            pacing_room_threshold=4,
            pacing_window_minutes=30,
        )
        worker = DementiaSignalWorker(traj_repo, sig_repo, cfg=cfg, baseline_repo=baseline_repo)

        # Today: 13 alternating points over ~24 min → 12 transitions, rate ≈ 0.5 tpm.
        for i in range(13):
            room = "kitchen" if i % 2 == 0 else "hallway"
            await traj_repo.save_trajectory_point(
                PersonTrajectoryPoint(
                    identity_id=_IDENTITY,
                    ph_id=_GT,
                    observed_at=now - timedelta(minutes=25 - i * 2),
                    room_name=room,
                    identity_confidence=0.9,
                )
            )

        signals = await worker.run_once(now=now)
        pacing = [s for s in signals if s.signal_kind == "pacing"]
        assert pacing, "expected pacing signal"
        assert pacing[0].z_score is not None

        # Hand-verify: baseline rates are all 2/30; robust_z(current_rate, [2/30]*10)
        # gives modified_z = inf (MAD=0, value ≠ median).
        current_rate = pacing[0].value
        baseline_samples = [2 / 30] * 10
        hand = _robust_z(current_rate, baseline_samples)
        import math

        assert math.isinf(hand.modified_z), "expected inf z with degenerate baseline"
        assert math.isinf(pacing[0].z_score)


class TestTimezoneEveningWindow:
    """Task 1.2: AT TIME ZONE -- 21:30 EST (02:30 UTC next day) lands in correct evening bucket."""

    @pytest.mark.asyncio
    async def test_new_york_21_30_est_lands_in_jan15_evening(self) -> None:
        # Jan 15 2026 21:30 EST = Jan 16 2026 02:30 UTC (UTC-5 in January).
        point_utc = datetime(2026, 1, 16, 2, 30, 0, tzinfo=UTC)
        pt = PersonTrajectoryPoint(
            identity_id=_IDENTITY,
            ph_id=_GT,
            observed_at=point_utc,
            room_name="kitchen",
            identity_confidence=0.9,
        )
        repo = InMemoryBehaviorBaselineRepository(points=[pt])
        since = datetime(2026, 1, 10, 0, 0, 0, tzinfo=UTC)
        until = datetime(2026, 1, 20, 0, 0, 0, tzinfo=UTC)

        samples = await repo.daily_window_rates(
            _IDENTITY, 17, 22, "America/New_York", since=since, until=until
        )

        assert len(samples) == 1, f"expected 1 evening sample, got {samples}"
        # Must resolve to Jan 15 local (EST), not Jan 16 UTC.
        assert samples[0].local_date == date(2026, 1, 15)
        assert samples[0].observed_points == 1


# ---------------------------------------------------------------------------
# Task 1.3 -- Hysteresis correctness: per-episode keying + per-run idempotency
# ---------------------------------------------------------------------------


def _still_dwell(
    room: str,
    entered_offset_minutes: float,
    still_minutes: int = 90,
    now: datetime = _NOW,
    ph_id: str = _GT,
) -> RoomDwell:
    """Open dwell with substantial still_seconds and below-floor motion energy."""
    entered_at = now - timedelta(minutes=entered_offset_minutes)
    return RoomDwell(
        dwell_id=str(uuid.uuid4()),
        identity_id=_IDENTITY,
        ph_id=ph_id,
        room_name=room,
        entered_at=entered_at,
        still_seconds=still_minutes * 60,
        min_motion_energy=0.001,  # below the 0.02 motion floor
    )


class TestHysteresisCorrectness:
    """Task 1.3 regression and correctness tests for SignalHysteresis."""

    # ------------------------------------------------------------------
    # Test 1 (regression): two qualifying stillness dwells in run 1 must
    # not satisfy min_consecutive=2 on their own.
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_two_dwells_in_run1_emit_nothing_run2_emits(self) -> None:
        """Bug regression: two qualifying dwells in one run must not satisfy debounce.

        With the old code a second dwell inside a single run incremented the
        shared (identity, kind) counter from 1 to 2 and triggered an immediate
        emission.  The correct behaviour: each dwell is its own episode; run 1
        advances both episode counters to 1 (below threshold=2) and emits nothing;
        run 2 advances both to 2 and emits.

        InMemoryTrajectoryRepository stores open dwells keyed by (identity_id, ph_id)
        so two concurrent open dwells must use distinct ph_ids.
        """
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                stillness_threshold_minutes=30,
                onset_consecutive_windows=2,
            )
        )
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=121))

        dwell_a = _still_dwell("kitchen", 100, still_minutes=60, ph_id="ph-a")
        dwell_b = _still_dwell("living_room", 90, still_minutes=60, ph_id="ph-b")
        for d in (dwell_a, dwell_b):
            await traj_repo.save_room_dwell(d)

        run1 = await worker.run_once(now=_NOW)
        stillness_run1 = [s for s in run1 if s.signal_kind == "stillness_anomaly"]
        assert len(stillness_run1) == 0, (
            f"Run 1 must not emit (debounce not yet satisfied), got {stillness_run1}"
        )

        run2 = await worker.run_once(now=_NOW + timedelta(minutes=2))
        stillness_run2 = [s for s in run2 if s.signal_kind == "stillness_anomaly"]
        assert len(stillness_run2) >= 1, "Run 2 must emit after two consecutive holds"

    # ------------------------------------------------------------------
    # Test 2: two concurrent distinct stillness episodes both emit (after debounce).
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_two_concurrent_episodes_both_emit(self) -> None:
        """Distinct episode keys track independently; both emit after debounce.

        Uses distinct ph_ids so both open dwells coexist in InMemoryTrajectoryRepository.
        """
        worker, traj_repo, _ = _make_worker(
            SignalConfig(
                stillness_threshold_minutes=30,
                onset_consecutive_windows=2,
            )
        )
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=121))

        dwell_a = _still_dwell("kitchen", 100, still_minutes=60, ph_id="ph-a")
        dwell_b = _still_dwell("living_room", 90, still_minutes=60, ph_id="ph-b")
        for d in (dwell_a, dwell_b):
            await traj_repo.save_room_dwell(d)

        await worker.run_once(now=_NOW)
        run2 = await worker.run_once(now=_NOW + timedelta(minutes=2))

        stillness = [s for s in run2 if s.signal_kind == "stillness_anomaly"]
        assert len(stillness) == 2, (
            f"Both episodes must emit on run 2, got {len(stillness)} stillness signal(s)"
        )
        rooms = {s.context["room_name"] for s in stillness}
        assert rooms == {"kitchen", "living_room"}

    # ------------------------------------------------------------------
    # Test 3: same episode_key evaluated twice within one run is idempotent;
    # subsequent run still requires the second consecutive hold.
    # ------------------------------------------------------------------

    def test_same_key_twice_in_run_is_idempotent(self) -> None:
        """Per-run idempotency: counter advances at most once per (identity, kind, ep) per run."""
        h = SignalHysteresis(min_consecutive=2)
        now = _NOW

        h.begin_run(now)
        r1 = h.should_emit(_IDENTITY, "stillness_anomaly", "warning", now, 60, episode_key="ep1")
        r2 = h.should_emit(_IDENTITY, "stillness_anomaly", "warning", now, 60, episode_key="ep1")

        # Both calls must return the same value (False: counter is 1, below threshold=2).
        assert r1 == r2 == False, (  # noqa: E712
            "First run with threshold=2 must return False; both calls must agree"
        )

        # Second run: counter advances from 1 to 2 -- must emit now.
        now2 = now + timedelta(minutes=2)
        h.begin_run(now2)
        r3 = h.should_emit(_IDENTITY, "stillness_anomaly", "warning", now2, 60, episode_key="ep1")
        assert r3 is True, "Second run must emit once onset threshold is reached"

    # ------------------------------------------------------------------
    # Test 4: severity escalation within an episode bypasses cooldown.
    # ------------------------------------------------------------------

    def test_severity_escalation_bypasses_cooldown(self) -> None:
        """Escalating severity within an open episode always emits regardless of cooldown."""
        h = SignalHysteresis(min_consecutive=1)
        now = _NOW

        # Episode fires on run 1 at "warning".
        h.begin_run(now)
        assert h.should_emit(_IDENTITY, "stillness_anomaly", "warning", now, 60, episode_key="ep")

        # Run 2, same severity: still in cooldown (2 minutes elapsed < 60-minute cooldown).
        now2 = now + timedelta(minutes=2)
        h.begin_run(now2)
        assert not h.should_emit(
            _IDENTITY, "stillness_anomaly", "warning", now2, 60, episode_key="ep"
        ), "Same severity within cooldown must not re-emit"

        # Run 2 (same run), escalated to "emergency": bypasses cooldown.
        assert h.should_emit(
            _IDENTITY, "stillness_anomaly", "emergency", now2, 60, episode_key="ep2"
        ), "New episode key with escalated severity must emit after fresh debounce"

        # For clarity: test escalation on the *same* episode key in a *later* run.
        now3 = now + timedelta(minutes=5)
        h.begin_run(now3)
        assert h.should_emit(
            _IDENTITY, "stillness_anomaly", "emergency", now3, 60, episode_key="ep"
        ), "Escalated severity on existing open episode must bypass cooldown"

    # ------------------------------------------------------------------
    # Test 5: eviction -- episode last emitted 49 h ago is forgotten after begin_run.
    # ------------------------------------------------------------------

    def test_eviction_after_48h(self) -> None:
        """Episode keys last emitted more than 48 h ago are evicted and require full debounce."""
        h = SignalHysteresis(min_consecutive=2)
        now = _NOW

        # Build up an emitting episode (2 consecutive runs).
        h.begin_run(now)
        h.should_emit(_IDENTITY, "stillness_anomaly", "warning", now, 60, episode_key="old")
        h.begin_run(now + timedelta(minutes=1))
        fired = h.should_emit(
            _IDENTITY,
            "stillness_anomaly",
            "warning",
            now + timedelta(minutes=1),
            60,
            episode_key="old",
        )
        assert fired, "Episode must have fired before eviction test"

        # Jump 49 hours: begin_run must evict the episode.
        now_evicted = now + timedelta(hours=49)
        h.begin_run(now_evicted)

        # First call after eviction: counter starts from 0 again, must not emit.
        r1 = h.should_emit(
            _IDENTITY, "stillness_anomaly", "warning", now_evicted, 60, episode_key="old"
        )
        assert not r1, "Evicted episode must require full debounce again (run 1 of 2)"

        # Second run: counter reaches threshold and emits.
        now_evicted2 = now_evicted + timedelta(minutes=2)
        h.begin_run(now_evicted2)
        r2 = h.should_emit(
            _IDENTITY, "stillness_anomaly", "warning", now_evicted2, 60, episode_key="old"
        )
        assert r2, "Evicted episode must emit again after full debounce"

    # ------------------------------------------------------------------
    # Test 6: cooldown across episodes is independent.
    # ------------------------------------------------------------------

    def test_episode_cooldown_independence(self) -> None:
        """Episode A's cooldown must not suppress a distinct new episode B."""
        h = SignalHysteresis(min_consecutive=1)
        now = _NOW

        # Episode A fires.
        h.begin_run(now)
        assert h.should_emit(_IDENTITY, "stillness_anomaly", "warning", now, 60, episode_key="A")

        # Episode B has a completely fresh key; must emit independently in the same run.
        assert h.should_emit(_IDENTITY, "stillness_anomaly", "warning", now, 60, episode_key="B")

        # Verify A is still in cooldown while B is also in cooldown (both fired this run).
        now2 = now + timedelta(minutes=5)
        h.begin_run(now2)
        assert not h.should_emit(
            _IDENTITY, "stillness_anomaly", "warning", now2, 60, episode_key="A"
        ), "Episode A must be in cooldown"
        assert not h.should_emit(
            _IDENTITY, "stillness_anomaly", "warning", now2, 60, episode_key="B"
        ), "Episode B must be in cooldown"
