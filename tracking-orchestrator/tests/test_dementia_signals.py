"""Unit tests for DementiaSignalWorker.

All tests use InMemory repositories — no DB required.
Each test seeds fixture trajectories/dwells and asserts the expected
signal kind and severity.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import PersonTrajectoryPoint, RoomDwell
from app.storage.base import InMemoryDementiaSignalRepository, InMemoryTrajectoryRepository
from app.trajectory.dementia_signals import DementiaSignalWorker, SignalConfig

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
        global_track_id=_GT,
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
        global_track_id=_GT,
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
        worker, traj_repo, _ = _make_worker(SignalConfig(pacing_room_threshold=4, onset_consecutive_windows=1))
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
        worker, traj_repo, _ = _make_worker(SignalConfig(pacing_room_threshold=10, onset_consecutive_windows=1))
        # Only 3 transitions.
        for room in ["kitchen", "hallway", "kitchen", "hallway"]:
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=5))

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "pacing" for s in signals)

    @pytest.mark.asyncio
    async def test_pacing_severity_emergency_at_high_rate(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(pacing_room_threshold=3, onset_consecutive_windows=1))
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
        worker, traj_repo, _ = _make_worker(SignalConfig(nighttime_transition_threshold=2, onset_consecutive_windows=1))
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
        worker, traj_repo, _ = _make_worker(SignalConfig(stillness_threshold_minutes=20, onset_consecutive_windows=1))
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=41))
        # Open dwell in kitchen for 40 minutes with high still_seconds.
        entered_at = _NOW - timedelta(minutes=40)
        open_dwell = RoomDwell(
            dwell_id=str(uuid.uuid4()),
            identity_id=_IDENTITY,
            global_track_id=_GT,
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
        worker, traj_repo, _ = _make_worker(SignalConfig(stillness_threshold_minutes=20, onset_consecutive_windows=1))
        await traj_repo.save_trajectory_point(_point("bedroom", offset_minutes=121))
        # Long dwell in bedroom — should be ignored.
        open_dwell = _dwell("bedroom", 120, duration_seconds=None, exited=False)
        await traj_repo.save_room_dwell(open_dwell)

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "stillness_anomaly" for s in signals)

    @pytest.mark.asyncio
    async def test_stillness_not_detected_below_threshold(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(stillness_threshold_minutes=30, onset_consecutive_windows=1))
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
        worker, traj_repo, _ = _make_worker(SignalConfig(absence_threshold_minutes=30, onset_consecutive_windows=1))
        # Last seen 90 minutes ago.
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=90))

        signals = await worker.run_once(now=_NOW)
        absence = [s for s in signals if s.signal_kind == "absence"]
        assert len(absence) == 1
        assert absence[0].value >= 90.0

    @pytest.mark.asyncio
    async def test_absence_severity_emergency_at_2h(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(absence_threshold_minutes=30, onset_consecutive_windows=1))
        # Last seen 130 minutes ago.
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=130))

        signals = await worker.run_once(now=_NOW)
        absence = [s for s in signals if s.signal_kind == "absence"]
        assert absence
        assert absence[0].severity == "emergency"

    @pytest.mark.asyncio
    async def test_no_absence_when_recently_seen(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(absence_threshold_minutes=30, onset_consecutive_windows=1))
        # Last seen 5 minutes ago.
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=5))

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "absence" for s in signals)

    @pytest.mark.asyncio
    async def test_no_absence_when_no_data(self):
        worker, _, _ = _make_worker(SignalConfig(absence_threshold_minutes=30, onset_consecutive_windows=1))
        # No trajectory data at all — should not emit absence.
        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "absence" for s in signals)


# ---------------------------------------------------------------------------
# run_once: persistence and multi-identity
# ---------------------------------------------------------------------------


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_signals_persisted_to_repo(self):
        worker, traj_repo, sig_repo = _make_worker(SignalConfig(pacing_room_threshold=3, onset_consecutive_windows=1))
        rooms = ["kitchen", "hallway"] * 4
        for i, room in enumerate(rooms):
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=20 - i * 2))

        await worker.run_once(now=_NOW)
        stored = await sig_repo.list_signals()
        assert len(stored) > 0

    @pytest.mark.asyncio
    async def test_multiple_identities_processed_independently(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(pacing_room_threshold=3, onset_consecutive_windows=1))
        # Identity A: pacing.
        rooms = ["kitchen", "hallway"] * 4
        for i, room in enumerate(rooms):
            pt = PersonTrajectoryPoint(
                identity_id="alice",
                global_track_id="gt-a",
                observed_at=_NOW - timedelta(minutes=20 - i * 2),
                room_name=room,
                identity_confidence=0.9,
            )
            await traj_repo.save_trajectory_point(pt)

        # Identity B: no pacing.
        pt_b = PersonTrajectoryPoint(
            identity_id="bob",
            global_track_id="gt-b",
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
        from app.storage.base import InMemoryBehaviorBaselineRepository

        self._inner = InMemoryBehaviorBaselineRepository(points, dwells)
        self.dwell_durations_calls = 0
        self.hourly_activity_calls = 0
        self.stillness_episodes_calls = 0

    async def dwell_durations(self, *args, **kwargs):
        self.dwell_durations_calls += 1
        return await self._inner.dwell_durations(*args, **kwargs)

    async def hourly_activity(self, *args, **kwargs):
        self.hourly_activity_calls += 1
        return await self._inner.hourly_activity(*args, **kwargs)

    async def stillness_episodes(self, *args, **kwargs):
        self.stillness_episodes_calls += 1
        return await self._inner.stillness_episodes(*args, **kwargs)


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
        from app.storage.base import InMemoryBehaviorBaselineRepository

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
        worker = DementiaSignalWorker(
            traj_repo, sig_repo, cfg=cfg, baseline_repo=spy
        )

        # First run populates cache.
        await worker.run_once(now=_NOW)
        calls_after_first = spy.dwell_durations_calls

        # Second run within TTL — should use cache.
        await worker.run_once(now=_NOW + timedelta(minutes=1))
        assert spy.dwell_durations_calls == calls_after_first  # no new calls
