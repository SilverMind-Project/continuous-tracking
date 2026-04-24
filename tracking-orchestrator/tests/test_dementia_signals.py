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
        worker, traj_repo, _ = _make_worker(SignalConfig(pacing_room_threshold=4))
        # 10 alternating room transitions in 20 minutes.
        rooms = ["kitchen", "hallway"] * 5
        for i, room in enumerate(rooms):
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=20 - i * 2))

        signals = await worker.run_once(now=_NOW)
        pacing = [s for s in signals if s.signal_kind == "pacing"]
        assert len(pacing) == 1
        assert pacing[0].identity_id == _IDENTITY
        assert pacing[0].value >= 4

    @pytest.mark.asyncio
    async def test_pacing_not_detected_below_threshold(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(pacing_room_threshold=10))
        # Only 3 transitions.
        for room in ["kitchen", "hallway", "kitchen", "hallway"]:
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=5))

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "pacing" for s in signals)

    @pytest.mark.asyncio
    async def test_pacing_severity_emergency_at_high_rate(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(pacing_room_threshold=3))
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
        worker, traj_repo, _ = _make_worker()
        now = datetime(2026, 4, 23, 20, 0, 0, tzinfo=UTC)  # 20:00

        # Afternoon (12-17): 2 transitions across 6 points (low rate).
        afternoon_rooms = ["kitchen", "kitchen", "living_room", "living_room", "kitchen", "kitchen"]
        for i, room in enumerate(afternoon_rooms):
            await traj_repo.save_trajectory_point(
                _point(room, offset_minutes=0, now=datetime(2026, 4, 23, 14, i * 5, 0, tzinfo=UTC))
            )

        # Evening (17-22): 8 transitions across 10 points (high rate — 4x afternoon).
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
        assert sundowning[0].value > 1.5

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
        worker, traj_repo, _ = _make_worker()

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
        worker, traj_repo, _ = _make_worker(SignalConfig(nighttime_transition_threshold=2))
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
        worker, traj_repo, _ = _make_worker(SignalConfig(stillness_threshold_minutes=20))
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=41))
        # Open dwell in kitchen for 40 minutes.
        open_dwell = _dwell("kitchen", 40, duration_seconds=None, exited=False)
        await traj_repo.save_room_dwell(open_dwell)

        signals = await worker.run_once(now=_NOW)
        stillness = [s for s in signals if s.signal_kind == "stillness_anomaly"]
        assert len(stillness) == 1
        assert stillness[0].value >= 20 * 60

    @pytest.mark.asyncio
    async def test_stillness_not_detected_in_bedroom(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(stillness_threshold_minutes=20))
        await traj_repo.save_trajectory_point(_point("bedroom", offset_minutes=121))
        # Long dwell in bedroom — should be ignored.
        open_dwell = _dwell("bedroom", 120, duration_seconds=None, exited=False)
        await traj_repo.save_room_dwell(open_dwell)

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "stillness_anomaly" for s in signals)

    @pytest.mark.asyncio
    async def test_stillness_not_detected_below_threshold(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(stillness_threshold_minutes=30))
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
        worker, traj_repo, _ = _make_worker(SignalConfig(absence_threshold_minutes=30))
        # Last seen 90 minutes ago.
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=90))

        signals = await worker.run_once(now=_NOW)
        absence = [s for s in signals if s.signal_kind == "absence"]
        assert len(absence) == 1
        assert absence[0].value >= 90.0

    @pytest.mark.asyncio
    async def test_absence_severity_emergency_at_2h(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(absence_threshold_minutes=30))
        # Last seen 130 minutes ago.
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=130))

        signals = await worker.run_once(now=_NOW)
        absence = [s for s in signals if s.signal_kind == "absence"]
        assert absence
        assert absence[0].severity == "emergency"

    @pytest.mark.asyncio
    async def test_no_absence_when_recently_seen(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(absence_threshold_minutes=30))
        # Last seen 5 minutes ago.
        await traj_repo.save_trajectory_point(_point("kitchen", offset_minutes=5))

        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "absence" for s in signals)

    @pytest.mark.asyncio
    async def test_no_absence_when_no_data(self):
        worker, _, _ = _make_worker(SignalConfig(absence_threshold_minutes=30))
        # No trajectory data at all — should not emit absence.
        signals = await worker.run_once(now=_NOW)
        assert not any(s.signal_kind == "absence" for s in signals)


# ---------------------------------------------------------------------------
# run_once: persistence and multi-identity
# ---------------------------------------------------------------------------


class TestRunOnce:
    @pytest.mark.asyncio
    async def test_signals_persisted_to_repo(self):
        worker, traj_repo, sig_repo = _make_worker(SignalConfig(pacing_room_threshold=3))
        rooms = ["kitchen", "hallway"] * 4
        for i, room in enumerate(rooms):
            await traj_repo.save_trajectory_point(_point(room, offset_minutes=20 - i * 2))

        await worker.run_once(now=_NOW)
        stored = await sig_repo.list_signals()
        assert len(stored) > 0

    @pytest.mark.asyncio
    async def test_multiple_identities_processed_independently(self):
        worker, traj_repo, _ = _make_worker(SignalConfig(pacing_room_threshold=3))
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
