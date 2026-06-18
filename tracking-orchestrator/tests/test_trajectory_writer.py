"""Unit tests for TrajectoryWriter.

Tests cover:
- Writing trajectory points (one per call).
- First observation opens a new room dwell.
- Same room: no new dwell is opened.
- Room change: old dwell is closed, new dwell is opened.
- close_track closes the open dwell.
- Multiple tracks are tracked independently.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import FloorPoint
from app.storage.base import InMemoryTrajectoryRepository
from app.trajectory.trajectory_writer import TrajectoryWriter


@pytest.fixture()
def repo() -> InMemoryTrajectoryRepository:
    return InMemoryTrajectoryRepository()


@pytest.fixture()
def writer(repo: InMemoryTrajectoryRepository) -> TrajectoryWriter:
    return TrajectoryWriter(repo=repo)


_T0 = datetime(2026, 4, 22, 10, 0, 0, tzinfo=UTC)


async def test_writes_trajectory_point(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    point = await writer.write(
        identity_id="alice",
        ph_id="gt-001",
        room_name="kitchen",
        floor_point=FloorPoint(3500, 2100),
        captured_at=_T0,
        identity_confidence=0.9,
    )
    assert point.identity_id == "alice"
    assert point.ph_id == "gt-001"
    assert point.room_name == "kitchen"
    assert abs(point.ground_x - 3.5) < 1e-6
    assert abs(point.ground_y - 2.1) < 1e-6
    assert point.identity_confidence == pytest.approx(0.9)
    assert point.posture == "unknown"

    points = await repo.list_trajectory_points()
    assert len(points) == 1


async def test_floor_point_converted_to_meters(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    await writer.write(
        identity_id="alice",
        ph_id="gt-001",
        room_name="bedroom",
        floor_point=FloorPoint(x_mm=1000, y_mm=2000),
        captured_at=_T0,
    )
    points = await repo.list_trajectory_points()
    assert points[0].ground_x == pytest.approx(1.0)
    assert points[0].ground_y == pytest.approx(2.0)


async def test_trajectory_point_persists_confidence_fields(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    point = await writer.write(
        identity_id="alice",
        ph_id="gt-001",
        room_name="bedroom",
        floor_point=FloorPoint(x_mm=1000, y_mm=2000),
        captured_at=_T0,
        position_sigma_m=0.42,
        primary_camera_id="cam-primary",
        contributing_camera_count=3,
        footpoint_reliable=False,
    )

    assert point.position_sigma_m == pytest.approx(0.42)
    assert point.primary_camera_id == "cam-primary"
    assert point.contributing_camera_count == 3
    assert point.footpoint_reliable is False

    points = await repo.list_trajectory_points()
    assert points[0].position_sigma_m == pytest.approx(0.42)
    assert points[0].primary_camera_id == "cam-primary"
    assert points[0].contributing_camera_count == 3
    assert points[0].footpoint_reliable is False


async def test_first_observation_opens_dwell(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    await writer.write(
        identity_id="alice",
        ph_id="gt-001",
        room_name="kitchen",
        floor_point=FloorPoint(0, 0),
        captured_at=_T0,
        identity_confidence=0.85,
    )
    open_dwell = await repo.get_open_dwell("alice", "gt-001")
    assert open_dwell is not None
    assert open_dwell.room_name == "kitchen"
    assert open_dwell.exited_at is None
    assert open_dwell.entry_confidence == pytest.approx(0.85)


async def test_same_room_no_new_dwell(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    t1 = _T0
    t2 = _T0 + timedelta(seconds=5)
    await writer.write("alice", "gt-001", "kitchen", FloorPoint(0, 0), t1)
    await writer.write("alice", "gt-001", "kitchen", FloorPoint(100, 100), t2)

    dwells = await repo.list_room_dwells(identity_id="alice")
    # Only one dwell should exist and it must still be open.
    assert len(dwells) == 1
    assert dwells[0].exited_at is None


async def test_room_change_closes_old_opens_new(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    t1 = _T0
    t2 = _T0 + timedelta(seconds=30)

    await writer.write("alice", "gt-001", "kitchen", FloorPoint(0, 0), t1)
    await writer.write("alice", "gt-001", "hallway", FloorPoint(500, 0), t2)

    dwells = await repo.list_room_dwells(identity_id="alice")
    # One closed dwell (kitchen) + one open dwell (hallway).
    assert len(dwells) == 2

    closed = next(d for d in dwells if d.room_name == "kitchen")
    assert closed.exited_at == t2
    assert closed.duration_seconds == 30

    open_dwell = await repo.get_open_dwell("alice", "gt-001")
    assert open_dwell is not None
    assert open_dwell.room_name == "hallway"
    assert open_dwell.exited_at is None


async def test_close_track_closes_open_dwell(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    t_enter = _T0
    t_close = _T0 + timedelta(minutes=5)

    await writer.write("alice", "gt-001", "bathroom", FloorPoint(0, 0), t_enter)
    await writer.close_track("gt-001", closed_at=t_close)

    open_dwell = await repo.get_open_dwell("alice", "gt-001")
    assert open_dwell is None

    dwells = await repo.list_room_dwells(identity_id="alice")
    assert len(dwells) == 1
    assert dwells[0].exited_at == t_close
    assert dwells[0].duration_seconds == 300


async def test_close_track_noop_if_no_open_dwell(
    writer: TrajectoryWriter,
) -> None:
    # Should not raise; idempotent for unknown ph_id.
    await writer.close_track("gt-nonexistent", closed_at=_T0)


async def test_multiple_tracks_independent(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    t1 = _T0
    t2 = _T0 + timedelta(seconds=10)

    await writer.write("alice", "gt-001", "kitchen", FloorPoint(0, 0), t1)
    await writer.write("bob", "gt-002", "bedroom", FloorPoint(100, 200), t2)
    await writer.write("alice", "gt-001", "hallway", FloorPoint(50, 0), t2)

    points = await repo.list_trajectory_points()
    assert len(points) == 3

    alice_dwells = await repo.list_room_dwells(identity_id="alice")
    bob_dwells = await repo.list_room_dwells(identity_id="bob")
    assert len(alice_dwells) == 2  # kitchen (closed) + hallway (open)
    assert len(bob_dwells) == 1


async def test_multiple_trajectory_points_stored(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    for i in range(5):
        await writer.write(
            "alice",
            "gt-001",
            "kitchen",
            FloorPoint(i * 100, i * 50),
            _T0 + timedelta(seconds=i),
        )

    points = await repo.list_trajectory_points(identity_id="alice")
    assert len(points) == 5


async def test_start_segment_closes_prior_open_dwell(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    """Reviving a PH must finalize the prior open dwell, not orphan it.

    Regression: start_segment previously discarded the open dwell without
    writing exited_at, leaving a dangling row that the stillness detector later
    read as immobility (the phantom-signal storm).
    """
    t_enter = _T0
    t_revive = _T0 + timedelta(minutes=3)

    await writer.write("alice", "gt-001", "kitchen", FloorPoint(0, 0), t_enter)
    await writer.start_segment(
        ph_id="gt-001", identity_id="alice", room_name="hallway", entered_at=t_revive
    )

    # Exactly one open dwell (the new segment); the old one is closed, not leaked.
    all_dwells = await repo.list_room_dwells(identity_id="alice")
    open_dwells = [d for d in all_dwells if d.exited_at is None]
    assert len(open_dwells) == 1
    assert open_dwells[0].room_name == "hallway"

    closed = [d for d in all_dwells if d.exited_at is not None]
    assert len(closed) == 1
    assert closed[0].room_name == "kitchen"
    assert closed[0].exited_at is not None


async def test_reconcile_open_dwells_closes_danglers(
    writer: TrajectoryWriter,
    repo: InMemoryTrajectoryRepository,
) -> None:
    """Restart reconciliation closes dwells left open by a previous lifecycle."""
    # Simulate a dwell open since a previous process, with a last observation.
    await writer.write("alice", "gt-001", "kitchen", FloorPoint(0, 0), _T0)
    last_obs = _T0 + timedelta(minutes=10)
    await writer.write("alice", "gt-001", "kitchen", FloorPoint(0, 0), last_obs)
    assert await repo.get_open_dwell("alice", "gt-001") is not None

    closed_count = await writer.reconcile_open_dwells(closed_at=_T0 + timedelta(hours=2))
    assert closed_count == 1
    assert await repo.get_open_dwell("alice", "gt-001") is None

    dwells = await repo.list_room_dwells(identity_id="alice")
    assert len(dwells) == 1
    # Exit stamped at the last observed point, not the (much later) closed_at.
    assert dwells[0].exited_at == last_obs
    assert dwells[0].duration_seconds == 600
