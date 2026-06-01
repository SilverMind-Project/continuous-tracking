"""ClosePHStage presence and dwell event emission tests."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain import WorldFrameSnapshot
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.trajectory import ClosePHStage
from app.trajectory.trajectory_writer import TrajectoryWriter


def _make_snap(
    ph_id: str,
    identity_id: str | None = None,
    camera_id: str = "cam-1",
    room_name: str = "living_room",
) -> WorldFrameSnapshot:
    return WorldFrameSnapshot(
        ph_id=ph_id,
        camera_id=camera_id,
        frame_index=1,
        captured_at=datetime.now(UTC),
        floor_x_m=1.0,
        floor_y_m=2.0,
        floor_vx_m_s=0.0,
        floor_vy_m_s=0.0,
        position_sigma_m=0.05,
        identity_id=identity_id,
        identity_confidence=0.9 if identity_id else 0.0,
        posterior_entropy=0.5,
        direct_face_evidence=bool(identity_id),
        bbox=None,
        detection_confidence=0.95,
        height_m=1.7,
        room_id=1,
        room_name=room_name,
    )


def _make_ctx(
    active_ph_ids: set[str],
    snapshots: list[WorldFrameSnapshot] | None = None,
) -> FrameContext:
    from app.transport.redis_streams import FrameReady

    frame = FrameReady(
        camera_id="cam-1",
        minio_key="test/key",
        width=640,
        height=480,
        frame_index=1,
        capture_time_unix_ns=int(datetime.now(UTC).timestamp() * 1e9),
    )
    ctx = FrameContext(
        frame=frame,
        event_time=datetime.now(UTC),
        capture_time=datetime.now(UTC),
    )
    ctx.active_ph_ids = active_ph_ids
    ctx.world_snapshots = snapshots or []
    return ctx


# ---------------------------------------------------------------------------
# Presence: appeared
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_presence_appeared_emitted_for_new_ph():
    """A new PH entering the active set emits exactly one appeared event."""
    presence = MagicMock()
    presence.publish_appeared = AsyncMock()
    presence.publish_disappeared = AsyncMock()

    snap = _make_snap("ph-1", identity_id="alice", room_name="kitchen")

    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids=set(),
        presence_publisher=presence,
    )

    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)

    presence.publish_appeared.assert_called_once()
    call_kwargs = presence.publish_appeared.call_args.kwargs
    assert call_kwargs["ph_id"] == "ph-1"
    assert call_kwargs["identity_id"] == "alice"
    assert call_kwargs["room_name"] == "kitchen"


@pytest.mark.asyncio
async def test_presence_appeared_not_emitted_for_existing_ph():
    """An already-seen PH does not emit another appeared event."""
    presence = MagicMock()
    presence.publish_appeared = AsyncMock()
    presence.publish_disappeared = AsyncMock()

    snap = _make_snap("ph-1", identity_id="alice")

    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids={"ph-1"},
        presence_publisher=presence,
    )
    stage._seen_ph_ids = {"ph-1"}  # Pre-seed as already seen.

    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)

    presence.publish_appeared.assert_not_called()


# ---------------------------------------------------------------------------
# Presence: disappeared
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_presence_disappeared_emitted_for_terminated_ph():
    """A terminated PH emits exactly one disappeared event."""
    presence = MagicMock()
    presence.publish_appeared = AsyncMock()
    presence.publish_disappeared = AsyncMock()

    traj_writer = MagicMock(spec=TrajectoryWriter)
    traj_writer.close_track = AsyncMock()

    snap = _make_snap("ph-1", identity_id="alice")

    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        prev_active_ph_ids={"ph-1", "ph-2"},
        presence_publisher=presence,
    )
    stage._seen_ph_ids = {"ph-1", "ph-2"}
    stage._last_identity_by_ph = {"ph-2": "bob"}
    stage._last_room_by_ph = {"ph-2": "kitchen"}

    # ph-2 disappeared this frame.
    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)

    presence.publish_disappeared.assert_called_once()
    call_kwargs = presence.publish_disappeared.call_args.kwargs
    assert call_kwargs["ph_id"] == "ph-2"
    assert call_kwargs["identity_id"] == "bob"
    assert call_kwargs["room_name"] == "kitchen"


@pytest.mark.asyncio
async def test_presence_disappeared_with_unknown_identity():
    """A terminated PH with no identity still emits disappeared."""
    presence = MagicMock()
    presence.publish_appeared = AsyncMock()
    presence.publish_disappeared = AsyncMock()

    traj_writer = MagicMock(spec=TrajectoryWriter)
    traj_writer.close_track = AsyncMock()

    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        prev_active_ph_ids={"ph-1"},
        presence_publisher=presence,
    )
    stage._seen_ph_ids = {"ph-1"}
    stage._last_identity_by_ph = {"ph-1": None}
    stage._last_room_by_ph = {"ph-1": ""}

    ctx = _make_ctx(set(), snapshots=[])
    await stage.run(ctx)

    presence.publish_disappeared.assert_called_once()
    call_kwargs = presence.publish_disappeared.call_args.kwargs
    assert call_kwargs["identity_id"] is None


# ---------------------------------------------------------------------------
# Dwell: started
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dwell_started_emitted_for_new_ph_with_room():
    """A new PH with a room emits a dwell-started event."""
    dwell = MagicMock()
    dwell.publish_started = AsyncMock()
    dwell.publish_ended = AsyncMock()

    snap = _make_snap("ph-1", identity_id="alice", room_name="kitchen")

    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids=set(),
        dwell_publisher=dwell,
    )

    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)

    dwell.publish_started.assert_called_once()
    call_kwargs = dwell.publish_started.call_args.kwargs
    assert call_kwargs["ph_id"] == "ph-1"
    assert call_kwargs["room_name"] == "kitchen"


@pytest.mark.asyncio
async def test_dwell_started_not_emitted_for_new_ph_without_room():
    """A new PH without a room name does not emit dwell-started."""
    dwell = MagicMock()
    dwell.publish_started = AsyncMock()
    dwell.publish_ended = AsyncMock()

    snap = _make_snap("ph-1", room_name="")  # No room.

    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids=set(),
        dwell_publisher=dwell,
    )

    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)

    dwell.publish_started.assert_not_called()


@pytest.mark.asyncio
async def test_dwell_ended_and_started_on_room_change():
    """Room change emits dwell-ended for old room AND dwell-started for new room."""
    dwell = MagicMock()
    dwell.publish_started = AsyncMock()
    dwell.publish_ended = AsyncMock()

    snap = _make_snap("ph-1", identity_id="alice", room_name="bathroom")

    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids={"ph-1"},
        dwell_publisher=dwell,
    )
    stage._seen_ph_ids = {"ph-1"}
    stage._last_room_by_ph = {"ph-1": "kitchen"}  # Previously in kitchen.
    # Set room entered at 10 seconds ago so duration is non-zero.
    from datetime import timedelta

    entered = _make_ctx({"ph-1"}, snapshots=[snap]).event_time - timedelta(seconds=10)
    stage._room_entered_at = {"ph-1": entered}

    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)

    # Room change: kitchen → bathroom emits dwell-ended(kitchen) + dwell-started(bathroom).
    dwell.publish_ended.assert_called_once()
    ended_kwargs = dwell.publish_ended.call_args.kwargs
    assert ended_kwargs["room_name"] == "kitchen"
    assert ended_kwargs["duration_s"] >= 10

    dwell.publish_started.assert_called_once()
    started_kwargs = dwell.publish_started.call_args.kwargs
    assert started_kwargs["room_name"] == "bathroom"


# ---------------------------------------------------------------------------
# Dwell: ended
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dwell_ended_emitted_for_terminated_ph():
    """A terminated PH emits dwell-ended."""
    dwell = MagicMock()
    dwell.publish_started = AsyncMock()
    dwell.publish_ended = AsyncMock()

    traj_writer = MagicMock(spec=TrajectoryWriter)
    traj_writer.close_track = AsyncMock()

    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        prev_active_ph_ids={"ph-1"},
        dwell_publisher=dwell,
    )
    stage._seen_ph_ids = {"ph-1"}
    stage._last_identity_by_ph = {"ph-1": "alice"}
    stage._last_room_by_ph = {"ph-1": "kitchen"}

    ctx = _make_ctx(set(), snapshots=[])
    await stage.run(ctx)

    dwell.publish_ended.assert_called_once()
    call_kwargs = dwell.publish_ended.call_args.kwargs
    assert call_kwargs["ph_id"] == "ph-1"
    assert call_kwargs["room_name"] == "kitchen"


# ---------------------------------------------------------------------------
# Presence: deferred until min_observations_to_publish
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_presence_appeared_deferred_until_snapshot_available():
    """Presence-appeared is NOT emitted until the PH has a snapshot.

    The snapshot proves min_observations_to_publish has been met.
    """
    presence = MagicMock()
    presence.publish_appeared = AsyncMock()
    presence.publish_disappeared = AsyncMock()

    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids=set(),
        presence_publisher=presence,
    )

    # Frame 1: PH enters active_ph_ids but has NO snapshot.
    ctx1 = _make_ctx({"ph-1"}, snapshots=[])  # No snapshots.
    await stage.run(ctx1)

    # Should NOT emit presence-appeared - PH not yet confirmed.
    presence.publish_appeared.assert_not_called()
    # PH should NOT be in _seen_ph_ids (keeps retrying each frame).
    assert "ph-1" not in stage._seen_ph_ids

    # Frame 2: PH now has enough observations - snapshot available.
    snap = _make_snap("ph-1", identity_id="alice", room_name="kitchen")
    ctx2 = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx2)

    # Now presence-appeared should emit.
    presence.publish_appeared.assert_called_once()
    call_kwargs = presence.publish_appeared.call_args.kwargs
    assert call_kwargs["ph_id"] == "ph-1"
    assert call_kwargs["identity_id"] == "alice"
    assert call_kwargs["room_name"] == "kitchen"
    # PH should now be in _seen_ph_ids.
    assert "ph-1" in stage._seen_ph_ids


# ---------------------------------------------------------------------------
# No per-frame emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_per_frame_presence_emission():
    """Presence events are NOT emitted when no PHs appear or disappear."""
    presence = MagicMock()
    presence.publish_appeared = AsyncMock()
    presence.publish_disappeared = AsyncMock()

    snap = _make_snap("ph-1", identity_id="alice")

    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids={"ph-1"},
        presence_publisher=presence,
    )
    stage._seen_ph_ids = {"ph-1"}
    stage._last_room_by_ph = {"ph-1": "kitchen"}

    # Same PH, no change.
    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)

    presence.publish_appeared.assert_not_called()
    presence.publish_disappeared.assert_not_called()


@pytest.mark.asyncio
async def test_no_per_frame_dwell_emission():
    """Dwell events are NOT emitted when no PHs change rooms or terminate."""
    dwell = MagicMock()
    dwell.publish_started = AsyncMock()
    dwell.publish_ended = AsyncMock()

    snap = _make_snap("ph-1", identity_id="alice", room_name="kitchen")

    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids={"ph-1"},
        dwell_publisher=dwell,
    )
    stage._seen_ph_ids = {"ph-1"}
    stage._last_room_by_ph = {"ph-1": "kitchen"}

    # Same PH, same room, no change.
    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)

    dwell.publish_started.assert_not_called()
    dwell.publish_ended.assert_not_called()


# ---------------------------------------------------------------------------
# Graceful: no publisher configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_presence_publisher_does_not_crash():
    """ClosePHStage runs normally without a presence publisher."""
    traj_writer = MagicMock(spec=TrajectoryWriter)
    traj_writer.close_track = AsyncMock()

    snap = _make_snap("ph-1")

    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        prev_active_ph_ids=set(),
        presence_publisher=None,
    )

    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)
    # Should not crash.


@pytest.mark.asyncio
async def test_no_dwell_publisher_does_not_crash():
    """ClosePHStage runs normally without a dwell publisher."""
    traj_writer = MagicMock(spec=TrajectoryWriter)
    traj_writer.close_track = AsyncMock()

    snap = _make_snap("ph-1")

    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        prev_active_ph_ids=set(),
        dwell_publisher=None,
    )

    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)
    # Should not crash.
