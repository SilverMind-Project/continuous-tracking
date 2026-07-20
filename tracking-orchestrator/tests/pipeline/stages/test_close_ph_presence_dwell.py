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
# State tracking lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ph_state_tracked_on_snapshot():
    """A new PH with a snapshot gets its identity and room tracked."""
    traj_writer = MagicMock(spec=TrajectoryWriter)
    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        prev_active_ph_ids=set(),
    )
    snap = _make_snap("ph-1", identity_id="alice", room_name="kitchen")
    ctx = _make_ctx({"ph-1"}, snapshots=[snap])
    await stage.run(ctx)
    
    assert "ph-1" in stage._seen_ph_ids
    assert stage._last_identity_by_ph.get("ph-1") == "alice"
    assert stage._last_room_by_ph.get("ph-1") == "kitchen"

@pytest.mark.asyncio
async def test_ph_state_cleaned_on_terminate():
    """A terminated PH has its state cleaned up."""
    traj_writer = MagicMock(spec=TrajectoryWriter)
    traj_writer.close_track = AsyncMock()
    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        prev_active_ph_ids={"ph-1"},
    )
    stage._seen_ph_ids = {"ph-1"}
    stage._last_identity_by_ph = {"ph-1": "alice"}
    stage._last_room_by_ph = {"ph-1": "kitchen"}

    ctx = _make_ctx(set(), snapshots=[])
    await stage.run(ctx)
    
    assert "ph-1" not in stage._seen_ph_ids
    assert "ph-1" not in stage._last_identity_by_ph
    assert "ph-1" not in stage._last_room_by_ph

