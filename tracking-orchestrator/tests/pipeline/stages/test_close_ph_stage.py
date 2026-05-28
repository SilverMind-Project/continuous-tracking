"""WTR3: ClosePHStage tests.

Tests that ClosePHStage closes trajectory, motion energy, and posture state
by PH id when a previously-active PH disappears from ctx.active_ph_ids.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.trajectory import ClosePHStage
from app.trajectory.motion_energy import MotionEnergyTracker
from app.trajectory.posture import GlobalPostureTracker
from app.trajectory.trajectory_writer import TrajectoryWriter


def _make_ctx(active_ph_ids: set[str]) -> FrameContext:
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
    return ctx


@pytest.mark.asyncio
async def test_previously_active_ph_absent_this_frame_closes():
    """A PH active last frame but absent this frame triggers close."""
    traj_writer = MagicMock(spec=TrajectoryWriter)
    traj_writer.close_track = AsyncMock()
    motion = MotionEnergyTracker()
    posture = GlobalPostureTracker(required_consecutive=2)

    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        motion_energy_tracker=motion,
        posture_tracker=posture,
        prev_active_ph_ids={"ph-1", "ph-2"},
    )

    # This frame: only ph-1 is active, ph-2 disappeared.
    ctx = _make_ctx({"ph-1"})
    await stage.run(ctx)

    traj_writer.close_track.assert_called_once()
    call_args = traj_writer.close_track.call_args
    assert call_args[0][0] == "ph-2"
    # prev_active_ph_ids should now be {"ph-1"}.
    assert stage._prev_active_ph_ids == {"ph-1"}


@pytest.mark.asyncio
async def test_open_phs_remain_open():
    """PHs active in both frames do not trigger close."""
    traj_writer = MagicMock(spec=TrajectoryWriter)
    traj_writer.close_track = MagicMock()

    stage = ClosePHStage(
        trajectory_writer=traj_writer,
        prev_active_ph_ids={"ph-1"},
    )

    ctx = _make_ctx({"ph-1"})
    await stage.run(ctx)

    # No PH disappeared, so no close calls.
    traj_writer.close_track.assert_not_called()
    assert stage._prev_active_ph_ids == {"ph-1"}


@pytest.mark.asyncio
async def test_empty_active_set_does_not_crash():
    """Empty active_ph_ids is handled gracefully."""
    stage = ClosePHStage(
        trajectory_writer=MagicMock(spec=TrajectoryWriter),
        prev_active_ph_ids=set(),
    )
    ctx = _make_ctx(set())
    await stage.run(ctx)
    assert stage._prev_active_ph_ids == set()
