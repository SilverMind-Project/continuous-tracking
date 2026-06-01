"""PublishStage per-camera throttle tests."""

from __future__ import annotations

import time as _time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.publish import PublishStage
from app.services.camera_room_map import CameraRoomMap
from app.transport.redis_streams import RedisStreamsTransport


def _make_ctx(camera_id: str = "cam-1") -> FrameContext:
    from app.transport.redis_streams import FrameReady

    frame = FrameReady(
        camera_id=camera_id,
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
    ctx.world_snapshots = []
    ctx.active_ph_ids = set()
    ctx.raw_detections = []
    ctx.domain_detections = []
    ctx.outcome_decisions = []
    return ctx


def _room_map() -> CameraRoomMap:
    return CameraRoomMap()


@pytest.mark.asyncio
async def test_publishes_first_frame():
    """First frame for a camera is always published."""
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, camera_room_map=_room_map(), live_publish_max_hz=3.0)
    ctx = _make_ctx("cam-1")
    await stage.run(ctx)

    transport.publish_event.assert_called_once()


@pytest.mark.asyncio
async def test_throttle_skips_intermediate_frames():
    """Frames within the throttle window are skipped."""
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, camera_room_map=_room_map(), live_publish_max_hz=3.0)
    # Publish first frame.
    ctx1 = _make_ctx("cam-1")
    await stage.run(ctx1)
    assert transport.publish_event.call_count == 1

    # Immediate second frame should be throttled.
    ctx2 = _make_ctx("cam-1")
    ctx2.frame.frame_index = 2
    await stage.run(ctx2)
    # Call count should still be 1.
    assert transport.publish_event.call_count == 1


@pytest.mark.asyncio
async def test_publishes_after_throttle_window_expires():
    """After the throttle window expires, the next frame is published."""
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, camera_room_map=_room_map(), live_publish_max_hz=10.0)
    # Publish first frame.
    ctx1 = _make_ctx("cam-1")
    await stage.run(ctx1)

    # Advance the per-camera timestamp past the throttle interval (0.1s for 10 Hz).
    stage._last_publish_time["cam-1"] = _time.monotonic() - 0.2

    # Next frame should publish.
    ctx2 = _make_ctx("cam-1")
    ctx2.frame.frame_index = 2
    await stage.run(ctx2)
    assert transport.publish_event.call_count == 2


@pytest.mark.asyncio
async def test_throttle_per_camera_independent():
    """Throttle is per camera; publishing on cam-2 does not block cam-1."""
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, camera_room_map=_room_map(), live_publish_max_hz=3.0)

    # First frame for cam-1: published.
    await stage.run(_make_ctx("cam-1"))
    # First frame for cam-2: also published (different camera).
    await stage.run(_make_ctx("cam-2"))
    assert transport.publish_event.call_count == 2


@pytest.mark.asyncio
async def test_zero_hz_disables_throttle():
    """live_publish_max_hz=0 disables throttling (every frame published)."""
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, camera_room_map=_room_map(), live_publish_max_hz=0.0)

    # Two frames back-to-back, both should publish.
    await stage.run(_make_ctx("cam-1"))
    await stage.run(_make_ctx("cam-1"))
    assert transport.publish_event.call_count == 2


@pytest.mark.asyncio
async def test_default_max_hz_is_3():
    """Default live_publish_max_hz is 3.0."""
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, camera_room_map=_room_map())
    assert stage._live_publish_max_hz == 3.0
    assert stage._throttle_interval_s == pytest.approx(1.0 / 3.0)
