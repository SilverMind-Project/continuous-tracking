"""PublishStage PH contract tests."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain import WorldFrameSnapshot
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.publish import PublishStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.transport.redis_streams import RedisStreamsTransport


def _live_config() -> LiveConfigHolder:
    return LiveConfigHolder(CameraRoomMap(), RoomPolygonMap())


def _make_snap(ph_id: str, identity_id: str | None = None) -> WorldFrameSnapshot:
    return WorldFrameSnapshot(
        ph_id=ph_id,
        camera_id="cam-1",
        frame_index=1,
        captured_at=datetime.now(UTC),
        floor_x_m=1.0,
        floor_y_m=2.0,
        floor_vx_m_s=0.1,
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
        room_name="living_room",
    )


def _make_ctx(snapshots: list[WorldFrameSnapshot]) -> FrameContext:
    from app.transport.redis_streams import FrameReady

    frame = FrameReady(
        camera_id="cam-1",
        minio_key="k",
        width=640,
        height=480,
        frame_index=1,
        capture_time_unix_ns=int(datetime.now(UTC).timestamp() * 1e9),
    )
    ctx = FrameContext(frame=frame, event_time=datetime.now(UTC), capture_time=datetime.now(UTC))
    ctx.world_snapshots = snapshots
    ctx.active_ph_ids = {s.ph_id for s in snapshots}
    return ctx


@pytest.mark.asyncio
async def test_publishes_identity_snapshots_for_known_and_unknown_phs():
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, live_config=_live_config())
    ctx = _make_ctx(
        [
            _make_snap("ph-1", "alice"),
            _make_snap("ph-2", None),
        ]
    )
    await stage.run(ctx)

    transport.publish_event.assert_called_once()
    kwargs = transport.publish_event.call_args.kwargs
    assert kwargs["identity_snapshots"] is not None
    snaps = kwargs["identity_snapshots"]
    assert len(snaps) == 2
    ph_ids = {s["ph_id"] for s in snaps}
    assert "ph-1" in ph_ids
    assert "ph-2" in ph_ids


@pytest.mark.asyncio
async def test_identities_built_from_snapshots():
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, live_config=_live_config())
    ctx = _make_ctx([_make_snap("ph-1", "alice")])
    await stage.run(ctx)

    kwargs = transport.publish_event.call_args.kwargs
    identities = kwargs["identities"]
    assert "ph-1" in identities
    assert identities["ph-1"][0] == "alice"


@pytest.mark.asyncio
async def test_wire_uses_ph_id_not_global_track_id():
    """T3 (R3): identity_snapshots dict uses ph_id, never global_track_id."""
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, live_config=_live_config())
    ctx = _make_ctx([_make_snap("ph-a", "alice")])
    await stage.run(ctx)

    kwargs = transport.publish_event.call_args.kwargs
    snaps = kwargs["identity_snapshots"]
    assert snaps is not None
    for snap in snaps:
        assert "ph_id" in snap, "identity_snapshot must use ph_id key"
        assert "global_track_id" not in snap, (
            "identity_snapshot must not use legacy global_track_id key"
        )
        assert snap["ph_id"] == "ph-a"


@pytest.mark.asyncio
async def test_identity_map_keyed_by_ph_id():
    """T3 (R3): identity map keyed by ph_id, not global_track_id."""
    transport = MagicMock(spec=RedisStreamsTransport)
    transport.publish_event = AsyncMock()

    stage = PublishStage(transport=transport, live_config=_live_config())
    ctx = _make_ctx([_make_snap("ph-b", "bob")])
    await stage.run(ctx)

    kwargs = transport.publish_event.call_args.kwargs
    identities = kwargs["identities"]
    assert "ph-b" in identities
    assert "global_track_id" not in identities
