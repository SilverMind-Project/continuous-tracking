"""Tests for FaceIdentityStage per-camera policy driven by role and face_id_enabled."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from cts_contracts import CameraRole

from app.domain import BoundingBox, CameraProfile, Detection
from app.inference.face_id_client import CropQueryResult, CropQueryStatus
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.face_identity import FaceIdentityStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_profile_map import CameraProfileMap
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.storage.face_stats import InMemoryFaceObservationStatsRepository
from app.transport.redis_streams import FrameReady


def _make_context(camera_id: str, frame_index: int = 1) -> FrameContext:
    now = datetime.now(UTC)
    frame = FrameReady(
        camera_id=camera_id,
        frame_index=frame_index,
        minio_key=f"frames/{camera_id}/{frame_index}.jpg",
        width=640,
        height=480,
        capture_time_unix_ns=int(now.timestamp() * 1e9),
    )
    bbox = BoundingBox(x_min=100, y_min=50, x_max=220, y_max=350)
    det = Detection(
        detection_id=f"det-{frame_index}",
        camera_id=camera_id,
        bbox=bbox,
        confidence=0.95,
        embedding=None,
        capture_time=now,
        event_time=now,
    )
    crop = np.zeros((300, 120, 3), dtype=np.uint8)
    ctx = FrameContext(frame=frame, event_time=now, capture_time=now)
    ctx.image = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx.effective_width = 640
    ctx.effective_height = 480
    ctx.domain_detections = [det]
    ctx.crops = [crop]
    return ctx


def _make_stage(
    profiles: list[CameraProfile],
    *,
    cooldown_s: float = 5.0,
    identification_point_cooldown_s: float = 1.0,
) -> tuple[FaceIdentityStage, MagicMock, CameraProfileMap]:
    client = MagicMock()
    client.identify_crops = AsyncMock(
        return_value=[
            CropQueryResult(
                crop_index=0,
                status=CropQueryStatus.NO_FACE,
                faces=[],
                crop_width_px=120,
                crop_height_px=300,
            )
        ]
    )
    pmap = CameraProfileMap(profiles)
    holder = LiveConfigHolder(
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
        camera_profile_map=pmap,
    )
    stage = FaceIdentityStage(
        face_stats_repo=InMemoryFaceObservationStatsRepository(),
        face_id_client=client,
        face_id_cooldown_s=cooldown_s,
        identification_point_cooldown_s=identification_point_cooldown_s,
        live_config=holder,
    )
    return stage, client, pmap


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [CameraRole.FACE_CAPABLE, CameraRole.MIXED, CameraRole.SURVEILLANCE],
)
async def test_face_id_disabled_makes_no_face_calls_in_any_role(role: CameraRole) -> None:
    stage, client, _ = _make_stage(
        [CameraProfile(camera_id="cam-off", role=role, face_id_enabled=False)]
    )

    policy = stage.face_id_policy_for("cam-off")
    assert policy.enabled is False
    assert policy.role is role

    await stage.run(_make_context("cam-off"))
    client.identify_crops.assert_not_awaited()


@pytest.mark.asyncio
async def test_face_capable_uses_short_identification_point_cooldown() -> None:
    stage, client, _ = _make_stage(
        [CameraProfile(camera_id="id-cam", role=CameraRole.FACE_CAPABLE, face_id_enabled=True)],
        cooldown_s=5.0,
        identification_point_cooldown_s=1.0,
    )

    policy = stage.face_id_policy_for("id-cam")
    assert policy.enabled is True
    assert policy.cooldown_s == 1.0
    assert policy.role is CameraRole.FACE_CAPABLE

    # First frame always queries.
    await stage.run(_make_context("id-cam", frame_index=1))
    assert client.identify_crops.await_count == 1

    # Simulate 0.5 s elapsed (< 1.0 s cooldown): skipped.
    track_id = next(iter(stage._last_face_id_by_tracklet))
    stage._last_face_id_by_tracklet[track_id] = datetime.now(UTC) - timedelta(seconds=0.5)
    await stage.run(_make_context("id-cam", frame_index=2))
    assert client.identify_crops.await_count == 1

    # Simulate 1.2 s elapsed (>= 1.0 s cooldown, still < 5.0 s standard): queries again.
    stage._last_face_id_by_tracklet[track_id] = datetime.now(UTC) - timedelta(seconds=1.2)
    await stage.run(_make_context("id-cam", frame_index=3))
    assert client.identify_crops.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [CameraRole.MIXED, CameraRole.SURVEILLANCE])
async def test_mixed_and_surveillance_reproduce_standard_cooldown(role: CameraRole) -> None:
    stage, client, _ = _make_stage(
        [CameraProfile(camera_id="room-cam", role=role, face_id_enabled=True)],
        cooldown_s=5.0,
        identification_point_cooldown_s=1.0,
    )

    policy = stage.face_id_policy_for("room-cam")
    assert policy.enabled is True
    assert policy.cooldown_s == 5.0
    assert policy.role is role

    await stage.run(_make_context("room-cam", frame_index=1))
    assert client.identify_crops.await_count == 1

    # 2.0 s elapsed: still within 5.0 s cooldown -> skipped.
    track_id = next(iter(stage._last_face_id_by_tracklet))
    stage._last_face_id_by_tracklet[track_id] = datetime.now(UTC) - timedelta(seconds=2.0)
    await stage.run(_make_context("room-cam", frame_index=2))
    assert client.identify_crops.await_count == 1

    # 5.5 s elapsed: cooldown expired -> queries again.
    stage._last_face_id_by_tracklet[track_id] = datetime.now(UTC) - timedelta(seconds=5.5)
    await stage.run(_make_context("room-cam", frame_index=3))
    assert client.identify_crops.await_count == 2


@pytest.mark.asyncio
async def test_policy_updates_dynamically_when_camera_profile_map_refreshes() -> None:
    stage, _, pmap = _make_stage(
        [CameraProfile(camera_id="cam-1", role=CameraRole.SURVEILLANCE, face_id_enabled=True)],
        cooldown_s=5.0,
        identification_point_cooldown_s=1.0,
    )
    assert stage.face_id_policy_for("cam-1").cooldown_s == 5.0

    await pmap.set_all(
        [
            CameraProfile(
                camera_id="cam-1",
                role=CameraRole.FACE_CAPABLE,
                face_id_enabled=True,
                face_id_min_raw_similarity=0.55,
            )
        ]
    )
    updated = stage.face_id_policy_for("cam-1")
    assert updated.cooldown_s == 1.0
    assert updated.min_raw_similarity == 0.55
    assert updated.role is CameraRole.FACE_CAPABLE
