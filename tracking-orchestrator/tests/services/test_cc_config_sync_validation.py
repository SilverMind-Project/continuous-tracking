"""CC config sync validation tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.calibration.validator import validate_homography


def _floor_plan_missing_scale() -> dict:
    return {
        "floor_plan_width": None,
        "floor_plan_height": None,
        "floor_meters_per_pixel": None,
    }


def test_error_homography_is_rejected():
    """A degenerate matrix (zero determinant) must return severity=error."""
    # Near-zero determinant (degenerate).
    degenerate = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1e-7],
    ]
    result = validate_homography(
        matrix=degenerate,
        residuals=[0.6],  # also high residual
    )
    assert result.severity == "error"
    assert any("degenerate" in i for i in result.issues)


def test_warning_homography_is_accepted_with_warning():
    """A matrix with moderate residual gets severity=warning."""
    matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result = validate_homography(
        matrix=matrix,
        residuals=[0.3],  # moderate residual (0.25 < 0.3 <= 0.5)
    )
    assert result.severity == "warning"
    assert any("warning" in i for i in result.issues)


def test_ok_homography_passes():
    """A valid matrix with low residuals gets severity=ok."""
    matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result = validate_homography(
        matrix=matrix,
        residuals=[0.1],
        image_width=1920,
        image_height=1080,
    )
    assert result.severity == "ok"
    assert result.issues == []


@pytest.mark.asyncio
async def test_rejected_homography_does_not_reach_calibration_state():
    """Error-severity homography must not call set_homography."""
    from app.calibration.state import CalibrationState
    from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
    from app.services.cc_config_sync import CCConfigSyncService

    cal_state = CalibrationState()
    cal_state.set_homography = AsyncMock()
    camera_room_map = CameraRoomMap()
    room_polygon_map = RoomPolygonMap()

    client = MagicMock()

    async def get(path: str):
        if path == "/api/v1/cts/cameras":
            return [
                {
                    "id": "cam-1",
                    "enabled": True,
                    "room_id": 1,
                    "room_name": "living_room",
                    "homography_matrix": [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 1e-7],
                    ],
                    "homography_residual_m": 0.6,
                    "homography_floor_plan_id": "floor-plans/main.png",
                },
            ]
        if path == "/api/v1/household/floor-plan":
            return _floor_plan_missing_scale()
        raise AssertionError(f"unexpected path {path}")

    client.get = AsyncMock(side_effect=get)

    svc = CCConfigSyncService(
        client=client,
        calibration_state=cal_state,
        camera_room_map=camera_room_map,
        room_polygon_map=room_polygon_map,
        poll_interval_s=999,
    )

    await svc._poll()

    # set_homography must NOT be called for an error-severity matrix.
    cal_state.set_homography.assert_not_called()
    # Camera→room binding should still be set even if homography was rejected.
    assert await camera_room_map.get("cam-1") is not None


@pytest.mark.asyncio
async def test_sync_restores_homography_with_floor_plan_metadata():
    """CC sync must restore enough metadata for projection and shared-floor checks."""
    from app.calibration.state import CalibrationState
    from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
    from app.services.cc_config_sync import CCConfigSyncService

    cal_state = CalibrationState()
    camera_room_map = CameraRoomMap()
    room_polygon_map = RoomPolygonMap()
    matrix = [[1.0, 0.0, 0.1], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]

    client = MagicMock()

    async def get(path: str):
        if path == "/api/v1/cts/cameras":
            return [
                {
                    "id": "cam-1",
                    "enabled": True,
                    "room_id": 1,
                    "room_name": "living_room",
                    "homography_matrix": matrix,
                    "homography_residuals": [0.01, 0.02, 0.01, 0.02],
                    "homography_residual_m": 0.02,
                    "homography_floor_plan_id": "floor-plans/main.png",
                    "frame_natural_width": 1920,
                    "frame_natural_height": 1080,
                },
            ]
        if path == "/api/v1/household/floor-plan":
            return _floor_plan_missing_scale()
        raise AssertionError(f"unexpected path {path}")

    client.get = AsyncMock(side_effect=get)

    svc = CCConfigSyncService(
        client=client,
        calibration_state=cal_state,
        camera_room_map=camera_room_map,
        room_polygon_map=room_polygon_map,
        poll_interval_s=999,
    )

    await svc._poll()

    assert cal_state.homographies["cam-1"] == matrix
    calibration = cal_state.calibrations["cam-1"]
    assert calibration.floor_plan_id == "floor-plans/main.png"
    assert calibration.image_width == 1920
    assert calibration.image_height == 1080
    assert calibration.quality.max_residual_m == pytest.approx(0.02)
    assert calibration.quality.mean_residual_m == pytest.approx(0.015)
    assert calibration.quality.point_count == 4


@pytest.mark.asyncio
async def test_sync_converts_room_polygons_to_floor_metres():
    """CC room polygons are normalized; CTS tracker needs floor-plan metres."""
    from app.calibration.state import CalibrationState
    from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
    from app.services.cc_config_sync import CCConfigSyncService

    cal_state = CalibrationState()
    camera_room_map = CameraRoomMap()
    room_polygon_map = RoomPolygonMap()

    async def get(path: str):
        if path == "/api/v1/cts/cameras":
            return []
        if path == "/api/v1/household/floor-plan":
            return {
                "floor_plan_width": 1000,
                "floor_plan_height": 500,
                "floor_meters_per_pixel": 0.02,
            }
        if path == "/api/v1/rooms":
            return [
                {
                    "id": 7,
                    "name": "Living Room",
                    "floor_polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
                }
            ]
        raise AssertionError(f"unexpected path {path}")

    client = MagicMock()
    client.get = AsyncMock(side_effect=get)
    svc = CCConfigSyncService(
        client=client,
        calibration_state=cal_state,
        camera_room_map=camera_room_map,
        room_polygon_map=room_polygon_map,
        poll_interval_s=999,
    )

    await svc._poll()

    polygons, names = await room_polygon_map.snapshot()
    assert names == {"7": "Living Room"}
    assert polygons["7"] == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


@pytest.mark.asyncio
async def test_config_fetch_retries_transient_failures_before_applying():
    """Transient CC fetch failures should not skip a sync cycle immediately."""
    from app.calibration.state import CalibrationState
    from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
    from app.services.cc_config_sync import CCConfigSyncService

    cal_state = CalibrationState()
    camera_room_map = CameraRoomMap()
    room_polygon_map = RoomPolygonMap()

    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[
            RuntimeError("temporary timeout"),
            RuntimeError("temporary 503"),
            [
                {
                    "id": "cam-1",
                    "enabled": True,
                    "room_id": 1,
                    "room_name": "living_room",
                }
            ],
            _floor_plan_missing_scale(),
        ]
    )

    svc = CCConfigSyncService(
        client=client,
        calibration_state=cal_state,
        camera_room_map=camera_room_map,
        room_polygon_map=room_polygon_map,
        poll_interval_s=999,
        fetch_retry_initial_s=0,
        fetch_retry_max_s=0,
    )

    await svc._poll()

    assert client.get.await_count == 4
    assert await camera_room_map.get("cam-1") is not None


@pytest.mark.asyncio
async def test_malformed_room_polygon_raises_without_applying_polygon_map():
    """Malformed configured polygons are contract failures, not skipped points."""
    from datetime import UTC, datetime

    from app.calibration.state import CalibrationState
    from app.services.camera_room_map import (
        CameraRoomBinding,
        CameraRoomMap,
        RoomPolygonBinding,
        RoomPolygonMap,
    )
    from app.services.cc_config_sync import CCConfigSyncContractError, CCConfigSyncService

    cal_state = CalibrationState()
    camera_room_map = CameraRoomMap()
    room_polygon_map = RoomPolygonMap()
    now = datetime.now(UTC)
    await camera_room_map.set_all(
        [CameraRoomBinding("old-cam", "old-room", "Old Room", now)]
    )
    await room_polygon_map.set_all(
        [RoomPolygonBinding("old-room", "Old Room", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], now)]
    )

    async def get(path: str):
        if path == "/api/v1/cts/cameras":
            return []
        if path == "/api/v1/household/floor-plan":
            return {
                "floor_plan_width": 1000,
                "floor_plan_height": 500,
                "floor_meters_per_pixel": 0.02,
            }
        if path == "/api/v1/rooms":
            return [{"id": 7, "name": "Living Room", "floor_polygon": [[0.0, 0.0], [0.5]]}]
        raise AssertionError(f"unexpected path {path}")

    client = MagicMock()
    client.get = AsyncMock(side_effect=get)
    svc = CCConfigSyncService(
        client=client,
        calibration_state=cal_state,
        camera_room_map=camera_room_map,
        room_polygon_map=room_polygon_map,
        poll_interval_s=999,
    )

    with pytest.raises(CCConfigSyncContractError):
        await svc._poll()

    polygons, names = await room_polygon_map.snapshot()
    assert await camera_room_map.get("old-cam") is None
    assert polygons == {}
    assert names == {}
