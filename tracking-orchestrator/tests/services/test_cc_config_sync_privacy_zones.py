"""Privacy zones restored from CC: at startup and on every config-sync poll.

The orchestrator keeps privacy zones only in ``CalibrationState`` memory, so CC is their source
of truth. These tests pin that a restart or a poll leaves exactly CC's zones in force, and that
malformed or missing data never silently drops a zone that is already being enforced.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.calibration.state import CalibrationState, PrivacyZoneConfig
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.services.cc_config_sync import (
    CCConfigSyncContractError,
    CCConfigSyncService,
    apply_privacy_zones,
    restore_privacy_zones,
)
from app.services.transit_zone_map import TransitZoneMap

TRIANGLE = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.5]]


def _zone(zone_id: str = "mirror", policy: str = "drop_detection", **over: Any) -> dict:
    return {
        "zone_id": zone_id,
        "name": "Mirror",
        "polygon": TRIANGLE,
        "policy": policy,
        "enabled": True,
        **over,
    }


def _camera(camera_id: str = "cam-1", zones: list[dict] | None = None, **over: Any) -> dict:
    cam: dict[str, Any] = {
        "id": camera_id,
        "enabled": True,
        "room_id": 1,
        "room_name": "living_room",
        "privacy_zones": zones if zones is not None else [_zone()],
    }
    cam.update(over)
    return cam


def _config(zone_id: str = "mirror", policy: str = "drop_detection") -> PrivacyZoneConfig:
    return PrivacyZoneConfig(
        zone_id=zone_id, polygon=TRIANGLE, policy=policy, name="Mirror", enabled=True
    )


@pytest.mark.asyncio
async def test_applies_each_cameras_zones() -> None:
    state = CalibrationState()

    applied = await apply_privacy_zones(
        [_camera("cam-1"), _camera("cam-2", [_zone("desk", "mask_region")])], state
    )

    assert applied is True
    assert state.privacy_zones == {
        "cam-1": [_config()],
        "cam-2": [_config("desk", "mask_region")],
    }


@pytest.mark.asyncio
async def test_disabled_cameras_keep_their_zones_for_when_they_return() -> None:
    state = CalibrationState()

    await apply_privacy_zones([_camera(enabled=False)], state)

    assert state.privacy_zones["cam-1"] == [_config()]


@pytest.mark.asyncio
async def test_an_empty_list_and_a_removed_camera_both_clear_zones() -> None:
    state = CalibrationState()
    await state.set_privacy_zones("cam-1", [_config()])
    await state.set_privacy_zones("gone-cam", [_config()])

    await apply_privacy_zones([_camera("cam-1", zones=[])], state)

    assert state.privacy_zones["cam-1"] == []
    assert state.privacy_zones["gone-cam"] == []


@pytest.mark.asyncio
async def test_unchanged_zones_are_not_rewritten() -> None:
    state = CalibrationState()
    await apply_privacy_zones([_camera()], state)
    version = state.version

    await apply_privacy_zones([_camera()], state)

    assert state.version == version


@pytest.mark.asyncio
async def test_a_cc_without_the_field_changes_nothing() -> None:
    state = CalibrationState()
    await state.set_privacy_zones("cam-1", [_config()])
    cam = _camera()
    del cam["privacy_zones"]

    applied = await apply_privacy_zones([cam], state)

    assert applied is False
    assert state.privacy_zones["cam-1"] == [_config()]


@pytest.mark.parametrize(
    "bad_zone",
    [
        _zone(policy="skip_detection"),
        _zone(polygon=[[0.1, 0.1], [0.4, 0.1]]),
        _zone(polygon=[[0.1, 0.1], [1.4, 0.1], [0.4, 0.5]]),
        _zone(polygon=[[0.1, 0.1], [0.4], [0.4, 0.5]]),
        _zone(zone_id=""),
        _zone(enabled="yes"),
    ],
)
@pytest.mark.asyncio
async def test_malformed_zones_raise_and_leave_enforced_zones_untouched(bad_zone: dict) -> None:
    state = CalibrationState()
    await state.set_privacy_zones("cam-1", [_config()])

    # The valid camera listed first must not be applied either: all zones validate first.
    with pytest.raises(CCConfigSyncContractError):
        await apply_privacy_zones(
            [_camera("cam-2", [_zone("desk")]), _camera("cam-1", [bad_zone])], state
        )

    assert state.privacy_zones == {"cam-1": [_config()]}


def _service(state: CalibrationState, cameras: list[dict]) -> CCConfigSyncService:
    client = MagicMock()

    async def get(path: str) -> Any:
        if path == "/api/v1/cts/cameras":
            return cameras
        if path == "/api/v1/household/floor-plan":
            return {
                "floor_plan_width": None,
                "floor_plan_height": None,
                "floor_meters_per_pixel": None,
            }
        raise AssertionError(f"unexpected path {path}")

    client.get = AsyncMock(side_effect=get)
    return CCConfigSyncService(
        client=client,
        calibration_state=state,
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
        transit_zone_map=TransitZoneMap(),
        poll_interval_s=999,
    )


@pytest.mark.asyncio
async def test_poll_applies_zones_even_when_the_homography_is_rejected() -> None:
    state = CalibrationState()
    degenerate = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1e-7]]
    svc = _service(
        state,
        [
            _camera(
                homography_matrix=degenerate,
                homography_residual_m=0.6,
                homography_floor_plan_id="floor-plans/main.png",
            )
        ],
    )

    await svc._poll()

    assert state.privacy_zones["cam-1"] == [_config()]
    assert "cam-1" not in state.homographies


@pytest.mark.asyncio
async def test_poll_with_malformed_zones_fails_and_keeps_enforced_zones() -> None:
    state = CalibrationState()
    await state.set_privacy_zones("cam-1", [_config()])
    svc = _service(state, [_camera(zones=[_zone(policy="blur_faces")])])

    with pytest.raises(CCConfigSyncContractError):
        await svc._poll()

    assert state.privacy_zones["cam-1"] == [_config()]


@pytest.mark.asyncio
async def test_restore_loads_zones_before_the_pipeline_starts() -> None:
    state = CalibrationState()
    client = MagicMock()
    client.get = AsyncMock(return_value=[_camera(), _camera("cam-2", [_zone("desk")])])

    restored = await restore_privacy_zones(client, state)

    client.get.assert_awaited_once_with("/api/v1/cts/cameras")
    assert restored == 2
    assert state.privacy_zones["cam-2"] == [_config("desk")]


@pytest.mark.asyncio
async def test_restore_refuses_to_start_without_zones() -> None:
    client = MagicMock()
    client.get = AsyncMock(side_effect=ConnectionError("cc down"))

    with pytest.raises(RuntimeError, match="Failed to fetch privacy zones"):
        await restore_privacy_zones(client, CalibrationState())

    client.get = AsyncMock(return_value=[_camera(zones=[_zone(policy="blur_faces")])])
    with pytest.raises(RuntimeError, match="malformed privacy zones"):
        await restore_privacy_zones(client, CalibrationState())


@pytest.mark.asyncio
async def test_restore_from_a_cc_without_the_field_starts_with_no_zones() -> None:
    cam = _camera()
    del cam["privacy_zones"]
    client = MagicMock()
    client.get = AsyncMock(return_value=[cam])
    state = CalibrationState()

    assert await restore_privacy_zones(client, state) == 0
    assert state.privacy_zones == {}
