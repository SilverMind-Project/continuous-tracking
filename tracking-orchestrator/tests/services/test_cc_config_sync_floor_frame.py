"""CC config sync keeps the runtime floor frame equal to CC's (homographies, overlap groups)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import CollectorRegistry

from app.calibration.state import CalibrationState
from app.domain import OverlapGroup
from app.observability import metrics as _metrics
from app.observability.metrics import build_metrics
from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
from app.services.cc_config_sync import CCConfigSyncService
from app.services.transit_zone_map import TransitZoneMap

_MATRIX = [[1.0, 0.0, 0.1], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]
_NO_SCALE = {"floor_plan_width": None, "floor_plan_height": None, "floor_meters_per_pixel": None}


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> CollectorRegistry:
    reg = CollectorRegistry()
    monkeypatch.setattr(_metrics, "metrics", build_metrics(registry=reg))
    return reg


def _camera(camera_id: str, *, matrix: list[list[float]] | None, enabled: bool = True) -> dict:
    return {
        "id": camera_id,
        "enabled": enabled,
        "room_id": 1,
        "room_name": "living_room",
        "homography_matrix": matrix,
        "homography_residual_m": 0.05 if matrix else None,
        "homography_floor_plan_id": "household:1",
        "frame_natural_width": 2560,
        "frame_natural_height": 1440,
    }


class _FakeCC:
    """Serves mutable CC responses so one test can change CC between polls."""

    def __init__(self) -> None:
        self.cameras: list[dict] = []
        self.overlap_groups: Any = []
        self.overlap_groups_error: Exception | None = None

    async def get(self, path: str) -> Any:
        if path == "/api/v1/cts/cameras":
            return self.cameras
        if path == "/api/v1/household/floor-plan":
            return _NO_SCALE
        if path == "/api/v1/cts/overlap_groups":
            if self.overlap_groups_error is not None:
                raise self.overlap_groups_error
            return self.overlap_groups
        raise AssertionError(f"unexpected path {path}")


def _service(
    cc: _FakeCC,
    state: CalibrationState,
    sink: MagicMock | None = None,
) -> CCConfigSyncService:
    client = MagicMock()
    client.get = AsyncMock(side_effect=cc.get)
    return CCConfigSyncService(
        client=client,
        calibration_state=state,
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
        transit_zone_map=TransitZoneMap(),
        poll_interval_s=999,
        fetch_retry_attempts=1,
        overlap_group_sink=sink,
    )


@pytest.mark.asyncio
async def test_homography_cleared_in_cc_is_removed_at_next_poll() -> None:
    cc = _FakeCC()
    state = CalibrationState()
    svc = _service(cc, state)
    cc.cameras = [_camera("cam-a", matrix=_MATRIX), _camera("cam-b", matrix=_MATRIX)]
    await svc._poll()
    assert set(state.homographies) == {"cam-a", "cam-b"}
    assert set(state.calibrations) == {"cam-a", "cam-b"}

    cc.cameras = [_camera("cam-a", matrix=None), _camera("cam-b", matrix=_MATRIX)]
    await svc._poll()

    assert set(state.homographies) == {"cam-b"}
    assert set(state.calibrations) == {"cam-b"}


@pytest.mark.asyncio
async def test_homography_of_disabled_or_deleted_camera_is_removed() -> None:
    cc = _FakeCC()
    state = CalibrationState()
    svc = _service(cc, state)
    cc.cameras = [_camera("cam-a", matrix=_MATRIX), _camera("cam-b", matrix=_MATRIX)]
    await svc._poll()

    cc.cameras = [_camera("cam-a", matrix=_MATRIX, enabled=False)]
    await svc._poll()

    assert state.homographies == {}


@pytest.mark.asyncio
async def test_rejected_replacement_does_not_leave_the_old_matrix_in_use() -> None:
    cc = _FakeCC()
    state = CalibrationState()
    svc = _service(cc, state)
    cc.cameras = [_camera("cam-a", matrix=_MATRIX)]
    await svc._poll()

    degenerate = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1e-7]]
    cc.cameras = [_camera("cam-a", matrix=degenerate)]
    await svc._poll()

    assert "cam-a" not in state.homographies


@pytest.mark.asyncio
async def test_unchanged_homography_keeps_state_version_stable() -> None:
    cc = _FakeCC()
    state = CalibrationState()
    svc = _service(cc, state)
    cc.cameras = [_camera("cam-a", matrix=_MATRIX), _camera("cam-b", matrix=None)]
    await svc._poll()
    version = state.version

    removed = await state.remove_homographies({"cam-b", "cam-unknown"})

    assert removed == []
    assert state.version == version


@pytest.mark.asyncio
async def test_overlap_groups_reach_the_sink_and_follow_cc_changes(
    registry: CollectorRegistry,
) -> None:
    cc = _FakeCC()
    sink = MagicMock()
    svc = _service(cc, CalibrationState(), sink)
    cc.cameras = [_camera("cam-a", matrix=None)]
    cc.overlap_groups = [{"id": 7, "name": "living", "camera_ids": ["cam-a", "cam-b"]}]

    await svc._poll()
    await svc._poll()  # unchanged: not re-applied

    sink.assert_called_once()
    applied: Sequence[OverlapGroup] = sink.call_args.args[0]
    assert [(g.group_id, g.camera_ids) for g in applied] == [("7", ("cam-a", "cam-b"))]
    assert registry.get_sample_value("cts_overlap_groups_loaded") == 1.0

    cc.overlap_groups = []
    await svc._poll()

    assert sink.call_count == 2
    assert list(sink.call_args.args[0]) == []
    assert registry.get_sample_value("cts_overlap_groups_loaded") == 0.0


@pytest.mark.asyncio
async def test_overlap_groups_kept_when_cc_fetch_fails_or_is_malformed() -> None:
    cc = _FakeCC()
    sink = MagicMock()
    svc = _service(cc, CalibrationState(), sink)
    cc.cameras = [_camera("cam-a", matrix=None)]
    cc.overlap_groups = [{"id": 1, "name": "g", "camera_ids": ["cam-a", "cam-b"]}]
    await svc._poll()

    cc.overlap_groups_error = RuntimeError("CC unavailable")
    await svc._poll()
    cc.overlap_groups_error = None
    cc.overlap_groups = {"not": "a list"}
    await svc._poll()

    sink.assert_called_once()


@pytest.mark.asyncio
async def test_overlap_groups_not_fetched_without_a_sink() -> None:
    cc = _FakeCC()
    cc.cameras = [_camera("cam-a", matrix=None)]
    cc.overlap_groups_error = AssertionError("must not be fetched")

    await _service(cc, CalibrationState())._poll()
