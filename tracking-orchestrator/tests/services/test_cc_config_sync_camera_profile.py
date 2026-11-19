"""Tests for CC config sync of hardware-neutral camera profiles and stream size reporting."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing
from cts_contracts import CameraRole
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from app.calibration.state import CalibrationState
from app.domain import CameraProfile
from app.observability import metrics as _metrics
from app.observability.metrics import build_metrics
from app.pipeline.frame_pipeline import FrameProcessingPipeline, PipelineConfig
from app.routers import calibration as calibration_router_mod
from app.services.camera_profile_map import CameraProfileMap
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


class _FakeCC:
    def __init__(self, cameras: list[dict[str, Any]]) -> None:
        self.cameras = cameras

    async def get(self, path: str) -> Any:
        if path == "/api/v1/cts/cameras":
            return self.cameras
        if path == "/api/v1/household/floor-plan":
            return _NO_SCALE
        raise AssertionError(f"unexpected path {path}")


def _make_sync(
    cc: _FakeCC,
    *,
    cal_state: CalibrationState | None = None,
    profile_map: CameraProfileMap | None = None,
) -> tuple[CCConfigSyncService, CalibrationState, CameraProfileMap]:
    state = cal_state or CalibrationState()
    pmap = profile_map or CameraProfileMap()
    client = MagicMock()
    client.get = AsyncMock(side_effect=cc.get)
    svc = CCConfigSyncService(
        client=client,
        calibration_state=state,
        camera_room_map=CameraRoomMap(),
        room_polygon_map=RoomPolygonMap(),
        transit_zone_map=TransitZoneMap(),
        poll_interval_s=999,
        fetch_retry_attempts=1,
        camera_profile_map=pmap,
    )
    return svc, state, pmap


@pytest.mark.asyncio
async def test_full_camera_profile_decodes_into_frozen_dataclass() -> None:
    cc = _FakeCC(
        [
            {
                "id": "hall-id-cam",
                "enabled": True,
                "room_id": 2,
                "room_name": "Hallway",
                "role": "face_capable",
                "face_id_enabled": True,
                "face_id_min_raw_similarity": 0.55,
                "horizontal_fov_deg": 62.0,
                "mounting_height_m": 1.65,
                "tilt_deg": -5.0,
                "ir_capable": True,
                "frame_natural_width": 2560,
                "frame_natural_height": 1440,
            }
        ]
    )
    svc, _, pmap = _make_sync(cc)
    await svc._poll()

    profile = await pmap.get("hall-id-cam")
    assert profile == CameraProfile(
        camera_id="hall-id-cam",
        role=CameraRole.FACE_CAPABLE,
        face_id_enabled=True,
        face_id_min_raw_similarity=0.55,
        horizontal_fov_deg=62.0,
        mounting_height_m=1.65,
        tilt_deg=-5.0,
        ir_capable=True,
        frame_natural_width_px=2560,
        frame_natural_height_px=1440,
    )
    assert pmap.get_sync("hall-id-cam") == profile


@pytest.mark.asyncio
async def test_missing_fields_decode_to_none_and_default_role() -> None:
    cc = _FakeCC(
        [
            {
                "id": "legacy-cam",
                "enabled": True,
                "room_id": 1,
                "room_name": "Living Room",
            }
        ]
    )
    svc, _, pmap = _make_sync(cc)
    await svc._poll()

    profile = await pmap.get("legacy-cam")
    assert profile == CameraProfile(
        camera_id="legacy-cam",
        role=CameraRole.SURVEILLANCE,
        face_id_enabled=True,
        face_id_min_raw_similarity=None,
        horizontal_fov_deg=None,
        mounting_height_m=None,
        tilt_deg=None,
        ir_capable=None,
        frame_natural_width_px=None,
        frame_natural_height_px=None,
    )


@pytest.mark.asyncio
async def test_unknown_role_fails_camera_sync_loudly_with_metric_and_warning() -> None:
    cc = _FakeCC(
        [
            {
                "id": "bad-role-cam",
                "enabled": True,
                "room_id": 1,
                "room_name": "Kitchen",
                "role": "ptz_magic_camera",
            },
            {
                "id": "good-cam",
                "enabled": True,
                "room_id": 1,
                "room_name": "Kitchen",
                "role": "mixed",
            },
        ]
    )
    svc, _, pmap = _make_sync(cc)

    with structlog.testing.capture_logs() as logs:
        await svc._poll()

    # Bad camera must NOT silently default into the profile map.
    assert await pmap.get("bad-role-cam") is None
    # Valid sibling camera still syncs normally.
    good = await pmap.get("good-cam")
    assert good is not None
    assert good.role is CameraRole.MIXED

    # Metric incremented with bounded reason.
    count = _metrics.metrics.camera_profile_rejected_total.labels(
        camera_id="bad-role-cam", reason="unknown_role"
    )._value.get()
    assert count == 1.0

    assert any(
        entry.get("event") == "cc_config_sync_camera_profile_rejected"
        and entry.get("camera_id") == "bad-role-cam"
        and entry.get("reason") == "unknown_role"
        for entry in logs
    )


@pytest.mark.asyncio
async def test_uncalibrated_identification_camera_logs_warning_once() -> None:
    cc = _FakeCC(
        [
            {
                "id": "id-cam",
                "enabled": True,
                "room_id": 1,
                "room_name": "Hallway",
                "role": "face_capable",
            }
        ]
    )
    svc, _, _ = _make_sync(cc)

    with structlog.testing.capture_logs() as first_logs:
        await svc._poll()
    assert (
        sum(
            1
            for entry in first_logs
            if entry.get("event") == "identification_camera_uncalibrated"
            and entry.get("camera_id") == "id-cam"
        )
        == 1
    )

    # Repeated poll while still uncalibrated does not spam the log.
    with structlog.testing.capture_logs() as second_logs:
        await svc._poll()
    assert not any(
        entry.get("event") == "identification_camera_uncalibrated" for entry in second_logs
    )

    # Calibrating clears the warning state; removing calibration logs once again.
    cc.cameras[0]["homography_matrix"] = _MATRIX
    cc.cameras[0]["homography_residual_m"] = 0.05
    cc.cameras[0]["homography_floor_plan_id"] = "household:1"
    cc.cameras[0]["frame_natural_width"] = 2560
    cc.cameras[0]["frame_natural_height"] = 1440
    await svc._poll()

    cc.cameras[0]["homography_matrix"] = None
    with structlog.testing.capture_logs() as third_logs:
        await svc._poll()
    assert (
        sum(
            1
            for entry in third_logs
            if entry.get("event") == "identification_camera_uncalibrated"
            and entry.get("camera_id") == "id-cam"
        )
        == 1
    )


def test_stream_size_mismatch_logs_once_and_exposes_on_calibration_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_state = CalibrationState()
    monkeypatch.setattr("app.pipeline.frame_pipeline.calibration_state", isolated_state)
    monkeypatch.setattr(calibration_router_mod, "_default_state", isolated_state)

    pmap = CameraProfileMap(
        [
            CameraProfile(
                camera_id="hall-cam",
                role=CameraRole.FACE_CAPABLE,
                frame_natural_width_px=1280,
                frame_natural_height_px=720,
            )
        ]
    )
    pipeline = FrameProcessingPipeline(PipelineConfig())
    pipeline.set_camera_profile_map(pmap)

    with structlog.testing.capture_logs() as logs:
        pipeline._record_observed_stream_size("hall-cam", 2560, 1440)
        pipeline._record_observed_stream_size("hall-cam", 2560, 1440)

    mismatch_logs = [e for e in logs if e.get("event") == "camera_stream_size_mismatch"]
    assert len(mismatch_logs) == 1
    assert mismatch_logs[0]["observed_width_px"] == 2560
    assert mismatch_logs[0]["configured_width_px"] == 1280

    app = FastAPI()
    app.include_router(calibration_router_mod.router)
    client = TestClient(app)
    resp = client.get("/internal/calibration/status")
    assert resp.status_code == 200
    assert resp.json()["observed_stream_sizes"] == {
        "hall-cam": {"width_px": 2560, "height_px": 1440}
    }
