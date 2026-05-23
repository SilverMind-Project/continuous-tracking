"""Tests for SpatialProjectionStage."""

from __future__ import annotations

import time

import pytest

from app.calibration.state import CalibrationState
from app.inference.schemas import DetectionBox
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.spatial_projection import SpatialProjectionStage
from app.tracking.spatial_projection import SpatialProjectionService
from app.transport.redis_streams import FrameReady

_NOW_NS = int(time.time() * 1e9)


def _make_ctx(camera_id: str = "cam-1") -> FrameContext:
    from datetime import UTC, datetime

    frame = FrameReady(
        camera_id=camera_id,
        minio_key="frames/cam-1/0.jpg",
        frame_index=0,
        capture_time_unix_ns=_NOW_NS,
        received_time_unix_ns=_NOW_NS + 100_000_000,
        width=640,
        height=480,
    )
    return FrameContext(
        frame=frame,
        event_time=datetime.now(UTC),
        capture_time=datetime.fromtimestamp(frame.capture_time_unix_ns / 1e9, tz=UTC),
    )


async def _add_homography(cal_state, camera_id):
    await cal_state.set_homography(
        camera_id=camera_id,
        matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        floor_plan_id="fp-test",
        image_width=640,
        image_height=480,
        max_residual_m=0.1,
        mean_residual_m=0.05,
        quality_status="ok",
        quality_point_count=8,
    )


class TestSpatialProjectionStage:
    @pytest.mark.asyncio
    async def test_detections_receive_floor_points(self) -> None:
        """Every detection should have a non-default FloorPoint after projection."""
        cal_state = CalibrationState()
        await _add_homography(cal_state, "cam-1")
        svc = SpatialProjectionService(cal_state)
        stage = SpatialProjectionStage(svc)

        ctx = _make_ctx("cam-1")
        ctx.effective_width = 640
        ctx.effective_height = 480
        ctx.raw_detections = [
            DetectionBox(x1=0.1, y1=0.1, x2=0.3, y2=0.5, confidence=0.9),
            DetectionBox(x1=0.5, y1=0.2, x2=0.7, y2=0.6, confidence=0.8),
        ]

        await stage.run(ctx)

        assert len(ctx._floor_points_by_index) == 2
        for idx in range(2):
            fp = ctx._floor_points_by_index[idx]
            assert fp.calibrated, f"detection {idx} floor point should be calibrated"
            assert fp.x_mm != 0 or fp.y_mm != 0, f"detection {idx} has zero floor point"

    @pytest.mark.asyncio
    async def test_uncalibrated_camera_returns_uncalibrated_points(self) -> None:
        """When no homography exists, floor points should be uncalibrated."""
        cal_state = CalibrationState()
        svc = SpatialProjectionService(cal_state)
        stage = SpatialProjectionStage(svc)

        ctx = _make_ctx("cam-uncalibrated")
        ctx.effective_width = 640
        ctx.effective_height = 480
        ctx.raw_detections = [
            DetectionBox(x1=0.1, y1=0.1, x2=0.3, y2=0.5, confidence=0.9),
        ]

        await stage.run(ctx)

        fp = ctx._floor_points_by_index[0]
        assert not fp.calibrated
        assert fp.x_mm == 0
        assert fp.y_mm == 0
