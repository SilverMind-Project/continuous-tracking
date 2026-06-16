"""Tests for WorldTrackingStage with uncalibrated cameras.

Ensures that detections from cameras without homography calibration still
reach the WorldTracker (via synthetic floor points) so that PHs can be
created and face-based identities can be committed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.calibration.state import CalibrationState
from app.domain import BoundingBox, Detection, FloorPoint, OrientationBin, tuple_to_cov2x2
from app.inference.schemas import Keypoint, PoseResult
from app.pipeline.stages.world_tracking import (
    _CAMERA_TILE_M,
    _VIRTUAL_ROOM_M,
    _stable_camera_hash,
    _synthetic_floor_point,
)
from app.tracking.floor_projector import FloorProjector
from app.tracking.world.observation_model import homography_jacobian, pixel_covariance
from app.tracking.world.tracker import WorldTrackerResult

# ---------------------------------------------------------------------------
# Unit tests for _stable_camera_hash
# ---------------------------------------------------------------------------


def test_stable_camera_hash_is_deterministic():
    assert _stable_camera_hash("cam-1") == _stable_camera_hash("cam-1")


def test_stable_camera_hash_differs_across_cameras():
    hashes = {_stable_camera_hash(f"cam{i}") for i in range(10)}
    assert len(hashes) > 5  # at least 6 distinct hashes from 10 cameras


def test_stable_camera_hash_is_16bit():
    for cam in ["cam-1", "camera-abc", "front-door", ""]:
        h = _stable_camera_hash(cam)
        assert 0 <= h <= 0xFFFF


# ---------------------------------------------------------------------------
# Unit tests for _synthetic_floor_point
# ---------------------------------------------------------------------------


def _make_bbox(x_min: int, y_min: int, x_max: int, y_max: int) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _pose() -> PoseResult:
    keypoints = [Keypoint(x=0.5, y=0.5, score=0.9) for _ in range(17)]
    keypoints[15] = Keypoint(x=0.45, y=0.95, score=0.9)
    keypoints[16] = Keypoint(x=0.55, y=0.95, score=0.9)
    return PoseResult(keypoints=tuple(keypoints))


def test_synthetic_floor_point_returns_floor_point():
    fp = _synthetic_floor_point(_make_bbox(100, 100, 200, 400), 640, 480, "cam-1")
    assert isinstance(fp, FloorPoint)


def test_synthetic_floor_point_calibrated_flag_is_false():
    fp = _synthetic_floor_point(_make_bbox(0, 0, 320, 240), 640, 480, "cam-1")
    assert not fp.calibrated


def test_synthetic_floor_point_nonzero_coords():
    fp = _synthetic_floor_point(_make_bbox(100, 100, 200, 300), 640, 480, "cam-1")
    assert fp.x_mm != 0 or fp.y_mm != 0


def test_synthetic_floor_point_within_tile():
    cam_h = _stable_camera_hash("cam-1")
    tile_x_mm = (cam_h % 256) * _CAMERA_TILE_M * 1000
    tile_y_mm = (cam_h >> 8) * _CAMERA_TILE_M * 1000

    fp = _synthetic_floor_point(_make_bbox(0, 0, 640, 480), 640, 480, "cam-1")
    assert tile_x_mm <= fp.x_mm <= tile_x_mm + int(_VIRTUAL_ROOM_M * 1000)
    assert tile_y_mm <= fp.y_mm <= tile_y_mm + int(_VIRTUAL_ROOM_M * 1000)


def test_synthetic_floor_points_for_different_cameras_are_far_apart():
    fp1 = _synthetic_floor_point(_make_bbox(320, 240, 400, 400), 640, 480, "cam-1")
    fp2 = _synthetic_floor_point(_make_bbox(320, 240, 400, 400), 640, 480, "cam-2")
    dx = abs(fp1.x_mm - fp2.x_mm) / 1000.0
    dy = abs(fp1.y_mm - fp2.y_mm) / 1000.0
    dist = (dx**2 + dy**2) ** 0.5
    # Must be > dedup_max_distance_m (0.6 m) — in practice many meters apart.
    assert dist > 1.0, f"Expected cameras to be far apart, got dist={dist:.2f}m"


def test_synthetic_floor_point_scales_with_bbox_position():
    # A bbox in the top-left should produce a smaller (or equal) position than top-right.
    fp_left = _synthetic_floor_point(_make_bbox(0, 0, 100, 100), 640, 480, "cam-test")
    fp_right = _synthetic_floor_point(_make_bbox(540, 0, 640, 100), 640, 480, "cam-test")
    assert fp_right.x_mm > fp_left.x_mm


# ---------------------------------------------------------------------------
# Integration test: WorldTrackingStage builds observations for uncalibrated
# cameras so the WorldTracker can create PHs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_world_tracking_stage_produces_observations_without_calibration():
    """Stage must NOT skip detections from uncalibrated cameras.

    Before the fix, WorldTrackingStage filtered out all detections where
    floor_point.calibrated=False, leaving the WorldTracker with zero
    observations and preventing PH creation.  After the fix, a synthetic
    floor point is generated for each uncalibrated detection.
    """
    from unittest.mock import AsyncMock, MagicMock

    from app.domain import FloorPoint
    from app.pipeline.frame_context import FrameContext
    from app.pipeline.stages.world_tracking import WorldTrackingStage
    from app.pipeline.types import LiveConfigHolder
    from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
    from app.transport.redis_streams import FrameReady

    now = datetime.now(UTC)
    frame = FrameReady(
        camera_id="cam-uncal",
        minio_key="",
        frame_index=1,
        capture_time_unix_ns=int(now.timestamp() * 1e9),
        received_time_unix_ns=int(now.timestamp() * 1e9),
        width=640,
        height=480,
    )
    ctx = FrameContext(
        frame=frame,
        event_time=now,
        capture_time=now,
        effective_width=640,
        effective_height=480,
    )
    ctx.domain_detections = [
        Detection(
            detection_id="det-1",
            camera_id="cam-uncal",
            bbox=BoundingBox(x_min=100, y_min=50, x_max=250, y_max=400),
            confidence=0.9,
            floor_point=FloorPoint(0, 0, calibrated=False),  # uncalibrated
            embedding=[],
            capture_time=now,
            event_time=now,
        )
    ]

    # Tracker mock captures what observations it receives.
    captured_observations: list = []

    empty_result = WorldTrackerResult(updated_phs=[], snapshots=[], continuations=[])

    tracker_mock = MagicMock()
    tracker_mock.step = AsyncMock(
        side_effect=lambda observations, **kw: (
            captured_observations.extend(observations) or empty_result
        )
    )

    stage = WorldTrackingStage(
        tracker=tracker_mock,
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
    )
    await stage.run(ctx)

    assert len(captured_observations) == 1, (
        "Uncalibrated detection must reach the WorldTracker; "
        "fix ensures a synthetic floor point is generated"
    )
    obs = captured_observations[0]
    assert obs.floor_point.x_mm != 0 or obs.floor_point.y_mm != 0, (
        "Synthetic floor point must have non-zero coordinates to prevent "
        "all detections collapsing to origin (0, 0)"
    )
    assert obs.floor_cov_random is None
    assert not obs.footpoint_reliable
    assert ctx.geometry_by_detection == {}


def test_build_observations_populates_geometry_for_calibrated_detection():
    from unittest.mock import MagicMock

    from app.pipeline.frame_context import FrameContext
    from app.pipeline.stages.world_tracking import WorldTrackingStage
    from app.pipeline.types import LiveConfigHolder
    from app.services.camera_room_map import CameraRoomMap, RoomPolygonMap
    from app.transport.redis_streams import FrameReady

    now = datetime.now(UTC)
    state = CalibrationState()
    h = np.array([[0.01, 0.0, -1.0], [0.0, 0.01, -2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    state.homographies["cam-cal"] = h.tolist()
    projector = FloorProjector(state)
    frame = FrameReady(
        camera_id="cam-cal",
        minio_key="",
        frame_index=1,
        capture_time_unix_ns=int(now.timestamp() * 1e9),
        received_time_unix_ns=int(now.timestamp() * 1e9),
        width=640,
        height=480,
    )
    ctx = FrameContext(
        frame=frame,
        event_time=now,
        capture_time=now,
        effective_width=640,
        effective_height=480,
    )
    bbox = BoundingBox(x_min=100, y_min=50, x_max=250, y_max=400)
    ctx.domain_detections = [
        Detection(
            detection_id="det-cal",
            camera_id="cam-cal",
            bbox=bbox,
            confidence=0.9,
            floor_point=projector.project("cam-cal", bbox),
            embedding=[0.1, 0.2],
            capture_time=now,
            event_time=now,
            crop_quality=0.8,
            floor_residual_m=0.05,
        )
    ]
    ctx.det_pose_result["det-cal"] = _pose()
    ctx.orientation_by_detection["det-cal"] = (OrientationBin.LEFT, 0.7)
    stage = WorldTrackingStage(
        tracker=MagicMock(),
        live_config=LiveConfigHolder(CameraRoomMap(), RoomPolygonMap()),
        floor_projector=projector,
    )

    observations, uncalibrated_count = stage._build_observations(ctx)

    assert uncalibrated_count == 0
    assert len(observations) == 1
    obs = observations[0]
    assert obs.floor_cov_random is not None
    assert obs.footpoint_reliable
    assert ctx.geometry_by_detection["det-cal"].footpoint_reliable
    assert ctx.geometry_by_detection["det-cal"].orientation == OrientationBin.LEFT
    expected = (
        homography_jacobian(h, bbox.center_x, float(bbox.y_max))
        @ pixel_covariance(ctx.geometry_by_detection["det-cal"])
        @ homography_jacobian(h, bbox.center_x, float(bbox.y_max)).T
    )
    np.testing.assert_allclose(tuple_to_cov2x2(obs.floor_cov_random), expected)
