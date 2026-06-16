"""Tests for occlusion-aware posture scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.domain import BoundingBox, Detection
from app.inference.schemas import COCO_KEYPOINTS, Keypoint, PoseResult
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.posture_stage import PostureStage
from app.pipeline.types import LiveConfigHolder
from app.services.camera_room_map import (
    CameraRoomBinding,
    CameraRoomMap,
    RoomPolygonMap,
)
from app.trajectory.posture import (
    PostureScores,
    _score_upper_torso_lying,
    score_posture,
)


async def _camera_room_map(camera_id: str, room_name: str) -> CameraRoomMap:
    room_map = CameraRoomMap()
    await room_map.set_all(
        [
            CameraRoomBinding(
                camera_id=camera_id,
                room_id=room_name,
                room_name=room_name,
                bound_at=datetime.now(UTC),
            )
        ]
    )
    return room_map


async def _live_config(camera_id: str, room_name: str) -> LiveConfigHolder:
    return LiveConfigHolder(
        camera_room_map=await _camera_room_map(camera_id, room_name),
        room_polygon_map=RoomPolygonMap(),
    )


def _kp(x: float = 0.5, y: float = 0.5, score: float = 0.9) -> Keypoint:
    return Keypoint(x=x, y=y, score=score)


def _low_kp(x: float = 0.5, y: float = 0.5) -> Keypoint:
    """Keypoint with confidence below _SCORE_FLOOR (invisible)."""
    return Keypoint(x=x, y=y, score=0.1)


def _pose(**overrides: Keypoint) -> PoseResult:
    """Build a PoseResult with defaults, overriding named keypoints."""
    kps = {name: _kp() for name in COCO_KEYPOINTS}
    for name, kp in overrides.items():
        kps[name] = kp
    return PoseResult(keypoints=tuple(kps[name] for name in COCO_KEYPOINTS))


class TestScoreUpperTorsoLying:
    def test_horizontal_shoulders_no_hips_returns_positive(self) -> None:
        """Visible horizontal shoulders + invisible hips → partial lying evidence."""
        pose = _pose(
            left_shoulder=_kp(50, 100),  # same y → horizontal line
            right_shoulder=_kp(150, 100),
            nose=_kp(100, 105),  # head at shoulder level
            left_hip=_low_kp(50, 180),  # invisible
            right_hip=_low_kp(150, 180),  # invisible
        )
        sc = _score_upper_torso_lying(pose)
        assert sc > 0.0, f"Expected positive partial lying score, got {sc}"

    def test_visible_hips_returns_zero(self) -> None:
        """When hips are visible, upper_torso_lying must return 0.0."""
        pose = _pose(
            left_shoulder=_kp(50, 100),
            right_shoulder=_kp(150, 100),
            left_hip=_kp(60, 180),  # visible
            right_hip=_kp(140, 180),  # visible
        )
        assert _score_upper_torso_lying(pose) == 0.0

    def test_no_shoulders_returns_zero(self) -> None:
        pose = _pose(
            left_shoulder=_low_kp(50, 100),
            right_shoulder=_low_kp(150, 100),
            left_hip=_low_kp(50, 180),
            right_hip=_low_kp(150, 180),
        )
        assert _score_upper_torso_lying(pose) == 0.0

    def test_score_capped_at_06(self) -> None:
        """Partial body score must never exceed 0.6."""
        pose = _pose(
            left_shoulder=_kp(0, 100),
            right_shoulder=_kp(200, 100),
            nose=_kp(100, 100),
            left_hip=_low_kp(0, 200),
            right_hip=_low_kp(200, 200),
        )
        sc = _score_upper_torso_lying(pose)
        assert sc <= 0.6, f"Expected score <= 0.6, got {sc}"


class TestScorePostureWithBedroomPrior:
    def test_bedroom_prior_increases_lying_score(self) -> None:
        """score_posture(bedroom_prior=0.18) must produce higher lying than without."""
        pose = _pose(
            left_shoulder=_kp(50, 100),
            right_shoulder=_kp(150, 100),
            nose=_kp(100, 105),
            left_hip=_low_kp(50, 180),
            right_hip=_low_kp(150, 180),
        )
        scores_no_prior = score_posture(pose, bedroom_prior=0.0)
        scores_with_prior = score_posture(pose, bedroom_prior=0.18)
        assert scores_with_prior.lying >= scores_no_prior.lying

    def test_bedroom_prior_does_not_exceed_10(self) -> None:
        """Prior must not push lying score above 1.0."""
        pose = _pose(
            left_shoulder=_kp(50, 100),
            right_shoulder=_kp(150, 100),
            nose=_kp(100, 100),
            left_hip=_low_kp(50, 180),
            right_hip=_low_kp(150, 180),
        )
        scores = score_posture(pose, bedroom_prior=10.0)  # absurd prior
        assert scores.lying <= 1.0

    def test_bedroom_prior_does_not_override_confident_standing(self) -> None:
        """A strong standing signal must not be overridden by a bedroom prior."""
        pose = _pose(
            nose=_kp(100, 10),
            left_shoulder=_kp(90, 30),
            right_shoulder=_kp(110, 30),
            left_hip=_kp(92, 80),
            right_hip=_kp(108, 80),
            left_knee=_kp(91, 110),
            right_knee=_kp(109, 110),
            left_ankle=_kp(91, 140),
            right_ankle=_kp(109, 140),
        )
        scores = score_posture(pose, bedroom_prior=0.18)
        # Standing signal must dominate lying even with prior.
        assert scores.standing_walking > scores.lying


def _make_det(camera_id: str = "cam-1") -> Detection:
    return Detection(
        detection_id="d1",
        camera_id=camera_id,
        bbox=BoundingBox(x_min=10, y_min=50, x_max=300, y_max=200),
        embedding=[],
        capture_time=datetime.now(UTC),
        event_time=datetime.now(UTC),
        confidence=0.8,
        tracklet_id="",
        ph_id=None,
    )


class TestPostureStageBedroomPrior:
    @pytest.mark.asyncio
    async def test_bedroom_camera_adds_prior_to_occluded_detection(self) -> None:
        """When camera room is 'bedroom' and kp_confidence < threshold, lying score increases."""
        mock_strategy = AsyncMock()
        # Fast path returns unknown (all zeros, low confidence) — simulating occlusion.
        mock_strategy.score = AsyncMock(
            return_value=PostureScores(
                lying=0.0, sitting=0.0, standing_walking=0.0, keypoint_confidence=0.1
            )
        )

        stage = PostureStage(
            live_config=await _live_config("cam-bedroom", "bedroom"),
            posture_strategy=mock_strategy,
        )

        det = _make_det(camera_id="cam-bedroom")

        frame_mock = MagicMock()
        frame_mock.camera_id = "cam-bedroom"
        ctx = FrameContext(
            frame=frame_mock,
            event_time=datetime.now(UTC),
            capture_time=datetime.now(UTC),
        )
        ctx.domain_detections = [det]
        ctx.det_pose_result = {}
        ctx.image = np.zeros((480, 640, 3), dtype=np.uint8)

        await stage.run(ctx)

        scores = ctx.det_posture_scores["d1"]
        # The bedroom prior + aspect ratio boost must have added lying evidence.
        assert scores.lying > 0.0, (
            f"Expected lying > 0 for bedroom occluded detection, got {scores.lying}"
        )

    @pytest.mark.asyncio
    async def test_non_bedroom_camera_no_prior(self) -> None:
        """A non-bedroom camera must not add any lying prior."""
        mock_strategy = AsyncMock()
        mock_strategy.score = AsyncMock(
            return_value=PostureScores(
                lying=0.0, sitting=0.0, standing_walking=0.0, keypoint_confidence=0.1
            )
        )

        stage = PostureStage(
            live_config=await _live_config("cam-living", "living room"),
            posture_strategy=mock_strategy,
        )

        det = _make_det(camera_id="cam-living")

        frame_mock = MagicMock()
        frame_mock.camera_id = "cam-living"
        ctx = FrameContext(
            frame=frame_mock,
            event_time=datetime.now(UTC),
            capture_time=datetime.now(UTC),
        )
        ctx.domain_detections = [det]
        ctx.image = np.zeros((480, 640, 3), dtype=np.uint8)

        await stage.run(ctx)

        scores = ctx.det_posture_scores["d1"]
        assert scores.lying == 0.0, (
            f"Expected no lying prior for non-bedroom camera, got {scores.lying}"
        )
