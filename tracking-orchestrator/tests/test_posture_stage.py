"""Tests for PostureStage — the new pipeline stage that replaces the posture half
of PostureAndTrailsStage."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from app.domain import BoundingBox, Detection
from app.inference.schemas import COCO_KEYPOINTS, Keypoint, PoseResult
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.posture_stage import PostureStage
from app.trajectory.posture import PostureScores


def _kp(x: float = 0.5, y: float = 0.5, score: float = 0.9) -> Keypoint:
    return Keypoint(x=x, y=y, score=score)


def _make_ctx(detections: list[Detection], pose_results: dict[str, PoseResult]) -> FrameContext:
    ctx = FrameContext(
        frame=MagicMock(),
        event_time=datetime.now(UTC),
        capture_time=datetime.now(UTC),
    )
    ctx.domain_detections = detections
    ctx.det_pose_result = pose_results
    ctx.image = np.zeros((480, 640, 3), dtype=np.uint8)
    return ctx


def _det(det_id: str) -> Detection:
    return Detection(
        detection_id=det_id,
        camera_id="cam-1",
        bbox=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=200),
        embedding=[],
        capture_time=datetime.now(UTC),
        event_time=datetime.now(UTC),
        confidence=0.9,
        tracklet_id="",
        ph_id=None,
    )


def _standing_pose() -> PoseResult:
    kps = {name: _kp() for name in COCO_KEYPOINTS}
    kps.update(
        {
            "nose": _kp(100, 10),
            "left_shoulder": _kp(90, 30),
            "right_shoulder": _kp(110, 30),
            "left_hip": _kp(92, 80),
            "right_hip": _kp(108, 80),
            "left_knee": _kp(91, 110),
            "right_knee": _kp(109, 110),
            "left_ankle": _kp(91, 140),
            "right_ankle": _kp(109, 140),
            "left_eye": _kp(96, 8),
            "right_eye": _kp(104, 8),
            "left_ear": _kp(88, 9),
            "right_ear": _kp(112, 9),
            "left_elbow": _kp(85, 50),
            "right_elbow": _kp(115, 50),
            "left_wrist": _kp(82, 70),
            "right_wrist": _kp(118, 70),
        }
    )
    return PoseResult(keypoints=tuple(kps[name] for name in COCO_KEYPOINTS))


@pytest.mark.asyncio
async def test_posture_stage_calls_strategy_for_every_detection() -> None:
    """PostureStage must call strategy.score() for every detection regardless of
    whether the detection is already in ctx.det_posture."""
    mock_strategy = AsyncMock()
    mock_strategy.score = AsyncMock(
        return_value=PostureScores(lying=0.0, sitting=0.0, standing_walking=0.8)
    )
    stage = PostureStage(posture_strategy=mock_strategy)
    dets = [_det("d1"), _det("d2")]
    ctx = _make_ctx(dets, {})
    ctx.det_posture["d1"] = "standing"

    await stage.run(ctx)

    assert mock_strategy.score.call_count == 2
    assert "d1" in ctx.det_posture_scores
    assert "d2" in ctx.det_posture_scores


@pytest.mark.asyncio
async def test_posture_stage_falls_back_to_score_posture_without_strategy() -> None:
    """When no strategy is provided, PostureStage uses score_posture() directly."""
    stage = PostureStage(posture_strategy=None)
    det = _det("d1")
    pose = _standing_pose()
    ctx = _make_ctx([det], {"d1": pose})
    await stage.run(ctx)
    assert "d1" in ctx.det_posture_scores
    s = ctx.det_posture_scores["d1"]
    assert isinstance(s, PostureScores)


@pytest.mark.asyncio
async def test_posture_stage_unknown_when_no_pose_and_no_strategy() -> None:
    stage = PostureStage(posture_strategy=None)
    det = _det("d1")
    ctx = _make_ctx([det], {})  # no pose result
    await stage.run(ctx)
    s = ctx.det_posture_scores["d1"]
    assert s.lying == 0.0
    assert s.sitting == 0.0
    assert s.standing_walking == 0.0
