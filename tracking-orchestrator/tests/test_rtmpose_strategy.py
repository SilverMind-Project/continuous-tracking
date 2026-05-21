"""Unit tests for RTMPosePostureStrategy."""

from __future__ import annotations

import numpy as np
import pytest

from app.domain import BoundingBox, Detection
from app.inference.schemas import Keypoint, PoseResult
from app.trajectory.posture_strategy import RTMPosePostureStrategy


def _make_detection() -> Detection:
    return Detection(
        detection_id="det-1",
        camera_id="cam-1",
        bbox=BoundingBox(x_min=100, y_min=100, x_max=300, y_max=400),
        embedding=[],
        capture_time=None,  # type: ignore[arg-type]
        event_time=None,  # type: ignore[arg-type]
    )


def _standing_pose() -> PoseResult:
    """Near-vertical torso, ankles below knees below hips."""
    _coco = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    )
    kps = {
        "nose": Keypoint(0.5, 0.15, 0.9),
        "left_eye": Keypoint(0.45, 0.12, 0.9),
        "right_eye": Keypoint(0.55, 0.12, 0.9),
        "left_ear": Keypoint(0.4, 0.15, 0.9),
        "right_ear": Keypoint(0.6, 0.15, 0.9),
        "left_shoulder": Keypoint(0.4, 0.3, 0.9),
        "right_shoulder": Keypoint(0.6, 0.3, 0.9),
        "left_elbow": Keypoint(0.35, 0.45, 0.9),
        "right_elbow": Keypoint(0.65, 0.45, 0.9),
        "left_wrist": Keypoint(0.3, 0.55, 0.9),
        "right_wrist": Keypoint(0.7, 0.55, 0.9),
        "left_hip": Keypoint(0.4, 0.55, 0.9),
        "right_hip": Keypoint(0.6, 0.55, 0.9),
        "left_knee": Keypoint(0.4, 0.75, 0.9),
        "right_knee": Keypoint(0.6, 0.75, 0.9),
        "left_ankle": Keypoint(0.4, 0.9, 0.9),
        "right_ankle": Keypoint(0.6, 0.9, 0.9),
    }
    return PoseResult(keypoints=tuple(kps[name] for name in _coco))


@pytest.mark.asyncio
async def test_rtmpose_strategy_delegates_to_classifier():
    strategy = RTMPosePostureStrategy()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = _make_detection()
    pose = _standing_pose()
    result = await strategy.infer(frame, detection, pose_result=pose)
    assert result == "standing"


@pytest.mark.asyncio
async def test_rtmpose_strategy_returns_unknown_when_no_pose():
    strategy = RTMPosePostureStrategy()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = _make_detection()
    result = await strategy.infer(frame, detection, pose_result=None)
    assert result == "unknown"


@pytest.mark.asyncio
async def test_rtmpose_strategy_evict_tracklet_is_noop():
    strategy = RTMPosePostureStrategy()
    strategy.evict_tracklet("t1")  # must not raise


def test_rtmpose_strategy_name():
    strategy = RTMPosePostureStrategy()
    assert strategy.name == "rtmpose"
