"""Table-driven tests for classify_posture."""

from __future__ import annotations

import pytest

from app.domain import BoundingBox
from app.inference.schemas import Keypoint, PoseResult
from app.trajectory.posture import classify_posture

# Helper: build a PoseResult with all 17 COCO keypoints at given positions.
# Unspecified keypoints default to (0.5, 0.5, score=1.0).

_COCO_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _keypoint(x: float = 0.5, y: float = 0.5, score: float = 0.9) -> Keypoint:
    return Keypoint(x=x, y=y, score=score)


def _pose(**overrides: Keypoint) -> PoseResult:
    """Build a PoseResult with defaults, overriding named keypoints."""
    kps = {name: _keypoint() for name in _COCO_NAMES}
    for name, kp in overrides.items():
        kps[name] = kp
    return PoseResult(keypoints=tuple(kps[name] for name in _COCO_NAMES))


_BBOX_PORTRAIT = BoundingBox(x_min=100, y_min=100, x_max=300, y_max=500)
_BBOX_WIDE = BoundingBox(x_min=100, y_min=100, x_max=500, y_max=250)


class TestClassifyPosture:
    def test_standing_vertical_torso(self) -> None:
        """Near-vertical torso with ankles below knees below hips."""
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.3),
            right_shoulder=_keypoint(0.6, 0.3),
            left_hip=_keypoint(0.4, 0.55),
            right_hip=_keypoint(0.6, 0.55),
            left_knee=_keypoint(0.4, 0.7),
            right_knee=_keypoint(0.6, 0.7),
            left_ankle=_keypoint(0.4, 0.85),
            right_ankle=_keypoint(0.6, 0.85),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "standing"

    def test_sitting_small_hip_knee_span(self) -> None:
        """Knees close to hips vertically → sitting."""
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.25),
            right_shoulder=_keypoint(0.6, 0.25),
            left_hip=_keypoint(0.4, 0.55),
            right_hip=_keypoint(0.6, 0.55),
            left_knee=_keypoint(0.4, 0.62),
            right_knee=_keypoint(0.6, 0.62),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"

    def test_lying_horizontal_torso(self) -> None:
        """Torso near horizontal → lying."""
        pose = _pose(
            left_shoulder=_keypoint(0.3, 0.5),
            right_shoulder=_keypoint(0.3, 0.55),
            left_hip=_keypoint(0.7, 0.5),
            right_hip=_keypoint(0.7, 0.55),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "lying"

    def test_lying_wide_bbox(self) -> None:
        """Wide-short bbox aspect ratio > 1.2 → lying."""
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.3),
            right_shoulder=_keypoint(0.6, 0.3),
            left_hip=_keypoint(0.4, 0.55),
            right_hip=_keypoint(0.6, 0.55),
            left_knee=_keypoint(0.4, 0.7),
            right_knee=_keypoint(0.6, 0.7),
            left_ankle=_keypoint(0.4, 0.85),
            right_ankle=_keypoint(0.6, 0.85),
        )
        # Normal pose but wide bbox → lying
        assert classify_posture(pose, _BBOX_WIDE) == "lying"

    def test_unknown_low_confidence_keypoints(self) -> None:
        """All keypoints below score floor → unknown."""
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.3, score=0.1),
            right_shoulder=_keypoint(0.6, 0.3, score=0.1),
            left_hip=_keypoint(0.4, 0.55, score=0.1),
            right_hip=_keypoint(0.6, 0.55, score=0.1),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "unknown"

    def test_standing_fallback_vertical_torso_only(self) -> None:
        """Near-vertical torso with no knee/ankle info → standing fallback."""
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.3),
            right_shoulder=_keypoint(0.6, 0.3),
            left_hip=_keypoint(0.4, 0.55),
            right_hip=_keypoint(0.6, 0.55),
            left_knee=_keypoint(0.4, 0.7, score=0.1),
            right_knee=_keypoint(0.6, 0.7, score=0.1),
            left_ankle=_keypoint(0.4, 0.85, score=0.1),
            right_ankle=_keypoint(0.6, 0.85, score=0.1),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "standing"
