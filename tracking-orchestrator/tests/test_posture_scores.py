"""Tests for PostureScores and score_posture()."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.inference.schemas import COCO_KEYPOINTS, Keypoint, PoseResult
from app.trajectory.posture import (
    GlobalPostureTracker,
    PostureScores,
    classify_posture,
    score_posture,
)


def _kp(x: float = 0.5, y: float = 0.5, score: float = 0.9) -> Keypoint:
    return Keypoint(x=x, y=y, score=score)


def _pose(**overrides: Keypoint) -> PoseResult:
    """Build a PoseResult with defaults, overriding named keypoints."""
    kps = {name: _kp() for name in COCO_KEYPOINTS}
    for name, kp in overrides.items():
        kps[name] = kp
    return PoseResult(keypoints=tuple(kps[name] for name in COCO_KEYPOINTS))


def _standing_pose() -> PoseResult:
    """Full-body upright pose with all keypoints visible."""
    return _pose(
        nose=_kp(100, 10),
        left_shoulder=_kp(90, 30),
        right_shoulder=_kp(110, 30),
        left_elbow=_kp(85, 50),
        right_elbow=_kp(115, 50),
        left_wrist=_kp(82, 70),
        right_wrist=_kp(118, 70),
        left_hip=_kp(92, 80),
        right_hip=_kp(108, 80),
        left_knee=_kp(91, 110),
        right_knee=_kp(109, 110),
        left_ankle=_kp(91, 140),
        right_ankle=_kp(109, 140),
        left_eye=_kp(96, 8),
        right_eye=_kp(104, 8),
        left_ear=_kp(88, 9),
        right_ear=_kp(112, 9),
    )


def _lying_pose() -> PoseResult:
    """Horizontal torso — person lying flat."""
    return _pose(
        nose=_kp(10, 100),
        left_shoulder=_kp(30, 92),
        right_shoulder=_kp(30, 108),
        left_hip=_kp(80, 92),
        right_hip=_kp(80, 108),
        left_knee=_kp(110, 91),
        right_knee=_kp(110, 109),
        left_ankle=_kp(140, 91),
        right_ankle=_kp(140, 109),
        left_elbow=_kp(50, 90),
        right_elbow=_kp(50, 110),
        left_wrist=_kp(20, 90),
        right_wrist=_kp(20, 110),
        left_eye=_kp(8, 96),
        right_eye=_kp(8, 104),
        left_ear=_kp(9, 88),
        right_ear=_kp(9, 112),
    )


class TestPostureScoresDataclass:
    def test_frozen(self) -> None:
        s = PostureScores(lying=0.3, sitting=0.7, standing_walking=0.1)
        with pytest.raises(FrozenInstanceError):
            s.lying = 0.5  # type: ignore[misc]

    def test_fields_bounded(self) -> None:
        s = PostureScores(lying=0.0, sitting=1.0, standing_walking=0.0, keypoint_confidence=0.85)
        assert 0.0 <= s.lying <= 1.0
        assert 0.0 <= s.sitting <= 1.0
        assert 0.0 <= s.standing_walking <= 1.0
        assert 0.0 <= s.keypoint_confidence <= 1.0


class TestScorePosture:
    def test_standing_pose_has_standing_walking_dominant(self) -> None:
        scores = score_posture(_standing_pose())
        assert scores.standing_walking > scores.lying
        assert scores.standing_walking > scores.sitting

    def test_lying_pose_has_lying_dominant(self) -> None:
        scores = score_posture(_lying_pose())
        assert scores.lying > scores.sitting
        assert scores.lying > scores.standing_walking

    def test_keypoint_confidence_positive_for_visible_pose(self) -> None:
        scores = score_posture(_standing_pose())
        assert scores.keypoint_confidence > 0.0

    def test_classify_posture_consistent_with_score_posture_standing(self) -> None:
        pose = _standing_pose()
        label = classify_posture(pose)
        scores = score_posture(pose)
        assert label in ("standing", "walking")
        assert scores.standing_walking >= scores.lying
        assert scores.standing_walking >= scores.sitting

    def test_classify_posture_consistent_with_score_posture_lying(self) -> None:
        pose = _lying_pose()
        label = classify_posture(pose)
        scores = score_posture(pose)
        assert label == "lying"
        assert scores.lying >= scores.sitting
        assert scores.lying >= scores.standing_walking


class TestGlobalPostureTrackerScoresInterface:
    def test_update_accepts_posture_scores(self) -> None:
        tracker = GlobalPostureTracker(required_consecutive=1)
        scores = PostureScores(
            lying=0.0, sitting=0.0, standing_walking=0.8, keypoint_confidence=0.9
        )
        result = tracker.update(
            global_track_id="gt-1",
            camera_id="cam-1",
            scores=scores,
            active_camera_ids=["cam-1"],
        )
        assert result in ("standing", "walking", "unknown")

    def test_update_lying_scores_resolves_to_lying(self) -> None:
        tracker = GlobalPostureTracker(required_consecutive=1)
        scores = PostureScores(lying=0.8, sitting=0.0, standing_walking=0.0)
        result = tracker.update("gt-1", "cam-1", scores, ["cam-1"])
        assert result == "lying"
