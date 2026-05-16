"""Tests for MotionEnergyTracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.inference.schemas import Keypoint, PoseResult
from app.trajectory.motion_energy import MotionEnergyTracker

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


def _kp(x: float, y: float, score: float = 0.9) -> Keypoint:
    return Keypoint(x=x, y=y, score=score)


def _pose(**overrides: Keypoint) -> PoseResult:
    kps = {name: _kp(0.5, 0.5) for name in _COCO_NAMES}
    for name, kp in overrides.items():
        kps[name] = kp
    return PoseResult(keypoints=tuple(kps[name] for name in _COCO_NAMES))


class TestMotionEnergyTracker:
    def test_single_frame_zero_energy(self) -> None:
        tracker = MotionEnergyTracker()
        pose = _pose()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        energy = tracker.update("gt-001", pose, t0, bbox_diag_px=200.0)
        assert energy.mean_keypoint_velocity_px_s == 0.0
        assert energy.still_fraction == 1.0
        assert energy.sample_count == 1

    def test_still_sequence_full_still_fraction(self) -> None:
        tracker = MotionEnergyTracker()
        bbox_diag = 50.0
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        # Feed identical poses → zero velocity → full still_fraction
        pose = _pose(
            left_shoulder=_kp(0.4, 0.3),
            right_shoulder=_kp(0.6, 0.3),
            left_hip=_kp(0.4, 0.55),
            right_hip=_kp(0.6, 0.55),
        )
        for i in range(10):
            t = t0 + timedelta(seconds=i * 0.2)
            tracker.update("gt-001", pose, t, bbox_diag_px=bbox_diag)
        energy = tracker.update("gt-001", pose, t0 + timedelta(seconds=2.0), bbox_diag_px=bbox_diag)
        # Identical poses produce zero velocity → all frames count as still
        assert energy.still_fraction == 1.0

    def test_moving_sequence_high_velocity(self) -> None:
        tracker = MotionEnergyTracker()
        # Small bbox diagonal → less normalization → higher normalized velocity.
        bbox_diag = 50.0
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        # Rapid joint movement: shift keypoints each frame
        for i in range(10):
            shift = i * 0.08
            pose = _pose(
                left_shoulder=_kp(0.4 + shift, 0.3),
                right_shoulder=_kp(0.6 + shift, 0.3),
                left_hip=_kp(0.4 + shift, 0.55),
                right_hip=_kp(0.6 + shift, 0.55),
                left_knee=_kp(0.4 + shift, 0.7),
                right_knee=_kp(0.6 + shift, 0.7),
                left_ankle=_kp(0.4 + shift, 0.85),
                right_ankle=_kp(0.6 + shift, 0.85),
            )
            t = t0 + timedelta(seconds=i * 0.2)
            tracker.update("gt-001", pose, t, bbox_diag_px=bbox_diag)
        energy = tracker.update("gt-001", pose, t0 + timedelta(seconds=2.0), bbox_diag_px=bbox_diag)
        # Should have significant velocity
        assert energy.max_joint_velocity_px_s > 0.0
        assert energy.still_fraction < 0.5

    def test_scale_invariance(self) -> None:
        """Same keypoint displacement at 2x bbox size → same normalized energy."""
        tracker_small = MotionEnergyTracker()
        tracker_large = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)

        pose_a = _pose(
            left_shoulder=_kp(0.4, 0.3),
            right_shoulder=_kp(0.6, 0.3),
            left_hip=_kp(0.4, 0.55),
            right_hip=_kp(0.6, 0.55),
            left_knee=_kp(0.4, 0.7),
            right_knee=_kp(0.6, 0.7),
            left_ankle=_kp(0.4, 0.85),
            right_ankle=_kp(0.6, 0.85),
        )
        pose_b = _pose(
            left_shoulder=_kp(0.5, 0.3),
            right_shoulder=_kp(0.7, 0.3),
            left_hip=_kp(0.5, 0.55),
            right_hip=_kp(0.7, 0.55),
            left_knee=_kp(0.5, 0.7),
            right_knee=_kp(0.7, 0.7),
            left_ankle=_kp(0.5, 0.85),
            right_ankle=_kp(0.7, 0.85),
        )

        tracker_small.update("gt-001", pose_a, t0, bbox_diag_px=50.0)
        e_small = tracker_small.update(
            "gt-001", pose_b, t0 + timedelta(seconds=0.2), bbox_diag_px=50.0
        )

        tracker_large.update("gt-001", pose_a, t0, bbox_diag_px=100.0)
        e_large = tracker_large.update(
            "gt-001", pose_b, t0 + timedelta(seconds=0.2), bbox_diag_px=100.0
        )

        # Same normalized displacement → roughly equal energy
        assert abs(e_small.mean_keypoint_velocity_px_s - e_large.mean_keypoint_velocity_px_s) < 0.01

    def test_evict_stale_track(self) -> None:
        tracker = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        pose = _pose()
        tracker.update("gt-001", pose, t0, bbox_diag_px=200.0)

        # Update another track far in the future → triggers eviction of gt-001
        t_future = t0 + timedelta(seconds=400)
        tracker.update("gt-002", pose, t_future, bbox_diag_px=200.0)

        # gt-001 should be evicted
        assert "gt-001" not in tracker._history
        assert "gt-002" in tracker._history

    def test_multiple_tracks_independent(self) -> None:
        tracker = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        pose_still = _pose()
        pose_move = _pose(
            left_shoulder=_kp(0.4, 0.3),
            right_shoulder=_kp(0.6, 0.3),
            left_hip=_kp(0.4, 0.55),
            right_hip=_kp(0.6, 0.55),
            left_knee=_kp(0.4, 0.7),
            right_knee=_kp(0.6, 0.7),
            left_ankle=_kp(0.4, 0.85),
            right_ankle=_kp(0.6, 0.85),
        )

        bbox_diag = 50.0
        tracker.update("gt-still", pose_still, t0, bbox_diag_px=bbox_diag)
        tracker.update("gt-move", pose_move, t0, bbox_diag_px=bbox_diag)

        # Still track: same pose
        e_still = tracker.update(
            "gt-still", pose_still, t0 + timedelta(seconds=0.2), bbox_diag_px=bbox_diag
        )
        # Moving track: large shift
        shift_pose = _pose(
            left_shoulder=_kp(0.5, 0.3),
            right_shoulder=_kp(0.7, 0.3),
            left_hip=_kp(0.5, 0.55),
            right_hip=_kp(0.7, 0.55),
            left_knee=_kp(0.5, 0.7),
            right_knee=_kp(0.7, 0.7),
            left_ankle=_kp(0.5, 0.85),
            right_ankle=_kp(0.7, 0.85),
        )
        e_move = tracker.update(
            "gt-move", shift_pose, t0 + timedelta(seconds=0.2), bbox_diag_px=bbox_diag
        )

        assert e_still.still_fraction > e_move.still_fraction
