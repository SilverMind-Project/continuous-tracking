"""Tests for MotionEnergyTracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain import BoundingBox
from app.inference.schemas import Keypoint, PoseResult
from app.trajectory.motion_energy import _STILL_VELOCITY_FLOOR_NU_S, MotionEnergyTracker

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


def _bbox(x_min: int = 100, y_min: int = 50, width: int = 120, height: int = 280) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_min + width, y_max=y_min + height)


class TestMotionEnergyTracker:
    def test_single_frame_zero_energy(self) -> None:
        tracker = MotionEnergyTracker()
        pose = _pose()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        energy = tracker.update("gt-001", pose, t0, _bbox())
        assert energy.mean_keypoint_velocity_nu_s == 0.0
        assert energy.still_fraction == 1.0
        assert energy.sample_count == 1

    def test_still_sequence_full_still_fraction(self) -> None:
        tracker = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        bbox = _bbox()
        pose = _pose(
            left_shoulder=_kp(0.4, 0.3),
            right_shoulder=_kp(0.6, 0.3),
            left_hip=_kp(0.4, 0.55),
            right_hip=_kp(0.6, 0.55),
        )
        for i in range(10):
            t = t0 + timedelta(seconds=i * 0.2)
            tracker.update("gt-001", pose, t, bbox)
        energy = tracker.update("gt-001", pose, t0 + timedelta(seconds=2.0), bbox)
        assert energy.still_fraction == 1.0

    def test_moving_sequence_high_velocity(self) -> None:
        tracker = MotionEnergyTracker()
        # Large absolute displacements via large bbox + shifting keypoints.
        bbox = _bbox(x_min=0, y_min=0, width=300, height=600)
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
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
            tracker.update("gt-001", pose, t, bbox)
        energy = tracker.update("gt-001", pose, t0 + timedelta(seconds=2.0), bbox)
        assert energy.max_joint_velocity_nu_s > 0.0
        assert energy.still_fraction < 0.5

    def test_scale_invariance(self) -> None:
        """Identical relative motion rendered at different bbox sizes produces equal energy."""
        tracker_small = MotionEnergyTracker()
        tracker_large = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)

        # pose_a and pose_b have the same normalized keypoints.
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

        # Small bbox: 50x100, origin (0,0)
        bbox_small = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=100)
        # Large bbox: 150x300, origin (0,0) -- same proportional motion, 3x scale
        bbox_large = BoundingBox(x_min=0, y_min=0, x_max=150, y_max=300)

        tracker_small.update("gt-001", pose_a, t0, bbox_small)
        e_small = tracker_small.update("gt-001", pose_b, t0 + timedelta(seconds=0.2), bbox_small)

        tracker_large.update("gt-001", pose_a, t0, bbox_large)
        e_large = tracker_large.update("gt-001", pose_b, t0 + timedelta(seconds=0.2), bbox_large)

        # Equal within floating-point rounding.
        assert abs(e_small.mean_keypoint_velocity_nu_s - e_large.mean_keypoint_velocity_nu_s) < 1e-6

    def test_crop_tracking_jitter_stationary_person(self) -> None:
        """Person stationary in world: bbox translates, keypoints constant in crop.

        Absolute position changes because the bbox shifted, so energy > 0.
        """
        tracker = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        # Same normalized keypoints in both frames (person hasn't moved in crop).
        pose = _pose(
            left_shoulder=_kp(0.4, 0.3),
            right_shoulder=_kp(0.6, 0.3),
            left_hip=_kp(0.4, 0.55),
            right_hip=_kp(0.6, 0.55),
        )
        # Frame 0: bbox at (100, 50)
        bbox0 = BoundingBox(x_min=100, y_min=50, x_max=220, y_max=330)
        # Frame 1: bbox translated 30px right -- crop jitter.
        bbox1 = BoundingBox(x_min=130, y_min=50, x_max=250, y_max=330)
        tracker.update("gt-001", pose, t0, bbox0)
        energy = tracker.update("gt-001", pose, t0 + timedelta(seconds=0.2), bbox1)
        # Person moved in absolute space (bbox shifted), but keypoints moved with the bbox:
        # absolute displacement ≠ zero, so energy > 0. This tests that we use absolute coords.
        assert energy.mean_keypoint_velocity_nu_s > 0.0

    def test_crop_tracking_jitter_moving_person_pinned_bbox(self) -> None:
        """Person moves in world, bbox is pinned: keypoint motion registers correctly."""
        tracker = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        bbox = BoundingBox(x_min=100, y_min=50, x_max=220, y_max=330)
        pose_a = _pose(left_shoulder=_kp(0.4, 0.3), right_shoulder=_kp(0.6, 0.3))
        pose_b = _pose(left_shoulder=_kp(0.5, 0.35), right_shoulder=_kp(0.7, 0.35))
        tracker.update("gt-001", pose_a, t0, bbox)
        energy = tracker.update("gt-001", pose_b, t0 + timedelta(seconds=0.2), bbox)
        assert energy.mean_keypoint_velocity_nu_s > 0.0

    def test_still_fraction_static_sequence(self) -> None:
        """Fully static sequence yields still_fraction == 1.0."""
        tracker = MotionEnergyTracker()
        bbox = _bbox()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        pose = _pose()
        for i in range(5):
            energy = tracker.update("gt-001", pose, t0 + timedelta(seconds=i * 0.2), bbox)
        assert energy.still_fraction == 1.0

    def test_still_fraction_mixed(self) -> None:
        """Still pairs and moving pairs produce a still_fraction strictly between 0 and 1."""
        tracker = MotionEnergyTracker()
        bbox = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=400)
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        pose_still = _pose()  # all keypoints at (0.5, 0.5) -- zero displacement between same pose
        # Moving pose: all 12 rapid-joint keypoints shifted by 0.15 crop units.
        # Abs displacement: sqrt((0.15*200)^2+(0.15*400)^2)=67px; diag≈447px; dt=0.2s
        # velocity = 67/447/0.2 ≈ 0.75 nu/s >> _STILL_VELOCITY_FLOOR_NU_S (0.05)
        pose_move = _pose(
            left_shoulder=_kp(0.65, 0.65),
            right_shoulder=_kp(0.65, 0.65),
            left_elbow=_kp(0.65, 0.65),
            right_elbow=_kp(0.65, 0.65),
            left_wrist=_kp(0.65, 0.65),
            right_wrist=_kp(0.65, 0.65),
            left_hip=_kp(0.65, 0.65),
            right_hip=_kp(0.65, 0.65),
            left_knee=_kp(0.65, 0.65),
            right_knee=_kp(0.65, 0.65),
            left_ankle=_kp(0.65, 0.65),
            right_ankle=_kp(0.65, 0.65),
        )
        # Sequence: still, still, move, move → 3 pairs: (still, move, still) → fraction = 2/3
        poses = [pose_still, pose_still, pose_move, pose_move]
        energy = None
        for i, p in enumerate(poses):
            energy = tracker.update("gt-001", p, t0 + timedelta(seconds=i * 0.2), bbox)
        assert energy is not None
        assert 0.0 < energy.still_fraction < 1.0

    def test_incremental_equals_naive(self) -> None:
        """Incremental per-pair output matches naive full-history recomputation."""
        import math

        import numpy as np

        from app.trajectory.motion_energy import _RAPID_JOINTS

        tracker = MotionEnergyTracker()
        bbox = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=400)
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        np.random.seed(42)

        history: list[tuple[datetime, np.ndarray]] = []
        for i in range(15):
            # Random keypoints in [0, 1]
            raw_kps = np.random.uniform(0.1, 0.9, (17, 2))
            abs_kps = raw_kps.copy()
            abs_kps[:, 0] = bbox.x_min + raw_kps[:, 0] * bbox.width
            abs_kps[:, 1] = bbox.y_min + raw_kps[:, 1] * bbox.height
            pose = PoseResult(
                keypoints=tuple(
                    Keypoint(x=float(raw_kps[j, 0]), y=float(raw_kps[j, 1]), score=0.9)
                    for j in range(17)
                )
            )
            ts = t0 + timedelta(seconds=i * 0.2)
            energy = tracker.update("gt-001", pose, ts, bbox)
            history.append((ts, abs_kps))

        # Naive recomputation from final history snapshot.
        diag = math.hypot(bbox.width, bbox.height)
        rapid = list(_RAPID_JOINTS)
        naive_vels: list[float] = []
        for idx in range(1, len(history)):
            prev_ts, prev_kp = history[idx - 1]
            curr_ts, curr_kp = history[idx]
            dt = (curr_ts - prev_ts).total_seconds()
            if dt <= 0:
                continue
            disp = np.linalg.norm(curr_kp[rapid] - prev_kp[rapid], axis=1)
            naive_vels.append(float(np.mean(disp / diag) / dt))

        assert naive_vels
        naive_mean = float(np.mean(naive_vels))
        assert abs(energy.mean_keypoint_velocity_nu_s - round(naive_mean, 6)) < 1e-5

    def test_evict_stale_track(self) -> None:
        tracker = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        pose = _pose()
        tracker.update("gt-001", pose, t0, _bbox())

        t_future = t0 + timedelta(seconds=400)
        tracker.update("gt-002", pose, t_future, _bbox())

        assert "gt-001" not in tracker._history
        assert "gt-002" in tracker._history

    def test_multiple_tracks_independent(self) -> None:
        tracker = MotionEnergyTracker()
        t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
        bbox = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=400)
        pose_still = _pose()
        pose_move = _pose(
            left_shoulder=_kp(0.4, 0.3),
            right_shoulder=_kp(0.6, 0.3),
            left_knee=_kp(0.4, 0.7),
            right_knee=_kp(0.6, 0.7),
        )

        tracker.update("gt-still", pose_still, t0, bbox)
        tracker.update("gt-move", pose_move, t0, bbox)

        e_still = tracker.update("gt-still", pose_still, t0 + timedelta(seconds=0.2), bbox)
        shift_pose = _pose(
            left_shoulder=_kp(0.6, 0.4),
            right_shoulder=_kp(0.8, 0.4),
            left_knee=_kp(0.6, 0.8),
            right_knee=_kp(0.8, 0.8),
        )
        e_move = tracker.update("gt-move", shift_pose, t0 + timedelta(seconds=0.2), bbox)

        assert e_still.still_fraction > e_move.still_fraction

    def test_threshold_relationship(self) -> None:
        """Still velocity floor < walking threshold (sanity check on calibration constants)."""
        from app.trajectory.posture import _WALKING_VELOCITY_NU_S

        assert _STILL_VELOCITY_FLOOR_NU_S < _WALKING_VELOCITY_NU_S
