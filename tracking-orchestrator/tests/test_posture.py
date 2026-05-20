"""Table-driven tests for classify_posture and PostureHysteresis."""

from __future__ import annotations

from app.domain import BoundingBox
from app.inference.schemas import Keypoint, PoseResult
from app.trajectory.posture import PostureHysteresis, classify_posture

# Helper: build a PoseResult with all 17 COCO keypoints at given positions.
# Unspecified keypoints default to (0.5, 0.5, score=0.9).

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


# ---------------------------------------------------------------------------
# classify_posture (stateless, per-frame)
# ---------------------------------------------------------------------------


class TestClassifyPosture:
    def test_standing_vertical_torso(self) -> None:
        """Near-vertical torso with ankles below knees below hips → standing."""
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

    def test_walking_from_motion_energy(self) -> None:
        """Same standing pose but with high motion energy → walking."""
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
        assert classify_posture(pose, _BBOX_PORTRAIT, motion_energy=0.012) == "walking"

    def test_walking_from_motion_energy_no_ankles(self) -> None:
        """No ankle keypoints visible, but vertical torso + high motion → walking."""
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.3),
            right_shoulder=_keypoint(0.6, 0.3),
            left_hip=_keypoint(0.4, 0.55),
            right_hip=_keypoint(0.6, 0.55),
            left_knee=_keypoint(0.4, 0.7),
            right_knee=_keypoint(0.6, 0.7),
            left_ankle=_keypoint(0.4, 0.85, score=0.1),
            right_ankle=_keypoint(0.6, 0.85, score=0.1),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT, motion_energy=0.012) == "walking"

    def test_sitting_from_bent_knee(self) -> None:
        """Knee bent ~90° (chair sitting pose)."""
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.2),
            right_shoulder=_keypoint(0.6, 0.2),
            left_hip=_keypoint(0.4, 0.5),
            right_hip=_keypoint(0.6, 0.5),
            # Thighs extend forward (x > hip), knees slightly below hips.
            left_knee=_keypoint(0.65, 0.55),
            right_knee=_keypoint(0.75, 0.55),
            # Shins point down from knees.
            left_ankle=_keypoint(0.65, 0.85),
            right_ankle=_keypoint(0.75, 0.85),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"

    def test_sitting_from_tilted_torso(self) -> None:
        """Torso tilted >30° with bent knees → sitting."""
        pose = _pose(
            left_shoulder=_keypoint(0.25, 0.2),
            right_shoulder=_keypoint(0.45, 0.2),
            left_hip=_keypoint(0.55, 0.55),
            right_hip=_keypoint(0.75, 0.55),
            left_knee=_keypoint(0.65, 0.55),
            right_knee=_keypoint(0.85, 0.55),
            left_ankle=_keypoint(0.65, 0.85),
            right_ankle=_keypoint(0.85, 0.85),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"

    def test_lying_horizontal_torso_with_head(self) -> None:
        """Torso near horizontal AND head in line with torso → lying."""
        pose = _pose(
            nose=_keypoint(0.25, 0.5),
            left_shoulder=_keypoint(0.25, 0.5),
            right_shoulder=_keypoint(0.35, 0.55),
            left_hip=_keypoint(0.65, 0.5),
            right_hip=_keypoint(0.75, 0.55),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "lying"

    def test_tilted_torso_not_lying_without_head_alignment(self) -> None:
        """Torso near horizontal but head well above torso → not lying.

        The tilted torso with visible knees close to hips triggers the
        weak sitting signal instead.
        """
        pose = _pose(
            nose=_keypoint(0.3, 0.15),  # head high up, not aligned with torso
            left_shoulder=_keypoint(0.25, 0.5),
            right_shoulder=_keypoint(0.35, 0.55),
            left_hip=_keypoint(0.65, 0.5),
            right_hip=_keypoint(0.75, 0.55),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"

    def test_wide_bbox_does_not_force_lying(self) -> None:
        """A wide bbox with standing pose is NOT classified as lying."""
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
        assert classify_posture(pose, _BBOX_WIDE) == "standing"

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

    def test_sitting_knee_only_needs_higher_confidence(self) -> None:
        """Bent knee with score < 0.5 → knee-only sitting path blocked.

        Knees spread wide (thighs angled out) with vertical torso, so the
        standing path is also blocked by the ``knees_bent`` guard.  Result:
        unknown.
        """
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.2, score=0.45),
            right_shoulder=_keypoint(0.6, 0.2, score=0.45),
            left_hip=_keypoint(0.4, 0.5, score=0.45),
            right_hip=_keypoint(0.6, 0.5, score=0.45),
            # Thighs spread laterally → bent knee angle.
            left_knee=_keypoint(0.2, 0.6, score=0.45),
            right_knee=_keypoint(0.8, 0.6, score=0.45),
            left_ankle=_keypoint(0.2, 0.85, score=0.45),
            right_ankle=_keypoint(0.8, 0.85, score=0.45),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "unknown"

    def test_sitting_upright_foreshortened_knee(self) -> None:
        """Upright sitting: vertical torso, knees near hip height, ankles hanging below.

        This is the camera-projection failure mode: a person sitting upright in a
        chair viewed from the front or overhead produces a 2D knee angle well above
        130° (the depth axis collapses a true 90° bend into 140°+).  The shin-drop
        signal (knees near hips, ankles clearly below) must classify this as
        sitting without any torso tilt or a small 2D knee angle.
        """
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.2),
            right_shoulder=_keypoint(0.6, 0.2),
            left_hip=_keypoint(0.4, 0.5),
            right_hip=_keypoint(0.6, 0.5),
            # Knees just below hips (thighs roughly horizontal).
            left_knee=_keypoint(0.4, 0.6),
            right_knee=_keypoint(0.6, 0.6),
            # Ankles well below, shins hanging near-vertical.
            left_ankle=_keypoint(0.4, 0.9),
            right_ankle=_keypoint(0.6, 0.9),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"


# ---------------------------------------------------------------------------
# PostureHysteresis
# ---------------------------------------------------------------------------


class TestPostureHysteresis:
    def test_first_frame_commits_immediately(self) -> None:
        hyst = PostureHysteresis(required_consecutive=2)
        assert hyst.update("gt-1", "standing") == "standing"

    def test_consecutive_same_raw_returns_committed(self) -> None:
        hyst = PostureHysteresis(required_consecutive=3)
        assert hyst.update("gt-1", "standing") == "standing"  # frame 1, committed
        assert hyst.update("gt-1", "standing") == "standing"  # frame 2
        assert hyst.update("gt-1", "standing") == "standing"  # frame 3

    def test_flips_after_required_consecutive(self) -> None:
        hyst = PostureHysteresis(required_consecutive=2)
        assert hyst.update("gt-1", "standing") == "standing"  # committed
        # Frame 2: new candidate "sitting", but not yet committed.
        assert hyst.update("gt-1", "sitting") == "standing"
        # Frame 3: second consecutive "sitting" → commit.
        assert hyst.update("gt-1", "sitting") == "sitting"

    def test_resets_candidate_on_interruption(self) -> None:
        hyst = PostureHysteresis(required_consecutive=3)
        assert hyst.update("gt-1", "standing") == "standing"
        # Candidate "sitting", count=1.
        assert hyst.update("gt-1", "sitting") == "standing"
        # Candidate "sitting", count=2.
        assert hyst.update("gt-1", "sitting") == "standing"
        # Interrupted by "walking" → new candidate, count=1.
        assert hyst.update("gt-1", "walking") == "standing"
        # Back to "sitting", count=1.
        assert hyst.update("gt-1", "sitting") == "standing"

    def test_evict_removes_state(self) -> None:
        hyst = PostureHysteresis(required_consecutive=2)
        hyst.update("gt-1", "standing")
        hyst.evict("gt-1")
        # Fresh start after eviction.
        assert hyst.update("gt-1", "sitting") == "sitting"

    def test_independent_tracks(self) -> None:
        hyst = PostureHysteresis(required_consecutive=2)
        assert hyst.update("gt-1", "standing") == "standing"
        assert hyst.update("gt-2", "sitting") == "sitting"
        assert hyst.update("gt-1", "sitting") == "standing"
        assert hyst.update("gt-2", "standing") == "sitting"
