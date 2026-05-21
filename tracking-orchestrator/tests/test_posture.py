"""Table-driven tests for classify_posture and PostureHysteresis."""

from __future__ import annotations

from app.domain import BoundingBox
from app.inference.schemas import Keypoint, PoseResult
from app.trajectory.posture import GlobalPostureTracker, PostureHysteresis, classify_posture

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
        """Wide-knees pose at moderate confidence correctly classifies as sitting.

        Knees spread wide (thighs angled out) triggers the lateral-spread signal
        (norm_knee_dx > 0.6) with knees near hip height.  At score 0.45 this
        still clears the 0.4 confidence threshold, so the result is 'sitting'.
        The old assumption that this would be 'unknown' no longer holds with the
        new ankle-free lateral-spread feature.
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
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"

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


# ---------------------------------------------------------------------------
# GlobalPostureTracker
# ---------------------------------------------------------------------------


class TestGlobalPostureTracker:
    def test_single_camera_updates(self) -> None:
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.2),
            right_shoulder=_keypoint(0.6, 0.2),
            left_hip=_keypoint(0.4, 0.5),
            right_hip=_keypoint(0.6, 0.5),
            left_knee=_keypoint(0.65, 0.55),
            right_knee=_keypoint(0.75, 0.55),
            left_ankle=_keypoint(0.65, 0.85),
            right_ankle=_keypoint(0.75, 0.85),
        )
        tracker = GlobalPostureTracker(required_consecutive=2)
        # First frame commits immediately
        assert tracker.update("gt-1", "cam-1", pose, _BBOX_PORTRAIT, ["cam-1"]) == "sitting"

    def test_walking_from_motion_energy(self) -> None:
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
        tracker = GlobalPostureTracker(required_consecutive=2)
        assert tracker.update("gt-1", "cam-1", pose, _BBOX_PORTRAIT, ["cam-1"], motion_energy=0.012) == "walking"

    def test_multi_camera_fusion_and_partial_view(self) -> None:
        # Camera 1 sees a clear sitting pose
        pose_sitting = _pose(
            left_shoulder=_keypoint(0.4, 0.2),
            right_shoulder=_keypoint(0.6, 0.2),
            left_hip=_keypoint(0.4, 0.5),
            right_hip=_keypoint(0.6, 0.5),
            left_knee=_keypoint(0.65, 0.55),
            right_knee=_keypoint(0.75, 0.55),
            left_ankle=_keypoint(0.65, 0.85),
            right_ankle=_keypoint(0.75, 0.85),
        )
        # Camera 2 has a partial view (knees and ankles missing/low confidence, torso tilted/leaning at ~37 degrees)
        pose_partial = _pose(
            left_shoulder=_keypoint(0.3, 0.2),
            right_shoulder=_keypoint(0.5, 0.2),
            left_hip=_keypoint(0.6, 0.6),
            right_hip=_keypoint(0.8, 0.6),
            left_knee=_keypoint(0.65, 0.55, score=0.1),
            right_knee=_keypoint(0.75, 0.55, score=0.1),
            left_ankle=_keypoint(0.65, 0.85, score=0.1),
            right_ankle=_keypoint(0.75, 0.85, score=0.1),
        )
        tracker = GlobalPostureTracker(required_consecutive=2)
        
        # Initialize track with Camera 2's partial view (starts as unknown)
        res1 = tracker.update("gt-1", "cam-2", pose_partial, _BBOX_PORTRAIT, ["cam-1", "cam-2"])
        assert res1 == "unknown"

        # Update with Camera 1's full sitting view (first flip frame -> remains unknown)
        res2 = tracker.update("gt-1", "cam-1", pose_sitting, _BBOX_PORTRAIT, ["cam-1", "cam-2"])
        assert res2 == "unknown"

        # Second frame of "sitting" on cam-1 commits it
        res3 = tracker.update("gt-1", "cam-1", pose_sitting, _BBOX_PORTRAIT, ["cam-1", "cam-2"])
        assert res3 == "sitting"

        # Update from cam-2's partial view (fused with cam-1's stored scores -> remains sitting)
        res4 = tracker.update("gt-1", "cam-2", pose_partial, _BBOX_PORTRAIT, ["cam-1", "cam-2"])
        assert res4 == "sitting"

    def test_eviction_removes_state(self) -> None:
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
        tracker = GlobalPostureTracker(required_consecutive=2)
        assert tracker.update("gt-1", "cam-1", pose, _BBOX_PORTRAIT, ["cam-1"]) == "standing"
        tracker.evict_track("gt-1")
        
        # New pose is sitting
        pose_sitting = _pose(
            left_shoulder=_keypoint(0.4, 0.2),
            right_shoulder=_keypoint(0.6, 0.2),
            left_hip=_keypoint(0.4, 0.5),
            right_hip=_keypoint(0.6, 0.5),
            left_knee=_keypoint(0.65, 0.55),
            right_knee=_keypoint(0.75, 0.55),
            left_ankle=_keypoint(0.65, 0.85),
            right_ankle=_keypoint(0.75, 0.85),
        )
        # Because we evicted gt-1, it is a first observation again, so it commits sitting immediately
        assert tracker.update("gt-1", "cam-1", pose_sitting, _BBOX_PORTRAIT, ["cam-1"]) == "sitting"


# ---------------------------------------------------------------------------
# Sitting-as-standing regression tests
# ---------------------------------------------------------------------------


class TestSittingNotMisclassifiedAsStanding:
    def test_upright_sitter_occluded_ankles(self) -> None:
        """Upright torso, knees at ~hip height, ankles not visible.

        This is the canonical failure mode: a seated person whose torso is
        nearly vertical and whose ankles are outside the camera frame.  The old
        classifier would score kinematic_ordering=True and torso near-vertical,
        producing a standing score of 1.0 with zero sitting evidence.

        With the new horizontal-thigh signal (knees near hips + bent knee),
        the sitting scorer fires without requiring ankle visibility.
        """
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.2),
            right_shoulder=_keypoint(0.6, 0.2),
            left_hip=_keypoint(0.4, 0.5),
            right_hip=_keypoint(0.6, 0.5),
            # Thighs horizontal: knees just slightly below hip height
            left_knee=_keypoint(0.2, 0.55),
            right_knee=_keypoint(0.8, 0.55),
            # Ankles occluded
            left_ankle=_keypoint(0.2, 0.6, score=0.1),
            right_ankle=_keypoint(0.8, 0.6, score=0.1),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"

    def test_upright_sitter_frontal_camera_full_body(self) -> None:
        """Front-on camera, person sitting on a chair, all keypoints visible.

        Both legs bent ~90°.  Knees are close to hip height (thighs horizontal).
        Ankles hang below.  This is the standard chair-sitting geometry from the
        front and should never be misclassified as standing.
        """
        pose = _pose(
            left_shoulder=_keypoint(0.35, 0.25),
            right_shoulder=_keypoint(0.65, 0.25),
            left_hip=_keypoint(0.38, 0.52),
            right_hip=_keypoint(0.62, 0.52),
            # Knees at roughly hip height (thighs horizontal, spread apart)
            left_knee=_keypoint(0.2, 0.56),
            right_knee=_keypoint(0.8, 0.56),
            # Shins vertical, ankles below
            left_ankle=_keypoint(0.2, 0.82),
            right_ankle=_keypoint(0.8, 0.82),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"

    def test_one_leg_extended_sitter(self) -> None:
        """Person sitting with one leg stretched out (e.g. on a sofa).

        min_knee_angle (straightest leg) is large (~160°), but max_knee_angle
        (bent leg) is ~90°.  The old classifier used the most-bent knee for
        BOTH the sitting score and the standing veto, so a large min_knee_angle
        would pass the standing veto test.  The new classifier correctly uses
        min_knee_angle for the standing veto (only blocking if BOTH legs are bent)
        while using max_knee_angle for the sitting score.
        """
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.25),
            right_shoulder=_keypoint(0.6, 0.25),
            left_hip=_keypoint(0.4, 0.52),
            right_hip=_keypoint(0.6, 0.52),
            # Left leg bent ~90° (thigh horizontal, shin vertical)
            left_knee=_keypoint(0.15, 0.56),
            left_ankle=_keypoint(0.15, 0.82),
            # Right leg extended forward (nearly straight)
            right_knee=_keypoint(0.8, 0.54),
            right_ankle=_keypoint(0.95, 0.56),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "sitting"

    def test_standing_is_not_suppressed_by_new_veto(self) -> None:
        """Confirm a genuine standing pose still classifies as standing.

        Knees well below hips (norm_knee_dy > 0.55) and both legs straight
        (min_knee_angle_deg > 130°) — the new vetoes should not fire.
        """
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.15),
            right_shoulder=_keypoint(0.6, 0.15),
            left_hip=_keypoint(0.4, 0.4),
            right_hip=_keypoint(0.6, 0.4),
            # Knees well below hips
            left_knee=_keypoint(0.4, 0.65),
            right_knee=_keypoint(0.6, 0.65),
            left_ankle=_keypoint(0.4, 0.88),
            right_ankle=_keypoint(0.6, 0.88),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT) == "standing"

    def test_walking_stride_not_classified_as_sitting(self) -> None:
        """Mid-stride walker: one knee bent (stride leg), one extended (support leg).

        min_knee_angle is large (support leg ~150°), so the standing veto
        (both-knees-bent check) does NOT fire.  Knees are well below hips.
        The kinematic ordering score fires, giving standing/walking evidence.
        With motion_energy above threshold → walking.
        """
        pose = _pose(
            left_shoulder=_keypoint(0.4, 0.18),
            right_shoulder=_keypoint(0.6, 0.18),
            left_hip=_keypoint(0.4, 0.42),
            right_hip=_keypoint(0.6, 0.42),
            # Support leg: nearly straight
            left_knee=_keypoint(0.38, 0.64),
            left_ankle=_keypoint(0.36, 0.87),
            # Stride leg: bent
            right_knee=_keypoint(0.62, 0.62),
            right_ankle=_keypoint(0.67, 0.82),
        )
        assert classify_posture(pose, _BBOX_PORTRAIT, motion_energy=0.015) == "walking"
