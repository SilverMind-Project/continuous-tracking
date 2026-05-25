"""Tests for GlobalPostureTracker quality-weighted fusion and staleness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.trajectory.posture import (
    GlobalPostureTracker,
    PostureScores,
    _CameraSnapshot,
)


def _scores(
    lying: float = 0.0, sitting: float = 0.0, sw: float = 0.0, kp: float = 0.8
) -> PostureScores:
    return PostureScores(lying=lying, sitting=sitting, standing_walking=sw, keypoint_confidence=kp)


class TestQualityWeightedFusion:
    def test_high_confidence_camera_dominates_low_confidence(self) -> None:
        """A full-body view at 0.9 confidence should dominate a partial view at 0.2."""
        tracker = GlobalPostureTracker(required_consecutive=1)
        # cam-1: weak sitting signal, low confidence
        tracker.update("gt-1", "cam-1", _scores(sitting=0.6, kp=0.2), ["cam-1", "cam-2"])
        # cam-2: strong standing signal, high confidence — needs 2 frames to flip hysteresis.
        tracker.update("gt-1", "cam-2", _scores(sw=0.9, kp=0.9), ["cam-1", "cam-2"])
        result = tracker.update("gt-1", "cam-2", _scores(sw=0.9, kp=0.9), ["cam-1", "cam-2"])
        assert result == "standing"

    def test_two_equal_cameras_agree_produce_correct_fusion(self) -> None:
        """Two cameras with equal confidence and the same posture should agree."""
        tracker = GlobalPostureTracker(required_consecutive=1)
        tracker.update("gt-1", "cam-1", _scores(sitting=0.8, kp=0.7), ["cam-1", "cam-2"])
        result = tracker.update("gt-1", "cam-2", _scores(sitting=0.75, kp=0.7), ["cam-1", "cam-2"])
        assert result == "sitting"

    def test_depth_camera_contributes_with_floor_weight(self) -> None:
        """Depth-only camera (kp_conf=0.0) must still contribute to fusion."""
        tracker = GlobalPostureTracker(required_consecutive=1)
        # Depth camera says lying
        tracker.update("gt-1", "cam-depth", _scores(lying=0.9, kp=0.0), ["cam-depth"])
        result = tracker.update(
            "gt-1",
            "cam-depth",
            _scores(lying=0.9, kp=0.0),
            ["cam-depth"],
        )
        assert result == "lying"

    def test_depth_does_not_override_high_confidence_keypoint(self) -> None:
        """Depth lying score must not override a high-confidence keypoint standing score."""
        tracker = GlobalPostureTracker(required_consecutive=1)
        # Keypoint camera: strong standing
        tracker.update("gt-1", "cam-kp", _scores(sw=0.95, kp=0.92), ["cam-kp", "cam-depth"])
        # Depth camera: lying (low weight because kp=0.0)
        result = tracker.update(
            "gt-1",
            "cam-depth",
            _scores(lying=0.9, kp=0.0),
            ["cam-kp", "cam-depth"],
        )
        assert result == "standing"


class TestCameraStaleness:
    def test_stale_camera_excluded_from_fusion(self) -> None:
        """A camera that hasn't sent a frame for > stale_after_s should be excluded."""
        tracker = GlobalPostureTracker(required_consecutive=1, camera_stale_after_s=5.0)

        past = datetime.now(UTC) - timedelta(seconds=6)
        # Inject a stale snapshot directly for cam-1.
        tracker._snapshots["gt-1"] = {}
        tracker._snapshots["gt-1"]["cam-1"] = _CameraSnapshot(
            lying=0.0,
            sitting=0.9,
            standing_walking=0.0,
            keypoint_confidence=0.8,
            captured_at=past,
        )

        # Now cam-2 sends a fresh standing signal
        result = tracker.update("gt-1", "cam-2", _scores(sw=0.85, kp=0.9), ["cam-1", "cam-2"])
        # cam-1 is stale (6s > 5s threshold), only cam-2 contributes
        assert result == "standing"

    def test_fresh_camera_included_in_fusion(self) -> None:
        """A camera updated just now must be included in fusion."""
        tracker = GlobalPostureTracker(required_consecutive=1, camera_stale_after_s=10.0)
        tracker.update("gt-1", "cam-1", _scores(sitting=0.8, kp=0.7), ["cam-1", "cam-2"])
        result = tracker.update("gt-1", "cam-2", _scores(sw=0.3, kp=0.4), ["cam-1", "cam-2"])
        # Both fresh; sitting from cam-1 should dominate
        assert result == "sitting"


class TestHysteresisInlining:
    def test_hysteresis_state_stored_directly(self) -> None:
        """After M2, _hysteresis_state is a plain dict, not dict-of-PostureHysteresis."""
        tracker = GlobalPostureTracker(required_consecutive=2)
        # No PostureHysteresis objects should exist.
        assert not hasattr(tracker, "_hysteresis")
        assert hasattr(tracker, "_hysteresis_state")

    def test_single_frame_does_not_commit_flip(self) -> None:
        """With required_consecutive=2, a single different frame must not flip posture."""
        tracker = GlobalPostureTracker(required_consecutive=2)
        # Establish committed posture as standing.
        tracker.update("gt-1", "cam-1", _scores(sw=0.8, kp=0.9), ["cam-1"])
        tracker.update("gt-1", "cam-1", _scores(sw=0.8, kp=0.9), ["cam-1"])
        # One sitting frame — must not flip.
        result = tracker.update("gt-1", "cam-1", _scores(sitting=0.8, kp=0.9), ["cam-1"])
        assert result == "standing"

    def test_two_consecutive_frames_commit_flip(self) -> None:
        """With required_consecutive=2, two consecutive different frames must flip posture."""
        tracker = GlobalPostureTracker(required_consecutive=2)
        tracker.update("gt-1", "cam-1", _scores(sw=0.8, kp=0.9), ["cam-1"])
        tracker.update("gt-1", "cam-1", _scores(sw=0.8, kp=0.9), ["cam-1"])
        # Two sitting frames.
        tracker.update("gt-1", "cam-1", _scores(sitting=0.8, kp=0.9), ["cam-1"])
        result = tracker.update("gt-1", "cam-1", _scores(sitting=0.8, kp=0.9), ["cam-1"])
        assert result == "sitting"


class TestEvictTrack:
    def test_evict_track_cleans_snapshots_and_hysteresis(self) -> None:
        tracker = GlobalPostureTracker(required_consecutive=1)
        tracker.update("gt-1", "cam-1", _scores(sw=0.8, kp=0.9), ["cam-1"])
        assert "gt-1" in tracker._snapshots
        assert "gt-1" in tracker._hysteresis_state
        tracker.evict_track("gt-1")
        assert "gt-1" not in tracker._snapshots
        assert "gt-1" not in tracker._hysteresis_state

    def test_evict_unknown_track_is_noop(self) -> None:
        tracker = GlobalPostureTracker(required_consecutive=1)
        tracker.evict_track("gt-nonexistent")  # Must not raise.

    def test_evict_does_not_call_inner_evict(self) -> None:
        """Regression test: evict_track must not call any intermediate object.evict() method.
        The old code called self._hysteresis[id].evict(id) before popping the instance,
        which was a no-op. After M2, there is no inner evict() call at all."""
        tracker = GlobalPostureTracker(required_consecutive=1)
        tracker.update("gt-1", "cam-1", _scores(sw=0.8), ["cam-1"])
        # If the old pattern is accidentally restored, this test catches it because
        # _hysteresis no longer exists as an attribute.
        assert not hasattr(tracker, "_hysteresis")
        tracker.evict_track("gt-1")


class TestMultiCameraFusionEndToEnd:
    def test_best_camera_evidence_wins_over_partial_view(self) -> None:
        """Regression: full sitting from cam-1 must win over partial lying from cam-2."""
        tracker = GlobalPostureTracker(required_consecutive=1)
        # cam-1 full body: strong sitting (0.85), high confidence
        tracker.update("gt-1", "cam-1", _scores(sitting=0.85, kp=0.88), ["cam-1", "cam-2"])
        # cam-2 partial body: weak lying (0.55), low confidence (head only visible)
        result = tracker.update("gt-1", "cam-2", _scores(lying=0.55, kp=0.22), ["cam-1", "cam-2"])
        assert result == "sitting"

    def test_walking_requires_motion_energy(self) -> None:
        tracker = GlobalPostureTracker(required_consecutive=1)
        scores = _scores(sw=0.8, kp=0.9)
        result_no_me = tracker.update("gt-1", "cam-1", scores, ["cam-1"], motion_energy=None)
        assert result_no_me == "standing"
        # Standing is committed; 2 consecutive walking frames needed to flip hysteresis.
        tracker.update("gt-1", "cam-1", scores, ["cam-1"], motion_energy=0.02)
        result_with_me = tracker.update("gt-1", "cam-1", scores, ["cam-1"], motion_energy=0.02)
        assert result_with_me == "walking"
