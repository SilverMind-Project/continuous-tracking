"""Unit tests for FallDetector and FallDetectorConfig."""

from __future__ import annotations

import pytest

from app.trajectory.fall_detector import FallDecision, FallDetector
from app.trajectory.fall_features import FallFeatures

_RESTING = ("bed", "bedroom")


def _features(
    *,
    samples: int = 10,
    descent_rate: float = 1.2,
    height_ratio: float = 0.3,
    lying_score: float = 0.7,
    pose_available: bool = True,
    floor_speed: float | None = 0.5,
    post_event_motion: float | None = 0.01,
) -> FallFeatures:
    """Build a FallFeatures that fires all rules by default."""
    return FallFeatures(
        max_descent_rate_hps=descent_rate,
        height_ratio_now=height_ratio,
        lying_score_now=lying_score,
        post_event_motion_nu_s=post_event_motion,
        floor_speed_at_event_m_s=floor_speed,
        samples=samples,
        pose_available_now=pose_available,
    )


class TestFallDetectorImpact:
    def test_all_rules_fire(self) -> None:
        detector = FallDetector()
        decision = detector.check_impact(_features(), "living_room", _RESTING)
        assert isinstance(decision, FallDecision)
        assert decision.descent_rate_hps == pytest.approx(1.2)

    def test_too_few_samples(self) -> None:
        detector = FallDetector()
        assert detector.check_impact(_features(samples=4), "living_room", _RESTING) is None

    def test_slow_descent(self) -> None:
        detector = FallDetector()
        assert detector.check_impact(_features(descent_rate=0.5), "living_room", _RESTING) is None

    def test_height_ratio_too_high(self) -> None:
        detector = FallDetector()
        assert detector.check_impact(_features(height_ratio=0.7), "living_room", _RESTING) is None

    def test_low_lying_score_with_pose_available(self) -> None:
        detector = FallDetector()
        assert (
            detector.check_impact(
                _features(lying_score=0.1, pose_available=True),
                "living_room",
                _RESTING,
            )
            is None
        )

    def test_no_pose_overrides_lying_score_rule(self) -> None:
        # No keypoints (pose unavailable) passes rule 4 regardless of lying score.
        detector = FallDetector()
        decision = detector.check_impact(
            _features(lying_score=0.0, pose_available=False),
            "living_room",
            _RESTING,
        )
        assert decision is not None

    def test_high_floor_speed_rejected(self) -> None:
        detector = FallDetector()
        assert detector.check_impact(_features(floor_speed=3.0), "living_room", _RESTING) is None

    def test_none_floor_speed_passes(self) -> None:
        detector = FallDetector()
        decision = detector.check_impact(_features(floor_speed=None), "living_room", _RESTING)
        assert decision is not None

    def test_resting_room_suppressed(self) -> None:
        detector = FallDetector()
        assert detector.check_impact(_features(), "bedroom", _RESTING) is None

    def test_resting_room_substring_match(self) -> None:
        detector = FallDetector()
        assert detector.check_impact(_features(), "master_bedroom", _RESTING) is None

    def test_non_resting_room_fires(self) -> None:
        detector = FallDetector()
        assert detector.check_impact(_features(), "hallway", _RESTING) is not None

    @pytest.mark.parametrize(
        "missing",
        [
            "samples",
            "descent_rate",
            "height_ratio",
            "floor_speed",
        ],
    )
    def test_each_condition_removal_suppresses(self, missing: str) -> None:
        """Removing any single blocking condition stops detection."""
        detector = FallDetector()
        kwargs: dict[str, object] = {}
        if missing == "samples":
            kwargs["samples"] = 2
        elif missing == "descent_rate":
            kwargs["descent_rate"] = 0.3
        elif missing == "height_ratio":
            kwargs["height_ratio"] = 0.8
        elif missing == "floor_speed":
            kwargs["floor_speed"] = 5.0
        assert detector.check_impact(_features(**kwargs), "kitchen", _RESTING) is None


class TestFallDetectorEscalation:
    def test_escalatable_when_post_event_still(self) -> None:
        detector = FallDetector()
        assert detector.is_escalatable(_features(post_event_motion=0.01))

    def test_not_escalatable_when_high_motion(self) -> None:
        detector = FallDetector()
        assert not detector.is_escalatable(_features(post_event_motion=0.5))

    def test_escalatable_without_post_window_uses_posture_proxy(self) -> None:
        detector = FallDetector()
        # post_event_motion_nu_s is None → use posture proxy
        f = _features(post_event_motion=None, height_ratio=0.3, lying_score=0.7)
        assert detector.is_escalatable(f)

    def test_not_escalatable_without_post_window_standing(self) -> None:
        detector = FallDetector()
        f = _features(post_event_motion=None, height_ratio=0.9, lying_score=0.1)
        assert not detector.is_escalatable(f)


class TestFallDetectorStandingClear:
    def test_standing_clears(self) -> None:
        detector = FallDetector()
        f = _features(height_ratio=0.9, lying_score=0.05)
        assert detector.is_standing_cleared(f)

    def test_still_lying_not_cleared(self) -> None:
        detector = FallDetector()
        f = _features(height_ratio=0.3, lying_score=0.7)
        assert not detector.is_standing_cleared(f)

    def test_height_ok_but_still_lying_not_cleared(self) -> None:
        detector = FallDetector()
        f = _features(height_ratio=0.9, lying_score=0.5)
        assert not detector.is_standing_cleared(f)
