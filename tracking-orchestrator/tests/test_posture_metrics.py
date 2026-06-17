"""Tests for posture Prometheus metrics registration and incrementing."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from app.observability.metrics import build_metrics
from app.trajectory.posture import GlobalPostureTracker, PostureScores


@pytest.fixture
def fresh_metrics():
    """Build metrics against a fresh registry so tests don't leak state."""
    return build_metrics(registry=CollectorRegistry())


class TestPostureMetricsRegistered:
    def test_hysteresis_flips_counter_registered(self, fresh_metrics) -> None:
        assert hasattr(fresh_metrics, "cts_posture_hysteresis_flips_total")

    def test_camera_contributions_counter_registered(self, fresh_metrics) -> None:
        assert hasattr(fresh_metrics, "cts_posture_camera_contributions_total")

    def test_cameras_fused_histogram_registered(self, fresh_metrics) -> None:
        assert hasattr(fresh_metrics, "cts_posture_cameras_fused")

    def test_view_weight_histogram_registered(self, fresh_metrics) -> None:
        assert hasattr(fresh_metrics, "cts_posture_view_weight")

    def test_fused_class_counter_registered(self, fresh_metrics) -> None:
        assert hasattr(fresh_metrics, "cts_posture_fused_class_total")


class TestGlobalPostureTrackerMetrics:
    def test_update_increments_camera_contribution(self) -> None:
        """Each call to update() increments the camera contributions counter."""
        from app.observability import metrics as _metrics

        registry = CollectorRegistry()
        _metrics.metrics = build_metrics(registry=registry)

        tracker = GlobalPostureTracker(required_consecutive=1)
        scores = PostureScores(
            lying=0.0, sitting=0.0, standing_walking=0.9, keypoint_confidence=0.8
        )

        before = _metrics.metrics.cts_posture_camera_contributions_total.labels(
            camera_id="cam-test"
        )._value.get()

        tracker.update("gt-1", "cam-test", scores, ["cam-test"])

        after = _metrics.metrics.cts_posture_camera_contributions_total.labels(
            camera_id="cam-test"
        )._value.get()
        assert after == before + 1

    def test_posture_flip_increments_hysteresis_counter(self) -> None:
        """A committed posture flip increments the hysteresis flip counter."""
        from app.observability import metrics as _metrics

        registry = CollectorRegistry()
        _metrics.metrics = build_metrics(registry=registry)

        tracker = GlobalPostureTracker(required_consecutive=2)
        sw = PostureScores(lying=0.0, sitting=0.0, standing_walking=0.9, keypoint_confidence=0.9)
        # Establish standing.
        tracker.update("gt-flip", "cam-1", sw, ["cam-1"])
        tracker.update("gt-flip", "cam-1", sw, ["cam-1"])

        # Transition to sitting.
        sit = PostureScores(lying=0.0, sitting=0.9, standing_walking=0.0, keypoint_confidence=0.9)
        before = _metrics.metrics.cts_posture_hysteresis_flips_total.labels(
            camera_id="global"
        )._value.get()

        tracker.update("gt-flip", "cam-1", sit, ["cam-1"])
        tracker.update("gt-flip", "cam-1", sit, ["cam-1"])  # second → commit

        after = _metrics.metrics.cts_posture_hysteresis_flips_total.labels(
            camera_id="global"
        )._value.get()
        assert after > before

    def test_update_observes_view_weight(self) -> None:
        """Fusion records the geometry suitability multiplier for each contribution."""
        from app.observability import metrics as _metrics

        registry = CollectorRegistry()
        _metrics.metrics = build_metrics(registry=registry)

        tracker = GlobalPostureTracker(required_consecutive=1)
        scores = PostureScores(
            lying=0.0, sitting=0.9, standing_walking=0.0, keypoint_confidence=0.8
        )

        before = _metrics.metrics.cts_posture_view_weight._sum.get()
        tracker.update("gt-1", "cam-test", scores, ["cam-test"], view_weight=0.6)
        after = _metrics.metrics.cts_posture_view_weight._sum.get()

        assert after == before + 0.6


class TestCommittedPosture:
    def test_committed_posture_returns_none_before_first_update(self) -> None:
        tracker = GlobalPostureTracker(required_consecutive=1)
        assert tracker.committed_posture("gt-unknown") is None

    def test_committed_posture_returns_first_label_immediately(self) -> None:
        tracker = GlobalPostureTracker(required_consecutive=1)
        scores = PostureScores(
            lying=0.0, sitting=0.9, standing_walking=0.0, keypoint_confidence=0.8
        )
        tracker.update("gt-1", "cam-1", scores, ["cam-1"])
        assert tracker.committed_posture("gt-1") == "sitting"

    def test_committed_posture_stable_during_candidate_accumulation(self) -> None:
        """committed_posture() returns the OLD posture while a flip is in progress."""
        tracker = GlobalPostureTracker(required_consecutive=2)
        sw = PostureScores(lying=0.0, sitting=0.0, standing_walking=0.9, keypoint_confidence=0.9)
        tracker.update("gt-1", "cam-1", sw, ["cam-1"])
        tracker.update("gt-1", "cam-1", sw, ["cam-1"])
        # One sitting frame — not yet committed.
        sit = PostureScores(lying=0.0, sitting=0.9, standing_walking=0.0, keypoint_confidence=0.9)
        tracker.update("gt-1", "cam-1", sit, ["cam-1"])
        assert tracker.committed_posture("gt-1") == "standing"
