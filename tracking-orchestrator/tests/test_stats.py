"""Tests for robust statistics module."""

from __future__ import annotations

from app.trajectory.stats import ewma, robust_z


class TestRobustZ:
    def test_basic(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = robust_z(10.0, samples)
        assert result.n == 5
        assert result.median == 3.0
        assert result.modified_z > 0

    def test_rejects_single_outlier(self) -> None:
        """A single 10x outlier should not produce a high z-score for normal values."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 50.0]
        result = robust_z(4.0, samples)
        # A value near the median should have a small modified_z
        # even though one extreme outlier is present.
        assert abs(result.modified_z) < 3.0

    def test_empty_samples(self) -> None:
        result = robust_z(5.0, [])
        assert result.n == 0
        assert result.median == 0.0
        assert result.mad == 0.0
        assert result.modified_z == 0.0

    def test_all_identical_samples(self) -> None:
        result = robust_z(5.0, [5.0, 5.0, 5.0, 5.0, 5.0])
        assert result.modified_z == 0.0

    def test_value_differs_from_identical_baseline(self) -> None:
        """When baseline has zero variance, a different value gets inf z."""
        result = robust_z(10.0, [5.0, 5.0, 5.0])
        assert result.modified_z == float("inf")

    def test_match_hand_computed(self) -> None:
        """Verify against a hand-computed fixture.

        Samples: [1, 2, 3, 4, 5], value = 6
        Median = 3, abs deviations = [2, 1, 0, 1, 2], MAD = 1
        Modified z = 0.6745 * (6-3) / 1 = 2.0235
        """
        result = robust_z(6.0, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.median == 3.0
        assert result.mad == 1.0
        assert round(result.modified_z, 4) == 2.0235


class TestEwma:
    def test_basic(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = ewma(samples, halflife=2.0)
        # EWMA of ascending values should be between the mean and the most recent.
        assert 2.0 < result < 5.0

    def test_empty(self) -> None:
        assert ewma([], halflife=2.0) == 0.0

    def test_single(self) -> None:
        assert ewma([7.0], halflife=2.0) == 7.0
