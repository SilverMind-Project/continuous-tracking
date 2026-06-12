"""Tests for robust statistics module."""

from __future__ import annotations

import pytest

from app.trajectory.stats import ewma, robust_z, weighted_median


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


class TestWeightedMedian:
    def test_equal_weights_matches_ordinary_median_odd(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert weighted_median(values, weights) == 3.0

    def test_equal_weights_matches_ordinary_median_even(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        weights = [1.0, 1.0, 1.0, 1.0]
        # Nearest cumulative-weight crossing: total=4, half=2; after value 2.0 cumulative=2 >= 2
        assert weighted_median(values, weights) == 2.0

    def test_single_value(self) -> None:
        assert weighted_median([0.8], [10.0]) == 0.8

    def test_heavy_tail_shifts_median(self) -> None:
        # Five short slow bouts vs one long fast bout.
        # Without weighting the median would be 0.4 (slow); with duration
        # weighting the heavy 30 s fast bout tips the median to 0.9.
        values = [0.4, 0.4, 0.4, 0.4, 0.4, 0.9]
        weights = [3.0, 3.0, 3.0, 3.0, 3.0, 30.0]
        result = weighted_median(values, weights)
        assert result == 0.9

    def test_unsorted_input(self) -> None:
        values = [5.0, 1.0, 3.0]
        weights = [1.0, 1.0, 1.0]
        assert weighted_median(values, weights) == 3.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            weighted_median([], [])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            weighted_median([1.0, 2.0], [1.0])

    def test_zero_weight_fallback_to_unweighted_median(self) -> None:
        values = [1.0, 5.0, 3.0]
        weights = [0.0, 0.0, 0.0]
        result = weighted_median(values, weights)
        assert result == 3.0


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
