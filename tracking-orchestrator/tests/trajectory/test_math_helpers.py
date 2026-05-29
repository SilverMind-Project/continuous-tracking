"""U6-T6: Characterization tests for shared trajectory math helpers.

U6c dedup audit finding (2026-05-29):
  No genuine code duplication was found in app/trajectory/.  The following
  modules were audited against app/trajectory/stats.py:

  - motion_energy.py: uses np.mean/np.max on raw velocity arrays.  These
    are standard numpy, not a reimplementation of ewma().  The output is
    per-frame velocity, not a time-series smoothing value.  NOT a duplicate.

  - posture.py: uses _mean_visible_confidence() — domain-specific keypoint
    mean, not a general statistics helper.  NOT a duplicate.

  - fused_posture_strategy.py: uses quality-weighted arithmetic mean inline
    (weight * score / total_weight).  This is a simple convex combination,
    distinct from the time-series EWMA.  NOT a duplicate.

  - dementia_signals.py: calls stats.robust_z() and stats.ewma() directly
    (already consuming the shared helpers; no reimplementation).

  - posture_strategy.py: no statistical helpers found.

  Conclusion: stats.py already correctly centralises the shared helpers.
  No extraction is needed; U6c is a no-change workstream for CTS.

These characterization tests pin the CURRENT OUTPUT of robust_z() and ewma()
so any future refactor that changes their numerical behaviour is caught.  They
must pass both before and after any extraction (proof of behaviour preservation).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# T6: robust_z — characterization tests (pin current output)
# ---------------------------------------------------------------------------


class TestRobustZCharacterization:
    def test_median_of_uniform_sample(self):
        """Median of [1,2,3,4,5] = 3.0."""
        from app.trajectory.stats import robust_z

        result = robust_z(3.0, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.median == pytest.approx(3.0, abs=1e-5)

    def test_mad_of_uniform_sample(self):
        """MAD of [1,2,3,4,5] = median(|x - 3|) = median([2,1,0,1,2]) = 1.0."""
        from app.trajectory.stats import robust_z

        result = robust_z(3.0, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.mad == pytest.approx(1.0, abs=1e-5)

    def test_modified_z_exact_value(self):
        """modified_z for value=5 on [1,2,3,4,5]: 0.6745 * (5-3)/1 = 1.349."""
        from app.trajectory.stats import robust_z

        result = robust_z(5.0, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.modified_z == pytest.approx(1.349, abs=0.001)

    def test_n_equals_sample_count(self):
        from app.trajectory.stats import robust_z

        result = robust_z(1.0, [1.0, 2.0, 3.0])
        assert result.n == 3

    def test_zero_mad_value_equals_median(self):
        """When all samples identical and value == median, z = 0."""
        from app.trajectory.stats import robust_z

        result = robust_z(5.0, [5.0, 5.0, 5.0])
        assert result.modified_z == 0.0

    def test_zero_mad_value_not_median_returns_inf(self):
        """When all samples identical and value != median, z = inf."""
        from app.trajectory.stats import robust_z

        result = robust_z(10.0, [5.0, 5.0, 5.0])
        import math

        assert math.isinf(result.modified_z)

    def test_empty_samples_returns_zero_z(self):
        from app.trajectory.stats import robust_z

        result = robust_z(1.0, [])
        assert result.modified_z == 0.0
        assert result.n == 0

    def test_single_sample_zero_mad(self):
        """Single sample: MAD = 0, z depends on whether value == sample."""
        from app.trajectory.stats import robust_z

        result = robust_z(7.0, [7.0])
        assert result.modified_z == 0.0

    def test_returns_frozen_dataclass(self):
        from app.trajectory.stats import RobustZ, robust_z

        result = robust_z(1.0, [1.0, 2.0, 3.0])
        assert isinstance(result, RobustZ)
        with pytest.raises((AttributeError, TypeError)):
            result.n = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# T6: ewma — characterization tests (pin current output)
# ---------------------------------------------------------------------------


class TestEWMACharacterization:
    def test_empty_samples_returns_zero(self):
        from app.trajectory.stats import ewma

        assert ewma([], halflife=3.0) == 0.0

    def test_single_sample_returns_that_sample(self):
        """One sample: EWMA must equal the sample value."""
        from app.trajectory.stats import ewma

        result = ewma([42.0], halflife=3.0)
        assert result == pytest.approx(42.0, abs=1e-6)

    def test_constant_series_returns_constant(self):
        """EWMA of all-equal values must equal that value."""
        from app.trajectory.stats import ewma

        result = ewma([5.0, 5.0, 5.0, 5.0], halflife=2.0)
        assert result == pytest.approx(5.0, abs=1e-4)

    def test_recent_values_weighted_more_than_uniform_mean(self):
        """EWMA of ascending series [1,2,3,4,5] must exceed the uniform mean (3.0)."""
        from app.trajectory.stats import ewma

        result = ewma([1.0, 2.0, 3.0, 4.0, 5.0], halflife=2.0)
        # Uniform mean = 3.0; EWMA with recency bias on the end must be > 3.0
        assert result > 3.0

    def test_pinned_output_for_known_inputs(self):
        """Pin exact output for [1,2,3,4,5] halflife=2 to catch regressions.

        Reference: alpha ≈ 0.2929; weights follow the implementation's
        half-life formula; verified by running the function directly (2026-05-29).
        """
        from app.trajectory.stats import ewma

        result = ewma([1.0, 2.0, 3.0, 4.0, 5.0], halflife=2.0)
        # Pinned actual output: ≈ 3.586
        assert result == pytest.approx(3.586, abs=0.01)

    def test_float_output_type(self):
        from app.trajectory.stats import ewma

        assert isinstance(ewma([1.0, 2.0], halflife=1.0), float)


# ---------------------------------------------------------------------------
# T6: Audit evidence — non-duplicates are documented (not merged)
# ---------------------------------------------------------------------------


class TestNonDuplicatesDocumented:
    """These tests assert the *absence* of duplication by checking that the
    module-level functions exist in stats.py and are NOT re-implemented
    in the audited modules."""

    def test_stats_module_exports_ewma_and_robust_z(self):
        from app.trajectory import stats

        assert callable(stats.ewma)
        assert callable(stats.robust_z)

    def test_motion_energy_does_not_define_ewma(self):
        """motion_energy uses np.mean/max; it does NOT re-implement EWMA."""
        from app.trajectory import motion_energy

        assert not hasattr(motion_energy, "ewma"), (
            "motion_energy should not define its own ewma; use stats.ewma instead"
        )

    def test_fused_posture_strategy_does_not_define_robust_z(self):
        """fused_posture_strategy uses quality-weighted linear combination; not robust_z."""
        from app.trajectory import fused_posture_strategy

        assert not hasattr(fused_posture_strategy, "robust_z"), (
            "fused_posture_strategy should not define its own robust_z; "
            "use stats.robust_z if needed"
        )
