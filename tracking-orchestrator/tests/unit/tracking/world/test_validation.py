"""Tests for pure floor/covariance validators (app/tracking/world/validation.py)."""

from __future__ import annotations

import math

import numpy as np

from app.tracking.world.validation import (
    DEFAULT_MAX_COV_TRACE_M2,
    is_finite_point,
    is_valid_covariance,
)


class TestFinitePoint:
    def test_finite_point_accepted(self) -> None:
        assert is_finite_point(1.0, -2.5)

    def test_nan_rejected(self) -> None:
        assert not is_finite_point(math.nan, 0.0)
        assert not is_finite_point(0.0, math.nan)

    def test_inf_rejected(self) -> None:
        assert not is_finite_point(math.inf, 0.0)
        assert not is_finite_point(0.0, -math.inf)


class TestValidCovariance:
    def test_well_formed_covariance_accepted(self) -> None:
        cov = np.array([[0.25, 0.01], [0.01, 0.30]], dtype=np.float64)
        assert is_valid_covariance(cov)

    def test_large_but_legitimate_covariance_accepted(self) -> None:
        # The largest green fixture (2 m sigma per axis, trace 8.0) stays valid.
        assert is_valid_covariance(4.0 * np.eye(2))

    def test_wrong_shape_rejected(self) -> None:
        assert not is_valid_covariance(np.eye(3))
        assert not is_valid_covariance(np.array([1.0, 2.0]))

    def test_non_finite_rejected(self) -> None:
        assert not is_valid_covariance(np.array([[math.nan, 0.0], [0.0, 1.0]]))
        assert not is_valid_covariance(np.array([[math.inf, 0.0], [0.0, 1.0]]))

    def test_asymmetric_rejected(self) -> None:
        cov = np.array([[1.0, 0.5], [0.4, 1.0]], dtype=np.float64)  # off-diag mismatch
        assert not is_valid_covariance(cov)

    def test_symmetric_within_tolerance_accepted(self) -> None:
        cov = np.array([[1.0, 0.5], [0.5 + 1e-9, 1.0]], dtype=np.float64)
        assert is_valid_covariance(cov)

    def test_negative_eigenvalue_rejected(self) -> None:
        # Symmetric but indefinite: eigenvalues are +/- 1.
        cov = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        assert not is_valid_covariance(cov)

    def test_negative_diagonal_rejected(self) -> None:
        cov = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        assert not is_valid_covariance(cov)

    def test_over_cap_trace_rejected(self) -> None:
        over = (DEFAULT_MAX_COV_TRACE_M2 + 10.0) / 2.0
        cov = np.array([[over, 0.0], [0.0, over]], dtype=np.float64)
        assert not is_valid_covariance(cov)

    def test_at_cap_trace_accepted(self) -> None:
        half = DEFAULT_MAX_COV_TRACE_M2 / 2.0
        cov = np.array([[half, 0.0], [0.0, half]], dtype=np.float64)
        assert is_valid_covariance(cov)
