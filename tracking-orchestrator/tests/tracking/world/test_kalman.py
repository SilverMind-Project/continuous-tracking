"""Tests for floor-plane Kalman filter (app/tracking/world/kalman.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.tracking.world.kalman import (
    initialize,
    isotropic_cov,
    mahalanobis2_position,
    predict,
    update,
)


def _now() -> datetime:
    return datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)


class TestInitialize:
    def test_initializes_at_given_position(self) -> None:
        state = initialize(3.5, -2.0, 1.0, 2.0, _now())
        np.testing.assert_array_almost_equal(state.mean, [3.5, -2.0, 0.0, 0.0])

    def test_initial_covariance_is_diagonal(self) -> None:
        state = initialize(0.0, 0.0, 1.0, 2.0, _now())
        assert state.covariance.shape == (4, 4)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert state.covariance[i, j] == 0.0

    def test_position_uncertainty_matches_sigma(self) -> None:
        sigma = 1.5
        state = initialize(0.0, 0.0, sigma, 2.0, _now())
        assert state.covariance[0, 0] == sigma * sigma
        assert state.covariance[1, 1] == sigma * sigma


class TestPredict:
    def test_predict_advances_position_by_velocity(self) -> None:
        state = initialize(0.0, 0.0, 0.1, 0.1, _now())
        future = datetime(2026, 5, 26, 12, 0, 2, tzinfo=UTC)  # 2 seconds later
        predicted = predict(state, future, 0.5)
        np.testing.assert_array_almost_equal(predicted.mean[:2], [0.0, 0.0])

    def test_predict_inflates_covariance(self) -> None:
        state = initialize(0.0, 0.0, 0.1, 0.1, _now())
        future = datetime(2026, 5, 26, 12, 0, 1, tzinfo=UTC)  # 1 second later
        predicted = predict(state, future, 0.5)
        init_trace = float(np.trace(state.covariance))
        pred_trace = float(np.trace(predicted.covariance))
        assert pred_trace > init_trace

    def test_predict_zero_dt_is_identity(self) -> None:
        state = initialize(1.0, 2.0, 0.5, 1.0, _now())
        predicted = predict(state, state.updated_at, 0.5)
        np.testing.assert_array_almost_equal(predicted.mean, state.mean)
        np.testing.assert_array_almost_equal(predicted.covariance, state.covariance)

    def test_predict_caps_covariance_trace(self) -> None:
        state = initialize(0.0, 0.0, 0.1, 0.1, _now())
        far_future = datetime(2026, 5, 26, 13, 0, 0, tzinfo=UTC)  # 1 hour later
        predicted = predict(state, far_future, 0.5)
        pos_trace = float(predicted.covariance[0, 0] + predicted.covariance[1, 1])
        assert pos_trace <= 100.0


class TestUpdate:
    def test_update_reduces_position_uncertainty(self) -> None:
        state = initialize(0.0, 0.0, 1.0, 2.0, _now())
        init_pos_trace = float(state.covariance[0, 0] + state.covariance[1, 1])
        updated = update(state, 0.1, 0.1, isotropic_cov(0.25))
        updated_pos_trace = float(updated.covariance[0, 0] + updated.covariance[1, 1])
        assert updated_pos_trace < init_pos_trace

    def test_update_moves_mean_toward_observation(self) -> None:
        state = initialize(0.0, 0.0, 1.0, 2.0, _now())
        updated = update(state, 0.5, 0.5, isotropic_cov(0.25))
        assert updated.mean[0] > 0.0
        assert updated.mean[1] > 0.0


class TestMahalanobis2:
    def test_zero_at_mean(self) -> None:
        state = initialize(0.0, 0.0, 0.5, 1.0, _now())
        d2 = mahalanobis2_position(state, 0.0, 0.0, isotropic_cov(0.25))
        assert d2 == 0.0

    def test_large_for_distant_point(self) -> None:
        state = initialize(0.0, 0.0, 0.1, 0.1, _now())
        d2 = mahalanobis2_position(state, 20.0, 20.0, isotropic_cov(0.25))
        assert d2 > 100.0

    def test_increases_with_distance(self) -> None:
        state = initialize(0.0, 0.0, 0.1, 0.1, _now())
        d2_close = mahalanobis2_position(state, 0.5, 0.0, isotropic_cov(0.25))
        d2_far = mahalanobis2_position(state, 5.0, 0.0, isotropic_cov(0.25))
        assert d2_far > d2_close


class TestIsotropicCov:
    def test_shape_is_2x2(self) -> None:
        R = isotropic_cov(0.25)  # noqa: N806
        assert R.shape == (2, 2)

    def test_diagonal_equals_sigma_squared(self) -> None:
        R = isotropic_cov(0.3)  # noqa: N806
        np.testing.assert_allclose(R[0, 0], 0.09)
        np.testing.assert_allclose(R[1, 1], 0.09)

    def test_off_diagonal_zero(self) -> None:
        R = isotropic_cov(0.25)  # noqa: N806
        assert R[0, 1] == 0.0
        assert R[1, 0] == 0.0


class TestUpdateMatrixRParityWithScalar:
    """isotropic_cov path must reproduce the old scalar path to machine precision."""

    def test_parity_mean(self) -> None:
        # Golden numbers captured from the pre-refactor scalar implementation.
        state = initialize(1.0, 2.0, 0.5, 1.0, _now())
        posterior = update(state, 1.3, 2.1, isotropic_cov(0.25))
        np.testing.assert_allclose(
            posterior.mean,
            [1.24, 2.08, 0.0, 0.0],
            atol=1e-12,
        )

    def test_parity_covariance(self) -> None:
        state = initialize(1.0, 2.0, 0.5, 1.0, _now())
        posterior = update(state, 1.3, 2.1, isotropic_cov(0.25))
        expected_cov = [
            [0.04999999999999999, 0.0, 0.0, 0.0],
            [0.0, 0.04999999999999999, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        np.testing.assert_allclose(posterior.covariance, expected_cov, atol=1e-12)

    def test_parity_mahalanobis(self) -> None:
        state = initialize(1.0, 2.0, 0.5, 1.0, _now())
        d2 = mahalanobis2_position(state, 1.3, 2.1, isotropic_cov(0.25))
        np.testing.assert_allclose(d2, 0.3200000000000002, atol=1e-12)


class TestUpdateAnisotropicR:
    """Anisotropic R must reach the filter and pull the posterior asymmetrically."""

    def test_pulls_more_along_low_variance_axis(self) -> None:
        # State at origin with isotropic prior. Observe at (1.0, 1.0).
        # R = diag(0.01, 1.0): x has low noise → filter trusts x observation more.
        state = initialize(0.0, 0.0, 0.5, 1.0, _now())
        R = np.diag([0.01, 1.0]).astype(np.float64)  # noqa: N806
        posterior = update(state, 1.0, 1.0, R)
        # x innovation and y innovation are equal, but x posterior move >> y.
        x_shift = float(posterior.mean[0])
        y_shift = float(posterior.mean[1])
        assert x_shift > y_shift * 2, (
            f"Expected x_shift ({x_shift:.4f}) >> y_shift ({y_shift:.4f}) under diag(0.01,1.0)"
        )

    def test_covariance_shrinks_less_along_high_noise_axis(self) -> None:
        state = initialize(0.0, 0.0, 0.5, 1.0, _now())
        R = np.diag([0.01, 1.0]).astype(np.float64)  # noqa: N806
        posterior = update(state, 1.0, 1.0, R)
        # x posterior variance shrinks more (more trusted); y shrinks less.
        assert posterior.covariance[0, 0] < posterior.covariance[1, 1]


class TestMahalanobisAnisotropic:
    """A fixed offset gates differently under transposed R matrices."""

    def test_offset_along_x_gates_differently_under_rotated_r(self) -> None:
        # Observation is 1 m along x axis only.
        state = initialize(0.0, 0.0, 0.1, 0.1, _now())
        r_x_tight = np.diag([0.01, 1.0]).astype(np.float64)  # x is precise
        r_y_tight = np.diag([1.0, 0.01]).astype(np.float64)  # y is precise
        # Offset is along x, so it is penalized more when x is tight.
        d2_x_tight = mahalanobis2_position(state, 1.0, 0.0, r_x_tight)
        d2_y_tight = mahalanobis2_position(state, 1.0, 0.0, r_y_tight)
        assert d2_x_tight > d2_y_tight, (
            f"x-tight ({d2_x_tight:.2f}) should be > y-tight ({d2_y_tight:.2f}) for a pure x offset"
        )
