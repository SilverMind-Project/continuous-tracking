"""Tests for floor-plane Kalman filter (app/tracking/world/kalman.py)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.tracking.world.kalman import (
    initialize,
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
        updated = update(state, 0.1, 0.1, 0.25)
        updated_pos_trace = float(updated.covariance[0, 0] + updated.covariance[1, 1])
        assert updated_pos_trace < init_pos_trace

    def test_update_moves_mean_toward_observation(self) -> None:
        state = initialize(0.0, 0.0, 1.0, 2.0, _now())
        updated = update(state, 0.5, 0.5, 0.25)
        assert updated.mean[0] > 0.0
        assert updated.mean[1] > 0.0


class TestMahalanobis2:
    def test_zero_at_mean(self) -> None:
        state = initialize(0.0, 0.0, 0.5, 1.0, _now())
        d2 = mahalanobis2_position(state, 0.0, 0.0, 0.25)
        assert d2 == 0.0

    def test_large_for_distant_point(self) -> None:
        state = initialize(0.0, 0.0, 0.1, 0.1, _now())
        d2 = mahalanobis2_position(state, 20.0, 20.0, 0.25)
        assert d2 > 100.0

    def test_increases_with_distance(self) -> None:
        state = initialize(0.0, 0.0, 0.1, 0.1, _now())
        d2_close = mahalanobis2_position(state, 0.5, 0.0, 0.25)
        d2_far = mahalanobis2_position(state, 5.0, 0.0, 0.25)
        assert d2_far > d2_close
