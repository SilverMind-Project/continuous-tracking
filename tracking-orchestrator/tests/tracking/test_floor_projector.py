"""Tests for homography floor projection helpers."""

from __future__ import annotations

import numpy as np

from app.calibration.state import CalibrationState
from app.domain import BoundingBox, tuple_to_cov2x2
from app.tracking.floor_projector import FloorProjector
from app.tracking.world.observation_model import homography_jacobian


def _bbox() -> BoundingBox:
    return BoundingBox(x_min=100, y_min=200, x_max=300, y_max=400)


def test_project_unchanged_behavior() -> None:
    state = CalibrationState()
    state.homographies["cam-1"] = [[0.01, 0.0, -1.0], [0.0, 0.02, -2.0], [0.0, 0.0, 1.0]]
    projector = FloorProjector(state)

    point = projector.project("cam-1", _bbox())

    assert point.calibrated
    assert point.x_mm == 1000
    assert point.y_mm == 6000


def test_project_with_covariance_uncalibrated_returns_none() -> None:
    projector = FloorProjector(CalibrationState())

    point, floor_cov_random = projector.project_with_covariance(
        "missing-cam",
        _bbox(),
        np.eye(2, dtype=np.float64),
    )

    assert not point.calibrated
    assert floor_cov_random is None


def test_project_with_covariance_matches_jacobian() -> None:
    state = CalibrationState()
    h = np.array(
        [
            [0.012, 0.001, -7.5],
            [0.0006, 0.018, -5.0],
            [0.00008, 0.00035, 1.0],
        ],
        dtype=np.float64,
    )
    state.homographies["cam-1"] = h.tolist()
    projector = FloorProjector(state)
    pixel_cov_px2 = np.array([[9.0, 1.0], [1.0, 16.0]], dtype=np.float64)

    point, floor_cov_random = projector.project_with_covariance(
        "cam-1",
        _bbox(),
        pixel_cov_px2,
    )

    assert point.calibrated
    assert floor_cov_random is not None
    footpoint_px = (200.0, 400.0)
    jacobian_m_per_px = homography_jacobian(h, *footpoint_px)
    expected = jacobian_m_per_px @ pixel_cov_px2 @ jacobian_m_per_px.T
    np.testing.assert_allclose(tuple_to_cov2x2(floor_cov_random), expected)
