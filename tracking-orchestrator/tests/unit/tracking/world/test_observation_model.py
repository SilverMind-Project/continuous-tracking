"""Tests for the pure observation geometry model."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from app.domain import ObservationGeometry, OrientationBin
from app.tracking.world.observation_model import (
    calibration_covariance,
    homography_jacobian,
    observation_covariance,
    pixel_covariance,
    posture_view_weight,
    primary_camera_score,
    random_covariance,
)

NDArrayF8 = npt.NDArray[np.float64]


def _geo(
    *,
    footpoint_px: tuple[float, float] = (960.0, 720.0),
    floor_residual_m: float = 0.04,
    footpoint_reliable: bool = True,
    detection_confidence: float = 0.9,
    crop_quality: float = 0.8,
    orientation: OrientationBin = OrientationBin.LEFT,
    orientation_confidence: float = 1.0,
) -> ObservationGeometry:
    return ObservationGeometry(
        footpoint_px=footpoint_px,
        floor_residual_m=floor_residual_m,
        footpoint_reliable=footpoint_reliable,
        detection_confidence=detection_confidence,
        crop_quality=crop_quality,
        orientation=orientation,
        orientation_confidence=orientation_confidence,
    )


def _project_m(h: NDArrayF8, px: float, py: float) -> NDArrayF8:
    point = h @ np.array([px, py, 1.0], dtype=np.float64)
    if abs(float(point[2])) < 1e-9:
        raise ValueError("degenerate projection")
    return np.array([point[0] / point[2], point[1] / point[2]], dtype=np.float64)


def _finite_difference_jacobian(h: NDArrayF8, px: float, py: float) -> NDArrayF8:
    eps_px = 1e-4
    dx = (_project_m(h, px + eps_px, py) - _project_m(h, px - eps_px, py)) / (2.0 * eps_px)
    dy = (_project_m(h, px, py + eps_px) - _project_m(h, px, py - eps_px)) / (2.0 * eps_px)
    return np.column_stack((dx, dy)).astype(np.float64)


def test_jacobian_matches_finite_difference() -> None:
    h = np.array(
        [
            [0.012, 0.001, -7.5],
            [0.0006, 0.018, -5.0],
            [0.00008, 0.00035, 1.0],
        ],
        dtype=np.float64,
    )
    px = 713.5
    py = 905.25

    analytic = homography_jacobian(h, px, py)
    finite_diff = _finite_difference_jacobian(h, px, py)

    np.testing.assert_allclose(analytic, finite_diff, rtol=1e-6, atol=1e-8)


def test_jacobian_degenerate_raises() -> None:
    h = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, -10.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="degenerate"):
        homography_jacobian(h, 10.0, 20.0)


def test_oblique_camera_r_elongated_along_ray() -> None:
    h = np.array(
        [
            [0.01, 0.0, -6.4],
            [0.0, 0.01, -3.6],
            [0.0, -0.0016, 1.5],
        ],
        dtype=np.float64,
    )
    geo = _geo(footpoint_px=(640.0, 900.0), floor_residual_m=0.0)

    cov_m2 = observation_covariance(h, geo)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_m2)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    camera_to_point = _project_m(h, *geo.footpoint_px)
    camera_to_point /= np.linalg.norm(camera_to_point)

    assert float(np.max(eigenvalues) / np.min(eigenvalues)) > 100.0
    assert abs(float(np.dot(major_axis, camera_to_point))) > 0.9


def test_unreliable_footpoint_inflates_sigma() -> None:
    h = np.eye(3, dtype=np.float64)
    reliable = _geo(footpoint_reliable=True)
    unreliable = _geo(footpoint_reliable=False)

    reliable_det = np.linalg.det(observation_covariance(h, reliable))
    unreliable_det = np.linalg.det(observation_covariance(h, unreliable))

    assert float(unreliable_det) > float(reliable_det)


def test_low_confidence_inflates_sigma() -> None:
    h = np.eye(3, dtype=np.float64)
    high_confidence = _geo(detection_confidence=0.95, crop_quality=0.9)
    low_confidence = _geo(detection_confidence=0.1, crop_quality=0.2)

    high_det = np.linalg.det(observation_covariance(h, high_confidence))
    low_det = np.linalg.det(observation_covariance(h, low_confidence))

    assert float(low_det) > float(high_det)


def test_calibration_covariance_scales_with_residual() -> None:
    small = calibration_covariance(_geo(floor_residual_m=0.05))
    large = calibration_covariance(_geo(floor_residual_m=0.10))

    np.testing.assert_allclose(large, 4.0 * small)


def test_random_vs_total_covariance_split() -> None:
    h = np.array(
        [
            [0.008, 0.001, -3.0],
            [0.0005, 0.009, -2.0],
            [0.00002, -0.0001, 1.0],
        ],
        dtype=np.float64,
    )
    geo = _geo()
    numeric_floor_m2 = 1e-4 * np.eye(2, dtype=np.float64)

    expected = random_covariance(h, geo) + calibration_covariance(geo) + numeric_floor_m2

    np.testing.assert_allclose(observation_covariance(h, geo), expected)


def test_posture_view_weight_penalizes_frontal_and_occluded() -> None:
    side = _geo(orientation=OrientationBin.LEFT, orientation_confidence=1.0)
    front = _geo(orientation=OrientationBin.FRONT, orientation_confidence=1.0)
    occluded_side = _geo(
        footpoint_reliable=False,
        orientation=OrientationBin.LEFT,
        orientation_confidence=1.0,
    )
    uncertain_front = _geo(orientation=OrientationBin.FRONT, orientation_confidence=0.0)

    assert posture_view_weight(front) < posture_view_weight(side)
    assert posture_view_weight(occluded_side) < posture_view_weight(side)
    assert math.isclose(posture_view_weight(uncertain_front), 1.0)


def test_primary_camera_score_prefers_reliable_high_quality() -> None:
    good = _geo(footpoint_reliable=True, detection_confidence=0.95, crop_quality=0.9)
    unreliable = _geo(footpoint_reliable=False, detection_confidence=0.95, crop_quality=0.9)
    low_quality = _geo(footpoint_reliable=True, detection_confidence=0.5, crop_quality=0.4)

    assert primary_camera_score(good) > primary_camera_score(unreliable)
    assert primary_camera_score(good) > primary_camera_score(low_quality)


def test_all_outputs_are_python_float_and_2x2_float64() -> None:
    h = np.eye(3, dtype=np.float64)
    geo = _geo()

    for cov_m2 in (
        pixel_covariance(geo),
        calibration_covariance(geo),
        random_covariance(h, geo),
        observation_covariance(h, geo),
    ):
        assert cov_m2.shape == (2, 2)
        assert cov_m2.dtype == np.float64

    assert isinstance(posture_view_weight(geo), float)
    assert isinstance(primary_camera_score(geo), float)
