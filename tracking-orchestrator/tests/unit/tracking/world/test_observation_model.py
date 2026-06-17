"""Tests for the pure observation geometry model."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from app.domain import ObservationGeometry, OrientationBin
from app.inference.schemas import Keypoint, PoseResult
from app.tracking.world.observation_model import (
    bias_floor_from_residual,
    calibration_covariance,
    footpoint_reliable,
    fuse_information_form,
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


def _pose(left_ankle_score: float = 0.9, right_ankle_score: float = 0.9) -> PoseResult:
    keypoints = [Keypoint(x=0.5, y=0.5, score=0.9) for _ in range(17)]
    keypoints[15] = Keypoint(x=0.45, y=0.95, score=left_ankle_score)
    keypoints[16] = Keypoint(x=0.55, y=0.95, score=right_ankle_score)
    return PoseResult(keypoints=tuple(keypoints))


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


def test_footpoint_reliable_false_when_bottom_truncated() -> None:
    assert not footpoint_reliable(
        _pose(),
        (100, 100, 220, 478),
        image_w=640,
        image_h=480,
        edge_margin_px=4,
    )


def test_footpoint_reliable_false_when_ankles_occluded() -> None:
    assert not footpoint_reliable(
        _pose(left_ankle_score=0.1, right_ankle_score=0.2),
        (100, 100, 220, 420),
        image_w=640,
        image_h=480,
    )


def test_footpoint_reliable_true_when_clean_and_ankle_visible() -> None:
    assert footpoint_reliable(
        _pose(left_ankle_score=0.1, right_ankle_score=0.8),
        (100, 100, 220, 420),
        image_w=640,
        image_h=480,
    )


def test_footpoint_reliable_true_without_pose_when_not_truncated() -> None:
    assert footpoint_reliable(
        None,
        (100, 100, 220, 420),
        image_w=640,
        image_h=480,
    )


# ---------------------------------------------------------------------------
# bias_floor_from_residual tests
# ---------------------------------------------------------------------------


def test_bias_floor_from_residual_zero_returns_numeric_floor() -> None:
    bf = bias_floor_from_residual(0.0)
    assert bf.shape == (2, 2)
    assert float(bf[0, 0]) >= 1e-4
    assert float(bf[1, 1]) >= 1e-4


def test_bias_floor_from_residual_scales_with_residual() -> None:
    small = bias_floor_from_residual(0.05)
    large = bias_floor_from_residual(0.10)
    assert float(large[0, 0]) > float(small[0, 0])


def test_bias_floor_from_residual_k_cal_scales_result() -> None:
    base = bias_floor_from_residual(0.1, k_cal=1.0)
    doubled = bias_floor_from_residual(0.1, k_cal=2.0)
    np.testing.assert_allclose(doubled[0, 0], 4.0 * base[0, 0])


def test_bias_floor_is_isotropic() -> None:
    bf = bias_floor_from_residual(0.2)
    np.testing.assert_allclose(bf[0, 0], bf[1, 1])
    np.testing.assert_allclose(bf[0, 1], 0.0)
    np.testing.assert_allclose(bf[1, 0], 0.0)


# ---------------------------------------------------------------------------
# fuse_information_form tests
# ---------------------------------------------------------------------------


def _iso_cov(sigma_m: float) -> NDArrayF8:
    return (sigma_m**2) * np.eye(2, dtype=np.float64)


def test_single_camera_returns_own_covariance_plus_bias_floor() -> None:
    r_rand = _iso_cov(0.1)
    bias_floor = _iso_cov(0.05)
    (x, y), cov_rm = fuse_information_form([(3.0, 4.0)], [r_rand], bias_floor)
    fused_cov = np.array(cov_rm).reshape(2, 2)
    # The implementation adds a numeric floor (1e-4) to r_rand before inversion for
    # numerical stability, so the result is r_rand + numeric_floor + bias_floor.
    numeric_floor = 1e-4 * np.eye(2, dtype=np.float64)
    expected = r_rand + numeric_floor + bias_floor
    np.testing.assert_allclose(x, 3.0, atol=1e-9)
    np.testing.assert_allclose(y, 4.0, atol=1e-9)
    np.testing.assert_allclose(fused_cov, expected, rtol=1e-6)


def test_fusion_weights_toward_low_covariance_camera() -> None:
    r_low = _iso_cov(0.01)  # very precise camera
    r_high = _iso_cov(1.0)  # very imprecise camera
    bias_floor = _iso_cov(0.001)
    points = [(0.0, 0.0), (10.0, 0.0)]
    covs = [r_low, r_high]
    (x, _y), _ = fuse_information_form(points, covs, bias_floor)
    # Fused x must be much closer to 0.0 (low-R camera) than to 10.0 (high-R camera).
    assert x < 1.0, f"fused x={x} should be near 0.0 (the low-covariance camera)"


def test_fusion_does_not_shrink_below_bias_floor() -> None:
    """Anti-jump guarantee: R* diagonal → bias_floor as N grows, never → 0."""
    bias_floor = _iso_cov(0.1)
    r_rand = _iso_cov(0.05)
    for n_cameras in [2, 5, 10, 50]:
        points = [(1.0, 2.0)] * n_cameras
        covs = [r_rand] * n_cameras
        _xy, cov_rm = fuse_information_form(points, covs, bias_floor)
        fused_cov = np.array(cov_rm).reshape(2, 2)
        assert float(fused_cov[0, 0]) >= float(bias_floor[0, 0]) - 1e-9, (
            f"R* diagonal {fused_cov[0, 0]} fell below bias_floor {bias_floor[0, 0]} "
            f"at N={n_cameras}"
        )


def test_fusion_symmetric_cameras_return_shared_position() -> None:
    r_rand = _iso_cov(0.1)
    bias_floor = _iso_cov(0.01)
    (x, y), _ = fuse_information_form([(3.0, 4.0), (3.0, 4.0)], [r_rand, r_rand], bias_floor)
    np.testing.assert_allclose(x, 3.0, atol=1e-9)
    np.testing.assert_allclose(y, 4.0, atol=1e-9)


def test_fuse_information_form_empty_raises() -> None:
    with pytest.raises(ValueError, match="no valid"):
        fuse_information_form([], [], _iso_cov(0.1))
