"""Pure observation geometry and uncertainty model for floor-plane tracking."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from app.domain import ObservationGeometry, OrientationBin

NDArrayF8 = npt.NDArray[np.float64]

# Detector bbox-bottom localization noise in raw image pixels (px, 1 sigma).
_BASE_FOOTPOINT_SIGMA_PX: float = 4.0
# Standard-deviation multiplier when feet are hidden or the bbox is truncated.
_OCCLUDED_INFLATION: float = 8.0
# Confidence floor for scaling sigma_px; avoids div-by-zero and unbounded R.
_MIN_CONF_FLOOR: float = 0.05
# Calibration residual gain; R_cal = (K_CAL * residual_m)^2 * I in m^2.
_K_CAL: float = 1.0
# Diagonal covariance floor in m^2 to keep observation covariance invertible.
_NUMERIC_FLOOR_M2: float = 1e-4

_DEGENERATE_HOMOGRAPHY_EPS: float = 1e-9


def _homography_matrix(h: npt.ArrayLike) -> NDArrayF8:
    matrix = np.asarray(h, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    return matrix


def homography_jacobian(h: npt.ArrayLike, px: float, py: float) -> NDArrayF8:
    """Return d(floor_m)/d(pixel_px), the 2x2 homography Jacobian in m/px.

    Args:
        h: 3x3 homography mapping raw pixels to floor-plan metres.
        px: Raw image x coordinate of the footpoint in px.
        py: Raw image y coordinate of the footpoint in px.

    Raises:
        ValueError: if the homography shape is not 3x3 or the projection is
            degenerate at the requested pixel.
    """
    matrix = _homography_matrix(h)
    nx_m = matrix[0, 0] * px + matrix[0, 1] * py + matrix[0, 2]
    ny_m = matrix[1, 0] * px + matrix[1, 1] * py + matrix[1, 2]
    denominator = matrix[2, 0] * px + matrix[2, 1] * py + matrix[2, 2]
    if abs(float(denominator)) < _DEGENERATE_HOMOGRAPHY_EPS:
        raise ValueError("homography projection is degenerate at footpoint")

    denominator2 = denominator * denominator
    jacobian_m_per_px = np.array(
        [
            [
                matrix[0, 0] * denominator - nx_m * matrix[2, 0],
                matrix[0, 1] * denominator - nx_m * matrix[2, 1],
            ],
            [
                matrix[1, 0] * denominator - ny_m * matrix[2, 0],
                matrix[1, 1] * denominator - ny_m * matrix[2, 1],
            ],
        ],
        dtype=np.float64,
    )
    scaled_jacobian_m_per_px: NDArrayF8 = jacobian_m_per_px / float(denominator2)
    return scaled_jacobian_m_per_px


def pixel_covariance(geo: ObservationGeometry) -> NDArrayF8:
    """Return the image-space footpoint covariance Σ_px in px^2."""
    sigma_px = _BASE_FOOTPOINT_SIGMA_PX
    if not geo.footpoint_reliable:
        sigma_px *= _OCCLUDED_INFLATION

    detection_confidence = max(geo.detection_confidence, _MIN_CONF_FLOOR)
    crop_quality = max(geo.crop_quality, _MIN_CONF_FLOOR)
    confidence_scale = math.sqrt(detection_confidence * crop_quality)
    sigma_px /= confidence_scale

    variance_px2 = sigma_px * sigma_px
    return variance_px2 * np.eye(2, dtype=np.float64)


def calibration_covariance(geo: ObservationGeometry) -> NDArrayF8:
    """Return the systematic calibration covariance R_cal in floor-plan m^2."""
    variance_m2 = (_K_CAL * geo.floor_residual_m) ** 2
    return variance_m2 * np.eye(2, dtype=np.float64)


def random_covariance(h: npt.ArrayLike, geo: ObservationGeometry) -> NDArrayF8:
    """Return the random projected covariance J·Σ_px·Jᵀ in floor-plan m^2."""
    jacobian_m_per_px = homography_jacobian(h, *geo.footpoint_px)
    sigma_px2 = pixel_covariance(geo)
    return jacobian_m_per_px @ sigma_px2 @ jacobian_m_per_px.T


def observation_covariance(h: npt.ArrayLike, geo: ObservationGeometry) -> NDArrayF8:
    """Return full single-camera observation covariance R in floor-plan m^2."""
    numeric_floor_m2 = _NUMERIC_FLOOR_M2 * np.eye(2, dtype=np.float64)
    return random_covariance(h, geo) + calibration_covariance(geo) + numeric_floor_m2


def posture_view_weight(geo: ObservationGeometry) -> float:
    """Return a [0, 1] posture multiplier based on view geometry.

    Side views best separate sit, stand, and lie. Frontal/back views are
    foreshortened, so their factor is lower when orientation confidence is
    high. Low orientation confidence blends the factor back toward 1.0 to
    avoid over-penalizing an uncertain pose estimate.
    """
    weight = 1.0
    if not geo.footpoint_reliable:
        weight *= 0.3

    orientation_factor = {
        OrientationBin.FRONT: 0.6,
        OrientationBin.BACK: 0.6,
        OrientationBin.LEFT: 1.0,
        OrientationBin.RIGHT: 1.0,
        OrientationBin.UNKNOWN: 0.5,
    }[geo.orientation]
    confidence = min(max(geo.orientation_confidence, 0.0), 1.0)
    blended_orientation_factor = (confidence * orientation_factor) + (1.0 - confidence)
    return float(min(max(weight * blended_orientation_factor, 0.0), 1.0))


def primary_camera_score(geo: ObservationGeometry) -> float:
    """Return a [0, 1] scalar view score for primary camera selection."""
    reliability_factor = 1.0 if geo.footpoint_reliable else 0.5
    score = reliability_factor * geo.crop_quality * geo.detection_confidence
    return float(min(max(score, 0.0), 1.0))
