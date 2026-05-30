"""Automatic homography estimation from a single camera frame.

Orchestrates the depth-estimation → floor-plane-fitting → homography-computation
pipeline.  The result is a draft homography that the operator should review in
the calibration UI before committing.

Accuracy notes
--------------
The computed homography uses an assumed horizontal FoV (default 70°, typical for
indoor surveillance cameras).  A 10° FoV error produces roughly 10-15% scale
error across the floor.  For precise metre-level accuracy, either:

* Ask the operator for the camera's published FoV specification, or
* Follow up with one or two manual calibration points to pin the scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from structlog import get_logger

from app.calibration.floor_plane import (
    FloorPlaneFitter,
    FloorPlaneResult,
    floor_plane_to_homography,
    sample_floor_plane_suggestions,
)
from app.inference.depth import DepthEstimator

logger = get_logger(__name__)

# Minimum inlier ratio before we declare the plane untrustworthy.
_MIN_CONFIDENCE = 0.10


@dataclass(frozen=True)
class AutoCalibrationResult:
    """Output from :class:`AutoCalibrator`."""

    #: Local pixel→camera-floor-metres homography (row-major nested list).
    #: This is not anchored to the shared household floor plan.
    draft_matrix: list[list[float]]
    #: Suggested floor inlier camera pixels for manual global anchoring.
    suggested_points: list[dict[str, list[float]]]
    #: Scalar confidence in [0, 1].  Values below 0.4 indicate a poor fit.
    confidence: float
    #: Number of inlier 3-D points used to compute the homography.
    inlier_count: int
    #: Total candidate points sampled from the depth map.
    sample_count: int
    #: Depth map shape used for fitting: (height, width).
    depth_shape: tuple[int, int]
    #: HoV value used for back-projection.
    fov_deg: float
    #: ``"depth_auto_draft"`` — must not be treated as committed calibration.
    method: str = "depth_auto_draft"


class AutoCalibrator:
    """Automatically estimate a pixel→floor-metres homography from one frame.

    Parameters
    ----------
    depth_estimator:
        A connected :class:`~app.inference.depth.DepthEstimator` instance.
    fov_deg:
        Camera's horizontal field of view in degrees.  Defaults to 70°.
    floor_region_fraction:
        Fraction of image height (from the bottom) treated as the floor
        candidate region.  Defaults to 0.6.
    """

    def __init__(
        self,
        depth_estimator: DepthEstimator,
        fov_deg: float = 70.0,
        floor_region_fraction: float = 0.60,
    ) -> None:
        self._depth_estimator = depth_estimator
        self._fov_deg = fov_deg
        self._fitter = FloorPlaneFitter(
            fov_deg=fov_deg,
            floor_region_fraction=floor_region_fraction,
        )

    async def calibrate(
        self,
        image: npt.NDArray[np.uint8],
        fov_deg: float | None = None,
    ) -> AutoCalibrationResult | None:
        """Run auto-calibration on one RGB frame.

        Args:
            image: ``(H, W, 3)`` uint8 RGB array.
            fov_deg: Override the instance-level FoV for this call only.
                Useful when the user supplies a per-request FoV value.

        Returns:
            :class:`AutoCalibrationResult` on success, or ``None`` when the
            floor plane cannot be detected reliably (confidence too low, or
            fewer than 4 inlier correspondences).
        """
        effective_fov = fov_deg if fov_deg is not None else self._fov_deg
        h, w = image.shape[:2]

        depth_map = await self._depth_estimator.estimate(image)

        depth_valid = (depth_map > 0.3) & (depth_map < 15.0)
        depth_stats = {
            "min": float(depth_map[depth_valid].min()) if depth_valid.any() else 0.0,
            "max": float(depth_map[depth_valid].max()) if depth_valid.any() else 0.0,
            "mean": float(depth_map[depth_valid].mean()) if depth_valid.any() else 0.0,
            "valid_fraction": float(depth_valid.mean()),
        }
        logger.debug("depth_map_computed", shape=depth_map.shape, **depth_stats)

        fitter = (
            self._fitter
            if effective_fov == self._fov_deg
            else FloorPlaneFitter(fov_deg=effective_fov)
        )
        plane_result: FloorPlaneResult | None = fitter.fit(depth_map)
        if plane_result is None:
            logger.warning(
                "auto_calibration_floor_fit_failed",
                reason="too_few_valid_pixels",
                fov_deg=effective_fov,
                **depth_stats,
            )
            return None

        if plane_result.confidence < _MIN_CONFIDENCE:
            logger.warning(
                "auto_calibration_confidence_low",
                confidence=round(plane_result.confidence, 3),
                min_confidence=_MIN_CONFIDENCE,
                inlier_ratio=round(plane_result.inlier_ratio, 3),
                mean_inlier_distance_m=round(plane_result.mean_inlier_distance, 3),
                fov_deg=effective_fov,
                **depth_stats,
            )
            return None

        matrix = floor_plane_to_homography(
            plane_result,
            image_h=h,
            image_w=w,
            fov_deg=effective_fov,
        )
        if matrix is None:
            logger.warning(
                "auto_calibration_homography_failed",
                reason="too_few_inlier_correspondences",
                inlier_count=int(plane_result.inlier_mask.sum()),
                fov_deg=effective_fov,
            )
            return None

        inlier_count = int(plane_result.inlier_mask.sum())
        return AutoCalibrationResult(
            draft_matrix=matrix,
            suggested_points=sample_floor_plane_suggestions(plane_result, count=9),
            confidence=float(plane_result.confidence),
            inlier_count=inlier_count,
            sample_count=len(plane_result.inlier_mask),
            depth_shape=(int(depth_map.shape[0]), int(depth_map.shape[1])),
            fov_deg=effective_fov,
        )
