"""Stateless homography-based floor projection.

Converts the footpoint (bottom-center) of a pixel-space BoundingBox to
millimetre floor-plane coordinates using a per-camera homography matrix
stored in CalibrationState.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from structlog import get_logger

from ..calibration.state import CalibrationState
from ..domain import BoundingBox, FloorPoint, cov2x2_to_tuple
from .world.observation_model import homography_jacobian

logger = get_logger(__name__)


def _project_raw(h: list[list[float]], px: float, py: float) -> tuple[float, float, float]:
    """Return dehomogenized floor metres and homogeneous w for a pixel."""
    x_h = h[0][0] * px + h[0][1] * py + h[0][2]
    y_h = h[1][0] * px + h[1][1] * py + h[1][2]
    w_h = h[2][0] * px + h[2][1] * py + h[2][2]
    if abs(w_h) < 1e-9:
        raise ValueError("homography projection is degenerate")
    return (x_h / w_h, y_h / w_h, w_h)


class FloorProjector:
    """Projects pixel bounding boxes to floor-plane coordinates.

    The calibration_state is read on each call so hot-reloaded homographies
    are picked up automatically.
    """

    def __init__(self, calibration_state: CalibrationState) -> None:
        self._state = calibration_state

    def project(self, camera_id: str, bbox: BoundingBox) -> FloorPoint:
        """Project a bounding box footpoint to the floor plane.

        The footpoint is the bottom-centre of the bounding box, which is the
        pixel closest to where the person's feet contact the ground.

        Args:
            camera_id: the camera whose homography to use.
            bbox: the pixel-space bounding box.

        Returns:
            A FloorPoint in mm.  ``calibrated=True`` when a homography was
            available; ``calibrated=False`` (with x_mm=y_mm=0) when not.
        """
        homography = self._state.homographies.get(camera_id)
        if not homography:
            return FloorPoint(x_mm=0, y_mm=0, calibrated=False)

        # Footpoint: bottom-centre of the bounding box.
        fx = (bbox.x_min + bbox.x_max) / 2.0
        fy = float(bbox.y_max)

        try:
            x_m, y_m, _w_h = _project_raw(homography, fx, fy)
        except ValueError:
            return FloorPoint(x_mm=0, y_mm=0, calibrated=False)

        # H maps raw pixels to metres; convert to mm for FloorPoint.
        return FloorPoint(
            x_mm=round(x_m * 1000.0),
            y_mm=round(y_m * 1000.0),
            calibrated=True,
        )

    def project_with_covariance(
        self,
        camera_id: str,
        bbox: BoundingBox,
        pixel_cov_px2: npt.NDArray[np.float64],
    ) -> tuple[FloorPoint, tuple[float, float, float, float] | None]:
        """Project the footpoint and propagate pixel covariance to floor m².

        Returns ``(point, None)`` when the camera has no valid homography or
        the homography is degenerate at the bbox bottom-centre.
        """
        homography = self._state.homographies.get(camera_id)
        if not homography:
            return (FloorPoint(x_mm=0, y_mm=0, calibrated=False), None)

        fx = (bbox.x_min + bbox.x_max) / 2.0
        fy = float(bbox.y_max)
        point = self.project(camera_id, bbox)
        if not point.calibrated:
            return (point, None)

        h = np.array(homography, dtype=np.float64)
        try:
            jacobian_m_per_px = homography_jacobian(h, fx, fy)
        except ValueError:
            logger.warning("floor_projector_jacobian_degenerate", camera_id=camera_id)
            return (point, None)
        floor_cov_m2 = jacobian_m_per_px @ pixel_cov_px2 @ jacobian_m_per_px.T
        return (point, cov2x2_to_tuple(floor_cov_m2))

    def estimate_height_mm(self, camera_id: str, bbox: BoundingBox) -> float | None:
        """Estimate person height from bbox geometry + homography.

        Projects the head point (top-centre) and footpoint (bottom-centre)
        through the camera homography and measures the Euclidean distance
        between them in millimetres.

        Returns:
            Height in mm, or None when the camera lacks a calibrated
            homography or the projection is degenerate (W ≈ 0).
        """
        homography = self._state.homographies.get(camera_id)
        if not homography:
            return None

        # Footpoint: bottom-centre.
        fx = (bbox.x_min + bbox.x_max) / 2.0
        fy = float(bbox.y_max)

        # Head point: top-centre.
        hx = (bbox.x_min + bbox.x_max) / 2.0
        hy = float(bbox.y_min)

        h = homography

        try:
            foot_x_m, foot_y_m, _fw_h = _project_raw(h, fx, fy)
        except ValueError:
            return None

        # Project head point.
        try:
            head_x_m, head_y_m, _hw_h = _project_raw(h, hx, hy)
        except ValueError:
            return None

        # Euclidean distance in mm.
        dx_m = head_x_m - foot_x_m
        dy_m = head_y_m - foot_y_m
        return math.hypot(dx_m, dy_m) * 1000.0

    @staticmethod
    def distance_m(a: FloorPoint, b: FloorPoint) -> float:
        """Euclidean distance between two floor points in metres."""
        dx = a.x_mm - b.x_mm
        dy = a.y_mm - b.y_mm
        return math.hypot(dx, dy) / 1000.0
