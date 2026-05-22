"""Stateless homography-based floor projection.

Converts the footpoint (bottom-center) of a pixel-space BoundingBox to
millimetre floor-plane coordinates using a per-camera homography matrix
stored in CalibrationState.
"""

from __future__ import annotations

import math

from ..calibration.state import CalibrationState
from ..domain import BoundingBox, FloorPoint


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

        # Homogeneous projection: [X, Y, W] = H @ [fx, fy, 1]
        h = homography  # list[list[float]], 3x3 row-major
        x_h = h[0][0] * fx + h[0][1] * fy + h[0][2]
        y_h = h[1][0] * fx + h[1][1] * fy + h[1][2]
        w_h = h[2][0] * fx + h[2][1] * fy + h[2][2]

        if abs(w_h) < 1e-9:
            return FloorPoint(x_mm=0, y_mm=0, calibrated=False)

        # H maps raw pixels to metres; convert to mm for FloorPoint.
        return FloorPoint(
            x_mm=round((x_h / w_h) * 1000.0),
            y_mm=round((y_h / w_h) * 1000.0),
            calibrated=True,
        )

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

        # Project footpoint.
        fx_h = h[0][0] * fx + h[0][1] * fy + h[0][2]
        fy_h = h[1][0] * fx + h[1][1] * fy + h[1][2]
        fw_h = h[2][0] * fx + h[2][1] * fy + h[2][2]

        if abs(fw_h) < 1e-9:
            return None

        foot_x_m = fx_h / fw_h
        foot_y_m = fy_h / fw_h

        # Project head point.
        hx_h = h[0][0] * hx + h[0][1] * hy + h[0][2]
        hy_h = h[1][0] * hx + h[1][1] * hy + h[1][2]
        hw_h = h[2][0] * hx + h[2][1] * hy + h[2][2]

        if abs(hw_h) < 1e-9:
            return None

        head_x_m = hx_h / hw_h
        head_y_m = hy_h / hw_h

        # Euclidean distance in mm.
        dx_m = head_x_m - foot_x_m
        dy_m = head_y_m - foot_y_m
        return math.sqrt(dx_m * dx_m + dy_m * dy_m) * 1000.0

    @staticmethod
    def distance_m(a: FloorPoint, b: FloorPoint) -> float:
        """Euclidean distance between two floor points in metres."""
        dx = a.x_mm - b.x_mm
        dy = a.y_mm - b.y_mm
        return math.sqrt(dx * dx + dy * dy) / 1000.0
