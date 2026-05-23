"""Spatial projection service: canonical floor-plane projection and comparison.

Replaces direct ``FloorProjector`` usage with a higher-level service that
knows about floor-plan coordinate systems and can gate cross-camera metric
comparisons on shared calibration.
"""

from __future__ import annotations

import math

from structlog import get_logger

from ..calibration.state import CalibrationState
from ..domain import BoundingBox, FloorPoint

logger = get_logger(__name__)


class SpatialProjectionService:
    """Projects detections to shared floor-plan coordinates.

    The calibration_state is read on each call so hot-reloaded homographies
    are picked up automatically.
    """

    def __init__(self, calibration_state: CalibrationState) -> None:
        self._state = calibration_state

    def project_detection(self, camera_id: str, bbox: BoundingBox) -> FloorPoint:
        """Project a detection footpoint to the shared floor plane.

        Uses the bottom-centre of *bbox* as the footpoint.  Returns a
        calibrated ``FloorPoint`` when a valid homography exists for
        *camera_id*; otherwise returns ``FloorPoint(0, 0, calibrated=False)``.
        """
        homography = self._state.homographies.get(camera_id)

        if not homography:
            return FloorPoint(x_mm=0, y_mm=0, calibrated=False)

        # Footpoint: bottom-centre of the bounding box.
        fx = (bbox.x_min + bbox.x_max) / 2.0
        fy = float(bbox.y_max)

        h = homography
        x_h = h[0][0] * fx + h[0][1] * fy + h[0][2]
        y_h = h[1][0] * fx + h[1][1] * fy + h[1][2]
        w_h = h[2][0] * fx + h[2][1] * fy + h[2][2]

        if abs(w_h) < 1e-9:
            return FloorPoint(x_mm=0, y_mm=0, calibrated=False)

        x_m = x_h / w_h
        y_m = y_h / w_h

        if not (math.isfinite(x_m) and math.isfinite(y_m)):
            return FloorPoint(x_mm=0, y_mm=0, calibrated=False)

        return FloorPoint(
            x_mm=round(x_m * 1000.0),
            y_mm=round(y_m * 1000.0),
            calibrated=True,
        )

    def can_compare(self, camera_a: str, camera_b: str) -> bool:
        """Return True when floor points from *camera_a* and *camera_b*
        can be compared metrically — both must be calibrated to the same
        non-empty ``floor_plan_id``.
        """
        cal_a = self._state.calibrations.get(camera_a)
        cal_b = self._state.calibrations.get(camera_b)

        if cal_a is None or cal_b is None:
            return False
        if not cal_a.floor_plan_id or not cal_b.floor_plan_id:
            return False
        return cal_a.floor_plan_id == cal_b.floor_plan_id

    def floor_plan_id_for(self, camera_id: str) -> str | None:
        """Return the floor_plan_id for *camera_id*, or None if uncalibrated."""
        cal = self._state.calibrations.get(camera_id)
        if cal is None:
            return None
        return cal.floor_plan_id or None

    @staticmethod
    def distance_m(a: FloorPoint, b: FloorPoint) -> float | None:
        """Euclidean distance between two floor points in metres.

        Returns None when either point is uncalibrated or the points
        are known to be in different coordinate systems (the caller
        must check ``can_compare`` first).
        """
        if not a.calibrated or not b.calibrated:
            return None
        dx = a.x_mm - b.x_mm
        dy = a.y_mm - b.y_mm
        return math.hypot(dx, dy) / 1000.0
