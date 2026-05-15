"""Privacy zone enforcement in the frame processing pipeline.

Constructed from CalibrationState and read fresh each frame (hot-reload).
Uses shapely for foot-point polygon containment.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from shapely import Point, Polygon
from structlog import get_logger

from ..calibration.state import CalibrationState, PrivacyZoneConfig
from ..domain import BoundingBox

logger = get_logger(__name__)


class PrivacyZoneFilter:
    """Enforces privacy zones on frame data.

    Constructed fresh each frame from the hot-reloaded CalibrationState.
    Zones with foot-point policies require a calibrated homography;
    when absent the filter fails closed for that zone.

    Usage::

        pzf = PrivacyZoneFilter.from_state(calibration_state, camera_id)
        # Check detection foot-point containment
        kept = [d for d in detections if not pzf.should_drop(d)]
        # Apply blur/mask to frame in place
        pzf.apply_blur_mask(frame_numpy)
    """

    def __init__(
        self,
        camera_id: str,
        zones: list[PrivacyZoneConfig],
        homography_available: bool,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> None:
        self._camera_id = camera_id
        self._enabled_zones = [z for z in zones if z.enabled]
        self._homography_available = homography_available
        self._frame_width = frame_width
        self._frame_height = frame_height

        # Pre-build shapely polygons for each zone (normalised [0,1] coords).
        self._zone_polygons: list[tuple[PrivacyZoneConfig, Polygon]] = []
        for z in self._enabled_zones:
            if len(z.polygon) < 3:
                continue
            poly = Polygon([(p[0], p[1]) for p in z.polygon])
            self._zone_polygons.append((z, poly))

    @classmethod
    def from_state(
        cls,
        state: CalibrationState,
        camera_id: str,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> PrivacyZoneFilter:
        zones = state.privacy_zones.get(camera_id, [])
        has_homography = camera_id in state.homographies
        return cls(camera_id, zones, has_homography, frame_width, frame_height)

    def is_active(self) -> bool:
        """True if there are any enabled zones to enforce."""
        return len(self._zone_polygons) > 0

    def should_drop(self, foot_point_norm: tuple[float, float] | None = None) -> bool:
        """Return True if the detection at *foot_point_norm* should be dropped.

        *foot_point_norm* is (x, y) normalised to [0, 1] in the frame.
        When the homography is unavailable (uncalibrated + foot-point policy),
        all detections are dropped (fail closed).

        Returns False if no enabled drop_detection zone contains the point.
        """
        if not self._zone_polygons:
            return False

        # If any drop_detection zone exists and we're uncalibrated, fail closed.
        has_drop = any(z.policy == "drop_detection" for z, _ in self._zone_polygons)
        if has_drop and not self._homography_available and foot_point_norm is None:
            logger.warning(
                "privacy_zone_uncalibrated",
                camera_id=self._camera_id,
                msg="camera has drop_detection zones but no homography; failing closed",
            )
            return True

        if foot_point_norm is None:
            return False

        for z, poly in self._zone_polygons:
            if z.policy != "drop_detection":
                continue
            if _point_in_poly(foot_point_norm, poly):
                logger.debug(
                    "privacy_detection_dropped",
                    camera_id=self._camera_id,
                    zone_id=z.zone_id,
                    policy=z.policy,
                )
                return True

        return False

    def should_drop_bbox(self, bbox: BoundingBox) -> bool:
        """Check if a detection bounding box foot-point is inside a drop zone.

        Uses the bbox bottom-center as the foot point, normalised to [0,1].
        """
        if not self._zone_polygons:
            return False

        foot_x = bbox.center_x / max(self._frame_width, 1)
        foot_y = bbox.y_max / max(self._frame_height, 1)
        return self.should_drop((foot_x, foot_y))

    def apply_blur_mask(
        self, frame: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """Apply blur_region and mask_region policies to *frame* in place.

        Returns *frame* for chaining.
        """
        if not self._zone_polygons:
            return frame

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame

        for z, poly in self._zone_polygons:
            if z.policy not in ("blur_region", "mask_region"):
                continue
            # Convert normalised polygon to pixel coordinates.
            px_poly = np.array(
                [[int(p[0] * w), int(p[1] * h)] for p in poly.exterior.coords],
                dtype=np.int32,
            )
            if z.policy == "mask_region":
                cv2.fillPoly(frame, [px_poly], (114, 114, 114))
            elif z.policy == "blur_region":
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [px_poly], 255)
                blurred = cv2.GaussianBlur(frame, (51, 51), 0)
                frame[mask == 255] = blurred[mask == 255]

        return frame


def _point_in_poly(point: tuple[float, float], poly: Polygon) -> bool:
    """Test if normalized point is inside polygon."""
    return poly.contains(Point(point[0], point[1]))


# Deferred import to avoid circular dependency at module level.
import cv2  # noqa: E402
