"""Tests for PrivacyZoneFilter."""

from __future__ import annotations

import numpy as np
import pytest

from app.calibration.state import CalibrationState, PrivacyZoneConfig
from app.domain import BoundingBox
from app.pipeline.privacy import PrivacyZoneFilter


def _zone(
    zone_id: str,
    polygon: list[list[float]],
    policy: str = "drop_detection",
) -> PrivacyZoneConfig:
    return PrivacyZoneConfig(
        zone_id=zone_id,
        polygon=polygon,
        policy=policy,
        enabled=True,
    )


class TestPrivacyZoneFilter:
    def test_empty_no_active(self) -> None:
        pzf = PrivacyZoneFilter("cam-1", [], True, 640, 480)
        assert not pzf.is_active()

    def test_disabled_zone_not_active(self) -> None:
        zone = PrivacyZoneConfig(
            zone_id="z1",
            polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            policy="drop_detection",
            enabled=False,
        )
        pzf = PrivacyZoneFilter("cam-1", [zone], True, 640, 480)
        assert not pzf.is_active()

    def test_point_inside_dropped(self) -> None:
        zone = _zone("z1", [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        pzf = PrivacyZoneFilter("cam-1", [zone], True, 640, 480)
        # Point at (0.25, 0.25) is inside the polygon.
        assert pzf.should_drop((0.25, 0.25))
        # Point outside.
        assert not pzf.should_drop((0.75, 0.75))

    def test_point_outside_kept(self) -> None:
        zone = _zone("z1", [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        pzf = PrivacyZoneFilter("cam-1", [zone], True, 640, 480)
        assert not pzf.should_drop((0.75, 0.75))

    def test_uncalibrated_fails_closed(self) -> None:
        """When homography is unavailable and drop_detection zones exist, fail closed."""
        zone = _zone("z1", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        pzf = PrivacyZoneFilter("cam-1", [zone], homography_available=False, frame_width=640, frame_height=480)
        # Without foot-point info (None), uncalibrated + drop zones → return True
        assert pzf.should_drop(None)

    def test_calibrated_no_drop_without_foot_point(self) -> None:
        """With homography available, None foot point means no drop check."""
        zone = _zone("z1", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        pzf = PrivacyZoneFilter("cam-1", [zone], homography_available=True, frame_width=640, frame_height=480)
        assert not pzf.should_drop(None)

    def test_blur_region_no_drop(self) -> None:
        """Blur_region zones should not drop detections."""
        zone = _zone("z1", [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]], policy="blur_region")
        pzf = PrivacyZoneFilter("cam-1", [zone], True, 640, 480)
        assert not pzf.should_drop((0.25, 0.25))

    def test_mask_region_no_drop(self) -> None:
        """Mask_region zones should not drop detections."""
        zone = _zone("z1", [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]], policy="mask_region")
        pzf = PrivacyZoneFilter("cam-1", [zone], True, 640, 480)
        assert not pzf.should_drop((0.25, 0.25))

    def test_multiple_zones_mixed_policies(self) -> None:
        drop_zone = _zone("z-drop", [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]], policy="drop_detection")
        blur_zone = _zone("z-blur", [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0]], policy="blur_region")
        pzf = PrivacyZoneFilter("cam-1", [drop_zone, blur_zone], True, 640, 480)
        # Point clearly inside drop zone (triangle) → dropped.
        assert pzf.should_drop((0.2, 0.1))
        # Point inside blur zone only → not dropped.
        assert not pzf.should_drop((0.75, 0.75))

    def test_apply_blur_mask_noop_for_empty(self) -> None:
        pzf = PrivacyZoneFilter("cam-1", [], True, 640, 480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pzf.apply_blur_mask(frame)
        np.testing.assert_array_equal(result, frame)

    def test_apply_mask_region_fills(self) -> None:
        """Mask_region should fill the polygon area."""
        zone = _zone("z1", [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], policy="mask_region")
        pzf = PrivacyZoneFilter("cam-1", [zone], True, 640, 480)
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        result = pzf.apply_blur_mask(frame.copy())
        # Top-left quadrant should now be gray (114,114,114)
        center = result[120, 160]  # roughly center of the masked region
        assert center[0] == 114
        # Bottom-right should remain white
        br = result[360, 480]
        assert br[0] == 255

    def test_apply_blur_region_modifies(self) -> None:
        """Blur_region should apply Gaussian blur to the polygon area."""
        zone = _zone("z1", [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]], policy="blur_region")
        pzf = PrivacyZoneFilter("cam-1", [zone], True, 640, 480)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pzf.apply_blur_mask(frame.copy())
        # The blurred region should differ from the original
        top_left = result[:240, :320]
        orig_tl = frame[:240, :320]
        assert not np.array_equal(top_left, orig_tl)

    def test_should_drop_bbox(self) -> None:
        zone = _zone("z1", [[0.0, 0.8], [1.0, 0.8], [1.0, 1.0], [0.0, 1.0]])
        pzf = PrivacyZoneFilter("cam-1", [zone], True, 640, 480)
        # Bbox near the bottom of the frame (foot at y_max within zone).
        bbox = BoundingBox(x_min=100, y_min=300, x_max=200, y_max=430)
        assert pzf.should_drop_bbox(bbox)
        # Bbox near the top.
        bbox_top = BoundingBox(x_min=100, y_min=10, x_max=200, y_max=100)
        assert not pzf.should_drop_bbox(bbox_top)
