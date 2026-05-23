"""Tests for SpatialProjectionService."""

from __future__ import annotations

import math

import pytest

from app.calibration.state import CalibrationState
from app.domain import BoundingBox, FloorPoint
from app.tracking.spatial_projection import SpatialProjectionService


def _identity_homography() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


async def _set_up_calibrated(cal_state, camera_id, floor_plan_id="fp-1"):
    await cal_state.set_homography(
        camera_id=camera_id,
        matrix=_identity_homography(),
        floor_plan_id=floor_plan_id,
        image_width=1920,
        image_height=1080,
        max_residual_m=0.1,
        mean_residual_m=0.05,
        quality_status="ok",
        quality_point_count=8,
    )


class TestSpatialProjectionService:
    def test_identity_homography_produces_expected_coords(self) -> None:
        """Identity homography: pixel footpoint maps to metres, then to mm."""
        state = CalibrationState()
        state.homographies["cam-1"] = _identity_homography()
        svc = SpatialProjectionService(state)

        # Footpoint at pixel (960, 1000) with identity homography
        # maps to floor-plane metres (0.960, 1.000), i.e. (960 mm, 1000 mm).
        bbox = BoundingBox(x_min=900, y_min=800, x_max=1020, y_max=1000)
        fp = svc.project_detection("cam-1", bbox)

        assert fp.calibrated
        foot_x_px = (900 + 1020) / 2.0  # = 960.0
        foot_y_px = 1000.0
        assert fp.x_mm == round(foot_x_px * 1000.0)
        assert fp.y_mm == round(foot_y_px * 1000.0)

    def test_missing_calibration_returns_uncalibrated(self) -> None:
        """No homography stored → FloorPoint(0, 0, calibrated=False)."""
        state = CalibrationState()
        svc = SpatialProjectionService(state)
        bbox = BoundingBox(x_min=100, y_min=200, x_max=300, y_max=400)
        fp = svc.project_detection("cam-unknown", bbox)

        assert not fp.calibrated
        assert fp.x_mm == 0
        assert fp.y_mm == 0

    @pytest.mark.asyncio
    async def test_can_compare_requires_shared_floor_plan(self) -> None:
        """Two cameras with the same floor_plan_id can be compared."""
        state = CalibrationState()
        await _set_up_calibrated(state, "cam-a", floor_plan_id="fp-1")
        await _set_up_calibrated(state, "cam-b", floor_plan_id="fp-1")
        svc = SpatialProjectionService(state)

        assert svc.can_compare("cam-a", "cam-b")

    @pytest.mark.asyncio
    async def test_floor_plan_mismatch_blocks_comparison(self) -> None:
        """Different floor plans → can_compare returns False."""
        state = CalibrationState()
        await _set_up_calibrated(state, "cam-a", floor_plan_id="fp-1")
        await _set_up_calibrated(state, "cam-b", floor_plan_id="fp-2")
        svc = SpatialProjectionService(state)

        assert not svc.can_compare("cam-a", "cam-b")

    @pytest.mark.asyncio
    async def test_can_compare_false_when_one_uncalibrated(self) -> None:
        """One camera uncalibrated → can_compare returns False."""
        state = CalibrationState()
        await _set_up_calibrated(state, "cam-a", floor_plan_id="fp-1")
        # cam-b has no calibration
        svc = SpatialProjectionService(state)

        assert not svc.can_compare("cam-a", "cam-b")

    def test_non_finite_projection_is_rejected(self) -> None:
        """Degenerate w ≈ 0 → returns uncalibrated FloorPoint."""
        # A homography that produces w=0 for the footpoint.
        zero_w_homography = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        state = CalibrationState()
        state.homographies["cam-1"] = zero_w_homography
        svc = SpatialProjectionService(state)

        bbox = BoundingBox(x_min=100, y_min=200, x_max=300, y_max=400)
        fp = svc.project_detection("cam-1", bbox)

        assert not fp.calibrated

    def test_distance_m_returns_none_for_uncalibrated(self) -> None:
        """distance_m returns None when either point is uncalibrated."""
        a = FloorPoint(x_mm=1000, y_mm=2000, calibrated=True)
        b = FloorPoint(x_mm=0, y_mm=0, calibrated=False)

        assert SpatialProjectionService.distance_m(a, b) is None
        assert SpatialProjectionService.distance_m(b, a) is None

    def test_distance_m_computes_correctly(self) -> None:
        """distance_m computes Euclidean distance in metres."""
        a = FloorPoint(x_mm=0, y_mm=0, calibrated=True)
        b = FloorPoint(x_mm=3000, y_mm=4000, calibrated=True)  # 3 m, 4 m

        dist = SpatialProjectionService.distance_m(a, b)
        assert dist is not None
        assert math.isclose(dist, 5.0)  # 3-4-5 triangle in metres

    @pytest.mark.asyncio
    async def test_floor_plan_id_for_returns_none_when_uncalibrated(self) -> None:
        """floor_plan_id_for returns None for uncalibrated cameras."""
        state = CalibrationState()
        svc = SpatialProjectionService(state)

        assert svc.floor_plan_id_for("unknown") is None

    @pytest.mark.asyncio
    async def test_floor_plan_id_for_returns_id_when_calibrated(self) -> None:
        """floor_plan_id_for returns the calibration's floor_plan_id."""
        state = CalibrationState()
        await _set_up_calibrated(state, "cam-1", floor_plan_id="fp-main")
        svc = SpatialProjectionService(state)

        assert svc.floor_plan_id_for("cam-1") == "fp-main"
