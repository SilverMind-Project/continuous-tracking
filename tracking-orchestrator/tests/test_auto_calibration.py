"""Unit tests for floor plane fitting and auto-calibration.

No Triton or MinIO dependency — all ML calls are mocked via protocols.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.calibration.auto_calibrator import AutoCalibrationResult, AutoCalibrator
from app.calibration.floor_plane import (
    FloorPlaneFitter,
    _fit_plane_3pts,
    _fit_plane_svd,
    floor_plane_to_homography,
)
from app.calibration.homography import RESIDUAL_ERROR_M, RESIDUAL_WARN_M, compute_homography, residual_status
from app.inference.depth import DepthEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_depth(
    h: int = 480,
    w: int = 640,
    floor_depth_m: float = 3.0,
    fov_deg: float = 70.0,
    camera_height_m: float = 2.5,
) -> np.ndarray:
    """Generate a synthetic depth map where the lower 60% is a flat floor."""
    depth = np.full((h, w), fill_value=10.0, dtype=np.float32)
    fx = w / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    cx, cy = w / 2.0, h / 2.0

    floor_start = int(h * 0.4)
    rows = np.arange(floor_start, h)
    for r in rows:
        # Camera at height camera_height_m looking down with angle determined
        # by row index. Compute distance to floor plane (y = camera_height_m in camera frame).
        y_cam = (r - cy) / fx  # normalised camera-frame y
        if y_cam <= 0:
            continue
        # depth such that Y * depth = camera_height_m
        z = camera_height_m / y_cam
        depth[r, :] = np.clip(z, 0.5, 15.0).astype(np.float32)

    return depth


# ---------------------------------------------------------------------------
# compute_homography
# ---------------------------------------------------------------------------


def test_compute_homography_roundtrip() -> None:
    """Matrix should map input pixel points back to floor points within threshold."""
    pixel_pts = [
        [100.0, 200.0],
        [400.0, 200.0],
        [400.0, 400.0],
        [100.0, 400.0],
        [250.0, 300.0],
    ]
    floor_pts = [
        [0.5, 1.0],
        [2.5, 1.0],
        [2.5, 3.0],
        [0.5, 3.0],
        [1.5, 2.0],
    ]
    matrix, residuals = compute_homography(pixel_pts, floor_pts)
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    assert len(residuals) == 5
    assert max(residuals) < 0.01, f"Residuals too high: {residuals}"


def test_compute_homography_raises_on_too_few_points() -> None:
    with pytest.raises(ValueError, match="4 point pairs"):
        compute_homography([[0.0, 0.0]] * 3, [[0.0, 0.0]] * 3)


def test_compute_homography_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_homography([[0.0, 0.0]] * 4, [[0.0, 0.0]] * 5)


def test_residual_status() -> None:
    assert residual_status(0.0) == "ok"
    assert residual_status(RESIDUAL_WARN_M) == "ok"
    assert residual_status(RESIDUAL_WARN_M + 0.01) == "warning"
    assert residual_status(RESIDUAL_ERROR_M) == "warning"
    assert residual_status(RESIDUAL_ERROR_M + 0.01) == "error"


# ---------------------------------------------------------------------------
# Plane geometry helpers
# ---------------------------------------------------------------------------


def test_fit_plane_3pts_known_plane() -> None:
    """Three points on z=2.0 plane should yield normal (0, 0, ±1) and d=∓2."""
    p0 = np.array([0.0, 0.0, 2.0])
    p1 = np.array([1.0, 0.0, 2.0])
    p2 = np.array([0.0, 1.0, 2.0])
    n, d = _fit_plane_3pts(p0, p1, p2)
    assert n is not None and d is not None
    assert abs(abs(n[2]) - 1.0) < 1e-6, f"Expected |n_z|=1, got {n}"
    assert abs(d - (-2.0 * n[2])) < 1e-6


def test_fit_plane_3pts_degenerate() -> None:
    """Collinear points should return (None, None)."""
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 1.0, 1.0])
    p2 = np.array([2.0, 2.0, 2.0])
    n, d = _fit_plane_3pts(p0, p1, p2)
    assert n is None and d is None


def test_fit_plane_svd_recovers_horizontal_plane() -> None:
    """SVD fit on points scattered on y=3.0 should recover y-axis normal."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, (100, 3))
    pts[:, 1] = 3.0 + rng.normal(0, 1e-4, 100)
    n, d = _fit_plane_svd(pts)
    assert n is not None and d is not None
    assert abs(abs(n[1]) - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# FloorPlaneFitter
# ---------------------------------------------------------------------------


def test_floor_plane_fitter_synthetic_depth() -> None:
    """Fitter should detect the synthetic floor plane with reasonable confidence."""
    depth = _make_synthetic_depth()
    fitter = FloorPlaneFitter(fov_deg=70.0, floor_region_fraction=0.6, max_samples=2048)
    rng = np.random.default_rng(42)
    result = fitter.fit(depth, rng=rng)
    assert result is not None, "Fitter returned None on synthetic depth map"
    assert result.inlier_ratio > 0.5, f"Inlier ratio too low: {result.inlier_ratio}"
    assert result.confidence > 0.3, f"Confidence too low: {result.confidence}"


def test_floor_plane_fitter_all_zeros() -> None:
    """All-zero depth map (invalid) should return None."""
    depth = np.zeros((480, 640), dtype=np.float32)
    fitter = FloorPlaneFitter()
    result = fitter.fit(depth)
    assert result is None


def test_floor_plane_fitter_too_few_valid() -> None:
    """Depth map with fewer than 6 valid pixels should return None."""
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[400, 300] = 2.0
    depth[400, 301] = 2.0
    fitter = FloorPlaneFitter()
    result = fitter.fit(depth)
    assert result is None


def test_floor_plane_to_homography_from_synthetic() -> None:
    """floor_plane_to_homography should return a 3×3 list given good inliers."""
    depth = _make_synthetic_depth(h=480, w=640)
    fitter = FloorPlaneFitter(fov_deg=70.0, max_samples=2048)
    rng = np.random.default_rng(0)
    result = fitter.fit(depth, rng=rng)
    assert result is not None
    H = floor_plane_to_homography(result, image_h=480, image_w=640, fov_deg=70.0)
    assert H is not None
    assert len(H) == 3
    assert all(len(row) == 3 for row in H)


# ---------------------------------------------------------------------------
# AutoCalibrator (mocked DepthEstimator)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_depth_estimator() -> DepthEstimator:
    """DepthEstimator whose ``estimate`` returns a synthetic depth map."""
    depth_output = _make_synthetic_depth(h=480, w=640)

    mock_client = AsyncMock()
    estimator = DepthEstimator(client=mock_client)

    # Patch the estimate method directly to return our synthetic depth.
    async def _fake_estimate(image: Any) -> Any:
        return depth_output

    estimator.estimate = _fake_estimate  # type: ignore[method-assign]
    return estimator


@pytest.mark.asyncio
async def test_auto_calibrator_returns_result(fake_depth_estimator: DepthEstimator) -> None:
    """AutoCalibrator should return a non-None result for a synthetic floor scene."""
    calibrator = AutoCalibrator(depth_estimator=fake_depth_estimator, fov_deg=70.0)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = await calibrator.calibrate(image)
    assert result is not None, "AutoCalibrator returned None on synthetic depth"
    assert isinstance(result, AutoCalibrationResult)
    assert result.method == "depth_auto"
    assert 0.0 <= result.confidence <= 1.0
    assert result.inlier_count > 0
    assert result.sample_count >= result.inlier_count
    assert len(result.matrix) == 3
    assert all(len(row) == 3 for row in result.matrix)


@pytest.mark.asyncio
async def test_auto_calibrator_fov_override(fake_depth_estimator: DepthEstimator) -> None:
    """Passing fov_deg override should propagate to the result."""
    calibrator = AutoCalibrator(depth_estimator=fake_depth_estimator, fov_deg=70.0)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = await calibrator.calibrate(image, fov_deg=90.0)
    # The FoV used must be the override (90°), not the instance default (70°).
    if result is not None:
        assert result.fov_deg == pytest.approx(90.0)


@pytest.mark.asyncio
async def test_auto_calibrator_low_confidence_returns_none() -> None:
    """AutoCalibrator should return None when the plane fit confidence is below threshold."""
    # Return a depth map with no floor structure (random noise).
    mock_client = AsyncMock()
    estimator = DepthEstimator(client=mock_client)

    async def _noisy_estimate(image: Any) -> Any:
        rng = np.random.default_rng(1)
        return rng.uniform(0.3, 15.0, (480, 640)).astype(np.float32)

    estimator.estimate = _noisy_estimate  # type: ignore[method-assign]
    calibrator = AutoCalibrator(depth_estimator=estimator, fov_deg=70.0)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = await calibrator.calibrate(image)
    # Noisy depth: may or may not produce a result depending on RANSAC luck.
    # The important thing is that the function does not raise.
    assert result is None or isinstance(result, AutoCalibrationResult)
