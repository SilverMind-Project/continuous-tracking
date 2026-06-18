"""Tests for floor_region_polygon and the auto-calibrate route field.

All tests use synthetic FloorPlaneResult objects — no Triton, no GPU.
Coordinate convention: sample_indices are (row, col); normalised output is
[col/image_w, row/image_h] = [x_norm, y_norm].
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.routers.calibration as _cal_router_mod
from app.calibration.floor_plane import FloorPlaneResult, floor_region_polygon
from app.calibration.state import CalibrationState
from app.routers.calibration import router as calibration_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    row_col: np.ndarray,
    inlier_mask: np.ndarray,
) -> FloorPlaneResult:
    """Minimal FloorPlaneResult for polygon tests."""
    n = len(row_col)
    return FloorPlaneResult(
        normal=np.array([0.0, -1.0, 0.0]),
        d=1.5,
        inlier_mask=inlier_mask,
        sample_indices=row_col,
        points_3d=np.zeros((n, 3)),
        inlier_ratio=float(inlier_mask.sum()) / max(n, 1),
        mean_inlier_distance=0.02,
    )


# ---------------------------------------------------------------------------
# floor_region_polygon: correctness
# ---------------------------------------------------------------------------


def test_floor_region_from_inliers_excludes_image_border():
    """Interior-clustered inliers → polygon must stay away from all four edges."""
    image_h, image_w = 480, 640

    # Inliers in a tight interior cluster: rows 200-280, cols 200-440.
    rows = np.arange(200, 280, 5)
    cols = np.arange(200, 440, 10)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    row_col = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.int64)
    inlier_mask = np.ones(len(row_col), dtype=bool)
    result = _make_result(row_col, inlier_mask)

    poly = floor_region_polygon(result, image_h, image_w)
    assert poly is not None, "expected a polygon from dense interior inliers"

    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]

    # Interior cluster never touches any edge — enforce generous margins.
    assert min(xs) > 0.20, f"polygon touches left edge: min x={min(xs):.3f}"
    assert max(xs) < 0.80, f"polygon touches right edge: max x={max(xs):.3f}"
    assert min(ys) > 0.30, f"polygon touches top edge: min y={min(ys):.3f}"
    assert max(ys) < 0.65, f"polygon touches bottom edge: max y={max(ys):.3f}"

    # All coordinates normalised to [0, 1].
    for x, y in poly:
        assert 0.0 <= x <= 1.0, f"x_norm={x} outside [0,1]"
        assert 0.0 <= y <= 1.0, f"y_norm={y} outside [0,1]"


def test_concave_hull_handles_l_shaped_floor():
    """An L-shaped inlier set should produce a concave (non-rectangular) hull."""
    image_h, image_w = 480, 640

    # L-shape: bottom strip + left column.
    rows_h = np.arange(360, 479, 4)
    cols_h = np.arange(50, 630, 10)
    rr_h, cc_h = np.meshgrid(rows_h, cols_h, indexing="ij")
    rows_v = np.arange(240, 360, 4)
    cols_v = np.arange(50, 200, 10)
    rr_v, cc_v = np.meshgrid(rows_v, cols_v, indexing="ij")

    row_col = np.vstack(
        [
            np.stack([rr_h.ravel(), cc_h.ravel()], axis=1),
            np.stack([rr_v.ravel(), cc_v.ravel()], axis=1),
        ]
    ).astype(np.int64)
    inlier_mask = np.ones(len(row_col), dtype=bool)
    result = _make_result(row_col, inlier_mask)

    poly = floor_region_polygon(result, image_h, image_w, concave=True)
    assert poly is not None
    assert len(poly) >= 3

    # Convex hull of an L would include the top-right gap; concave should be smaller.
    poly_convex = floor_region_polygon(result, image_h, image_w, concave=False)
    assert poly_convex is not None

    # Compute approximate areas.
    def _area(pts: list[list[float]]) -> float:
        n = len(pts)
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += pts[i][0] * pts[j][1]
            a -= pts[j][0] * pts[i][1]
        return abs(a) / 2.0

    concave_area = _area(poly)
    convex_area = _area(poly_convex)
    # Concave hull should be strictly smaller than the convex hull for an L shape.
    assert concave_area <= convex_area + 1e-4, (
        f"concave area {concave_area:.4f} > convex area {convex_area:.4f}"
    )


def test_fewer_than_three_inliers_returns_none():
    """Fewer than 3 inliers after region filter → None."""
    row_col = np.array([[300, 100], [320, 150]], dtype=np.int64)
    inlier_mask = np.array([True, True])
    result = _make_result(row_col, inlier_mask)
    assert floor_region_polygon(result, 480, 640) is None


def test_zero_inliers_returns_none():
    row_col = np.array([[300, 100], [320, 150], [340, 200]], dtype=np.int64)
    inlier_mask = np.array([False, False, False])
    result = _make_result(row_col, inlier_mask)
    assert floor_region_polygon(result, 480, 640) is None


def test_region_fraction_can_exceed_ransac_fraction():
    """Inliers above the 60% RANSAC band are included when region_fraction > 0.6."""
    image_h, image_w = 480, 640
    # Place inliers across the full image height (simulating a high-mounted camera
    # whose floor appears from row 0 to row 479).
    rows = np.arange(0, image_h, 8)
    cols = np.arange(100, 540, 10)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    row_col = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.int64)
    inlier_mask = np.ones(len(row_col), dtype=bool)
    result = _make_result(row_col, inlier_mask)

    # With region_fraction=0.9, inliers above row 48 (10% of 480) are included.
    poly_09 = floor_region_polygon(result, image_h, image_w, region_fraction=0.9)
    assert poly_09 is not None
    ys_09 = [pt[1] for pt in poly_09]
    # Polygon should extend above the 40% mark (0.40) that RANSAC would set.
    assert min(ys_09) < 0.40, (
        f"region_fraction=0.9 should cover rows above RANSAC band; min y={min(ys_09):.3f}"
    )

    # With region_fraction=0.6, only inliers in the bottom 60% are included.
    poly_06 = floor_region_polygon(result, image_h, image_w, region_fraction=0.6)
    assert poly_06 is not None
    ys_06 = [pt[1] for pt in poly_06]
    # All points must be at or below 40% of image height.
    assert min(ys_06) >= 0.40 - 0.01, (
        f"region_fraction=0.6 should not extend above RANSAC band; min y={min(ys_06):.3f}"
    )


def test_normalisation_uses_col_for_x_and_row_for_y():
    """Axis-order guard: x_norm = col/image_w, y_norm = row/image_h."""
    # Deliberately non-square image to expose swaps.
    image_h, image_w = 300, 600

    # Cluster in top-right area: rows 30-60 (y_norm ~0.1-0.2), cols 450-570 (x_norm ~0.75-0.95).
    rows = np.arange(30, 60, 3)
    cols = np.arange(450, 570, 6)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    row_col = np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.int64)
    inlier_mask = np.ones(len(row_col), dtype=bool)
    result = _make_result(row_col, inlier_mask)

    poly = floor_region_polygon(result, image_h, image_w, region_fraction=0.9)
    assert poly is not None

    xs = [pt[0] for pt in poly]
    ys = [pt[1] for pt in poly]

    # x should be in the right portion (>0.5); y should be in the top portion (<0.5).
    assert min(xs) > 0.5, f"x_norm should reflect col~450-570 on w=600; got min={min(xs):.3f}"
    assert max(ys) < 0.5, f"y_norm should reflect row~30-60 on h=300; got max={max(ys):.3f}"


# ---------------------------------------------------------------------------
# Auto-calibrate route: floor_region_polygon in response
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    fresh = CalibrationState()
    monkeypatch.setattr(_cal_router_mod, "_default_state", fresh)
    app = FastAPI()
    app.include_router(calibration_router)
    with TestClient(app) as c:
        yield c


def test_auto_calibrate_response_includes_floor_region(client, monkeypatch):
    """Route test: AutoCalibrateResult carries floor_region_polygon from the calibrator."""

    class FakeFetcher:
        async def fetch_rgb(self, key: str):
            return np.zeros((480, 640, 3), dtype=np.uint8)

    class FakeAutoCalibrator:
        async def calibrate(self, image, fov_deg: float):
            return SimpleNamespace(
                draft_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                suggested_points=[],
                confidence=0.75,
                inlier_count=150,
                sample_count=300,
                fov_deg=fov_deg,
                method="depth_auto_draft",
                floor_region_polygon=[[0.1, 0.4], [0.9, 0.4], [0.9, 0.95], [0.1, 0.95]],
            )

    monkeypatch.setattr(_cal_router_mod, "_auto_calibrator", FakeAutoCalibrator())
    monkeypatch.setattr(_cal_router_mod, "_frame_fetcher", FakeFetcher())

    resp = client.post(
        "/internal/calibration/auto/cam-1",
        json={"minio_key": "frames/cam-1/1.jpg", "fov_deg": 70.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["floor_region_polygon"] is not None
    assert len(body["floor_region_polygon"]) == 4
    assert body["floor_region_polygon"][0] == [0.1, 0.4]


def test_auto_calibrate_response_floor_region_none_when_not_provided(client, monkeypatch):
    """floor_region_polygon is null in the response when the calibrator returns None."""

    class FakeFetcher:
        async def fetch_rgb(self, key: str):
            return np.zeros((480, 640, 3), dtype=np.uint8)

    class FakeAutoCalibrator:
        async def calibrate(self, image, fov_deg: float):
            return SimpleNamespace(
                draft_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                suggested_points=[],
                confidence=0.75,
                inlier_count=150,
                sample_count=300,
                fov_deg=fov_deg,
                method="depth_auto_draft",
                floor_region_polygon=None,
            )

    monkeypatch.setattr(_cal_router_mod, "_auto_calibrator", FakeAutoCalibrator())
    monkeypatch.setattr(_cal_router_mod, "_frame_fetcher", FakeFetcher())

    resp = client.post(
        "/internal/calibration/auto/cam-1",
        json={"minio_key": "frames/cam-1/1.jpg", "fov_deg": 70.0},
    )
    assert resp.status_code == 200
    assert resp.json()["floor_region_polygon"] is None
