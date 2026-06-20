"""Tests for camera-drift detection (app/calibration/drift.py).

All tests use synthetic textured images (checkerboard + Gaussian noise) to
ensure ORB finds reliable features.  Flat/gradient images would yield too few
keypoints and trigger the insufficient_features guard, making tests vacuous.

Key assertions:
  - identical frames → drifted=False  (sanity)
  - geometrically shifted frame → drifted=True  (detection sensitivity)
  - brightness-only change → drifted=False  (lighting false-positive guard)
  - drift endpoint round-trips the score through the FastAPI route
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.routers.calibration as _cal_mod
from app.calibration.drift import DriftResult, drift_score
from app.calibration.state import CalibrationState
from app.routers.calibration import router as calibration_router

# ---------------------------------------------------------------------------
# Helpers: synthetic textured images
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _checkerboard(height: int = 240, width: int = 320, square: int = 20) -> np.ndarray:
    """Return a BGR uint8 checkerboard with Gaussian noise for ORB richness."""
    board = np.zeros((height, width), dtype=np.uint8)
    for r in range(0, height, square):
        for c in range(0, width, square):
            if ((r // square) + (c // square)) % 2 == 0:
                board[r : r + square, c : c + square] = 200
    noise = _RNG.integers(0, 30, size=board.shape, dtype=np.uint8)
    gray = np.clip(board.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _shifted(bgr: np.ndarray, dx: int = 0, dy: int = 0, angle_deg: float = 0.0) -> np.ndarray:
    """Apply translation + rotation to a BGR image."""
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    mat[0, 2] += dx
    mat[1, 2] += dy
    return cv2.warpAffine(bgr, mat, (w, h))


# ---------------------------------------------------------------------------
# Pure unit tests for drift_score
# ---------------------------------------------------------------------------


def test_identical_frames_not_drifted():
    """Same image as reference and current → not drifted, high inlier ratio."""
    img = _checkerboard()
    result = drift_score(img, img.copy())
    assert isinstance(result, DriftResult)
    assert not result.drifted, f"expected not drifted, got reason={result.reason!r}"
    assert result.inlier_ratio > 0.8, f"expected high inlier ratio, got {result.inlier_ratio:.3f}"
    assert result.ssim > 0.9, f"expected high SSIM on identical frames, got {result.ssim:.3f}"


def test_shifted_frame_flagged():
    """Rotating the reference by 5° must be detected as drift."""
    ref = _checkerboard()
    # 5° rotation — well above the 1.5° threshold.
    shifted = _shifted(ref, angle_deg=5.0)
    result = drift_score(ref, shifted)
    assert result.drifted, (
        f"expected drifted=True for 5° rotation, got reason={result.reason!r}, "
        f"inlier_ratio={result.inlier_ratio:.3f}"
    )


def test_lighting_change_not_flagged():
    """Brightness/contrast change with unchanged geometry → not drifted.

    This is the key false-positive guard: SSIM would flag this as structural
    difference, but ORB binary descriptors are intensity-invariant so the
    inlier ratio stays high.
    """
    ref = _checkerboard()
    # Simulate overexposure: multiply pixel values by 1.4, cap at 255.
    brighter = np.clip(ref.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
    result = drift_score(ref, brighter)
    assert not result.drifted, (
        f"expected not drifted for lighting change, got reason={result.reason!r}, "
        f"inlier_ratio={result.inlier_ratio:.3f}"
    )
    assert result.inlier_ratio > 0.5, (
        f"ORB should survive brightness change, got inlier_ratio={result.inlier_ratio:.3f}"
    )


def test_large_translation_flagged():
    """A 40-pixel translation must be detected as drift."""
    ref = _checkerboard()
    translated = _shifted(ref, dx=40, dy=0)
    result = drift_score(ref, translated)
    assert result.drifted, (
        f"expected drifted=True for 40px translation, got reason={result.reason!r}"
    )


# ---------------------------------------------------------------------------
# Router-level test for the drift endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def drift_client(monkeypatch):
    """Minimal app with mocked MinioFrameFetcher for the drift endpoint."""
    fresh_state = CalibrationState()
    monkeypatch.setattr(_cal_mod, "_default_state", fresh_state)

    ref_img = _checkerboard()
    cur_img = ref_img.copy()

    # fetch_rgb returns RGB; endpoint converts to BGR internally.
    ref_rgb = ref_img[:, :, ::-1].copy()
    cur_rgb = cur_img[:, :, ::-1].copy()

    mock_fetcher = MagicMock()
    mock_fetcher.fetch_rgb = AsyncMock(side_effect=[ref_rgb, cur_rgb])
    monkeypatch.setattr(_cal_mod, "_frame_fetcher", mock_fetcher)

    app = FastAPI()
    app.include_router(calibration_router)
    with TestClient(app) as c:
        yield c


def test_drift_endpoint_returns_score(drift_client: TestClient):
    """POST /internal/calibration/drift/{camera_id} returns a valid drift score."""
    resp = drift_client.post(
        "/internal/calibration/drift/cam-1",
        json={
            "reference_key": "calibration-refs/cam-1/ref.jpg",
            "current_key": "frames/cam-1/cur.jpg",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["camera_id"] == "cam-1"
    assert "inlier_ratio" in data
    assert "ssim" in data
    assert "drifted" in data
    assert "reason" in data
    assert not data["drifted"], f"identical frames should not be drifted, got {data}"


def test_drift_endpoint_503_without_fetcher(monkeypatch):
    """Returns 503 when MinIO fetcher is not wired."""
    monkeypatch.setattr(_cal_mod, "_frame_fetcher", None)
    fresh_state = CalibrationState()
    monkeypatch.setattr(_cal_mod, "_default_state", fresh_state)

    app = FastAPI()
    app.include_router(calibration_router)
    with TestClient(app) as c:
        resp = c.post(
            "/internal/calibration/drift/cam-1",
            json={"reference_key": "ref.jpg", "current_key": "cur.jpg"},
        )
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "calibration.minio_unavailable"
