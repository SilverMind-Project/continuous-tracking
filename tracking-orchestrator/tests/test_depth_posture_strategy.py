"""Unit tests for DepthPostureStrategy._classify_from_depth."""

from __future__ import annotations

import numpy as np

from app.domain import BoundingBox, Detection
from app.trajectory.depth_posture_strategy import DepthPostureStrategy


def _make_detection(bbox: tuple[int, int, int, int]) -> Detection:
    return Detection(
        detection_id="det-1",
        camera_id="cam-1",
        bbox=BoundingBox(x_min=bbox[0], y_min=bbox[1], x_max=bbox[2], y_max=bbox[3]),
        embedding=[],
        capture_time=None,  # type: ignore[arg-type]
        event_time=None,  # type: ignore[arg-type]
    )


def test_classify_vertical_blob_is_standing():
    """Tall narrow depth blob -> standing."""
    strategy = DepthPostureStrategy(depth_estimator=None)  # type: ignore[arg-type]
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[140:340, 300:340] = 1.5
    depth_map[140:340, 300:340] += np.random.normal(0, 0.1, (200, 40)).astype(np.float32)
    detection = _make_detection(bbox=(300, 140, 340, 340))
    result = strategy._classify_from_depth(depth_map, detection)
    assert result == "standing"


def test_classify_horizontal_blob_is_lying():
    """Wide flat depth blob -> lying."""
    strategy = DepthPostureStrategy(depth_estimator=None)  # type: ignore[arg-type]
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[230:270, 100:500] = 0.8
    depth_map[230:270, 100:500] += np.random.normal(0, 0.1, (40, 400)).astype(np.float32)
    detection = _make_detection(bbox=(100, 230, 500, 270))
    result = strategy._classify_from_depth(depth_map, detection)
    assert result == "lying"


def test_uniform_depth_returns_unknown():
    """Uniform depth blob -> unknown (person may not be present)."""
    strategy = DepthPostureStrategy(depth_estimator=None)  # type: ignore[arg-type]
    depth_map = np.full((480, 640), 2.0, dtype=np.float32)
    detection = _make_detection(bbox=(100, 100, 200, 300))
    result = strategy._classify_from_depth(depth_map, detection)
    assert result == "unknown"


def test_empty_bbox_returns_unknown():
    """Zero-area bbox -> unknown."""
    strategy = DepthPostureStrategy(depth_estimator=None)  # type: ignore[arg-type]
    depth_map = np.zeros((480, 640), dtype=np.float32)
    detection = _make_detection(bbox=(300, 300, 300, 300))
    result = strategy._classify_from_depth(depth_map, detection)
    assert result == "unknown"


def test_out_of_bounds_bbox_clamped():
    """Bbox extending beyond depth map is clamped."""
    strategy = DepthPostureStrategy(depth_estimator=None)  # type: ignore[arg-type]
    depth_map = np.zeros((480, 640), dtype=np.float32)
    depth_map[0:200, 0:200] = 1.0
    depth_map[0:200, 0:200] += np.random.normal(0, 0.1, (200, 200)).astype(np.float32)
    detection = _make_detection(bbox=(-10, -10, 100, 100))
    result = strategy._classify_from_depth(depth_map, detection)
    assert result in ("standing", "sitting", "lying", "unknown")
