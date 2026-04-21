"""Unit tests for the per-camera tracker (BoT-SORT-like association)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.domain import BoundingBox, Detection, FloorPoint
from app.inference.schemas import Embedding
from app.tracking.tracker import (
    PerCameraTracker,
    PerCameraTrackers,
    SimpleKalmanFilter,
    TrackerConfig,
    _embedding_distance,
    _iou_matrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    detection_id: str,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        camera_id="cam-1",
        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        embedding=[0.0] * 768,
        capture_time=datetime.now(UTC),
        event_time=datetime.now(UTC),
        confidence=confidence,
        floor_point=FloorPoint(0, 0),
    )


def _make_embedding(dim: int = 768) -> Embedding:
    return np.random.randn(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Kalman filter tests
# ---------------------------------------------------------------------------


class TestSimpleKalmanFilter:
    def test_initialize(self) -> None:
        kf = SimpleKalmanFilter()
        assert not kf.initialized
        obs = np.array([100.0, 100.0, 1.5, 50.0], dtype=np.float64)
        kf.initialize(obs)
        assert kf.initialized
        state = kf.state()
        assert state[0] == pytest.approx(100.0, abs=0.01)
        assert state[1] == pytest.approx(100.0, abs=0.01)

    def test_predict(self) -> None:
        kf = SimpleKalmanFilter()
        obs = np.array([100.0, 100.0, 1.5, 50.0], dtype=np.float64)
        kf.initialize(obs)
        pred = kf.predict()
        assert pred[0] == pytest.approx(100.0, abs=1.0)  # close to initial

    def test_update_after_initialize(self) -> None:
        kf = SimpleKalmanFilter()
        obs = np.array([100.0, 100.0, 1.5, 50.0], dtype=np.float64)
        kf.initialize(obs)
        new_obs = np.array([102.0, 101.0, 1.5, 51.0], dtype=np.float64)
        state = kf.update(new_obs)
        assert state[0] == pytest.approx(102.0, abs=2.0)

    def test_predict_before_init(self) -> None:
        kf = SimpleKalmanFilter()
        state = kf.predict()
        assert all(s == 0.0 for s in state)


# ---------------------------------------------------------------------------
# IoU matrix tests
# ---------------------------------------------------------------------------


class TestIoUMatrix:
    def test_identical_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        iou = _iou_matrix(boxes, boxes)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        a = np.array([[0.0, 0.0, 50.0, 50.0]], dtype=np.float64)
        b = np.array([[100.0, 100.0, 150.0, 150.0]], dtype=np.float64)
        iou = _iou_matrix(a, b)
        assert iou[0, 0] == pytest.approx(0.0)

    def test_overlapping_boxes(self) -> None:
        a = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        b = np.array([[50.0, 0.0, 150.0, 100.0]], dtype=np.float64)
        iou = _iou_matrix(a, b)
        # Overlap: 50x100 = 5000, Union: 10000 + 10000 - 5000 = 15000
        # IoU = 5000/15000 = 0.333
        assert iou[0, 0] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_empty_boxes_a(self) -> None:
        a: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        b = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        iou = _iou_matrix(a, b)
        assert iou.shape == (0, 1)

    def test_empty_boxes_b(self) -> None:
        a = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float64)
        b: np.ndarray = np.zeros((0, 4), dtype=np.float64)
        iou = _iou_matrix(a, b)
        assert iou.shape == (1, 0)


# ---------------------------------------------------------------------------
# Embedding distance tests
# ---------------------------------------------------------------------------


class TestEmbeddingDistance:
    def test_identical_embeddings(self) -> None:
        emb = np.random.randn(768).astype(np.float64)
        dist = _embedding_distance(emb.reshape(1, -1), emb.reshape(1, -1))
        assert dist[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_embeddings(self) -> None:
        emb = np.random.randn(768).astype(np.float64)
        neg_emb = -emb
        dist = _embedding_distance(emb.reshape(1, -1), neg_emb.reshape(1, -1))
        assert dist[0, 0] == pytest.approx(2.0, abs=1e-6)

    def test_orthogonal_embeddings(self) -> None:
        emb_a = np.zeros(768, dtype=np.float64)
        emb_a[0] = 1.0
        emb_b = np.zeros(768, dtype=np.float64)
        emb_b[1] = 1.0
        dist = _embedding_distance(emb_a.reshape(1, -1), emb_b.reshape(1, -1))
        assert dist[0, 0] == pytest.approx(1.0)

    def test_empty_embeddings(self) -> None:
        a: np.ndarray = np.zeros((0, 768), dtype=np.float64)
        b = np.random.randn(10, 768).astype(np.float64)
        dist = _embedding_distance(a, b)
        assert dist.shape == (0, 10)


# ---------------------------------------------------------------------------
# PerCameraTracker tests
# ---------------------------------------------------------------------------


class TestPerCameraTracker:
    def test_empty_frame(self) -> None:
        tracker = PerCameraTracker()
        tracks = tracker.update([], frame_index=0)
        assert tracks == []
        assert tracker.active_track_count == 0

    def test_single_detection_creates_track(self) -> None:
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5))
        det = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb = _make_embedding()
        tracks = tracker.update([det], [emb], frame_index=0)
        assert len(tracks) == 1
        assert tracks[0].local_track_id.startswith("track-")
        assert not tracks[0].confirmed  # needs min_hits=3

    def test_same_detection_over_frames_creates_confirmed_track(self) -> None:
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5, min_hits=3))
        det = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb = _make_embedding()

        # Frame 0
        tracks = tracker.update([det], [emb], frame_index=0)
        assert len(tracks) == 1
        assert not tracks[0].confirmed

        # Frame 1
        tracks = tracker.update([det], [emb], frame_index=1)
        assert len(tracks) == 1
        assert tracks[0].hit_count == 2

        # Frame 2: should be confirmed
        tracks = tracker.update([det], [emb], frame_index=2)
        assert len(tracks) == 1
        assert tracks[0].confirmed

    def test_low_confidence_detection_not_tracked(self) -> None:
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5))
        det = _make_detection("d1", 0, 0, 100, 100, confidence=0.2)
        tracks = tracker.update([det], [], frame_index=0)
        assert tracks == []

    def test_multiple_detections(self) -> None:
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5))
        d1 = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        d2 = _make_detection("d2", 200, 0, 300, 100, confidence=0.9)
        emb = _make_embedding()
        tracks = tracker.update([d1, d2], [emb, emb], frame_index=0)
        assert len(tracks) == 2

    def test_track_termination_after_lost_frames(self) -> None:
        config = TrackerConfig(new_track_thresh=0.5, max_time_lost=3, min_hits=1)
        tracker = PerCameraTracker(config)
        det = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb = _make_embedding()

        # Create track
        tracks = tracker.update([det], [emb], frame_index=0)
        assert len(tracks) == 1
        assert tracker.active_track_count == 1

        # No detections for max_time_lost frames
        for i in range(4):
            tracks = tracker.update([], [], frame_index=i + 1)
        assert tracker.active_track_count == 0

    def test_embedding_in_tracks(self) -> None:
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5))
        det = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb = _make_embedding()
        tracks = tracker.update([det], [emb], frame_index=0)
        assert len(tracks) == 1
        assert tracks[0].embedding is not None
        assert len(tracks[0].embedding) == 768


# ---------------------------------------------------------------------------
# PerCameraTrackers tests
# ---------------------------------------------------------------------------


class TestPerCameraTrackers:
    def test_isolated_cameras(self) -> None:
        registry = PerCameraTrackers(TrackerConfig(new_track_thresh=0.5))
        det = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb = _make_embedding()

        # Camera 1
        tracks_1 = registry.update("cam-1", [det], [emb], frame_index=0)
        assert len(tracks_1) == 1
        assert registry.get_active_count("cam-1") == 1

        # Camera 2 should be independent
        assert registry.get_active_count("cam-2") == 0

    def test_none_detections_no_error(self) -> None:
        registry = PerCameraTrackers()
        tracks = registry.update("cam-1", [], None, frame_index=0)
        assert tracks == []
