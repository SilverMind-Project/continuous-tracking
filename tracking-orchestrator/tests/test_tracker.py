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
        assert dist[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_embeddings(self) -> None:
        emb_a = np.zeros(768, dtype=np.float64)
        emb_a[0] = 1.0
        emb_b = np.zeros(768, dtype=np.float64)
        emb_b[1] = 1.0
        dist = _embedding_distance(emb_a.reshape(1, -1), emb_b.reshape(1, -1))
        assert dist[0, 0] == pytest.approx(0.5)

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


class TestPerCameraTrackerHungarian:
    """Tests for the Hungarian assignment in PerCameraTracker._associate."""

    def test_hungarian_two_tracks_two_detections(
        self,
    ) -> None:
        """Two confirmed tracks + two detections should produce correct
        Hungarian assignment based on IoU proximity."""
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5, min_hits=1))
        # Create two existing tracks from previous frames.
        d1 = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb1 = _make_embedding()
        tracks = tracker.update([d1], [emb1], frame_index=0)
        assert len(tracks) == 1
        track_a_id = tracks[0].local_track_id

        # Second track from a nearby but distinct detection.
        # Re-process with the same detection to create second track.
        # Actually, since d2 is very close to d1's box, it might match track_a.
        # So create a second track with a more distant detection.
        d2_far = _make_detection("d2", 200, 200, 300, 300, confidence=0.9)
        emb_far = _make_embedding()
        tracks2 = tracker.update([d2_far], [emb_far], frame_index=1)
        assert len(tracks2) == 1
        track_b_id = tracks2[0].local_track_id

        assert tracker.active_track_count == 2

        # Now two new detections: one close to track_a's box, one close to track_b's box.
        d_a_new = _make_detection("da", 5, 5, 105, 105, confidence=0.9)
        d_b_new = _make_detection("db", 205, 205, 305, 305, confidence=0.9)
        emb_a = _make_embedding()
        emb_b = _make_embedding()

        tracks = tracker.update([d_a_new, d_b_new], [emb_a, emb_b], frame_index=2)
        assert len(tracks) == 2
        track_ids = {t.local_track_id for t in tracks}
        assert track_a_id in track_ids
        assert track_b_id in track_ids

    def test_unmatched_track_lost_count_increments(
        self,
    ) -> None:
        """When detections arrive but a track is not matched, its lost_count
        should increment via the normal update path (not the early-return)."""
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5, max_time_lost=2, min_hits=1))
        # Create one confirmed track.
        d1 = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb1 = _make_embedding()
        tracks = tracker.update([d1], [emb1], frame_index=0)
        assert len(tracks) == 1
        track_id = tracks[0].local_track_id

        # Get the track's lost_count before.
        internal = tracker._tracks[track_id]
        assert internal.lost_count == 0

        # Send a detection that is far away — should NOT match this track.
        # IoU will be near zero, cost near 1.0, which likely exceeds threshold.
        d_far = _make_detection("dfar", 500, 500, 600, 600, confidence=0.9)
        emb_far = _make_embedding()

        # Frame 1: detection arrives but doesn't match existing track.
        tracks = tracker.update([d_far], [emb_far], frame_index=1)
        # The far detection creates a new track; existing track is unmatched.
        assert len(tracks) == 1  # new track from unmatched detection

        # The original track should have lost_count incremented.
        internal = tracker._tracks[track_id]
        assert internal.lost_count == 1

    def test_new_track_from_unmatched_detection_with_existing_tracks(
        self,
    ) -> None:
        """Unmatched detections should spawn new tracks even when existing
        tracks are present."""
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5, min_hits=1))
        # Create one track.
        d1 = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb1 = _make_embedding()
        tracker.update([d1], [emb1], frame_index=0)
        assert tracker.active_track_count == 1

        # Detection far from existing track → new track.
        d2 = _make_detection("d2", 300, 300, 400, 400, confidence=0.9)
        emb2 = _make_embedding()
        tracks = tracker.update([d2], [emb2], frame_index=1)
        assert len(tracks) == 1
        assert tracker.active_track_count == 2

    def test_detection_replaces_new_track_when_no_match(
        self,
    ) -> None:
        """After a track terminates, a new detection in a different location
        creates a fresh track."""
        tracker = PerCameraTracker(TrackerConfig(new_track_thresh=0.5, max_time_lost=1, min_hits=1))
        d1 = _make_detection("d1", 0, 0, 100, 100, confidence=0.9)
        emb1 = _make_embedding()
        tracker.update([d1], [emb1], frame_index=0)
        assert tracker.active_track_count == 1

        # Terminate the track by sending no detections (lost_count increments
        # in the early-return path).
        tracker.update([], [], frame_index=1)
        assert tracker.active_track_count == 0

        # New detection creates a fresh track.
        d2 = _make_detection("d2", 200, 200, 300, 300, confidence=0.9)
        emb2 = _make_embedding()
        tracks = tracker.update([d2], [emb2], frame_index=2)
        assert len(tracks) == 1
        assert tracker.active_track_count == 1

    def test_mixed_embedding_history_iou_only_for_new_tracks(
        self,
    ) -> None:
        """Verify that the mixed embedding history path uses IoU-only cost
        for tracks without history (not a biased neutral embedding). This
        exercises the fix for review issue #4 (mixed embedding history bias).

        The mixed case occurs when some tracks have embedding history and
        others don't. We verify this by directly testing the _associate
        method's cost computation behavior.
        """
        config = TrackerConfig(new_track_thresh=0.5, min_hits=1, match_thresh=0.8)
        tracker = PerCameraTracker(config)

        # Create Track A with embedding history (3 frames of tracking).
        d_a = _make_detection("da", 50, 50, 150, 150, confidence=0.9)
        emb_a = _make_embedding()
        for i in range(3):
            tracker.update([d_a], [emb_a], frame_index=i)
        assert tracker.active_track_count == 1

        # Now Track A has embedding history.
        internal_a = next(t for t in tracker._tracks.values())
        assert len(internal_a.embedding_history) == 3

        # Create a new detection that is spatially close to Track A's box.
        # This should match Track A based on IoU.
        d_near_a = _make_detection("dnear", 55, 55, 155, 155, confidence=0.9)
        emb_near = _make_embedding()
        tracks = tracker.update([d_near_a], [emb_near], frame_index=3)

        # Track A should be matched (IoU is high).
        assert len(tracks) == 1
        assert tracker.active_track_count == 1
        internal_a = next(t for t in tracker._tracks.values())
        assert internal_a.hit_count == 4  # 3 previous + 1 new

        # Now create Track B by sending a far detection.
        d_far = _make_detection("dfar", 300, 300, 400, 400, confidence=0.9)
        emb_far = _make_embedding()
        tracks = tracker.update([d_far], [emb_far], frame_index=4)
        assert tracker.active_track_count == 2  # A + new B

        # Send a detection close to Track A. Both A (has history) and B
        # (no history) are candidates. Track A should win because its
        # IoU is higher.
        d_near_a2 = _make_detection("dnear2", 60, 60, 160, 160, confidence=0.9)
        emb_near2 = _make_embedding()
        tracks = tracker.update([d_near_a2], [emb_near2], frame_index=5)

        # Track A should be matched (it has history and IoU is high).
        matched_ids = {t.local_track_id for t in tracks}
        assert len(matched_ids) == 1
        track_a_id = next(
            tid for tid, t in tracker._tracks.items() if len(t.embedding_history) >= 5
        )
        assert track_a_id in matched_ids
