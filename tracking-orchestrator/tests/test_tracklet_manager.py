"""Unit tests for the TrackletManager."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest

from app.domain import (
    BoundingBox,
    CameraConfig,
    Detection,
    FloorPoint,
    GalleryEmbedding,
    TrackingEvent,
    Tracklet,
)
from app.inference.schemas import Embedding
from app.storage.base import InMemoryGalleryRepository, InMemoryTrackingRepository
from app.tracking.tracklet_manager import TrackletConfig, TrackletManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detection(
    detection_id: str,
    x_min: int = 0,
    y_min: int = 0,
    x_max: int = 100,
    y_max: int = 100,
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


@dataclass
class _MockLocalTrack:
    local_track_id: str
    detection: Detection
    bbox: BoundingBox
    confidence: float
    age: int
    hit_count: int
    lost_count: int
    confirmed: bool
    embedding: list[float] | None = None


def _make_local_track(
    local_track_id: str,
    detection: Detection,
    confirmed: bool = True,
    hit_count: int = 3,
    lost_count: int = 0,
    age: int = 3,
) -> _MockLocalTrack:
    return _MockLocalTrack(
        local_track_id=local_track_id,
        detection=detection,
        bbox=detection.bbox,
        confidence=detection.confidence,
        age=age,
        hit_count=hit_count,
        lost_count=lost_count,
        confirmed=confirmed,
    )


# ---------------------------------------------------------------------------
# TrackletManager tests
# ---------------------------------------------------------------------------


class TestTrackletManager:
    @pytest.fixture
    def manager(self) -> TrackletManager:
        repo = InMemoryTrackingRepository()
        gallery = InMemoryGalleryRepository()
        # Disable stability gate so existing tests don't have to call step() N times.
        return TrackletManager(repo, gallery, TrackletConfig(min_frames_to_publish=0))

    @pytest.fixture
    def camera(self) -> CameraConfig:
        return CameraConfig(camera_id="cam-1", name="Test Camera")

    @pytest.fixture
    def event_time(self) -> datetime:
        return datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)

    async def test_empty_frame_produces_nothing(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        tracklets, entries, events = await manager.step(
            camera=camera,
            local_tracks=[],
            detections=[],
            embeddings=[],
            event_time=event_time,
            frame_index=0,
        )
        assert tracklets == []
        assert entries == []
        assert events == []

    async def test_confirmed_creates_tracklet(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        det = _make_detection("d1", confidence=0.9)
        lt = _make_local_track("lt-1", det, confirmed=True)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        tracklets, _entries, _events = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )

        assert len(tracklets) == 1
        tracklet = tracklets[0]
        assert isinstance(tracklet, Tracklet)
        assert tracklet.camera_id == "cam-1"
        assert tracklet.state == "active"
        assert len(tracklet.detection_ids) == 1
        assert tracklet.detection_ids[0] == "d1"

    async def test_unconfirmed_not_created(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        det = _make_detection("d1", confidence=0.9)
        lt = _make_local_track(
            "lt-1",
            det,
            confirmed=False,
            hit_count=1,
            age=2,
        )

        tracklets, _, _ = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[],
            event_time=event_time,
            frame_index=0,
        )
        assert tracklets == []

    async def test_persisted_to_repo(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        det = _make_detection("d1", confidence=0.9)
        lt = _make_local_track("lt-1", det, confirmed=True)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        tracklets, _, _ = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )

        persisted = await manager._repo.get_tracklet(tracklets[0].tracklet_id)
        assert persisted is not None
        assert persisted.tracklet_id == tracklets[0].tracklet_id

    async def test_event_produced(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        det = _make_detection("d1", confidence=0.9)
        lt = _make_local_track("lt-1", det, confirmed=True)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        _, _, events = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, TrackingEvent)
        assert event.camera_id == "cam-1"
        assert event.frame_index == 0
        assert event.detections == [det]

    async def test_gallery_entry_for_high_quality(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        det = _make_detection(
            "d1",
            x_min=0,
            y_min=0,
            x_max=500,
            y_max=500,
            confidence=0.95,
        )
        lt = _make_local_track("lt-1", det, confirmed=True)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        _tracklets, gallery_entries, _ = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )

        assert len(gallery_entries) >= 1
        entry = gallery_entries[0]
        assert isinstance(entry, GalleryEmbedding)
        assert entry.quality > 0

    async def test_disabled_produces_nothing(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        manager._config = TrackletConfig(enabled=False)
        det = _make_detection("d1", confidence=0.9)
        lt = _make_local_track("lt-1", det, confirmed=True)

        tracklets, entries, events = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[],
            event_time=event_time,
            frame_index=0,
        )
        assert tracklets == []
        assert entries == []
        assert events == []

    async def test_get_active_tracklets(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        det = _make_detection("d1", confidence=0.9)
        lt = _make_local_track("lt-1", det, confirmed=True)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        tracklets, _, _ = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )

        active = manager.get_active_tracklets()
        assert len(active) == 1
        assert active[0].tracklet_id == tracklets[0].tracklet_id

    async def test_multiple_local_tracks(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        d1 = _make_detection(
            "d1",
            x_min=0,
            y_min=0,
            x_max=100,
            y_max=100,
            confidence=0.9,
        )
        d2 = _make_detection(
            "d2",
            x_min=200,
            y_min=0,
            x_max=300,
            y_max=100,
            confidence=0.85,
        )
        lt1 = _make_local_track("lt-1", d1, confirmed=True)
        lt2 = _make_local_track("lt-2", d2, confirmed=True)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        tracklets, _, events = await manager.step(
            camera=camera,
            local_tracks=[lt1, lt2],
            detections=[d1, d2],
            embeddings=[emb, emb],
            event_time=event_time,
            frame_index=0,
        )

        assert len(tracklets) == 2
        assert len(events) == 1
        assert events[0].detections == [d1, d2]

    async def test_tracklet_extension(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        """A confirmed local track across multiple frames should extend the
        same tracklet, growing its detection_ids."""
        det1 = _make_detection("d1", confidence=0.9)
        lt = _make_local_track("lt-1", det1, confirmed=True)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        # Frame 0: creates tracklet.
        tracklets_0, _, _ = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det1],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )
        assert len(tracklets_0) == 1
        tracklet_id = tracklets_0[0].tracklet_id
        assert len(tracklets_0[0].detection_ids) == 1

        # Frame 1: same local track, new detection → extension.
        det2 = _make_detection("d2", confidence=0.9)
        # The local track must have the same local_track_id to be recognized as
        # the same tracklet, but the detection changes.
        lt2 = _make_local_track("lt-1", det2, confirmed=True)

        tracklets_1, _, _ = await manager.step(
            camera=camera,
            local_tracks=[lt2],
            detections=[det2],
            embeddings=[emb],
            event_time=event_time,
            frame_index=1,
        )
        assert len(tracklets_1) == 1
        assert tracklets_1[0].tracklet_id == tracklet_id
        assert len(tracklets_1[0].detection_ids) == 2

    async def test_tracklet_closure_after_lost_frames(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        """A tracklet that is no longer alive should close after the grace
        window and return a closed-tracklet update."""
        det = _make_detection("d1", confidence=0.9)
        lt = _make_local_track("lt-1", det, confirmed=True)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        # Create tracklet.
        tracklets, _, _ = await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )
        assert len(tracklets) == 1
        tracklet_id = tracklets[0].tracklet_id

        # Stop sending the local track — tracklet is no longer in alive set.
        # Use a short grace window for faster testing.
        manager._config = TrackletConfig(close_grace_frames=2, min_frames_to_publish=0)

        # Frame 1: lost_count = 1, still alive.
        _, _, _ = await manager.step(
            camera=camera,
            local_tracks=[],
            detections=[],
            embeddings=[],
            event_time=event_time,
            frame_index=1,
        )
        active = manager.get_active_tracklets()
        assert len(active) == 1

        # Frame 2: lost_count = 2 >= grace, tracklet closes.
        closed_tracklets, _, _ = await manager.step(
            camera=camera,
            local_tracks=[],
            detections=[],
            embeddings=[],
            event_time=event_time,
            frame_index=2,
        )
        assert len(closed_tracklets) == 1
        assert closed_tracklets[0].tracklet_id == tracklet_id
        assert closed_tracklets[0].state == "terminated"
        assert len(manager.get_active_tracklets()) == 0

    async def test_embedding_indexing_with_different_embeddings(
        self,
        manager: TrackletManager,
        camera: CameraConfig,
        event_time: datetime,
    ) -> None:
        """Verify that each detection gets its correct embedding when multiple
        detections have different embeddings. This exercises the _find_embedding_index
        fix (review issue #1)."""
        d1 = _make_detection("d1", x_min=0, y_min=0, x_max=100, y_max=100, confidence=0.9)
        d2 = _make_detection("d2", x_min=200, y_min=0, x_max=300, y_max=100, confidence=0.85)
        lt1 = _make_local_track("lt-1", d1, confirmed=True)
        lt2 = _make_local_track("lt-2", d2, confirmed=True)
        # Different embeddings: d1 gets all-ones, d2 gets all-zeros.
        emb1: Embedding = np.ones(768, dtype=np.float32) * 1.0
        emb2: Embedding = np.zeros(768, dtype=np.float32)

        tracklets, gallery_entries, _ = await manager.step(
            camera=camera,
            local_tracks=[lt1, lt2],
            detections=[d1, d2],
            embeddings=[emb1, emb2],
            event_time=event_time,
            frame_index=0,
        )

        assert len(tracklets) == 2
        # Collect unique embedding signatures from gallery entries
        unique_embs: list[tuple[float, ...]] = []
        for entry in gallery_entries:
            sig = tuple(round(v, 4) for v in entry.embedding[:4])
            unique_embs.append(sig)
        # If embedding indexing is correct, we should see at least two
        # distinct signatures (one for all-ones, one for all-zeros).
        assert len(set(unique_embs)) > 1

    async def test_compute_quality_uses_camera_resolution(
        self,
        manager: TrackletManager,
        event_time: datetime,
    ) -> None:
        """Verify that _compute_quality uses the camera's resolution for
        normalization, not a hard-coded 1920x1080 (review issue #8)."""
        # Small detection: 20x20 = 400 px.
        det = _make_detection(
            "d1",
            x_min=0,
            y_min=0,
            x_max=20,
            y_max=20,
            confidence=1.0,
        )

        # 4K camera: 3840x2160 → max_area = 8294400
        cam_4k = CameraConfig(camera_id="cam-4k", resolution_width=3840, resolution_height=2160)
        quality_4k = manager._compute_quality(det, cam_4k)

        # HD camera: 1920x1080 → max_area = 2073600
        cam_hd = CameraConfig(camera_id="cam-hd", resolution_width=1920, resolution_height=1080)
        quality_hd = manager._compute_quality(det, cam_hd)

        # The same detection should score higher on the smaller (HD) camera
        # because it occupies a larger fraction of the frame.
        assert quality_hd > quality_4k
        # Both should be > 0 since confidence is 1.0
        assert quality_hd > 0
        assert quality_4k > 0

    async def test_compute_quality_default_resolution(
        self,
        manager: TrackletManager,
    ) -> None:
        """Verify that CameraConfig without explicit resolution defaults to
        1920x1080 for backward compatibility."""
        det = _make_detection(
            "d1",
            x_min=0,
            y_min=0,
            x_max=192,  # 10% of 1920 width
            y_max=108,  # 10% of 1080 height
            confidence=1.0,
        )
        cam_default = CameraConfig(camera_id="cam-default")
        quality = manager._compute_quality(det, cam_default)
        # Box area = 192*108 = 20736. max_area = 1920*1080 = 2073600.
        # size_score = min(20736 / (2073600 * 0.001), 1.0) = min(10.0, 1.0) = 1.0
        # quality = 0.4 * 1.0 + 0.6 * 1.0 = 1.0
        assert quality == pytest.approx(1.0)
