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
        return TrackletManager(repo, gallery, TrackletConfig())

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

        persisted = manager.get_tracklet(tracklets[0].tracklet_id)
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
