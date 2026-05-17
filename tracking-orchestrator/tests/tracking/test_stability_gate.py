"""Unit tests for the stability gate in TrackletManager (Phase 1 §3.3.3)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.domain import BoundingBox, CameraConfig, Detection
from app.inference.schemas import Embedding
from app.storage.base import InMemoryGalleryRepository, InMemoryTrackingRepository
from app.tracking.tracker import LocalTrack
from app.tracking.tracklet_manager import TrackletConfig, TrackletManager


def _make_detection(did: str, confidence: float = 0.9) -> Detection:
    bbox = BoundingBox(x_min=10, y_min=10, x_max=100, y_max=200)
    now = datetime.now(UTC)
    return Detection(
        detection_id=did,
        camera_id="cam-1",
        bbox=bbox,
        embedding=[],
        capture_time=now,
        event_time=now,
        confidence=confidence,
    )


def _make_local_track(lid: str, det: Detection, confirmed: bool = True) -> LocalTrack:
    return LocalTrack(
        local_track_id=lid,
        detection=det,
        bbox=det.bbox,
        confidence=det.confidence,
        age=1,
        hit_count=1,
        lost_count=0,
        confirmed=confirmed,
    )


@pytest.fixture
def camera() -> CameraConfig:
    return CameraConfig(camera_id="cam-1", resolution_width=640, resolution_height=480)


@pytest.fixture
def event_time() -> datetime:
    return datetime.now(UTC)


class TestStabilityGate:
    @pytest.mark.asyncio
    async def test_tracklet_hidden_below_gate(
        self, camera: CameraConfig, event_time: datetime
    ) -> None:
        """A new tracklet with frames_alive < min_frames_to_publish is not returned."""
        repo = InMemoryTrackingRepository()
        gallery = InMemoryGalleryRepository()
        manager = TrackletManager(repo, gallery, TrackletConfig(min_frames_to_publish=3))

        det = _make_detection("d1")
        lt = _make_local_track("lt-1", det)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )
        # frames_alive = 1, gate = 3 → not yet published
        assert manager.get_active_tracklets() == []
        assert manager.get_held_count_by_camera() == {"cam-1": 1}

    @pytest.mark.asyncio
    async def test_tracklet_exposed_after_gate(
        self, camera: CameraConfig, event_time: datetime
    ) -> None:
        """After frames_alive >= min_frames_to_publish, the tracklet appears."""
        repo = InMemoryTrackingRepository()
        gallery = InMemoryGalleryRepository()
        manager = TrackletManager(repo, gallery, TrackletConfig(min_frames_to_publish=3))

        det = _make_detection("d1")
        lt = _make_local_track("lt-1", det)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        for frame_idx in range(3):
            await manager.step(
                camera=camera,
                local_tracks=[lt],
                detections=[det],
                embeddings=[emb],
                event_time=event_time,
                frame_index=frame_idx,
            )

        # frames_alive = 3, gate = 3 → now published
        active = manager.get_active_tracklets()
        assert len(active) == 1
        assert manager.get_held_count_by_camera() == {}

    @pytest.mark.asyncio
    async def test_gate_zero_disables_filter(
        self, camera: CameraConfig, event_time: datetime
    ) -> None:
        """Setting min_frames_to_publish=0 exposes tracklets immediately."""
        repo = InMemoryTrackingRepository()
        gallery = InMemoryGalleryRepository()
        manager = TrackletManager(repo, gallery, TrackletConfig(min_frames_to_publish=0))

        det = _make_detection("d1")
        lt = _make_local_track("lt-1", det)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )
        assert len(manager.get_active_tracklets()) == 1

    @pytest.mark.asyncio
    async def test_false_positive_suppressed_before_gate(
        self, camera: CameraConfig, event_time: datetime
    ) -> None:
        """A tracklet that dies before reaching the gate produces no active entries."""
        repo = InMemoryTrackingRepository()
        gallery = InMemoryGalleryRepository()
        manager = TrackletManager(
            repo,
            gallery,
            TrackletConfig(min_frames_to_publish=3, close_grace_frames=1),
        )

        det = _make_detection("d1")
        lt = _make_local_track("lt-1", det)
        emb: Embedding = np.zeros(768, dtype=np.float32)

        # Frame 0: tracklet created, frames_alive=1
        await manager.step(
            camera=camera,
            local_tracks=[lt],
            detections=[det],
            embeddings=[emb],
            event_time=event_time,
            frame_index=0,
        )

        # Frame 1: no local tracks → lost_count=1 >= grace=1 → tracklet closed
        await manager.step(
            camera=camera,
            local_tracks=[],
            detections=[],
            embeddings=[],
            event_time=event_time,
            frame_index=1,
        )

        # Tracklet never crossed the gate; no active, no held
        assert manager.get_active_tracklets() == []
        assert manager.get_held_count_by_camera() == {}
