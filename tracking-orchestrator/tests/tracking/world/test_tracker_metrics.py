"""U1-T11: dedup metrics increment correctly."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

_ROOM_POLYGONS = {
    "room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
}


def _obs(
    camera_id: str,
    x_m: float,
    y_m: float,
    detection_id: str,
    captured_at: datetime,
    calibrated: bool = True,
    quality: float = 0.5,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=1,
        captured_at=captured_at,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=calibrated),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=[0.9, 0.1, 0.0],
        detection_confidence=0.92,
        detection_id=detection_id,
        quality=quality,
    )


@pytest.mark.asyncio
async def test_dedup_counter_increments_once_per_collapsed_observation(monkeypatch):
    """worldtracker_observations_deduped_total increments by (cluster_size - 1) per cluster."""
    from app.observability import metrics as _metrics

    fake_deduped = MagicMock()
    fake_clusters = MagicMock()
    fake_missing = MagicMock()

    class _FakeMetrics:
        worldtracker_observations_deduped_total = fake_deduped
        worldtracker_dedup_clusters_total = fake_clusters
        worldtracker_observation_missing_floorpoint_total = fake_missing

    monkeypatch.setattr(_metrics, "metrics", _FakeMetrics())

    cfg = WorldTrackerConfig(
        dedup_enabled=True, dedup_max_distance_m=0.6, min_observations_to_publish=1
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    now = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    # Two overlapping cross-camera observations → one cluster → deduped_total.inc(1)
    observations = [
        _obs("cam-1", 5.0, 5.0, "d1", now, quality=0.7),
        _obs("cam-2", 5.05, 5.0, "d2", now, quality=0.5),
    ]
    await tracker.step(observations, now=now, room_polygons=_ROOM_POLYGONS)

    fake_deduped.inc.assert_called_once_with(1)
    fake_clusters.inc.assert_called_once()


@pytest.mark.asyncio
async def test_missing_floorpoint_counter_increments(monkeypatch):
    """worldtracker_observation_missing_floorpoint_total increments for uncalibrated obs."""
    from app.observability import metrics as _metrics

    fake_deduped = MagicMock()
    fake_clusters = MagicMock()
    fake_missing = MagicMock()

    class _FakeMetrics:
        worldtracker_observations_deduped_total = fake_deduped
        worldtracker_dedup_clusters_total = fake_clusters
        worldtracker_observation_missing_floorpoint_total = fake_missing

    monkeypatch.setattr(_metrics, "metrics", _FakeMetrics())

    cfg = WorldTrackerConfig(
        dedup_enabled=True, dedup_max_distance_m=0.6, min_observations_to_publish=1
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    now = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    observations = [
        _obs("cam-1", 5.0, 5.0, "d1", now, calibrated=False),
        _obs("cam-2", 5.5, 5.0, "d2", now, calibrated=True),
    ]
    await tracker.step(observations, now=now, room_polygons=_ROOM_POLYGONS)

    fake_missing.inc.assert_called_once()
