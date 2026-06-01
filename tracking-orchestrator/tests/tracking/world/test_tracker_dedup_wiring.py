"""Tests that the dedup knob and wiring work end-to-end (InMemory repos)."""

from __future__ import annotations

from datetime import UTC, datetime

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
    quality: float = 0.5,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=1,
        captured_at=captured_at,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=[0.9, 0.1, 0.0],
        detection_confidence=0.92,
        detection_id=detection_id,
        quality=quality,
    )


@pytest.mark.asyncio
async def test_dedup_enabled_two_simultaneous_cameras_one_ph():
    """With dedup_enabled=True, two overlapping cameras produce one PH."""
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        dedup_max_distance_m=0.6,
        min_observations_to_publish=1,
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    now = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    observations = [
        _obs("cam-1", 5.0, 5.0, "d1", now),
        _obs("cam-2", 5.05, 5.0, "d2", now),
    ]
    await tracker.step(observations, now=now, room_polygons=_ROOM_POLYGONS)

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1, f"dedup enabled: expected 1 PH, got {len(open_phs)}"
    ph = open_phs[0]
    assert "cam-1" in ph.active_cameras
    assert "cam-2" in ph.active_cameras


@pytest.mark.asyncio
async def test_dedup_disabled_two_simultaneous_cameras_two_phs():
    """With dedup_enabled=False, two overlapping cameras produce two PHs (pre-U1 behaviour)."""
    cfg = WorldTrackerConfig(
        dedup_enabled=False,
        min_observations_to_publish=1,
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    now = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    observations = [
        _obs("cam-1", 5.0, 5.0, "d1", now),
        _obs("cam-2", 5.05, 5.0, "d2", now),
    ]
    await tracker.step(observations, now=now, room_polygons=_ROOM_POLYGONS)

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 2, f"dedup disabled: expected 2 PHs, got {len(open_phs)}"


@pytest.mark.asyncio
async def test_dedup_both_cameras_obs_stored_on_single_ph():
    """With dedup enabled, observations from both cameras are stored against the one PH."""
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        dedup_max_distance_m=0.6,
        min_observations_to_publish=1,
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    now = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    observations = [
        _obs("cam-1", 5.0, 5.0, "d1", now),
        _obs("cam-2", 5.05, 5.0, "d2", now),
    ]
    await tracker.step(observations, now=now, room_polygons=_ROOM_POLYGONS)

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1
    ph = open_phs[0]
    obs_list = await obs_repo.list_by_ph(ph.ph_id, limit=10)
    camera_ids_in_obs = {o.camera_id for o in obs_list}
    assert "cam-1" in camera_ids_in_obs, "cam-1 observation must be stored on the PH"
    assert "cam-2" in camera_ids_in_obs, "cam-2 observation must be stored on the PH"
