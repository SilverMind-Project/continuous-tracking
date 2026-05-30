"""U1-T10: Tests that mean_quality EMA is updated on PHs."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.domain import BoundingBox, FaceAnchor, FloorPoint, WorldObservation
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
    quality: float,
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
async def test_mean_quality_ema_updated_across_frames():
    """mean_quality converges toward successive observation qualities via EMA."""
    cfg = WorldTrackerConfig(
        dedup_enabled=False,
        min_observations_to_publish=1,
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    t0 = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    from datetime import timedelta

    # Frame 1: spawn with quality=1.0 → mean_quality starts at 1.0
    await tracker.step(
        [_obs("cam-1", 5.0, 5.0, "d1", t0, quality=1.0)],
        now=t0,
        room_polygons=_ROOM_POLYGONS,
    )
    phs = await ph_repo.list_open()
    assert len(phs) == 1
    assert abs(phs[0].mean_quality - 1.0) < 1e-6, "first observation sets mean_quality=1.0"

    # Frame 2: update with quality=0.0 → EMA: 0.1*0.0 + 0.9*1.0 = 0.9
    t1 = t0 + timedelta(seconds=1)
    await tracker.step(
        [_obs("cam-1", 5.1, 5.0, "d2", t1, quality=0.0)],
        now=t1,
        room_polygons=_ROOM_POLYGONS,
    )
    phs = await ph_repo.list_open()
    assert len(phs) == 1
    assert abs(phs[0].mean_quality - 0.9) < 1e-6, "EMA after quality=0: 0.1*0 + 0.9*1 = 0.9"

    # Frame 3: update with quality=0.0 again → EMA: 0.1*0.0 + 0.9*0.9 = 0.81
    t2 = t1 + timedelta(seconds=1)
    await tracker.step(
        [_obs("cam-1", 5.2, 5.0, "d3", t2, quality=0.0)],
        now=t2,
        room_polygons=_ROOM_POLYGONS,
    )
    phs = await ph_repo.list_open()
    assert len(phs) == 1
    assert abs(phs[0].mean_quality - 0.81) < 1e-6, "EMA after second quality=0: 0.81"


@pytest.mark.asyncio
async def test_new_ph_mean_quality_set_from_first_observation():
    """A newly spawned PH gets mean_quality equal to the spawning observation's quality."""
    cfg = WorldTrackerConfig(dedup_enabled=False, min_observations_to_publish=1)
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    t0 = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    await tracker.step(
        [_obs("cam-1", 5.0, 5.0, "d1", t0, quality=0.75)],
        now=t0,
        room_polygons=_ROOM_POLYGONS,
    )
    phs = await ph_repo.list_open()
    assert len(phs) == 1
    assert abs(phs[0].mean_quality - 0.75) < 1e-6


@pytest.mark.asyncio
async def test_no_room_polygons_still_spawns_ph_and_snapshot_identity():
    """Unconfigured room polygons must not block live PH/identity publication."""
    cfg = WorldTrackerConfig(dedup_enabled=False, min_observations_to_publish=1)
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)

    t0 = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)
    obs = replace(
        _obs("cam-1", 5.0, 5.0, "det-grandma", t0, quality=0.9),
        face_anchor=FaceAnchor(
            person_id="grandma",
            confidence=0.94,
            tracklet_id="",
            detection_id="det-grandma",
            camera_id="cam-1",
            captured_at=t0,
        ),
    )

    result = await tracker.step([obs], now=t0, room_polygons={})

    phs = await ph_repo.list_open()
    assert len(phs) == 1
    assert result.det_to_ph == {"det-grandma": phs[0].ph_id}
    assert len(result.snapshots) == 1
    assert result.snapshots[0].identity_id == "grandma"
