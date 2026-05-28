"""WTR10: Replay — two people crossing, no identity swap.

Two people with distinct embeddings observed near each other
must produce two distinct PHs without identity merging.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker


def _make_obs(
    camera_id: str, frame_index: int, fx: float, fy: float,
    embedding: list[float],
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=int(fx * 1000), y_mm=int(fy * 1000), calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=embedding,
        detection_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_two_people_produce_two_distinct_phs():
    """Two people with different embeddings must be tracked as separate PHs."""
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()

    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)
    now = datetime.now(UTC)
    room_polygons = {"living_room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}

    # Frame 1: person A at (1,1), person B at (7,7) — far apart.
    result1 = await tracker.step(
        observations=[
            _make_obs("cam-1", 1, 1.0, 1.0, embedding=[1.0, 0.0]),
            _make_obs("cam-1", 1, 7.0, 7.0, embedding=[0.0, 1.0]),
        ],
        now=now, room_polygons=room_polygons,
    )
    new_phs = [ph for ph in result1.updated_phs if ph.observation_count == 1]
    assert len(new_phs) == 2, f"Expected 2 new PHs, got {len(new_phs)}"

    # Frame 2: both people move slightly but stay distinct.
    result2 = await tracker.step(
        observations=[
            _make_obs("cam-1", 2, 1.1, 1.1, embedding=[0.95, 0.05]),
            _make_obs("cam-1", 2, 6.9, 6.9, embedding=[0.05, 0.95]),
        ],
        now=now, room_polygons=room_polygons,
    )

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 2, f"Expected 2 open PHs, got {len(open_phs)}"
    for ph in open_phs:
        assert ph.observation_count == 2
