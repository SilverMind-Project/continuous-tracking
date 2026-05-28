"""WTR10: Replay — one person across two cameras produces one PH.

Uses InMemory repos. Verifies one PH lifecycle across two calibrated cameras.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.tracker import WorldTracker

_ROOM = {"living_room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}


def _make_obs(
    camera_id: str,
    frame_index: int,
    fx: float,
    fy: float,
    embedding: list[float] | None = None,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=int(fx * 1000), y_mm=int(fy * 1000), calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=embedding or [1.0, 0.0, 0.0],
        detection_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_one_person_two_cameras_one_ph():
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)
    now = datetime.now(UTC)

    # Camera 1: person at (1, 2).
    result1 = await tracker.step(
        observations=[_make_obs("cam-1", 1, 1.0, 2.0)],
        now=now,
        room_polygons=_ROOM,
    )
    assert len(result1.updated_phs) >= 1, f"Expected >=1 PH, got {len(result1.updated_phs)}"
    ph_id = result1.updated_phs[0].ph_id

    # Camera 2: same person nearby.
    result2 = await tracker.step(
        observations=[_make_obs("cam-2", 1, 1.2, 2.1, embedding=[0.9, 0.1, 0.0])],
        now=now,
        room_polygons=_ROOM,
    )
    matched = {ph.ph_id for ph in result2.updated_phs}
    assert ph_id in matched, f"PH {ph_id} must match cam-2 observation"

    open_phs = await ph_repo.list_open()
    assert len(open_phs) == 1
    ph = open_phs[0]
    assert "cam-1" in ph.active_cameras
