"""U1-T9: quality field round-trips through InMemoryWorldObservationRepository."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.storage.base import InMemoryWorldObservationRepository

_NOW = datetime(2026, 5, 29, 10, 0, 0, tzinfo=UTC)


def _make_obs(quality: float, detection_id: str = "d1") -> WorldObservation:
    return WorldObservation(
        camera_id="cam-1",
        frame_index=1,
        captured_at=_NOW,
        floor_point=FloorPoint(x_mm=5000, y_mm=5000, calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=[0.9, 0.1, 0.0],
        detection_confidence=0.92,
        detection_id=detection_id,
        quality=quality,
    )


@pytest.mark.asyncio
async def test_quality_round_trips_through_inmemory_repo():
    repo = InMemoryWorldObservationRepository()
    obs = _make_obs(quality=0.73)
    await repo.save(obs, ph_id="ph-1")
    stored = await repo.list_by_ph("ph-1", limit=10)
    assert len(stored) == 1
    assert abs(stored[0].quality - 0.73) < 1e-6


@pytest.mark.asyncio
async def test_quality_default_is_zero():
    obs_default = WorldObservation(
        camera_id="cam-1",
        frame_index=1,
        captured_at=_NOW,
        floor_point=FloorPoint(x_mm=5000, y_mm=5000, calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=[],
        detection_confidence=0.5,
    )
    assert obs_default.quality == 0.0


@pytest.mark.asyncio
async def test_quality_preserved_across_multiple_saves():
    repo = InMemoryWorldObservationRepository()
    for i, q in enumerate([0.1, 0.5, 0.9]):
        obs = _make_obs(quality=q, detection_id=f"d{i}")
        await repo.save(obs, ph_id="ph-1")
    stored = await repo.list_by_ph("ph-1", limit=10)
    stored_qualities = sorted(o.quality for o in stored)
    assert abs(stored_qualities[0] - 0.1) < 1e-6
    assert abs(stored_qualities[1] - 0.5) < 1e-6
    assert abs(stored_qualities[2] - 0.9) < 1e-6
