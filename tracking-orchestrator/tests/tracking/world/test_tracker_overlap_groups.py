"""Overlap groups replaced at runtime take effect on the next tracker round."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import BoundingBox, FloorPoint, OverlapGroup, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

_NOW = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
_ROOMS = {"room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}
_LOOK = [0.9, 0.1, 0.0]


def _uncalibrated(camera_id: str, x_m: float, detection_id: str, at: datetime) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=1,
        captured_at=at,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=5000, calibrated=False),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=_LOOK,
        detection_confidence=0.92,
        detection_id=detection_id,
        quality=0.6,
    )


@pytest.mark.asyncio
async def test_set_overlap_groups_enables_and_disables_group_dedup() -> None:
    cfg = WorldTrackerConfig(
        dedup_enabled=True,
        enable_group_appearance_dedup=True,
        dedup_group_appearance_min_sim=0.75,
        min_observations_to_publish=1,
    )
    tracker = WorldTracker(
        ph_repo=InMemoryPHRepository(),
        obs_repo=InMemoryWorldObservationRepository(),
        config=cfg,
    )

    async def tracks_for_two_cameras(offset_s: float, tag: str) -> int:
        # One person on two cameras: distinct PHs the two detections were assigned to.
        at = _NOW + timedelta(seconds=offset_s)
        result = await tracker.step(
            [
                _uncalibrated("cam-a", 1.0, f"{tag}a", at),
                _uncalibrated("cam-b", 9.0, f"{tag}b", at),
            ],
            now=at,
            room_polygons=_ROOMS,
        )
        return len({result.det_to_ph[f"{tag}a"], result.det_to_ph[f"{tag}b"]})

    assert await tracks_for_two_cameras(0, "r1") == 2

    tracker.set_overlap_groups([OverlapGroup(group_id="1", camera_ids=("cam-a", "cam-b"))])
    assert await tracks_for_two_cameras(600, "r2") == 1

    tracker.set_overlap_groups([])
    assert await tracks_for_two_cameras(1200, "r3") == 2
