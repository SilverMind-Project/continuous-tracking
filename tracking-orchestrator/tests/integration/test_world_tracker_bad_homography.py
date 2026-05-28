"""WTR10: Replay — bad homography camera must not produce floor markers.

Uncalibrated detections must be counted as diagnostics, not placed on the floor plan.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.tracker import WorldTracker


def _make_calibrated_obs(
    camera_id: str,
    frame_index: int,
    fx: float,
    fy: float,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=int(fx * 1000), y_mm=int(fy * 1000), calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=[1.0, 0.0],
        detection_confidence=0.9,
    )


def _make_uncalibrated_obs(
    camera_id: str,
    frame_index: int,
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=frame_index,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=0, y_mm=0, calibrated=False),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=[1.0, 0.0],
        detection_confidence=0.9,
    )


@pytest.mark.asyncio
async def test_uncalibrated_detections_not_tracked():
    """Uncalibrated detections must not spawn PHs or become floor markers."""
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()

    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)
    now = datetime.now(UTC)

    # Uncalibrated observation from cam-3.
    result = await tracker.step(
        observations=[
            _make_uncalibrated_obs("cam-3", 1),
        ],
        now=now,
    )

    # Uncalibrated observations should be filtered out before association.
    # No PHs should be spawned.
    new_phs = [ph for ph in result.updated_phs if ph.observation_count == 1]
    uncalibrated_phs = [ph for ph in new_phs if "cam-3" in ph.active_cameras]
    assert len(uncalibrated_phs) == 0, "Uncalibrated detections must not spawn PHs"


@pytest.mark.asyncio
async def test_calibrated_detections_are_tracked():
    """Calibrated detections from good cameras must be tracked normally."""
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()

    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)
    now = datetime.now(UTC)

    room_polygons = {"room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]}
    result = await tracker.step(
        observations=[
            _make_calibrated_obs("cam-1", 1, 1.0, 2.0),
        ],
        now=now,
        room_polygons=room_polygons,
    )

    calibrated_phs = [ph for ph in result.updated_phs if ph.observation_count >= 1]
    assert len(calibrated_phs) >= 1, "Calibrated detections must produce PHs"
