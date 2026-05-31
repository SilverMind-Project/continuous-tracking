"""WTR10: WorldTracker behaviour with uncalibrated observations.

WorldTrackingStage creates synthetic floor points for uncalibrated cameras so
that PHs (and identity resolution) still work without a homography.  This file
tests the tracker contract directly: uncalibrated observations passed to
WorldTracker.step are tracked normally (the stage owns the synthetic-floor-point
conversion; the tracker trusts whatever floor position it receives).
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
async def test_uncalibrated_detections_are_tracked():
    """Uncalibrated observations passed to the tracker do spawn PHs.

    WorldTrackingStage converts uncalibrated detections to synthetic floor
    points before calling WorldTracker.step.  The tracker itself does not
    filter by calibration status: it trusts whatever floor position it
    receives.  The resulting PH has calibrated=False floor coordinates and
    the CC side decides whether to render it on the floor plan.
    """
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()

    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo)
    now = datetime.now(UTC)

    result = await tracker.step(
        observations=[_make_uncalibrated_obs("cam-3", 1)],
        now=now,
    )

    # The tracker spawns a PH even for an uncalibrated observation.
    new_phs = [ph for ph in result.updated_phs if ph.observation_count == 1]
    cam3_phs = [ph for ph in new_phs if "cam-3" in ph.active_cameras]
    assert len(cam3_phs) == 1, (
        "Tracker must spawn a PH for an uncalibrated observation; "
        "WorldTrackingStage is responsible for synthetic floor points"
    )


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
