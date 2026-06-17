"""WorldTracker stabilized primary-camera selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker


def _obs(
    camera_id: str,
    x_m: float,
    y_m: float,
    detection_id: str,
    captured_at: datetime,
    *,
    primary_score: float,
    quality: float = 0.8,
    footpoint_reliable: bool = True,
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
        primary_score=primary_score,
        footpoint_reliable=footpoint_reliable,
    )


def _tracker(
    *,
    primary_switch_frames: int = 3,
    ph_close_grace_s: float = 5.0,
    dedup_enabled: bool = False,
) -> tuple[WorldTracker, InMemoryPHRepository]:
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(
        ph_repo=ph_repo,
        obs_repo=obs_repo,
        config=WorldTrackerConfig(
            dedup_enabled=dedup_enabled,
            min_observations_to_publish=1,
            primary_switch_frames=primary_switch_frames,
            ph_close_grace_s=ph_close_grace_s,
        ),
    )
    return tracker, ph_repo


@pytest.mark.asyncio
async def test_primary_initializes_to_first_camera() -> None:
    tracker, _ph_repo = _tracker()
    t0 = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    result = await tracker.step([_obs("cam-a", 5.0, 5.0, "d1", t0, primary_score=0.4)], now=t0)

    assert len(result.snapshots) == 1
    assert result.snapshots[0].camera_id == "cam-a"


@pytest.mark.asyncio
async def test_primary_stays_until_challenger_persists() -> None:
    tracker, _ph_repo = _tracker(primary_switch_frames=3)
    t0 = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    result = await tracker.step([_obs("cam-a", 5.0, 5.0, "d1", t0, primary_score=0.4)], now=t0)
    assert result.snapshots[0].camera_id == "cam-a"

    for idx in range(1, 3):
        now = t0 + timedelta(seconds=idx)
        result = await tracker.step(
            [_obs("cam-b", 5.02, 5.0, f"d{idx + 1}", now, primary_score=0.9)],
            now=now,
        )
        assert result.snapshots[0].camera_id == "cam-a"

    now = t0 + timedelta(seconds=3)
    result = await tracker.step([_obs("cam-b", 5.03, 5.0, "d4", now, primary_score=0.9)], now=now)

    assert result.snapshots[0].camera_id == "cam-b"


@pytest.mark.asyncio
async def test_primary_uses_best_source_camera_from_dedup_cluster() -> None:
    tracker, _ph_repo = _tracker(primary_switch_frames=3, dedup_enabled=True)
    t0 = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    result = await tracker.step(
        [
            _obs("cam-a", 5.0, 5.0, "d1", t0, primary_score=0.2, quality=0.9),
            _obs("cam-b", 5.03, 5.0, "d2", t0, primary_score=0.95, quality=0.1),
        ],
        now=t0,
    )

    assert len(result.snapshots) == 1
    assert result.snapshots[0].camera_id == "cam-b"


@pytest.mark.asyncio
async def test_snapshot_records_cluster_count_and_representative_reliability() -> None:
    tracker, _ph_repo = _tracker(primary_switch_frames=1, dedup_enabled=True)
    t0 = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    result = await tracker.step(
        [
            _obs(
                "cam-a",
                5.0,
                5.0,
                "d1",
                t0,
                primary_score=0.2,
                quality=0.9,
                footpoint_reliable=False,
            ),
            _obs(
                "cam-b",
                5.03,
                5.0,
                "d2",
                t0,
                primary_score=0.95,
                quality=0.1,
                footpoint_reliable=True,
            ),
        ],
        now=t0,
    )

    assert len(result.snapshots) == 1
    snapshot = result.snapshots[0]
    assert snapshot.contributing_camera_count == 2
    assert snapshot.footpoint_reliable is False


@pytest.mark.asyncio
async def test_snapshot_room_fallback_uses_primary_camera() -> None:
    tracker, _ph_repo = _tracker(primary_switch_frames=3)
    t0 = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    room_map = {"cam-a": "Kitchen", "cam-b": "Hall"}

    await tracker.step(
        [_obs("cam-a", 5.0, 5.0, "d1", t0, primary_score=0.4)],
        now=t0,
        camera_room_map=room_map,
    )
    result = await tracker.step(
        [_obs("cam-b", 5.02, 5.0, "d2", t0 + timedelta(seconds=1), primary_score=0.9)],
        now=t0 + timedelta(seconds=1),
        camera_room_map=room_map,
    )

    snapshot = result.snapshots[0]
    assert snapshot.camera_id == "cam-a"
    assert snapshot.room_id == "Kitchen"
    assert snapshot.room_name == "Kitchen"


@pytest.mark.asyncio
async def test_primary_evicted_on_close() -> None:
    tracker, ph_repo = _tracker(ph_close_grace_s=0.5)
    t0 = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    await tracker.step([_obs("cam-a", 5.0, 5.0, "d1", t0, primary_score=0.4)], now=t0)
    phs = await ph_repo.list_open()
    assert len(phs) == 1
    assert tracker._primary_camera == {phs[0].ph_id: "cam-a"}

    await tracker.step([], now=t0 + timedelta(seconds=1))

    assert await ph_repo.list_open() == []
    assert tracker._primary_camera == {}
    assert tracker._primary_challenger == {}


@pytest.mark.asyncio
async def test_position_unchanged_by_primary_selection() -> None:
    t0 = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)

    primary_b_tracker, _repo_b = _tracker(primary_switch_frames=1, dedup_enabled=True)
    primary_a_tracker, _repo_a = _tracker(primary_switch_frames=1, dedup_enabled=True)

    result_b = await primary_b_tracker.step(
        [
            _obs("cam-a", 5.0, 5.0, "a1", t0, primary_score=0.2),
            _obs("cam-b", 5.04, 5.0, "b1", t0, primary_score=0.9),
        ],
        now=t0,
    )
    result_a = await primary_a_tracker.step(
        [
            _obs("cam-a", 5.0, 5.0, "a1", t0, primary_score=0.9),
            _obs("cam-b", 5.04, 5.0, "b1", t0, primary_score=0.2),
        ],
        now=t0,
    )

    snap_b = result_b.snapshots[0]
    snap_a = result_a.snapshots[0]
    assert snap_b.camera_id == "cam-b"
    assert snap_a.camera_id == "cam-a"
    assert snap_b.floor_x_m == pytest.approx(snap_a.floor_x_m)
    assert snap_b.floor_y_m == pytest.approx(snap_a.floor_y_m)
