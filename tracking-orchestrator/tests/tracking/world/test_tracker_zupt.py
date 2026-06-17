"""WorldTracker ZUPT stationarity behavior."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.storage.base import InMemoryPHRepository, InMemoryWorldObservationRepository
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker

_ROOM_POLYGONS = {
    "room": [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
}


def _obs(x_m: float, y_m: float, detection_id: str, captured_at: datetime) -> WorldObservation:
    return WorldObservation(
        camera_id="cam-1",
        frame_index=1,
        captured_at=captured_at,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=[0.9, 0.1, 0.0],
        detection_confidence=0.92,
        detection_id=detection_id,
        quality=0.9,
    )


async def _run_sequence(
    cfg: WorldTrackerConfig,
    points: list[tuple[float, float]],
) -> tuple[list[float], list[tuple[float, float]], WorldTracker]:
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)
    t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    speeds: list[float] = []
    positions: list[tuple[float, float]] = []

    for idx, (x_m, y_m) in enumerate(points):
        captured_at = t0 + timedelta(seconds=idx)
        await tracker.step(
            [_obs(x_m, y_m, f"det-{idx}", captured_at)],
            now=captured_at,
            room_polygons=_ROOM_POLYGONS,
        )
        phs = await ph_repo.list_open()
        assert len(phs) == 1
        ph = phs[0]
        speeds.append(ph.last_floor_speed_m_s)
        positions.append((ph.state_mean[0], ph.state_mean[1]))

    return speeds, positions, tracker


@pytest.mark.asyncio
async def test_stationary_person_velocity_converges_to_zero_and_variance_drops() -> None:
    points = [
        (5.00, 5.00),
        (5.04, 4.97),
        (4.97, 5.02),
        (5.02, 5.01),
        (4.98, 4.98),
        (5.01, 5.03),
        (4.99, 4.99),
        (5.03, 4.98),
        (4.97, 5.02),
        (5.00, 5.00),
        (5.02, 4.99),
        (4.98, 5.01),
    ]
    cfg = WorldTrackerConfig(
        dedup_enabled=False,
        min_observations_to_publish=1,
        observation_noise_m=0.2,
        zupt_consecutive_frames=3,
    )
    no_zupt_cfg = replace(cfg, zupt_consecutive_frames=999)

    speeds, positions, _tracker = await _run_sequence(cfg, points)
    no_zupt_speeds, no_zupt_positions, _no_zupt_tracker = await _run_sequence(
        no_zupt_cfg,
        points,
    )

    zupt_variance = float(np.var(np.array(positions[-6:], dtype=np.float64), axis=0).sum())
    no_zupt_variance = float(
        np.var(np.array(no_zupt_positions[-6:], dtype=np.float64), axis=0).sum()
    )
    assert speeds[-1] < 0.005
    assert no_zupt_speeds[-1] > speeds[-1] * 10
    assert zupt_variance < no_zupt_variance * 0.1


@pytest.mark.asyncio
async def test_slow_shuffle_not_clamped() -> None:
    cfg = WorldTrackerConfig(
        dedup_enabled=False,
        min_observations_to_publish=1,
        zupt_consecutive_frames=3,
    )
    points = [(5.0 + 0.3 * idx, 5.0) for idx in range(10)]

    speeds, _positions, tracker = await _run_sequence(cfg, points)

    assert speeds[-1] == pytest.approx(0.3, abs=0.03)
    assert tracker._still_counter == {}


@pytest.mark.asyncio
async def test_zupt_debounce_single_still_frame_mid_walk_does_not_trigger() -> None:
    cfg = WorldTrackerConfig(
        dedup_enabled=False,
        min_observations_to_publish=1,
        zupt_consecutive_frames=3,
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)
    t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)

    points = [(5.0, 5.0), (5.3, 5.0), (5.3, 5.0)]
    for idx, (x_m, y_m) in enumerate(points):
        captured_at = t0 + timedelta(seconds=idx)
        await tracker.step(
            [_obs(x_m, y_m, f"det-{idx}", captured_at)],
            now=captured_at,
            room_polygons=_ROOM_POLYGONS,
        )

    phs = await ph_repo.list_open()
    assert len(phs) == 1
    assert tracker._still_counter == {phs[0].ph_id: 1}

    captured_at = t0 + timedelta(seconds=3)
    await tracker.step(
        [_obs(5.6, 5.0, "det-3", captured_at)],
        now=captured_at,
        room_polygons=_ROOM_POLYGONS,
    )
    phs = await ph_repo.list_open()
    assert len(phs) == 1
    speed_after_resume = phs[0].last_floor_speed_m_s

    assert speed_after_resume > cfg.zupt_speed_exit_m_s
    assert tracker._still_counter == {}


@pytest.mark.asyncio
async def test_zupt_evicted_on_close() -> None:
    cfg = WorldTrackerConfig(
        dedup_enabled=False,
        min_observations_to_publish=1,
        ph_close_grace_s=0.5,
        zupt_consecutive_frames=2,
    )
    ph_repo = InMemoryPHRepository()
    obs_repo = InMemoryWorldObservationRepository()
    tracker = WorldTracker(ph_repo=ph_repo, obs_repo=obs_repo, config=cfg)
    t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)

    for idx in range(3):
        captured_at = t0 + timedelta(seconds=idx)
        await tracker.step(
            [_obs(5.0, 5.0, f"det-{idx}", captured_at)],
            now=captured_at,
            room_polygons=_ROOM_POLYGONS,
        )

    phs = await ph_repo.list_open()
    assert len(phs) == 1
    assert tracker._still_counter == {phs[0].ph_id: 2}

    close_at = t0 + timedelta(seconds=4)
    await tracker.step([], now=close_at, room_polygons=_ROOM_POLYGONS)

    assert await ph_repo.list_open() == []
    assert tracker._still_counter == {}
