"""Integration and in-memory round-trip tests for device-free WorldObservation persistence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from app.domain import (
    FloorPoint,
    ObservationSource,
    PersonHypothesis,
    WorldObservation,
)
from app.storage.base import InMemoryWorldObservationRepository
from app.storage.postgres.ph_repo import PostgresPHRepository, PostgresWorldObservationRepository

_T0 = datetime(2026, 11, 25, 11, 0, 0, tzinfo=UTC)


def _ph(ph_id: str) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(2.5, 4.0, 0.0, 0.0),
        state_cov=(
            0.09,
            0.0,
            0.0,
            0.0,
            0.0,
            0.09,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
        ),
        born_at=_T0,
        last_seen_at=_T0,
        last_seen_camera="sensor:radar-bathroom-1",
        observation_count=1,
        active_cameras=frozenset({"sensor:radar-bathroom-1"}),
    )


def _device_free_obs() -> WorldObservation:
    return WorldObservation(
        camera_id="sensor:radar-bathroom-1",
        frame_index=7,
        captured_at=_T0,
        floor_point=FloorPoint(x_mm=2450, y_mm=4100, calibrated=True),
        bbox=None,
        embedding=[],
        detection_confidence=1.0,
        detection_id="pos:radar-bathroom-1:t1:1000",
        quality=0.0,
        floor_residual_m=0.12,
        floor_cov_random=(0.04, 0.01, 0.01, 0.05),
        footpoint_reliable=True,
        source=ObservationSource.DEVICE_FREE,
    )


@pytest.mark.asyncio
async def test_in_memory_world_observation_device_free_round_trip() -> None:
    obs_repo = InMemoryWorldObservationRepository()
    ph_id = str(uuid.uuid4())
    await obs_repo.save(_device_free_obs(), ph_id=ph_id)

    rows = await obs_repo.list_by_ph(ph_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.source == ObservationSource.DEVICE_FREE
    assert row.device_free is True
    assert row.bbox is None
    assert row.camera_id == "sensor:radar-bathroom-1"
    assert row.floor_cov_random == pytest.approx((0.04, 0.01, 0.01, 0.05))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_world_observation_device_free_round_trip_and_legacy_default(
    db_pool: Any,
) -> None:
    ph_repo = PostgresPHRepository(db_pool)
    obs_repo = PostgresWorldObservationRepository(db_pool)
    ph_id = str(uuid.uuid4())
    await ph_repo.save(_ph(ph_id))
    await obs_repo.save(_device_free_obs(), ph_id=ph_id)

    # Legacy row written before source metadata existed must read back as CAMERA.
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO continuous_tracking.world_observations (
                observation_id, ph_id, camera_id, frame_index, captured_at,
                floor_x_m, floor_y_m, detection_confidence, bbox, height_m, quality, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            str(uuid.uuid4()),
            ph_id,
            "cam-1",
            8,
            _T0.replace(second=5),
            1.0,
            2.0,
            0.9,
            json.dumps({"x_min": 10.0, "y_min": 20.0, "x_max": 50.0, "y_max": 120.0}),
            None,
            0.8,
            json.dumps({"floor_residual_m": 0.05, "footpoint_reliable": True}),
        )

    rows = await obs_repo.list_by_ph(ph_id)
    by_cam = {r.camera_id: r for r in rows}

    df = by_cam["sensor:radar-bathroom-1"]
    assert df.source == ObservationSource.DEVICE_FREE
    assert df.device_free is True
    assert df.bbox is None
    assert df.floor_point.x_mm == 2450
    assert df.floor_point.y_mm == 4100
    assert df.floor_cov_random == pytest.approx((0.04, 0.01, 0.01, 0.05))

    legacy = by_cam["cam-1"]
    assert legacy.source == ObservationSource.CAMERA
    assert legacy.device_free is False
    assert legacy.bbox is not None
