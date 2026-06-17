"""Integration tests for trajectory confidence persistence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from app.domain import PersonTrajectoryPoint
from app.storage.trajectory import InMemoryTrajectoryRepository

_DUMMY_STATE_MEAN = [0.0, 0.0, 0.0, 0.0]
_DUMMY_STATE_COV = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


async def _ensure_identity_and_ph(
    conn: Any,
    *,
    identity_id: str,
    ph_id: str,
    now: datetime,
) -> None:
    await conn.execute(
        """
        INSERT INTO continuous_tracking.identities (identity_id, display_name)
        VALUES ($1, $1)
        """,
        identity_id,
    )
    await conn.execute(
        """
        INSERT INTO continuous_tracking.person_hypotheses
            (ph_id, born_at, last_seen_at, last_seen_camera,
             observation_count, state_mean, state_cov, metadata)
        VALUES ($1, $2, $2, 'cam-primary', 1, $3, $4, '{}')
        """,
        ph_id,
        now,
        _DUMMY_STATE_MEAN,
        _DUMMY_STATE_COV,
    )


def _point(*, identity_id: str, ph_id: str, now: datetime) -> PersonTrajectoryPoint:
    return PersonTrajectoryPoint(
        identity_id=identity_id,
        ph_id=ph_id,
        observed_at=now,
        room_name="kitchen",
        ground_x=1.25,
        ground_y=2.5,
        posture="standing",
        identity_confidence=0.88,
        position_sigma_m=0.37,
        primary_camera_id="cam-primary",
        contributing_camera_count=2,
        footpoint_reliable=False,
        motion_energy=0.012,
        floor_speed_m_s=0.24,
    )


@pytest.mark.integration
async def test_postgres_trajectory_roundtrip_confidence(db_pool: asyncpg.Pool) -> None:
    from app.storage.postgres.trajectory_repo import PostgresTrajectoryRepository

    identity_id = "resident-" + str(uuid.uuid4())[:8]
    ph_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    async with db_pool.acquire() as conn:
        await _ensure_identity_and_ph(conn, identity_id=identity_id, ph_id=ph_id, now=now)

    repo = PostgresTrajectoryRepository(db_pool)
    expected = _point(identity_id=identity_id, ph_id=ph_id, now=now)
    await repo.save_trajectory_point(expected)

    points = await repo.list_trajectory_points(ph_id=ph_id, limit=1)

    assert len(points) == 1
    actual = points[0]
    assert actual.position_sigma_m == pytest.approx(expected.position_sigma_m)
    assert actual.primary_camera_id == expected.primary_camera_id
    assert actual.contributing_camera_count == expected.contributing_camera_count
    assert actual.footpoint_reliable is expected.footpoint_reliable


@pytest.mark.integration
async def test_inmemory_postgres_trajectory_confidence_parity(db_pool: asyncpg.Pool) -> None:
    from app.storage.postgres.trajectory_repo import PostgresTrajectoryRepository

    identity_id = "resident-" + str(uuid.uuid4())[:8]
    ph_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    async with db_pool.acquire() as conn:
        await _ensure_identity_and_ph(conn, identity_id=identity_id, ph_id=ph_id, now=now)

    expected = _point(identity_id=identity_id, ph_id=ph_id, now=now)
    in_memory = InMemoryTrajectoryRepository()
    postgres = PostgresTrajectoryRepository(db_pool)

    await in_memory.save_trajectory_point(expected)
    await postgres.save_trajectory_point(expected)

    in_memory_point = (await in_memory.list_trajectory_points(ph_id=ph_id, limit=1))[0]
    postgres_point = (await postgres.list_trajectory_points(ph_id=ph_id, limit=1))[0]

    assert postgres_point.position_sigma_m == pytest.approx(in_memory_point.position_sigma_m)
    assert postgres_point.primary_camera_id == in_memory_point.primary_camera_id
    assert postgres_point.contributing_camera_count == in_memory_point.contributing_camera_count
    assert postgres_point.footpoint_reliable is in_memory_point.footpoint_reliable
