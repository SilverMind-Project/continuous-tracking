"""PostgresFaceObservationStatsRepository parity with the InMemory peer, plus migrations 0002-0004.

Runs against the session testcontainer (migrated schema).
Marked @pytest.mark.integration: skipped by make check, included by make ci.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import asyncpg
import pytest

from app.domain import FaceGateOutcome
from app.storage.face_stats import InMemoryFaceObservationStatsRepository
from app.storage.migrations import MigrationRunner
from app.storage.postgres.face_stats_repo import PostgresFaceObservationStatsRepository
from tests.fixtures.face_stats import T0 as _T0
from tests.fixtures.face_stats import face_stat

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

_ROWS = [
    face_stat(0, "cam-1"),
    face_stat(5, "cam-2", FaceGateOutcome.DROPPED_MIN_SIMILARITY),
    replace(face_stat(9, "cam-1"), interocular_px=None, best_candidate_id=None),
    face_stat(3600, "cam-1"),
    face_stat(12, "cam-1", FaceGateOutcome.DROPPED_NOT_IN_HEAD_REGION),
    face_stat(13, "cam-2", FaceGateOutcome.DROPPED_AMBIGUOUS),
]


@pytest.mark.integration
class TestFaceStatsRepoParity:
    @pytest.mark.parametrize("camera_id", [None, "cam-1", "cam-2"])
    async def test_list_between_matches_inmemory(
        self, db_pool: asyncpg.Pool, camera_id: str | None
    ) -> None:
        postgres = PostgresFaceObservationStatsRepository(db_pool)
        inmemory = InMemoryFaceObservationStatsRepository()
        await postgres.append_many(_ROWS)
        await inmemory.append_many(_ROWS)

        window = (_T0, _T0 + timedelta(minutes=30))
        assert await postgres.list_between(*window, camera_id=camera_id) == (
            await inmemory.list_between(*window, camera_id=camera_id)
        )

    async def test_round_trip_preserves_every_field(self, db_pool: asyncpg.Pool) -> None:
        repo = PostgresFaceObservationStatsRepository(db_pool)
        await repo.append_many(_ROWS)
        stored = await repo.list_between(_T0, _T0 + timedelta(hours=2))
        assert stored == sorted(_ROWS, key=lambda r: r.observed_at)

    async def test_check_constraint_rejects_unknown_gate_outcome(
        self, db_pool: asyncpg.Pool
    ) -> None:
        async with db_pool.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.face_observation_stats (
                        observed_at, camera_id, detection_id, crop_width_px, crop_height_px,
                        face_width_px, face_height_px, interocular_px, det_score, yaw_deg,
                        pitch_deg, raw_similarity, recognition_state, best_candidate_id,
                        gate_outcome
                    ) VALUES ($1, 'cam-1', 'd', 10, 10, 1, 1, NULL, 0.9, 0, 0, 0.5,
                              'recognized', NULL, 'emitted_somehow')
                    """,
                    _T0,
                )


async def _face_stats_objects(conn: asyncpg.Connection) -> tuple[bool, int]:
    table_exists = await conn.fetchval(
        "SELECT to_regclass('continuous_tracking.face_observation_stats') IS NOT NULL"
    )
    retention_jobs = await conn.fetchval(
        """
        SELECT count(*) FROM timescaledb_information.jobs
        WHERE proc_name = 'policy_retention'
          AND hypertable_schema = 'continuous_tracking'
          AND hypertable_name = 'face_observation_stats'
        """
    )
    return bool(table_exists), int(retention_jobs)


@pytest.mark.integration
async def test_migration_0002_down_then_up(db_pool: asyncpg.Pool) -> None:
    runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
    async with db_pool.acquire() as conn:
        assert await _face_stats_objects(conn) == (True, 1)

    try:
        # 0003 (attribution outcomes) and 0004 (outcome vocabulary) sit on top of 0002.
        assert await runner.rollback(3) == 3
        async with db_pool.acquire() as conn:
            assert await _face_stats_objects(conn) == (False, 0)
    finally:
        # Restore the session schema for the remaining integration tests.
        await runner.migrate()

    async with db_pool.acquire() as conn:
        assert await _face_stats_objects(conn) == (True, 1)


async def _insert_outcome(conn: asyncpg.Connection, outcome: str) -> None:
    await conn.execute(
        """
        INSERT INTO continuous_tracking.face_observation_stats (
            observed_at, camera_id, detection_id, crop_width_px, crop_height_px,
            face_width_px, face_height_px, interocular_px, det_score, yaw_deg,
            pitch_deg, raw_similarity, recognition_state, best_candidate_id, gate_outcome
        ) VALUES ($1, 'cam-1', 'd', 10, 10, 1, 1, NULL, 0.9, 0, 0, 0.5,
                  'recognized', NULL, $2)
        """,
        _T0,
        outcome,
    )


@pytest.mark.integration
async def test_migration_0003_widens_and_restores_outcome_check(db_pool: asyncpg.Pool) -> None:
    runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
    async with db_pool.acquire() as conn:
        await _insert_outcome(conn, "dropped_ambiguous")

    try:
        # 0004 (outcome vocabulary) sits on top of 0003.
        assert await runner.rollback(2) == 2
        async with db_pool.acquire() as conn:
            # The down migration removed the row it could no longer represent.
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM continuous_tracking.face_observation_stats"
                )
                == 0
            )
            with pytest.raises(asyncpg.CheckViolationError):
                await _insert_outcome(conn, "dropped_not_in_head_region")
            await _insert_outcome(conn, "emitted")
    finally:
        await runner.migrate()

    async with db_pool.acquire() as conn:
        await _insert_outcome(conn, "dropped_not_in_head_region")


async def _outcomes(conn: asyncpg.Connection) -> list[str]:
    rows = await conn.fetch(
        "SELECT gate_outcome FROM continuous_tracking.face_observation_stats ORDER BY gate_outcome"
    )
    return [row["gate_outcome"] for row in rows]


@pytest.mark.integration
async def test_migration_0004_renames_the_similarity_floor_outcome(db_pool: asyncpg.Pool) -> None:
    runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
    async with db_pool.acquire() as conn:
        await _insert_outcome(conn, "dropped_min_similarity")
        await _insert_outcome(conn, "emitted")
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_outcome(conn, "dropped_min_confidence")

    try:
        assert await runner.rollback(1) == 1
        async with db_pool.acquire() as conn:
            assert await _outcomes(conn) == ["dropped_min_confidence", "emitted"]
            with pytest.raises(asyncpg.CheckViolationError):
                await _insert_outcome(conn, "dropped_min_similarity")
    finally:
        await runner.migrate()

    async with db_pool.acquire() as conn:
        assert await _outcomes(conn) == ["dropped_min_similarity", "emitted"]
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_outcome(conn, "dropped_min_confidence")
