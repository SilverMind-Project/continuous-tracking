"""Integration tests for PostgresGaitBoutRepository.

Proves InMemory and Postgres behave identically across the full list_bouts
interface.  Runs against a real testcontainer (migrated schema).
Marked @pytest.mark.integration — skipped by make check, included by make ci.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from app.storage.gait import InMemoryGaitBoutRepository
from app.storage.postgres.gait_repo import PostgresGaitBoutRepository
from app.trajectory.gait import WalkingBout

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)


def _t(seconds: float) -> datetime:
    return _T0 + timedelta(seconds=seconds)


def _bout(
    identity_id: str,
    offset_s: float,
    duration_s: float = 10.0,
    speed: float = 0.7,
) -> WalkingBout:
    started = _t(offset_s)
    return WalkingBout(
        identity_id=identity_id,
        started_at=started,
        ended_at=started + timedelta(seconds=duration_s),
        duration_s=duration_s,
        distance_m=speed * duration_s,
        median_speed_m_s=speed,
        p95_speed_m_s=speed + 0.1,
        sample_count=int(duration_s),
        rooms=["hallway"],
    )


ALICE = "test-gait-alice"
BOB = "test-gait-bob"

SCENARIO_BOUTS = [
    _bout(ALICE, 0),
    _bout(ALICE, 60),
    _bout(ALICE, 120),
    _bout(BOB, 30),
    _bout(BOB, 90),
]


async def _ensure_identity(conn: asyncpg.Connection, identity_id: str) -> None:
    await conn.execute(
        """
        INSERT INTO continuous_tracking.identities (identity_id, display_name)
        VALUES ($1, $1)
        ON CONFLICT (identity_id) DO NOTHING
        """,
        identity_id,
    )


@pytest.fixture
def inmemory_repo() -> InMemoryGaitBoutRepository:
    return InMemoryGaitBoutRepository()


@pytest.fixture
async def postgres_repo(db_pool: asyncpg.Pool) -> PostgresGaitBoutRepository:
    async with db_pool.acquire() as conn:
        for identity_id in {ALICE, BOB}:
            await _ensure_identity(conn, identity_id)
    return PostgresGaitBoutRepository(db_pool)


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------


async def _seed(repo: InMemoryGaitBoutRepository | PostgresGaitBoutRepository) -> None:
    for bout in SCENARIO_BOUTS:
        await repo.upsert_bout(bout)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGaitBoutRepoParity:
    """InMemory and Postgres must return identical results for every query."""

    async def test_list_all(
        self,
        inmemory_repo: InMemoryGaitBoutRepository,
        postgres_repo: PostgresGaitBoutRepository,
    ) -> None:
        await _seed(inmemory_repo)
        await _seed(postgres_repo)

        mem = await inmemory_repo.list_bouts(limit=100)
        pg = await postgres_repo.list_bouts(limit=100)

        assert len(mem) == len(pg) == len(SCENARIO_BOUTS)

    async def test_filter_by_identity(
        self,
        inmemory_repo: InMemoryGaitBoutRepository,
        postgres_repo: PostgresGaitBoutRepository,
    ) -> None:
        await _seed(inmemory_repo)
        await _seed(postgres_repo)

        mem = await inmemory_repo.list_bouts(identity_id=ALICE)
        pg = await postgres_repo.list_bouts(identity_id=ALICE)

        assert len(mem) == len(pg) == 3
        assert all(b.identity_id == ALICE for b in mem)
        assert all(b.identity_id == ALICE for b in pg)

    async def test_filter_after(
        self,
        inmemory_repo: InMemoryGaitBoutRepository,
        postgres_repo: PostgresGaitBoutRepository,
    ) -> None:
        await _seed(inmemory_repo)
        await _seed(postgres_repo)

        cutoff = _t(50)
        mem = await inmemory_repo.list_bouts(after=cutoff, limit=100)
        pg = await postgres_repo.list_bouts(after=cutoff, limit=100)

        assert len(mem) == len(pg)
        assert all(b.started_at >= cutoff for b in mem)
        assert all(b.started_at >= cutoff for b in pg)

    async def test_idempotent_upsert(
        self,
        inmemory_repo: InMemoryGaitBoutRepository,
        postgres_repo: PostgresGaitBoutRepository,
    ) -> None:
        """Processing the same bout twice produces exactly one row."""
        bout = SCENARIO_BOUTS[0]
        await inmemory_repo.upsert_bout(bout)
        await inmemory_repo.upsert_bout(bout)
        await postgres_repo.upsert_bout(bout)
        await postgres_repo.upsert_bout(bout)

        mem = await inmemory_repo.list_bouts(identity_id=ALICE)
        pg = await postgres_repo.list_bouts(identity_id=ALICE)
        assert len(mem) == len(pg) == 1

    async def test_sorted_newest_first(
        self,
        inmemory_repo: InMemoryGaitBoutRepository,
        postgres_repo: PostgresGaitBoutRepository,
    ) -> None:
        await _seed(inmemory_repo)
        await _seed(postgres_repo)

        mem = await inmemory_repo.list_bouts(identity_id=ALICE)
        pg = await postgres_repo.list_bouts(identity_id=ALICE)

        for bouts in (mem, pg):
            times = [b.started_at for b in bouts]
            assert times == sorted(times, reverse=True), "not sorted newest first"

    async def test_rooms_round_trip(
        self,
        inmemory_repo: InMemoryGaitBoutRepository,
        postgres_repo: PostgresGaitBoutRepository,
    ) -> None:
        bout = WalkingBout(
            identity_id=ALICE,
            started_at=_t(200),
            ended_at=_t(215),
            duration_s=15.0,
            distance_m=9.0,
            median_speed_m_s=0.6,
            p95_speed_m_s=0.8,
            sample_count=15,
            rooms=["bedroom", "hallway", "living_room"],
        )
        await inmemory_repo.upsert_bout(bout)
        await postgres_repo.upsert_bout(bout)

        mem = await inmemory_repo.list_bouts(identity_id=ALICE)
        pg = await postgres_repo.list_bouts(identity_id=ALICE)

        assert mem[0].rooms == bout.rooms
        assert pg[0].rooms == bout.rooms
