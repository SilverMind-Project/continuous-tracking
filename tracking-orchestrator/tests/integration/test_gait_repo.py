"""Integration tests for PostgresGaitBoutRepository and PostgresGaitDailyRepository.

Proves InMemory and Postgres behave identically across the full repository
interface.  Runs against a real testcontainer (migrated schema).
Marked @pytest.mark.integration — skipped by make check, included by make ci.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import asyncpg
import pytest

from app.storage.gait import InMemoryGaitBoutRepository, InMemoryGaitDailyRepository
from app.storage.postgres.gait_repo import PostgresGaitBoutRepository, PostgresGaitDailyRepository
from app.trajectory.gait import GaitDailyRecord, WalkingBout

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


# ---------------------------------------------------------------------------
# GaitDailyRepository parity
# ---------------------------------------------------------------------------

_DATE_1 = date(2026, 1, 14)
_DATE_2 = date(2026, 1, 15)
_DATE_3 = date(2026, 1, 16)


def _daily(identity_id: str, local_date: date, bout_count: int = 4) -> GaitDailyRecord:
    return GaitDailyRecord(
        identity_id=identity_id,
        local_date=local_date,
        bout_count=bout_count,
        total_walking_s=bout_count * 30.0,
        total_distance_m=bout_count * 20.0,
        median_speed_m_s=0.65,
        mad_speed_m_s=0.05,
        p95_speed_m_s=0.85,
        sample_bout_ids=[str(uuid.uuid4()) for _ in range(bout_count)],
        computed_at=_T0,
    )


@pytest.fixture
def inmemory_daily_repo() -> InMemoryGaitDailyRepository:
    return InMemoryGaitDailyRepository()


@pytest.fixture
async def postgres_daily_repo(db_pool: asyncpg.Pool) -> PostgresGaitDailyRepository:
    async with db_pool.acquire() as conn:
        for identity_id in {ALICE, BOB}:
            await _ensure_identity(conn, identity_id)
    return PostgresGaitDailyRepository(db_pool)


DAILY_SCENARIO = [
    _daily(ALICE, _DATE_1, bout_count=5),
    _daily(ALICE, _DATE_2, bout_count=3),
    _daily(BOB, _DATE_2, bout_count=6),
    _daily(BOB, _DATE_3, bout_count=4),
]


async def _seed_daily(
    repo: InMemoryGaitDailyRepository | PostgresGaitDailyRepository,
) -> None:
    for record in DAILY_SCENARIO:
        await repo.upsert_day(record)


@pytest.mark.integration
class TestGaitDailyRepoParity:
    """InMemory and Postgres must return identical results for every query."""

    async def test_list_all_for_identity(
        self,
        inmemory_daily_repo: InMemoryGaitDailyRepository,
        postgres_daily_repo: PostgresGaitDailyRepository,
    ) -> None:
        await _seed_daily(inmemory_daily_repo)
        await _seed_daily(postgres_daily_repo)

        mem = await inmemory_daily_repo.list_days(ALICE)
        pg = await postgres_daily_repo.list_days(ALICE)

        assert len(mem) == len(pg) == 2
        assert all(r.identity_id == ALICE for r in mem)
        assert all(r.identity_id == ALICE for r in pg)

    async def test_date_filter_since(
        self,
        inmemory_daily_repo: InMemoryGaitDailyRepository,
        postgres_daily_repo: PostgresGaitDailyRepository,
    ) -> None:
        await _seed_daily(inmemory_daily_repo)
        await _seed_daily(postgres_daily_repo)

        mem = await inmemory_daily_repo.list_days(ALICE, since=_DATE_2)
        pg = await postgres_daily_repo.list_days(ALICE, since=_DATE_2)

        assert len(mem) == len(pg) == 1
        assert mem[0].local_date == _DATE_2
        assert pg[0].local_date == _DATE_2

    async def test_date_filter_until(
        self,
        inmemory_daily_repo: InMemoryGaitDailyRepository,
        postgres_daily_repo: PostgresGaitDailyRepository,
    ) -> None:
        await _seed_daily(inmemory_daily_repo)
        await _seed_daily(postgres_daily_repo)

        mem = await inmemory_daily_repo.list_days(ALICE, until=_DATE_1)
        pg = await postgres_daily_repo.list_days(ALICE, until=_DATE_1)

        assert len(mem) == len(pg) == 1
        assert mem[0].local_date == _DATE_1
        assert pg[0].local_date == _DATE_1

    async def test_sorted_oldest_first(
        self,
        inmemory_daily_repo: InMemoryGaitDailyRepository,
        postgres_daily_repo: PostgresGaitDailyRepository,
    ) -> None:
        await _seed_daily(inmemory_daily_repo)
        await _seed_daily(postgres_daily_repo)

        for repo in (inmemory_daily_repo, postgres_daily_repo):
            rows = await repo.list_days(ALICE)
            dates = [r.local_date for r in rows]
            assert dates == sorted(dates), "list_days must return oldest-first"

    async def test_upsert_overwrites(
        self,
        inmemory_daily_repo: InMemoryGaitDailyRepository,
        postgres_daily_repo: PostgresGaitDailyRepository,
    ) -> None:
        original = _daily(ALICE, _DATE_1, bout_count=5)
        updated = _daily(ALICE, _DATE_1, bout_count=9)

        for repo in (inmemory_daily_repo, postgres_daily_repo):
            await repo.upsert_day(original)
            await repo.upsert_day(updated)

        mem = await inmemory_daily_repo.list_days(ALICE, since=_DATE_1, until=_DATE_1)
        pg = await postgres_daily_repo.list_days(ALICE, since=_DATE_1, until=_DATE_1)

        assert len(mem) == len(pg) == 1
        assert mem[0].bout_count == 9
        assert pg[0].bout_count == 9

    async def test_sample_bout_ids_round_trip(
        self,
        inmemory_daily_repo: InMemoryGaitDailyRepository,
        postgres_daily_repo: PostgresGaitDailyRepository,
    ) -> None:
        ids = [str(uuid.uuid4()) for _ in range(3)]
        record = GaitDailyRecord(
            identity_id=ALICE,
            local_date=_DATE_3,
            bout_count=3,
            total_walking_s=90.0,
            total_distance_m=60.0,
            median_speed_m_s=0.7,
            mad_speed_m_s=0.04,
            p95_speed_m_s=0.9,
            sample_bout_ids=ids,
            computed_at=_T0,
        )
        await inmemory_daily_repo.upsert_day(record)
        await postgres_daily_repo.upsert_day(record)

        mem = await inmemory_daily_repo.list_days(ALICE, since=_DATE_3)
        pg = await postgres_daily_repo.list_days(ALICE, since=_DATE_3)

        assert mem[0].sample_bout_ids == ids
        assert pg[0].sample_bout_ids == ids
