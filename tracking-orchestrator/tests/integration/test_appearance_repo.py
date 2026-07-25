"""Integration tests for PostgresDailyAppearanceRepo.

Proves InMemory and Postgres behave identically across the full repository
interface. Runs against a real testcontainer (migrated schema).
Marked @pytest.mark.integration -- skipped by make check, included by make ci.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import asyncpg
import pytest

from app.storage.appearance import InMemoryDailyAppearanceRepo
from app.storage.postgres.appearance_repo import PostgresDailyAppearanceRepo
from app.trajectory.appearance_profile import DailyAppearanceProfile

_T0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)

ALICE = "test-appearance-alice"
BOB = "test-appearance-bob"

_DAY_1 = date(2026, 1, 14)
_DAY_2 = date(2026, 1, 15)
_DAY_3 = date(2026, 1, 16)


def _profile(
    identity_id: str,
    day: date,
    *,
    centroid: tuple[float, ...] = (0.6, 0.8),
    sample_count: int = 5,
    mean_quality: float = 0.7,
    best_keyframe_objects: tuple[str, ...] = ("cam01/frame_1.jpg",),
) -> DailyAppearanceProfile:
    return DailyAppearanceProfile(
        identity_id=identity_id,
        day=day,
        centroid=centroid,
        sample_count=sample_count,
        mean_quality=mean_quality,
        best_keyframe_objects=best_keyframe_objects,
        created_at=_T0,
    )


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
def inmemory_repo() -> InMemoryDailyAppearanceRepo:
    return InMemoryDailyAppearanceRepo()


@pytest.fixture
async def postgres_repo(db_pool: asyncpg.Pool) -> PostgresDailyAppearanceRepo:
    async with db_pool.acquire() as conn:
        for identity_id in {ALICE, BOB}:
            await _ensure_identity(conn, identity_id)
    return PostgresDailyAppearanceRepo(db_pool)


SCENARIO = [
    _profile(ALICE, _DAY_1, sample_count=5),
    _profile(ALICE, _DAY_2, sample_count=7),
    _profile(BOB, _DAY_2, sample_count=6),
    _profile(BOB, _DAY_3, sample_count=4),
]


async def _seed(
    repo: InMemoryDailyAppearanceRepo | PostgresDailyAppearanceRepo,
) -> None:
    for profile in SCENARIO:
        await repo.upsert_profile(profile)


@pytest.mark.integration
class TestDailyAppearanceRepoParity:
    """InMemory and Postgres must return identical results for every query."""

    async def test_get_profile(
        self,
        inmemory_repo: InMemoryDailyAppearanceRepo,
        postgres_repo: PostgresDailyAppearanceRepo,
    ) -> None:
        await _seed(inmemory_repo)
        await _seed(postgres_repo)

        mem = await inmemory_repo.get_profile(ALICE, _DAY_1)
        pg = await postgres_repo.get_profile(ALICE, _DAY_1)

        assert mem is not None and pg is not None
        assert mem.sample_count == pg.sample_count == 5
        assert mem.centroid == (0.6, 0.8)
        # FLOAT4[] is single precision (mirrors ph.gallery_mean's storage type).
        assert pg.centroid == pytest.approx((0.6, 0.8), abs=1e-6)

    async def test_get_profile_missing_returns_none(
        self,
        inmemory_repo: InMemoryDailyAppearanceRepo,
        postgres_repo: PostgresDailyAppearanceRepo,
    ) -> None:
        assert await inmemory_repo.get_profile(ALICE, _DAY_3) is None
        assert await postgres_repo.get_profile(ALICE, _DAY_3) is None

    async def test_list_days_for_identity(
        self,
        inmemory_repo: InMemoryDailyAppearanceRepo,
        postgres_repo: PostgresDailyAppearanceRepo,
    ) -> None:
        await _seed(inmemory_repo)
        await _seed(postgres_repo)

        mem = await inmemory_repo.list_days(ALICE)
        pg = await postgres_repo.list_days(ALICE)

        assert len(mem) == len(pg) == 2
        assert all(p.identity_id == ALICE for p in mem)
        assert all(p.identity_id == ALICE for p in pg)

    async def test_list_days_since_filter(
        self,
        inmemory_repo: InMemoryDailyAppearanceRepo,
        postgres_repo: PostgresDailyAppearanceRepo,
    ) -> None:
        await _seed(inmemory_repo)
        await _seed(postgres_repo)

        mem = await inmemory_repo.list_days(ALICE, since_day=_DAY_2)
        pg = await postgres_repo.list_days(ALICE, since_day=_DAY_2)

        assert len(mem) == len(pg) == 1
        assert mem[0].day == pg[0].day == _DAY_2

    async def test_sorted_oldest_first(
        self,
        inmemory_repo: InMemoryDailyAppearanceRepo,
        postgres_repo: PostgresDailyAppearanceRepo,
    ) -> None:
        await _seed(inmemory_repo)
        await _seed(postgres_repo)

        for repo in (inmemory_repo, postgres_repo):
            rows = await repo.list_days(ALICE)
            days = [r.day for r in rows]
            assert days == sorted(days), "list_days must return oldest-first"

    async def test_upsert_overwrites(
        self,
        inmemory_repo: InMemoryDailyAppearanceRepo,
        postgres_repo: PostgresDailyAppearanceRepo,
    ) -> None:
        original = _profile(ALICE, _DAY_1, sample_count=5)
        updated = _profile(ALICE, _DAY_1, sample_count=9)

        for repo in (inmemory_repo, postgres_repo):
            await repo.upsert_profile(original)
            await repo.upsert_profile(updated)

        mem = await inmemory_repo.get_profile(ALICE, _DAY_1)
        pg = await postgres_repo.get_profile(ALICE, _DAY_1)
        assert mem is not None and pg is not None
        assert mem.sample_count == pg.sample_count == 9

    async def test_best_keyframe_objects_round_trip(
        self,
        inmemory_repo: InMemoryDailyAppearanceRepo,
        postgres_repo: PostgresDailyAppearanceRepo,
    ) -> None:
        objects = ("cam01/a.jpg", "cam03/b.jpg")
        profile = _profile(ALICE, _DAY_3, best_keyframe_objects=objects)

        await inmemory_repo.upsert_profile(profile)
        await postgres_repo.upsert_profile(profile)

        mem = await inmemory_repo.get_profile(ALICE, _DAY_3)
        pg = await postgres_repo.get_profile(ALICE, _DAY_3)
        assert mem is not None and pg is not None
        assert mem.best_keyframe_objects == objects
        assert pg.best_keyframe_objects == objects
