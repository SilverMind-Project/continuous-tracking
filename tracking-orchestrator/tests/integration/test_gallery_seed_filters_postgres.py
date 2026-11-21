"""Gallery provenance filters, Postgres half, plus migration 0009 down and up.

Runs the cases of ``tests/storage/_gallery_seed_cases.py`` against the session testcontainer so a
divergence between peers cannot hide behind a divergent fixture. Marked integration.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from app.storage.migrations import MigrationRunner
from app.storage.postgres.gallery_repo import PostgresGalleryRepository
from tests.storage import _gallery_seed_cases as cases
from tests.storage._gallery_seed_fixtures import FILTER_CASES

pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _repo(db_pool: Any) -> PostgresGalleryRepository:
    repo = PostgresGalleryRepository(db_pool)
    await cases.seed(repo)
    return repo


@pytest.mark.parametrize(("exclude", "seen_before", "expected"), FILTER_CASES)
async def test_search_similar_provenance_filters(
    db_pool: Any, exclude: frozenset[str], seen_before: datetime | None, expected: frozenset[str]
) -> None:
    await cases.assert_search_filter(await _repo(db_pool), exclude, seen_before, expected)


@pytest.mark.parametrize(("exclude", "seen_before", "expected"), FILTER_CASES)
async def test_tracklet_read_provenance_filters(
    db_pool: Any, exclude: frozenset[str], seen_before: datetime | None, expected: frozenset[str]
) -> None:
    await cases.assert_tracklet_filter(await _repo(db_pool), exclude, seen_before, expected)


async def test_seed_source_reaches_the_vote_path(db_pool: Any) -> None:
    await cases.assert_seed_source_reaches_vote_path(await _repo(db_pool))


async def test_count_seeded_entries(db_pool: Any) -> None:
    await cases.assert_count_seeded(await _repo(db_pool))


async def test_review_queue_seed_source_filter(db_pool: Any) -> None:
    await cases.assert_review_seed_source_filter(await _repo(db_pool))


async def test_seed_source_check_rejects_unknown_value(db_pool: asyncpg.Pool) -> None:
    await _repo(db_pool)
    async with db_pool.acquire() as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                "UPDATE continuous_tracking.reid_gallery SET seed_source = 'guess' "
                "WHERE identity_id = 'seed-member'"
            )


async def test_migration_0009_down_and_up_round_trip(db_pool: asyncpg.Pool) -> None:
    runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
    steps = sum(1 for p in _MIGRATIONS_DIR.glob("*.up.sql") if int(p.name.split("_", 1)[0]) >= 9)
    column_sql = (
        "SELECT count(*) FROM information_schema.columns WHERE table_schema = "
        "'continuous_tracking' AND table_name = 'reid_gallery' AND column_name = 'seed_source'"
    )
    try:
        assert await runner.rollback(steps) == steps
        async with db_pool.acquire() as conn:
            assert await conn.fetchval(column_sql) == 0
    finally:
        await runner.migrate()
    async with db_pool.acquire() as conn:
        assert await conn.fetchval(column_sql) == 1


async def test_state_counts_by_seed_source(db_pool: Any) -> None:
    await cases.assert_state_counts_by_seed_source(await _repo(db_pool))
