"""Image-only evaluation totals: Postgres and InMemory parity, cascade, and migration 0010.

Marked integration: runs against the session testcontainer in make ci.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import asyncpg
import pytest

from app.domain import Identity, ImageOnlyDailyTotal, WearableAttestationGrade
from app.storage.image_only_evaluation import (
    ImageOnlyEvaluationRepository,
    InMemoryImageOnlyEvaluationRepository,
)
from app.storage.migrations import MigrationRunner
from app.storage.postgres.gallery_repo import PostgresGalleryRepository
from app.storage.postgres.image_only_evaluation_repo import PostgresImageOnlyEvaluationRepository

pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_T0 = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
_DAY = _T0.date()


def _total(
    identity_id: str, arm: str, *, day: date = _DAY, grade: str = "trajectory", s: float = 1.0
) -> ImageOnlyDailyTotal:
    return ImageOnlyDailyTotal(
        day=day,
        identity_id=identity_id,
        arm=arm,
        attestation_grade=WearableAttestationGrade(grade),
        correct_s=s,
        wrong_s=2 * s,
        unknown_s=0.5,
    )


async def _exercise(repo: ImageOnlyEvaluationRepository) -> list[tuple[object, ...]]:
    await repo.add_totals(
        [
            _total("alice", "control"),
            _total("alice", "treatment", s=3.0),
            _total("alice", "control", grade="room"),
            _total("bob", "control", day=_DAY + timedelta(days=1)),
        ]
    )
    await repo.add_totals([_total("alice", "control")])
    rows = await repo.list_totals(since=_DAY, until=_DAY)
    removed = await repo.purge_identity("bob")
    remaining = await repo.list_totals()
    return [
        tuple(
            (
                r.day,
                r.identity_id,
                r.arm,
                str(r.attestation_grade),
                r.correct_s,
                r.wrong_s,
                r.unknown_s,
            )
            for r in rows
        ),
        (removed,),
        tuple(r.identity_id for r in remaining),
    ]


async def test_postgres_matches_in_memory(db_pool: asyncpg.Pool) -> None:
    gallery = PostgresGalleryRepository(db_pool)
    for uid in ("alice", "bob"):
        await gallery.upsert_identity(Identity(identity_id=uid, display_name=uid, enrolled_at=_T0))
    postgres = await _exercise(PostgresImageOnlyEvaluationRepository(db_pool))
    memory = await _exercise(InMemoryImageOnlyEvaluationRepository())
    assert postgres == memory
    # The additive upsert summed the repeated control row.
    assert (_DAY, "alice", "control", "trajectory", 2.0, 4.0, 1.0) in postgres[0]


async def test_identity_delete_cascades(db_pool: asyncpg.Pool) -> None:
    gallery = PostgresGalleryRepository(db_pool)
    await gallery.upsert_identity(Identity(identity_id="carol", display_name="c", enrolled_at=_T0))
    repo = PostgresImageOnlyEvaluationRepository(db_pool)
    await repo.add_totals([_total("carol", "treatment")])
    assert await gallery.purge_identity("carol") is True
    assert await repo.list_totals() == []


async def test_checks_reject_unknown_arm_and_negative_seconds(db_pool: asyncpg.Pool) -> None:
    gallery = PostgresGalleryRepository(db_pool)
    await gallery.upsert_identity(Identity(identity_id="dave", display_name="d", enrolled_at=_T0))
    repo = PostgresImageOnlyEvaluationRepository(db_pool)
    with pytest.raises(asyncpg.CheckViolationError):
        await repo.add_totals([_total("dave", "placebo")])
    with pytest.raises(asyncpg.CheckViolationError):
        await repo.add_totals([_total("dave", "control", s=-1.0)])


async def test_migration_0010_down_and_up_round_trip(db_pool: asyncpg.Pool) -> None:
    runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
    steps = sum(1 for p in _MIGRATIONS_DIR.glob("*.up.sql") if int(p.name.split("_", 1)[0]) >= 10)
    sql = "SELECT to_regclass('continuous_tracking.image_only_evaluation_daily') IS NOT NULL"
    try:
        assert await runner.rollback(steps) == steps
        async with db_pool.acquire() as conn:
            assert await conn.fetchval(sql) is False
    finally:
        await runner.migrate()
    async with db_pool.acquire() as conn:
        assert await conn.fetchval(sql) is True
