"""PostgresWearableAssociationRepository parity, purge, cascade, and migration 0008 tests.

Runs against the session testcontainer (migrated schema).
Marked @pytest.mark.integration: skipped by make check, included by make ci.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
import pytest

from app.domain import (
    Identity,
    PersonHypothesis,
    WearableAssociationSegment,
    WearableAttestationGrade,
)
from app.storage.migrations import MigrationRunner
from app.storage.postgres.gallery_repo import PostgresGalleryRepository
from app.storage.postgres.ph_repo import PostgresPHRepository
from app.storage.postgres.wearable_association_repo import PostgresWearableAssociationRepository
from app.storage.wearable_associations import InMemoryWearableAssociationRepository

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_T0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)


def _make_ph(
    ph_id: str, *, closed_at: datetime | None = None, last_seen_at: datetime = _T0
) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(2.0, 2.0, 0.0, 0.0),
        state_cov=(
            0.04,
            0.0,
            0.0,
            0.0,
            0.0,
            0.04,
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
        born_at=_T0 - timedelta(minutes=5),
        last_seen_at=last_seen_at,
        last_seen_camera="cam-1",
        observation_count=5,
        closed_at=closed_at,
    )


def _make_segment(
    *,
    ph_id: str,
    identity_id: str,
    tag_id: str = "tag-1",
    started_at: datetime = _T0,
    ended_at: datetime | None = None,
    grade: WearableAttestationGrade = WearableAttestationGrade.TRAJECTORY,
    mean_residual_m: float | None = 0.14,
    swap_suspected: bool = False,
    close_reason: str | None = None,
) -> WearableAssociationSegment:
    return WearableAssociationSegment(
        segment_id=str(uuid.uuid4()),
        ph_id=ph_id,
        tag_id=tag_id,
        identity_id=identity_id,
        started_at=started_at,
        ended_at=ended_at,
        attestation_grade=grade,
        window_llr=18.25,
        margin_llr=9.5,
        matched_fixes=9,
        mean_residual_m=mean_residual_m,
        swap_suspected=swap_suspected,
        close_reason=close_reason,
    )


async def _seed_identity_and_ph(
    pool: asyncpg.Pool,
    *,
    identity_id: str,
    ph_id: str,
    closed_at: datetime | None = None,
    last_seen_at: datetime = _T0,
) -> None:
    gallery = PostgresGalleryRepository(pool)
    ph_repo = PostgresPHRepository(pool)
    await gallery.upsert_identity(
        Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=_T0)
    )
    await ph_repo.save(_make_ph(ph_id, closed_at=closed_at, last_seen_at=last_seen_at))


@pytest.mark.integration
class TestWearableAssociationRepoParity:
    async def test_crud_upsert_and_filters_match_inmemory(self, db_pool: asyncpg.Pool) -> None:
        pg = PostgresWearableAssociationRepository(db_pool)
        mem = InMemoryWearableAssociationRepository()

        ph1 = str(uuid.uuid4())
        ph2 = str(uuid.uuid4())
        await _seed_identity_and_ph(db_pool, identity_id="alice-w", ph_id=ph1)
        await _seed_identity_and_ph(db_pool, identity_id="bob-w", ph_id=ph2)

        seg1 = _make_segment(
            ph_id=ph1,
            identity_id="alice-w",
            tag_id="tag-a",
            started_at=_T0,
            ended_at=None,
            grade=WearableAttestationGrade.NONE,
        )
        seg2 = _make_segment(
            ph_id=ph2,
            identity_id="bob-w",
            tag_id="tag-b",
            started_at=_T0 + timedelta(seconds=5),
            ended_at=_T0 + timedelta(seconds=65),
            grade=WearableAttestationGrade.ROOM,
            mean_residual_m=None,
            close_reason="ph_closed",
        )
        for seg in (seg1, seg2):
            await pg.save_segment(seg)
            await mem.save_segment(seg)

        # Update seg1 via upsert (grade promoted + closed).
        seg1_updated = replace(
            seg1,
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            window_llr=24.0,
            swap_suspected=True,
        )
        await pg.save_segment(seg1_updated)
        await mem.save_segment(seg1_updated)

        assert await pg.get_segment(seg1.segment_id) == await mem.get_segment(seg1.segment_id)
        assert await pg.list_by_ph(ph1) == await mem.list_by_ph(ph1)
        assert await pg.list_active() == await mem.list_active()
        assert await pg.list_recent(tag_id="tag-b") == await mem.list_recent(tag_id="tag-b")
        assert await pg.list_recent() == await mem.list_recent()

    async def test_check_constraints_reject_invalid_grade_and_times(
        self, db_pool: asyncpg.Pool
    ) -> None:
        ph_id = str(uuid.uuid4())
        await _seed_identity_and_ph(db_pool, identity_id="check-user", ph_id=ph_id)

        async with db_pool.acquire() as conn:
            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.wearable_associations (
                        segment_id, ph_id, tag_id, identity_id, started_at,
                        attestation_grade, window_llr, margin_llr, matched_fixes
                    ) VALUES ($1::uuid, $2::uuid, 'tag-x', 'check-user', $3, 'bogus', 10, 5, 5)
                    """,
                    str(uuid.uuid4()),
                    ph_id,
                    _T0,
                )

            with pytest.raises(asyncpg.CheckViolationError):
                await conn.execute(
                    """
                    INSERT INTO continuous_tracking.wearable_associations (
                        segment_id, ph_id, tag_id, identity_id, started_at, ended_at,
                        attestation_grade, window_llr, margin_llr, matched_fixes
                    ) VALUES ($1::uuid, $2::uuid, 'tag-x', 'check-user', $3, $4, 'none', 10, 5, 5)
                    """,
                    str(uuid.uuid4()),
                    ph_id,
                    _T0,
                    _T0 - timedelta(seconds=1),
                )

    async def test_identity_purge_and_unknown_ph_cascade_delete_rows(
        self, db_pool: asyncpg.Pool
    ) -> None:
        pg = PostgresWearableAssociationRepository(db_pool)
        gallery = PostgresGalleryRepository(db_pool)
        ph_repo = PostgresPHRepository(db_pool)

        # 1. GalleryRepository.purge_identity removes wearable_associations for that identity.
        ph1 = str(uuid.uuid4())
        await _seed_identity_and_ph(db_pool, identity_id="purge-me", ph_id=ph1)
        seg1 = _make_segment(ph_id=ph1, identity_id="purge-me", tag_id="tag-p1")
        await pg.save_segment(seg1)
        assert await pg.get_segment(seg1.segment_id) is not None

        assert await gallery.purge_identity("purge-me") is True
        assert await pg.get_segment(seg1.segment_id) is None

        # 2. PHMaintenanceService / purge_unknown_older_than cascades via ON DELETE CASCADE.
        old_time = _T0 - timedelta(days=10)
        ph_old = str(uuid.uuid4())
        await _seed_identity_and_ph(
            db_pool,
            identity_id="keep-user",
            ph_id=ph_old,
            closed_at=old_time,
            last_seen_at=old_time,
        )
        seg_old = _make_segment(
            ph_id=ph_old,
            identity_id="keep-user",
            tag_id="tag-old",
            started_at=old_time,
            ended_at=old_time,
        )
        await pg.save_segment(seg_old)
        deleted_phs = await ph_repo.purge_unknown_older_than(_T0 - timedelta(days=7))
        assert deleted_phs >= 1
        assert await pg.get_segment(seg_old.segment_id) is None

    async def test_migration_0008_down_and_up_round_trip(self, db_pool: asyncpg.Pool) -> None:
        runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
        steps = sum(
            1 for p in _MIGRATIONS_DIR.glob("*.up.sql") if int(p.name.split("_", 1)[0]) >= 8
        )
        try:
            assert await runner.rollback(steps) == steps
            async with db_pool.acquire() as conn:
                exists_after_down = await conn.fetchval(
                    "SELECT to_regclass('continuous_tracking.wearable_associations') IS NOT NULL"
                )
                assert exists_after_down is False
        finally:
            await runner.migrate()

        async with db_pool.acquire() as conn:
            exists_after_up = await conn.fetchval(
                "SELECT to_regclass('continuous_tracking.wearable_associations') IS NOT NULL"
            )
            assert exists_after_up is True
