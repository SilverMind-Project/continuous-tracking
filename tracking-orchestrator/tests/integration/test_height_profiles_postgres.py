"""Postgres integration tests for height profiles, PH height samples, and attested sample joins."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from app.domain import IdentityHeightProfile, PersonHypothesis, PHHeightSample
from app.services.height_profile_learner import PostgresHeightSampleSource
from app.storage.height_profiles import (
    InMemoryIdentityHeightProfileRepository,
    InMemoryPHHeightSampleRepository,
)
from app.storage.migrations import MigrationRunner
from app.storage.postgres.height_profile_repo import (
    PostgresIdentityHeightProfileRepository,
    PostgresPHHeightSampleRepository,
)
from app.storage.postgres.ph_repo import PostgresPHRepository

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _seed_identity(pool: asyncpg.Pool, identity_id: str, name: str = "Resident") -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO continuous_tracking.identities (identity_id, display_name, is_active)
            VALUES ($1, $2, TRUE)
            """,
            identity_id,
            name,
        )


async def _seed_ph(pool: asyncpg.Pool, ph_id: str, now: datetime) -> None:
    ph_repo = PostgresPHRepository(pool)
    ph = PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
        ),
        born_at=now - timedelta(minutes=10),
        last_seen_at=now,
        last_seen_camera="cam-a",
        observation_count=5,
        height_estimate_m=1.63,
        height_sigma_m=0.045,
        active_cameras=frozenset({"cam-a"}),
    )
    await ph_repo.save(ph)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_identity_height_profile_repo_parity(db_pool: asyncpg.Pool) -> None:
    now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    id_a = str(uuid4())
    id_b = str(uuid4())
    await _seed_identity(db_pool, id_a, "Alice")
    await _seed_identity(db_pool, id_b, "Bob")

    pg_repo = PostgresIdentityHeightProfileRepository(db_pool)
    mem_repo = InMemoryIdentityHeightProfileRepository()

    for repo in (pg_repo, mem_repo):
        # 1. Operator insert succeeds when no profile exists.
        assert (
            await repo.upsert_operator_if_not_learned(
                id_a, height_m=1.58, height_sigma_m=0.05, updated_at=now
            )
            is True
        )
        p_a = await repo.get(id_a)
        assert p_a is not None
        assert p_a.source == "operator"
        assert p_a.height_m == pytest.approx(1.58)

        # 2. Learned upsert replaces operator profile.
        learned = IdentityHeightProfile(
            identity_id=id_a,
            height_m=1.55,
            height_sigma_m=0.03,
            source="learned",
            samples=240,
            cameras=2,
            updated_at=now + timedelta(minutes=5),
        )
        await repo.upsert(learned)

        # 3. Subsequent operator upsert MUST NOT overwrite learned.
        assert (
            await repo.upsert_operator_if_not_learned(
                id_a, height_m=1.62, height_sigma_m=0.05, updated_at=now + timedelta(minutes=10)
            )
            is False
        )
        p_a_after = await repo.get(id_a)
        assert p_a_after == learned

        # 4. delete_operator_profile never removes learned, but removes operator.
        assert await repo.delete_operator_profile(id_a) is False
        assert (
            await repo.upsert_operator_if_not_learned(
                id_b, height_m=1.80, height_sigma_m=0.05, updated_at=now
            )
            is True
        )
        assert await repo.delete_operator_profile(id_b) is True
        assert await repo.get(id_b) is None

        # 5. delete() removes learned on identity purge.
        assert await repo.delete(id_a) is True
        assert await repo.list_all() == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ph_height_sample_repo_parity_and_ph_sigma_roundtrip(
    db_pool: asyncpg.Pool,
) -> None:
    now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    ph1 = str(uuid4())
    ph2 = str(uuid4())
    await _seed_ph(db_pool, ph1, now)
    await _seed_ph(db_pool, ph2, now)

    # Verify PostgresPHRepository round-trips height_sigma_m cleanly.
    ph_repo = PostgresPHRepository(db_pool)
    loaded = await ph_repo.get(ph1)
    assert loaded is not None
    assert loaded.height_estimate_m == pytest.approx(1.63)
    assert loaded.height_sigma_m == pytest.approx(0.045)

    rows = [
        PHHeightSample(
            ph_id=ph1,
            camera_id="cam-a",
            observed_at=now - timedelta(seconds=30),
            height_m=1.62,
            height_sigma_m=0.05,
        ),
        PHHeightSample(
            ph_id=ph1,
            camera_id="cam-b",
            observed_at=now - timedelta(seconds=20),
            height_m=1.64,
            height_sigma_m=0.04,
        ),
        PHHeightSample(
            ph_id=ph2,
            camera_id="cam-a",
            observed_at=now - timedelta(seconds=10),
            height_m=1.81,
            height_sigma_m=0.06,
        ),
    ]

    pg_samples = PostgresPHHeightSampleRepository(db_pool)
    mem_samples = InMemoryPHHeightSampleRepository()
    for repo in (pg_samples, mem_samples):
        await repo.append_many(rows)
        all_rows = await repo.list_between(now - timedelta(minutes=1), now)
        assert all_rows == rows
        ph1_rows = await repo.list_between(now - timedelta(minutes=1), now, ph_id=ph1)
        assert ph1_rows == rows[:2]
        cam_a_rows = await repo.list_between(now - timedelta(minutes=1), now, camera_id="cam-a")
        assert cam_a_rows == [rows[0], rows[2]]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_height_sample_source_filters_high_trust_intervals(
    db_pool: asyncpg.Pool,
) -> None:
    """Only trajectory-grade unrevoked non-swap wearables and conflict-free direct faces qualify."""
    now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
    id_alice = str(uuid4())
    id_bob = str(uuid4())
    await _seed_identity(db_pool, id_alice, "Alice")
    await _seed_identity(db_pool, id_bob, "Bob")

    ph_good_wear = str(uuid4())
    ph_revoked_wear = str(uuid4())
    ph_room_wear = str(uuid4())
    ph_direct_face = str(uuid4())
    ph_conflict_face = str(uuid4())
    for ph_id in (ph_good_wear, ph_revoked_wear, ph_room_wear, ph_direct_face, ph_conflict_face):
        await _seed_ph(db_pool, ph_id, now)

    sample_repo = PostgresPHHeightSampleRepository(db_pool)
    t_sample = now - timedelta(minutes=2)
    await sample_repo.append_many(
        [
            PHHeightSample(
                ph_id=ph_good_wear,
                camera_id="cam-a",
                observed_at=t_sample,
                height_m=1.55,
                height_sigma_m=0.04,
            ),
            PHHeightSample(
                ph_id=ph_revoked_wear,
                camera_id="cam-a",
                observed_at=t_sample,
                height_m=1.99,
                height_sigma_m=0.04,
            ),
            PHHeightSample(
                ph_id=ph_room_wear,
                camera_id="cam-a",
                observed_at=t_sample,
                height_m=1.98,
                height_sigma_m=0.04,
            ),
            PHHeightSample(
                ph_id=ph_direct_face,
                camera_id="cam-b",
                observed_at=t_sample,
                height_m=1.80,
                height_sigma_m=0.05,
            ),
            PHHeightSample(
                ph_id=ph_conflict_face,
                camera_id="cam-b",
                observed_at=t_sample,
                height_m=1.97,
                height_sigma_m=0.05,
            ),
        ]
    )

    async with db_pool.acquire() as conn:
        # 1. Qualifying trajectory-grade wearable for Alice.
        await conn.execute(
            """
            INSERT INTO continuous_tracking.wearable_associations
                (segment_id, ph_id, tag_id, identity_id, started_at, ended_at,
                 attestation_grade, window_llr, margin_llr, matched_fixes,
                 swap_suspected, revoked_at, revoked_by, revoke_reason)
            VALUES ($1::uuid, $2::uuid, 'tag-1', $3, $4, $5,
                    'trajectory', 12.0, 8.0, 10, FALSE, NULL, NULL, NULL)
            """,
            str(uuid4()),
            ph_good_wear,
            id_alice,
            t_sample - timedelta(minutes=1),
            t_sample + timedelta(minutes=1),
        )
        # 2. Revoked wearable -> MUST be excluded.
        await conn.execute(
            """
            INSERT INTO continuous_tracking.wearable_associations
                (segment_id, ph_id, tag_id, identity_id, started_at, ended_at,
                 attestation_grade, window_llr, margin_llr, matched_fixes,
                 swap_suspected, revoked_at, revoked_by, revoke_reason)
            VALUES ($1::uuid, $2::uuid, 'tag-2', $3, $4, $5,
                    'trajectory', 12.0, 8.0, 10, FALSE, $6, 'admin', 'operator')
            """,
            str(uuid4()),
            ph_revoked_wear,
            id_alice,
            t_sample - timedelta(minutes=1),
            t_sample + timedelta(minutes=1),
            now,
        )
        # 3. Room-grade wearable -> MUST be excluded.
        await conn.execute(
            """
            INSERT INTO continuous_tracking.wearable_associations
                (segment_id, ph_id, tag_id, identity_id, started_at, ended_at,
                 attestation_grade, window_llr, margin_llr, matched_fixes,
                 swap_suspected, revoked_at, revoked_by, revoke_reason)
            VALUES ($1::uuid, $2::uuid, 'tag-3', $3, $4, $5,
                    'room', 6.0, 4.0, 6, FALSE, NULL, NULL, NULL)
            """,
            str(uuid4()),
            ph_room_wear,
            id_alice,
            t_sample - timedelta(minutes=1),
            t_sample + timedelta(minutes=1),
        )
        # 4. Conflict-free direct face decision for Bob within 5 s -> MUST be included.
        await conn.execute(
            """
            INSERT INTO continuous_tracking.identity_decisions
                (decision_id, ph_id, captured_at, effective_identity_id,
                 authority, decision_source, conflict_kind, top_probability)
            VALUES ($1::uuid, $2::uuid, $3, $4, 'direct_face', 'resolver', NULL, 0.95)
            """,
            str(uuid4()),
            ph_direct_face,
            t_sample + timedelta(seconds=2),
            id_bob,
        )
        # 5. Conflicted face decision -> MUST be excluded.
        await conn.execute(
            """
            INSERT INTO continuous_tracking.identity_decisions
                (decision_id, ph_id, captured_at, effective_identity_id,
                 authority, decision_source, conflict_kind, top_probability)
            VALUES ($1::uuid, $2::uuid, $3, $4, 'direct_face', 'resolver', 'face_vs_reid', 0.50)
            """,
            str(uuid4()),
            ph_conflict_face,
            t_sample + timedelta(seconds=1),
            id_bob,
        )

    source = PostgresHeightSampleSource(db_pool)
    attested = await source.list_attested_samples(
        since=now - timedelta(hours=1),
        until=now + timedelta(minutes=1),
    )
    got = {(s.identity_id, s.camera_id, round(s.height_m, 2)) for s in attested}
    assert got == {
        (id_alice, "cam-a", 1.55),
        (id_bob, "cam-b", 1.80),
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_migration_0013_down_and_up_roundtrip(db_pool: asyncpg.Pool) -> None:
    runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
    try:
        applied = (await runner.status())["applied"]
        above_0012 = sum(1 for name in applied if name >= "0013_identity_height_profiles")
        await runner.rollback(above_0012)

        async with db_pool.acquire() as conn:
            tbl = await conn.fetchval(
                "SELECT to_regclass('continuous_tracking.identity_height_profiles')"
            )
            assert tbl is None
            col = await conn.fetchval(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'continuous_tracking'
                  AND table_name = 'person_hypotheses'
                  AND column_name = 'height_sigma_m'
                """
            )
            assert col is None
    finally:
        await runner.migrate()

    async with db_pool.acquire() as conn:
        tbl_after = await conn.fetchval(
            "SELECT to_regclass('continuous_tracking.identity_height_profiles')"
        )
        assert tbl_after is not None
        samples_tbl = await conn.fetchval(
            "SELECT to_regclass('continuous_tracking.ph_height_samples')"
        )
        assert samples_tbl is not None
        # TimescaleDB 14-day retention policy is registered on ph_height_samples.
        policy_count = await conn.fetchval(
            """
            SELECT count(*) FROM timescaledb_information.jobs
            WHERE proc_name = 'policy_retention'
              AND hypertable_schema = 'continuous_tracking'
              AND hypertable_name = 'ph_height_samples'
            """
        )
        assert int(policy_count) >= 1
