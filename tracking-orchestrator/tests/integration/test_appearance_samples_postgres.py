"""Postgres integration tests for appearance_training_samples and migrations 0014/0015.

Proves InMemory/Postgres repository parity, ON DELETE CASCADE and purge_identity,
wearable association revocation, TimescaleDB 60-day retention policy registration,
and image_only_evaluation_daily classifier-arm CHECK constraint round-trip.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from app.domain import (
    AppearanceLabelSource,
    AppearanceTrainingSample,
    Identity,
    ImageOnlyDailyTotal,
    OrientationBin,
    PersonHypothesis,
    WearableAssociationSegment,
    WearableAttestationGrade,
)
from app.storage.appearance_samples import (
    AppearanceSampleRepository,
    InMemoryAppearanceSampleRepository,
)
from app.storage.migrations import MigrationRunner
from app.storage.postgres.appearance_sample_repo import PostgresAppearanceSampleRepository
from app.storage.postgres.gallery_repo import PostgresGalleryRepository
from app.storage.postgres.image_only_evaluation_repo import PostgresImageOnlyEvaluationRepository
from app.storage.postgres.ph_repo import PostgresPHRepository
from app.storage.postgres.wearable_association_repo import PostgresWearableAssociationRepository

pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_T0 = datetime(2026, 11, 21, 14, 0, 0, tzinfo=UTC)
_DAY0 = date(2026, 11, 21)


def _emb768(val: float) -> tuple[float, ...]:
    vec = [0.0] * 768
    vec[0] = val
    vec[1] = (1.0 - val * val) ** 0.5
    return tuple(vec)


def _sample(
    *,
    observed_at: datetime = _T0,
    day: date = _DAY0,
    identity_id: str | None = "alice",
    label_source: AppearanceLabelSource = AppearanceLabelSource.WEARABLE_ATTESTED,
    feature_version: str = "reid-solider",
    seed_association_id: str | None = None,
) -> AppearanceTrainingSample:
    return AppearanceTrainingSample(
        sample_id=str(uuid4()),
        observed_at=observed_at,
        day=day,
        identity_id=identity_id,
        label_source=label_source,
        camera_id="cam-living",
        orientation=OrientationBin.FRONT,
        quality=0.8125,
        feature_version=feature_version,
        embedding=_emb768(0.6),
        seed_association_id=seed_association_id,
    )


async def _exercise(
    repo: AppearanceSampleRepository,
    samples: list[AppearanceTrainingSample],
    assoc_id: str,
) -> tuple[object, ...]:
    inserted = await repo.append_many(samples)
    dup_inserted = await repo.append_many(samples[:1])
    listed_active = await repo.list_samples(feature_version="reid-solider")
    neg_count = await repo.count_negatives_for_day(_DAY0, feature_version="reid-solider")
    revoked = await repo.revoke_by_association(assoc_id, revoked_at=_T0 + timedelta(minutes=5))
    after_revoke = await repo.list_samples(feature_version="reid-solider")
    with_revoked = await repo.list_samples(feature_version="reid-solider", include_revoked=True)
    purged = await repo.purge_identity("bob")
    after_purge = await repo.list_samples(include_revoked=True)
    return (
        inserted,
        dup_inserted,
        [(s.sample_id, s.identity_id, str(s.label_source), s.quality) for s in listed_active],
        neg_count,
        revoked,
        [s.sample_id for s in after_revoke],
        len(with_revoked),
        purged,
        [s.identity_id for s in after_purge],
    )


async def test_appearance_sample_repo_postgres_inmemory_parity(db_pool: asyncpg.Pool) -> None:
    gallery = PostgresGalleryRepository(db_pool)
    for uid in ("alice", "bob"):
        await gallery.upsert_identity(Identity(identity_id=uid, display_name=uid, enrolled_at=_T0))

    ph_id = str(uuid4())
    await PostgresPHRepository(db_pool).save(
        PersonHypothesis(
            ph_id=ph_id,
            state_mean=(1.0, 2.0, 0.0, 0.0),
            state_cov=(0.2, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5),
            born_at=_T0 - timedelta(minutes=5),
            last_seen_at=_T0,
            last_seen_camera="cam-living",
            observation_count=5,
            active_cameras=frozenset({"cam-living"}),
        )
    )
    assoc_id = str(uuid4())
    await PostgresWearableAssociationRepository(db_pool).save_segment(
        WearableAssociationSegment(
            segment_id=assoc_id,
            ph_id=ph_id,
            tag_id="tag-1",
            identity_id="alice",
            started_at=_T0 - timedelta(minutes=2),
            ended_at=_T0,
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            window_llr=18.0,
            margin_llr=10.0,
            matched_fixes=10,
            mean_residual_m=0.1,
        )
    )

    samples = [
        _sample(
            observed_at=_T0,
            identity_id="alice",
            label_source=AppearanceLabelSource.WEARABLE_ATTESTED,
            seed_association_id=assoc_id,
        ),
        _sample(
            observed_at=_T0 + timedelta(seconds=15),
            identity_id="bob",
            label_source=AppearanceLabelSource.DIRECT_FACE,
        ),
        _sample(
            observed_at=_T0 + timedelta(seconds=30),
            identity_id=None,
            label_source=AppearanceLabelSource.NEGATIVE,
        ),
        _sample(
            observed_at=_T0 + timedelta(seconds=45),
            identity_id="alice",
            label_source=AppearanceLabelSource.DIRECT_FACE,
            feature_version="dinov2-v1",
        ),
    ]

    pg_res = await _exercise(PostgresAppearanceSampleRepository(db_pool), samples, assoc_id)
    mem_res = await _exercise(InMemoryAppearanceSampleRepository(), samples, assoc_id)
    assert pg_res == mem_res
    assert pg_res[0] == 4
    assert pg_res[1] == 0
    assert pg_res[3] == 1
    assert pg_res[4] == 1
    assert pg_res[7] == 1


async def test_gallery_purge_identity_cascades_appearance_samples(db_pool: asyncpg.Pool) -> None:
    gallery = PostgresGalleryRepository(db_pool)
    await gallery.upsert_identity(
        Identity(identity_id="carol", display_name="Carol", enrolled_at=_T0)
    )
    repo = PostgresAppearanceSampleRepository(db_pool)
    await repo.append_many(
        [
            _sample(identity_id="carol", label_source=AppearanceLabelSource.DIRECT_FACE),
            _sample(
                observed_at=_T0 + timedelta(seconds=20),
                identity_id=None,
                label_source=AppearanceLabelSource.NEGATIVE,
            ),
        ]
    )
    assert await gallery.purge_identity("carol") is True
    remaining = await repo.list_samples()
    assert all(s.identity_id != "carol" for s in remaining)
    assert any(s.label_source == AppearanceLabelSource.NEGATIVE for s in remaining)


async def test_migrations_0014_and_0015_retention_checks_and_roundtrip(
    db_pool: asyncpg.Pool,
) -> None:
    # 1. Verify TimescaleDB 60-day retention policy on appearance_training_samples.
    async with db_pool.acquire() as conn:
        policy_count = await conn.fetchval(
            """
            SELECT count(*) FROM timescaledb_information.jobs
            WHERE proc_name = 'policy_retention'
              AND hypertable_schema = 'continuous_tracking'
              AND hypertable_name = 'appearance_training_samples'
            """
        )
        assert int(policy_count) >= 1

    # 2. Verify image_only_evaluation_daily accepts 'classifier' arm and rejects unknown arms.
    gallery = PostgresGalleryRepository(db_pool)
    await gallery.upsert_identity(
        Identity(identity_id="dave", display_name="Dave", enrolled_at=_T0)
    )
    eval_repo = PostgresImageOnlyEvaluationRepository(db_pool)
    await eval_repo.add_totals(
        [
            ImageOnlyDailyTotal(
                day=_DAY0,
                identity_id="dave",
                arm="classifier",
                attestation_grade=WearableAttestationGrade.TRAJECTORY,
                correct_s=12.5,
                wrong_s=0.0,
                unknown_s=2.5,
            )
        ]
    )
    rows = await eval_repo.list_totals(since=_DAY0, until=_DAY0)
    assert any(r.identity_id == "dave" and r.arm == "classifier" for r in rows)
    with pytest.raises(asyncpg.CheckViolationError):
        await eval_repo.add_totals(
            [
                ImageOnlyDailyTotal(
                    day=_DAY0,
                    identity_id="dave",
                    arm="invalid_arm",
                    attestation_grade=WearableAttestationGrade.TRAJECTORY,
                    correct_s=1.0,
                )
            ]
        )

    # 3. Roll back migrations 0015 and 0014 and re-apply cleanly.
    runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
    try:
        applied = (await runner.status())["applied"]
        above_0013 = sum(1 for name in applied if name >= "0014_appearance_training_samples")
        assert above_0013 == 2
        assert await runner.rollback(above_0013) == 2
        async with db_pool.acquire() as conn:
            tbl = await conn.fetchval(
                "SELECT to_regclass('continuous_tracking.appearance_training_samples')"
            )
            assert tbl is None
    finally:
        await runner.migrate()

    async with db_pool.acquire() as conn:
        tbl_after = await conn.fetchval(
            "SELECT to_regclass('continuous_tracking.appearance_training_samples')"
        )
        assert tbl_after is not None
