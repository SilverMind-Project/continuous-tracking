"""Postgres integration tests for migrations 0011 & 0012, retention, and revocation parity."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.domain import (
    GallerySeedSource,
    Identity,
    IdentityRevisionRange,
    NewReviewCandidate,
    PersonHypothesis,
    WearableAssociationSegment,
    WearableAttestationGrade,
)
from app.storage.migrations import MigrationRunner
from app.storage.postgres.correction_repo import PostgresIdentityCorrectionRepository
from app.storage.postgres.gallery_repo import PostgresGalleryRepository
from app.storage.postgres.ph_repo import PostgresPHRepository
from app.storage.postgres.wearable_association_repo import PostgresWearableAssociationRepository
from app.storage.postgres.wearable_teaching_hold_repo import PostgresWearableTeachingHoldRepository

pytestmark = pytest.mark.integration

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_T0 = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)
_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "gallery-retention-revocation-pg")


def _uid(name: str) -> str:
    return str(uuid.uuid5(_NS, name))


def _cand(
    name: str,
    *,
    seen_at: datetime,
    state: str = "auto_verified",
    association_id: str | None = None,
) -> NewReviewCandidate:
    vec = [0.0] * 768
    vec[0] = 1.0
    return NewReviewCandidate(
        candidate_id=_uid(name),
        identity_id="alice",
        embedding=vec,
        quality=0.9,
        orientation=2,
        camera_id="cam-1",
        capture_time=seen_at,
        ph_id=_uid("ph-1"),
        observation_id=_uid(f"obs-{name}"),
        origin_tracklet_id=_uid(f"obs-{name}"),
        keyframe_id=None,
        crop_key=f"reid/{name}.jpg",
        source_frame_key=None,
        crop_hash="h",
        frame_hash=None,
        dimensions=(64, 128),
        is_truncated=False,
        is_occluded=False,
        candidate_reason="wearable_attested" if association_id else "face_derived",
        source_episode_id=_uid(f"ep-{name}"),
        created_actor="pipeline",
        model_version="reid",
        preprocessing_version="v1",
        confidence=None if association_id else 0.95,
        state=state,
        seed_source=(
            GallerySeedSource.WEARABLE_ATTESTED if association_id else GallerySeedSource.FACE
        ),
        seed_association_id=association_id,
    )


@pytest.mark.asyncio
async def test_count_gallery_entries_seen_since_and_retention_postgres(db_pool: Any) -> None:
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0))
    assoc = _uid("assoc-1")

    await repo.create_review_candidate(
        _cand(
            "old-row", seen_at=_T0 - timedelta(days=20), state="auto_verified", association_id=assoc
        )
    )
    await repo.create_review_candidate(
        _cand(
            "fresh-row",
            seen_at=_T0 - timedelta(hours=2),
            state="auto_verified",
            association_id=assoc,
        )
    )

    # Working-set count with seen_since excludes old-row
    assert await repo.count_gallery_entries("alice", 2) == 2
    assert await repo.count_gallery_entries("alice", 2, seen_since=_T0 - timedelta(hours=12)) == 1

    # Association filtering and counting
    by_assoc, total_assoc = await repo.list_review_candidates(
        state="auto_verified", seed_association_id=assoc
    )
    assert total_assoc == 2
    assert len(by_assoc) == 2
    assert (await repo.count_by_seed_association(assoc))["auto_verified"] == 2

    # Retention count & expire
    cutoff = _T0 - timedelta(days=14)
    assert await repo.count_past_retention(state="auto_verified", seen_before=cutoff) == 1
    crops = await repo.expire_past_retention(state="auto_verified", seen_before=cutoff, limit=100)
    assert crops == ["reid/old-row.jpg"]

    counts = await repo.count_review_queue()
    assert counts["expired"] == 1
    assert counts["auto_verified"] == 1
    events = await repo.list_review_events(_uid("old-row"))
    assert events[-1].actor == "system:retention"
    assert events[-1].new_state == "expired"


@pytest.mark.asyncio
async def test_wearable_revocation_holds_and_source_association_ranges_postgres(
    db_pool: Any,
) -> None:
    assoc_repo = PostgresWearableAssociationRepository(db_pool)
    holds_repo = PostgresWearableTeachingHoldRepository(db_pool)
    gallery_repo = PostgresGalleryRepository(db_pool)
    ph_repo = PostgresPHRepository(db_pool)
    corr_repo = PostgresIdentityCorrectionRepository(db_pool)

    await gallery_repo.upsert_identity(
        Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0)
    )

    seg1_id = _uid("seg-1")
    seg2_id = _uid("seg-2")
    ph_id = _uid("ph-rev")
    await ph_repo.save(
        PersonHypothesis(
            ph_id=ph_id,
            state_mean=(1.0, 1.0, 0.0, 0.0),
            state_cov=(1.0,) * 16,
            born_at=_T0,
            last_seen_at=_T0 + timedelta(minutes=5),
            last_seen_camera="cam-1",
            observation_count=5,
        )
    )

    seg1 = WearableAssociationSegment(
        segment_id=seg1_id,
        ph_id=ph_id,
        tag_id="tag-pg-1",
        identity_id="alice",
        started_at=_T0,
        ended_at=_T0 + timedelta(minutes=2),
        attestation_grade=WearableAttestationGrade.TRAJECTORY,
        window_llr=20.0,
        margin_llr=10.0,
        matched_fixes=12,
        mean_residual_m=0.25,
    )
    seg2 = WearableAssociationSegment(
        segment_id=seg2_id,
        ph_id=ph_id,
        tag_id="tag-pg-1",
        identity_id="alice",
        started_at=_T0 + timedelta(minutes=3),
        ended_at=_T0 + timedelta(minutes=4),
        attestation_grade=WearableAttestationGrade.TRAJECTORY,
        window_llr=20.0,
        margin_llr=10.0,
        matched_fixes=12,
        mean_residual_m=0.25,
        swap_suspected=True,
    )
    await assoc_repo.save_segment(seg1)
    await assoc_repo.save_segment(seg2)

    overlapping = await assoc_repo.list_unrevoked_overlapping(
        "tag-pg-1", since=_T0 - timedelta(minutes=1), until=_T0 + timedelta(minutes=5)
    )
    assert {s.segment_id for s in overlapping} == {seg1_id, seg2_id}

    assert (
        await assoc_repo.mark_revoked(
            seg1_id,
            revoked_at=_T0 + timedelta(minutes=6),
            revoked_by="op",
            revoke_reason="operator",
        )
        is True
    )
    assert (
        await assoc_repo.mark_revoked(
            seg1_id,
            revoked_at=_T0 + timedelta(minutes=7),
            revoked_by="op2",
            revoke_reason="operator",
        )
        is False
    )
    # Subsequent save_segment preserves revocation metadata.
    await assoc_repo.save_segment(seg1)
    reloaded = await assoc_repo.get_segment(seg1_id)
    assert reloaded is not None and reloaded.revoked_by == "op"
    assert await assoc_repo.count_revoked_since(_T0) == 1

    # Holds parity & idempotency
    h1 = await holds_repo.create_hold(tag_id="tag-pg-1", reason="operator", held_at=_T0)
    h2 = await holds_repo.create_hold(tag_id="tag-pg-1", reason="swap_suspected", held_at=_T0)
    assert h1.hold_id == h2.hold_id
    assert await holds_repo.is_held("tag-pg-1") is True
    assert len(await holds_repo.list_open_holds()) == 1
    cleared = await holds_repo.clear_hold("tag-pg-1", cleared_by="op", cleared_at=_T0)
    assert cleared is not None and cleared.cleared_by == "op"
    assert await holds_repo.is_held("tag-pg-1") is False

    # Inferred ranges by source_association_id
    rng = IdentityRevisionRange(
        range_id=_uid("rng-1"),
        revision_id=_uid("rev-1"),
        ph_id=ph_id,
        authority="inferred",
        range_start=_T0,
        range_end=_T0 + timedelta(minutes=2),
        effective_identity_id="alice",
        source_association_id=seg1_id,
        created_at=_T0,
    )
    await corr_repo.save_range(rng)
    found = await corr_repo.list_ranges_by_source_association(seg1_id, live_only=True)
    assert [r.range_id for r in found] == [rng.range_id]
    await corr_repo.mark_range_compensated(rng.range_id, by_revision_id=_uid("rev-comp-1"))
    assert await corr_repo.list_ranges_by_source_association(seg1_id, live_only=True) == []


@pytest.mark.asyncio
async def test_migrations_0011_and_0012_down_refusal_and_round_trip(db_pool: Any) -> None:
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0))
    await repo.create_review_candidate(
        _cand("exp-mig", seen_at=_T0 - timedelta(days=20), state="auto_verified")
    )
    await repo.expire_past_retention(
        state="auto_verified", seen_before=_T0 - timedelta(days=14), limit=10
    )

    runner = MigrationRunner(db_pool, _MIGRATIONS_DIR)
    try:
        # Roll back every migration newer than 0011 (0012, 0013, ...) so 0011 is next.
        applied = (await runner.status())["applied"]
        above_0011 = sum(1 for name in applied if name > "0011_reid_gallery_expired_state")
        if above_0011 > 0:
            await runner.rollback(above_0011)

        # Rolling back 0011 MUST fail while an expired row exists.
        with pytest.raises(Exception, match="expired"):
            await runner.rollback(1)

        # Clean up the expired row and audit events, then 0011 down succeeds.
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM continuous_tracking.gallery_review_events WHERE entry_id = $1::uuid",
                _uid("exp-mig"),
            )
            await conn.execute(
                "DELETE FROM continuous_tracking.reid_gallery WHERE id = $1::uuid",
                _uid("exp-mig"),
            )
        await runner.rollback(1)
    finally:
        await runner.migrate()
