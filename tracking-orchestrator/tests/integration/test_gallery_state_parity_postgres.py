"""M03: gallery state-filter parity matrix (Postgres half).

Proves ``PostgresGalleryRepository`` honors the identical explicit state contract as
``InMemoryGalleryRepository`` (``tests/storage/test_gallery_state_parity.py``, fast gate):
``VERIFIED_ONLY`` default, ``states``/``allowed_states=None`` means no filter. Both suites seed the
same fixture (``tests/storage/_gallery_parity_fixtures.py``) and assert against the same
expected-ID sets, so a divergence between implementations cannot hide behind a divergent fixture.

Marked @pytest.mark.integration; CI selects this marker against a testcontainer
(follows tests/integration/test_correction_repo_postgres.py conventions).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.domain import NewReviewCandidate
from app.storage.gallery import PENDING_AND_VERIFIED
from app.storage.postgres.gallery_repo import PostgresGalleryRepository
from tests.storage._gallery_parity_fixtures import (
    ALICE,
    ALL_TRACKLET_IDS,
    STATE_FILTER_CASES,
    expected_ids,
    make_entries,
    make_identities,
    query_embedding,
)

pytestmark = pytest.mark.integration


async def _seeded_repo(db_pool: Any) -> PostgresGalleryRepository:
    repo = PostgresGalleryRepository(db_pool)
    for identity in make_identities():
        await repo.upsert_identity(identity)
    for entry in make_entries():
        await repo.upsert_gallery_entry(entry)
    return repo


@pytest.mark.asyncio
@pytest.mark.parametrize("states", STATE_FILTER_CASES)
async def test_list_gallery_entries_state_filter(
    db_pool: Any, states: frozenset[str] | None
) -> None:
    repo = await _seeded_repo(db_pool)
    entries = await repo.list_gallery_entries(states=states)
    assert {str(e.gallery_entry_id) for e in entries} == expected_ids(states=states)


@pytest.mark.asyncio
async def test_list_gallery_entries_identity_filter_combines_with_state_filter(
    db_pool: Any,
) -> None:
    repo = await _seeded_repo(db_pool)
    entries = await repo.list_gallery_entries(identity_id=ALICE, states=None)
    got = {str(e.gallery_entry_id) for e in entries}
    assert got == expected_ids(states=None, identity_id=ALICE)


@pytest.mark.asyncio
@pytest.mark.parametrize("states", STATE_FILTER_CASES)
async def test_search_similar_state_filter(db_pool: Any, states: frozenset[str] | None) -> None:
    repo = await _seeded_repo(db_pool)
    hits = await repo.search_similar(query_embedding(), limit=10, states=states)
    got = {str(entry.gallery_entry_id) for entry, _sim in hits}
    assert got == expected_ids(states=states)


@pytest.mark.asyncio
@pytest.mark.parametrize("states", STATE_FILTER_CASES)
async def test_list_gallery_entries_for_tracklets_state_filter(
    db_pool: Any, states: frozenset[str] | None
) -> None:
    repo = await _seeded_repo(db_pool)
    entries = await repo.list_gallery_entries_for_tracklets(
        ALL_TRACKLET_IDS, limit=20, allowed_states=states
    )
    assert {str(e.gallery_entry_id) for e in entries} == expected_ids(states=states)


@pytest.mark.asyncio
async def test_gallery_similarity_default_excludes_pending(db_pool: Any) -> None:
    """Regression guard: gallery_similarity's own default must independently be
    VERIFIED_ONLY (see the M03 milestone's dated correction: it forwards
    allowed_states to list_gallery_entries_for_tracklets, whose `None` now means
    "no filter", so a bare `None` default would have silently flipped it to
    unfiltered voting in production)."""
    repo = await _seeded_repo(db_pool)
    entries = make_entries()
    alice_verified = next(
        e for e in entries if e.identity_id == ALICE and e.state == "operator_verified"
    )
    bob_pending = next(e for e in entries if e.identity_id == "bob" and e.state == "pending_review")

    sim = await repo.gallery_similarity(
        {alice_verified.origin_tracklet_id}, {bob_pending.origin_tracklet_id}
    )
    assert sim == 0.5


def _candidate(
    candidate_id: str, *, identity_id: str, orientation: int, state: str = "pending_review"
) -> NewReviewCandidate:
    return NewReviewCandidate(
        candidate_id=candidate_id,
        identity_id=identity_id,
        embedding=[0.1] * 768,
        quality=0.9,
        orientation=orientation,
        camera_id="cam01",
        capture_time=datetime.now(UTC),
        ph_id="00000000-0000-0000-0000-000000000301",
        observation_id="00000000-0000-0000-0000-000000000302",
        origin_tracklet_id="00000000-0000-0000-0000-000000000302",
        keyframe_id=None,
        crop_key=f"reid-candidates/v1/{candidate_id}.jpg",
        source_frame_key=None,
        crop_hash="deadbeef",
        frame_hash=None,
        dimensions=(128, 256),
        is_truncated=False,
        is_occluded=False,
        candidate_reason="face_derived",
        source_episode_id="00000000-0000-0000-0000-000000000301",
        created_actor="pipeline",
        model_version="reid-solider",
        preprocessing_version="v1",
        confidence=0.9,
        state=state,
    )


@pytest.mark.asyncio
async def test_create_review_candidate_round_trips_and_is_idempotent(db_pool: Any) -> None:
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(make_identities()[0])
    identity_id = make_identities()[0].identity_id
    candidate = _candidate(
        "11111111-2222-3333-4444-555555555601", identity_id=identity_id, orientation=0
    )

    first_id = await repo.create_review_candidate(candidate)
    second_id = await repo.create_review_candidate(candidate)  # retry, same id

    assert first_id == second_id == candidate.candidate_id
    row = await repo.get_review_candidate(candidate.candidate_id)
    assert row is not None
    assert row.state == "pending_review"
    assert row.ph_id == "00000000-0000-0000-0000-000000000301"
    assert row.observation_id == "00000000-0000-0000-0000-000000000302"
    assert row.crop_hash == "deadbeef"
    assert row.model_version == "reid-solider"

    entries = await repo.list_gallery_entries(identity_id=identity_id, states=None)
    matching = [e for e in entries if str(e.gallery_entry_id) == candidate.candidate_id]
    assert len(matching) == 1  # one row exposed to both readers, no duplicate on retry

    # F5 closure: once verified, the resolver's real query path
    # (list_gallery_entries_for_tracklets, keyed by origin_tracklet_id ==
    # PersonHypothesis.observation_ids) must see the row. Pending must not.
    obs_ids = {candidate.origin_tracklet_id}
    assert await repo.list_gallery_entries_for_tracklets(obs_ids) == []
    await repo.apply_review_action(
        candidate.candidate_id, action="approve", actor="operator", base_audit_version=1
    )
    verified_hits = await repo.list_gallery_entries_for_tracklets(obs_ids)
    assert {str(e.gallery_entry_id) for e in verified_hits} == {candidate.candidate_id}


@pytest.mark.asyncio
async def test_count_gallery_entries_defaults_to_pending_and_verified(db_pool: Any) -> None:
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(make_identities()[0])
    identity_id = make_identities()[0].identity_id

    await repo.create_review_candidate(
        _candidate("11111111-2222-3333-4444-555555555602", identity_id=identity_id, orientation=1)
    )
    verified = _candidate(
        "11111111-2222-3333-4444-555555555603", identity_id=identity_id, orientation=1
    )
    await repo.create_review_candidate(verified)
    await repo.apply_review_action(
        verified.candidate_id, action="approve", actor="operator", base_audit_version=1
    )
    rejected = _candidate(
        "11111111-2222-3333-4444-555555555604", identity_id=identity_id, orientation=1
    )
    await repo.create_review_candidate(rejected)
    await repo.apply_review_action(
        rejected.candidate_id, action="reject", actor="operator", base_audit_version=1
    )

    count = await repo.count_gallery_entries(identity_id, 1)
    assert count == 2  # pending + verified, never rejected
    assert count == await repo.count_gallery_entries(identity_id, 1, states=PENDING_AND_VERIFIED)
    assert await repo.count_gallery_entries(identity_id, 1, states=None) == 3


@pytest.mark.asyncio
async def test_auto_verified_created_row_visible_to_both_readers(db_pool: Any) -> None:
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(make_identities()[0])
    identity_id = make_identities()[0].identity_id
    candidate = _candidate(
        "11111111-2222-3333-4444-555555555610",
        identity_id=identity_id,
        orientation=0,
        state="auto_verified",
    )

    await repo.create_review_candidate(candidate)

    row = await repo.get_review_candidate(candidate.candidate_id)
    assert row is not None
    assert row.state == "auto_verified"
    entries = await repo.list_gallery_entries(identity_id=identity_id, states=None)
    matching = [e for e in entries if str(e.gallery_entry_id) == candidate.candidate_id]
    assert len(matching) == 1
    assert matching[0].state == "auto_verified"


@pytest.mark.asyncio
async def test_demote_restores_pending_and_keeps_vector(db_pool: Any) -> None:
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(make_identities()[0])
    identity_id = make_identities()[0].identity_id
    candidate = _candidate(
        "11111111-2222-3333-4444-555555555611",
        identity_id=identity_id,
        orientation=0,
        state="auto_verified",
    )
    await repo.create_review_candidate(candidate)

    updated = await repo.apply_review_action(
        candidate.candidate_id, action="demote", actor="op", base_audit_version=1
    )
    assert updated.state == "pending_review"

    entries = await repo.list_gallery_entries(identity_id=identity_id, states=None)
    matching = next(e for e in entries if str(e.gallery_entry_id) == candidate.candidate_id)
    assert matching.state == "pending_review"
    assert len(matching.embedding) > 0  # vector survives, unlike reject


@pytest.mark.asyncio
async def test_reject_from_auto_verified_deletes_vector(db_pool: Any) -> None:
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(make_identities()[0])
    identity_id = make_identities()[0].identity_id
    candidate = _candidate(
        "11111111-2222-3333-4444-555555555612",
        identity_id=identity_id,
        orientation=0,
        state="auto_verified",
    )
    await repo.create_review_candidate(candidate)

    updated = await repo.apply_review_action(
        candidate.candidate_id,
        action="reject",
        actor="op",
        base_audit_version=1,
        reason="wrong_person",
    )
    assert updated.state == "rejected"

    entries = await repo.list_gallery_entries(identity_id=identity_id, states=None)
    matching = next(e for e in entries if str(e.gallery_entry_id) == candidate.candidate_id)
    assert matching.state == "rejected"
    assert matching.embedding == []


@pytest.mark.asyncio
async def test_compensate_approve_from_auto_verified_restores_auto_verified(db_pool: Any) -> None:
    """Undo must read the event trail, not assume pending_review."""
    repo = PostgresGalleryRepository(db_pool)
    await repo.upsert_identity(make_identities()[0])
    identity_id = make_identities()[0].identity_id
    candidate = _candidate(
        "11111111-2222-3333-4444-555555555613",
        identity_id=identity_id,
        orientation=0,
        state="auto_verified",
    )
    await repo.create_review_candidate(candidate)
    await repo.apply_review_action(
        candidate.candidate_id, action="approve", actor="op", base_audit_version=1
    )

    restored = await repo.compensate_review(
        candidate.candidate_id, actor="op2", base_audit_version=2
    )
    assert restored.state == "auto_verified"

    entries = await repo.list_gallery_entries(identity_id=identity_id, states=None)
    matching = next(e for e in entries if str(e.gallery_entry_id) == candidate.candidate_id)
    assert matching.state == "auto_verified"
