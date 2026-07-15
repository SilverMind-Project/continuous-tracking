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

from typing import Any

import pytest

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
