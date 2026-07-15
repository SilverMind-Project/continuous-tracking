"""M03: gallery state-filter parity matrix (InMemory half).

Proves ``InMemoryGalleryRepository`` honors the explicit, governance-safe state
contract (``VERIFIED_ONLY`` default; ``states=None``/``allowed_states=None`` means no
filter) for every state-sensitive read. The Postgres half of the same scenario lives in
``tests/integration/test_gallery_state_parity_postgres.py`` (``pytest -m integration``,
``make ci``); both suites assert against the identical expected-ID sets defined in
``_gallery_parity_fixtures.py`` so a divergence between implementations cannot hide
behind a divergent fixture.
"""

from __future__ import annotations

import inspect

import pytest

from app.storage.gallery import VERIFIED_ONLY, InMemoryGalleryRepository
from app.storage.postgres.gallery_repo import PostgresGalleryRepository
from tests.storage._gallery_parity_fixtures import (
    ALICE,
    ALL_TRACKLET_IDS,
    STATE_FILTER_CASES,
    entry_for,
    expected_ids,
    make_entries,
    make_identities,
    query_embedding,
)


async def _seeded_repo() -> InMemoryGalleryRepository:
    repo = InMemoryGalleryRepository()
    for identity in make_identities():
        await repo.upsert_identity(identity)
    for entry in make_entries():
        await repo.upsert_gallery_entry(entry)
    return repo


class TestListGalleryEntriesParity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("states", STATE_FILTER_CASES)
    async def test_state_filter(self, states: frozenset[str] | None) -> None:
        repo = await _seeded_repo()
        entries = await repo.list_gallery_entries(states=states)
        assert {e.gallery_entry_id for e in entries} == expected_ids(states=states)

    @pytest.mark.asyncio
    async def test_default_excludes_pending_and_rejected(self) -> None:
        repo = await _seeded_repo()
        entries = await repo.list_gallery_entries()
        assert {e.gallery_entry_id for e in entries} == expected_ids(states=VERIFIED_ONLY)

    @pytest.mark.asyncio
    async def test_identity_filter_combines_with_state_filter(self) -> None:
        repo = await _seeded_repo()
        entries = await repo.list_gallery_entries(identity_id=ALICE, states=None)
        assert {e.gallery_entry_id for e in entries} == expected_ids(states=None, identity_id=ALICE)


class TestSearchSimilarParity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("states", STATE_FILTER_CASES)
    async def test_state_filter(self, states: frozenset[str] | None) -> None:
        repo = await _seeded_repo()
        hits = await repo.search_similar(query_embedding(), limit=10, states=states)
        assert {entry.gallery_entry_id for entry, _sim in hits} == expected_ids(states=states)

    @pytest.mark.asyncio
    async def test_default_excludes_pending_and_rejected(self) -> None:
        repo = await _seeded_repo()
        hits = await repo.search_similar(query_embedding(), limit=10)
        got = {entry.gallery_entry_id for entry, _sim in hits}
        assert got == expected_ids(states=VERIFIED_ONLY)


class TestListGalleryEntriesForTrackletsParity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("states", STATE_FILTER_CASES)
    async def test_state_filter(self, states: frozenset[str] | None) -> None:
        repo = await _seeded_repo()
        entries = await repo.list_gallery_entries_for_tracklets(
            ALL_TRACKLET_IDS, limit=20, allowed_states=states
        )
        assert {e.gallery_entry_id for e in entries} == expected_ids(states=states)

    @pytest.mark.asyncio
    async def test_default_excludes_pending_and_rejected(self) -> None:
        repo = await _seeded_repo()
        entries = await repo.list_gallery_entries_for_tracklets(ALL_TRACKLET_IDS, limit=20)
        assert {e.gallery_entry_id for e in entries} == expected_ids(states=VERIFIED_ONLY)


class TestGallerySimilarityDefault:
    @pytest.mark.asyncio
    async def test_default_is_verified_only(self) -> None:
        """Regression guard: gallery_similarity's own default must independently be
        VERIFIED_ONLY. It forwards allowed_states to list_gallery_entries_for_tracklets,
        whose `None` now means "no filter" -- if gallery_similarity kept a bare `None`
        default it would silently flip from verified-only to unfiltered voting."""
        repo = InMemoryGalleryRepository()
        for identity in make_identities():
            await repo.upsert_identity(identity)
        # Only a pending_review entry for bob: with the verified-only default this
        # must NOT be treated as a match against alice's verified entry.
        alice_verified = entry_for(ALICE, "operator_verified")
        bob_pending = entry_for("bob", "pending_review")
        await repo.upsert_gallery_entry(alice_verified)
        await repo.upsert_gallery_entry(bob_pending)

        sim = await repo.gallery_similarity(
            {alice_verified.origin_tracklet_id}, {bob_pending.origin_tracklet_id}
        )
        # bob's only entry is pending -> invisible at the default -> the "only one
        # side has entries" 0.5 fallback, never a real (and here, unrelated) cosine
        # similarity computed from a vector that should not have voted.
        assert sim == 0.5


class TestSignatureDefaultsGuardAgainstDrift:
    """mypy proves the type signatures; this proves the *default value* stays
    VERIFIED_ONLY on every state-sensitive method of both implementations."""

    def test_verified_only_is_pinned_to_the_literal(self) -> None:
        """Every other assertion in this module compares a result against
        ``VERIFIED_ONLY`` itself, which would stay green even if the constant's
        definition silently widened (e.g. to include ``pending_review``) --
        exactly the M00 pollution-incident failure mode. Pin the constant to
        its literal value independently of every other test in this file."""
        assert frozenset({"operator_verified"}) == VERIFIED_ONLY

    @pytest.mark.parametrize(
        ("repo_cls", "method_name", "param_name"),
        [
            (InMemoryGalleryRepository, "list_gallery_entries", "states"),
            (InMemoryGalleryRepository, "search_similar", "states"),
            (InMemoryGalleryRepository, "list_gallery_entries_for_tracklets", "allowed_states"),
            (InMemoryGalleryRepository, "gallery_similarity", "allowed_states"),
            (PostgresGalleryRepository, "list_gallery_entries", "states"),
            (PostgresGalleryRepository, "search_similar", "states"),
            (PostgresGalleryRepository, "list_gallery_entries_for_tracklets", "allowed_states"),
            (PostgresGalleryRepository, "gallery_similarity", "allowed_states"),
        ],
    )
    def test_default_is_verified_only(
        self, repo_cls: type, method_name: str, param_name: str
    ) -> None:
        method = getattr(repo_cls, method_name)
        sig = inspect.signature(method)
        assert sig.parameters[param_name].default == VERIFIED_ONLY
