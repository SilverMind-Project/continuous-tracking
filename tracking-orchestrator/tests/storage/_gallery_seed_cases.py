"""The provenance-filter assertions, run unchanged against each gallery peer."""

from __future__ import annotations

from datetime import datetime, timedelta

from app.domain import GallerySeedSource
from app.storage.gallery import GalleryRepository
from tests.storage._gallery_seed_fixtures import (
    MEMBER,
    SEGMENT,
    T0,
    all_tracklets,
    candidates,
    identity,
    names_for,
    query_embedding,
    row_id,
    seed_source_of,
)


async def seed(repo: GalleryRepository) -> None:
    await repo.upsert_identity(identity())
    for candidate in candidates():
        await repo.create_review_candidate(candidate)


async def assert_search_filter(
    repo: GalleryRepository,
    exclude: frozenset[str],
    seen_before: datetime | None,
    expected: frozenset[str],
) -> None:
    hits = await repo.search_similar(
        query_embedding(), limit=20, exclude_seed_sources=exclude, seen_before=seen_before
    )
    assert names_for({str(e.gallery_entry_id) for e, _ in hits}) == expected


async def assert_tracklet_filter(
    repo: GalleryRepository,
    exclude: frozenset[str],
    seen_before: datetime | None,
    expected: frozenset[str],
) -> None:
    rows = await repo.list_gallery_entries_for_tracklets(
        all_tracklets(), limit=20, exclude_seed_sources=exclude, seen_before=seen_before
    )
    assert names_for({str(e.gallery_entry_id) for e in rows}) == expected


async def assert_seed_source_reaches_vote_path(repo: GalleryRepository) -> None:
    """The scorer's trust depends on seed_source, so both vote-path reads must return it."""
    hits = await repo.search_similar(query_embedding(), limit=20)
    by_name = {next(iter(names_for({str(e.gallery_entry_id)}))): e for e, _ in hits}
    for name, entry in by_name.items():
        assert entry.seed_source == seed_source_of(name)
    assert by_name["wear-late"].seed_association_id == SEGMENT
    assert by_name["face-late"].seed_association_id is None
    rows = await repo.list_gallery_entries_for_tracklets(all_tracklets(), limit=20)
    assert {e.seed_source for e in rows} == set(GallerySeedSource) - {GallerySeedSource.OPERATOR}


async def assert_count_seeded(repo: GalleryRepository) -> None:
    count = await repo.count_seeded_entries(
        identity_id=MEMBER,
        seed_source=GallerySeedSource.WEARABLE_ATTESTED,
        seen_since=T0 + timedelta(minutes=5),
    )
    assert count == 2
    later = await repo.count_seeded_entries(
        identity_id=MEMBER,
        seed_source=GallerySeedSource.WEARABLE_ATTESTED,
        seen_since=T0 + timedelta(minutes=6),
    )
    assert later == 1


async def assert_review_seed_source_filter(repo: GalleryRepository) -> None:
    rows, total = await repo.list_review_candidates(
        state="auto_verified", seed_source="wearable_attested"
    )
    assert total == 2
    assert {r.candidate_id for r in rows} == {row_id("wear-early"), row_id("wear-late")}
    assert all(r.seed_association_id == SEGMENT for r in rows)


async def assert_state_counts_by_seed_source(repo: GalleryRepository) -> None:
    wearable = await repo.count_review_queue(
        seed_sources=frozenset({"wearable_attested", "wearable_room_attested"})
    )
    assert wearable["auto_verified"] == 3
    assert (await repo.count_review_queue())["auto_verified"] == 5
