"""Gallery provenance filters, InMemory half (the Postgres half runs the same cases in make ci)."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.storage.gallery import InMemoryGalleryRepository
from tests.storage import _gallery_seed_cases as cases
from tests.storage._gallery_seed_fixtures import FILTER_CASES


async def _repo() -> InMemoryGalleryRepository:
    repo = InMemoryGalleryRepository()
    await cases.seed(repo)
    return repo


@pytest.mark.parametrize(("exclude", "seen_before", "expected"), FILTER_CASES)
async def test_search_similar_provenance_filters(
    exclude: frozenset[str], seen_before: datetime | None, expected: frozenset[str]
) -> None:
    await cases.assert_search_filter(await _repo(), exclude, seen_before, expected)


@pytest.mark.parametrize(("exclude", "seen_before", "expected"), FILTER_CASES)
async def test_tracklet_read_provenance_filters(
    exclude: frozenset[str], seen_before: datetime | None, expected: frozenset[str]
) -> None:
    await cases.assert_tracklet_filter(await _repo(), exclude, seen_before, expected)


async def test_seed_source_reaches_the_vote_path() -> None:
    await cases.assert_seed_source_reaches_vote_path(await _repo())


async def test_count_seeded_entries() -> None:
    await cases.assert_count_seeded(await _repo())


async def test_review_queue_seed_source_filter() -> None:
    await cases.assert_review_seed_source_filter(await _repo())


async def test_state_counts_by_seed_source() -> None:
    await cases.assert_state_counts_by_seed_source(await _repo())
