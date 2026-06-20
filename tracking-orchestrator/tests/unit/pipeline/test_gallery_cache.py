"""Unit tests for GalleryCache."""

from __future__ import annotations

import asyncio

import pytest

from app.pipeline.gallery_cache import GalleryCache
from app.storage.gallery import InMemoryGalleryRepository

class _CountingRepo(InMemoryGalleryRepository):
    """InMemoryGalleryRepository that counts list_gallery_entries_for_tracklets calls."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> list:
        self.call_count += 1
        return await super().list_gallery_entries_for_tracklets(tracklet_ids, limit, allowed_states, model_versions)



class TestGalleryCacheReadThrough:
    @pytest.mark.asyncio
    async def test_first_read_fetches_from_repo(self) -> None:
        repo = _CountingRepo()
        cache = GalleryCache(repo)
        cache.invalidate()

        await cache.list_gallery_entries_for_tracklets({"t1"})

        assert repo.call_count == 1

    @pytest.mark.asyncio
    async def test_second_read_same_key_is_cached(self) -> None:
        repo = _CountingRepo()
        cache = GalleryCache(repo)
        cache.invalidate()

        await cache.list_gallery_entries_for_tracklets({"t1"})
        await cache.list_gallery_entries_for_tracklets({"t1"})

        assert repo.call_count == 1

    @pytest.mark.asyncio
    async def test_different_keys_are_cached_independently(self) -> None:
        repo = _CountingRepo()
        cache = GalleryCache(repo)
        cache.invalidate()

        await cache.list_gallery_entries_for_tracklets({"t1"})
        await cache.list_gallery_entries_for_tracklets({"t2"})
        await cache.list_gallery_entries_for_tracklets({"t1"})
        await cache.list_gallery_entries_for_tracklets({"t2"})

        assert repo.call_count == 2

    @pytest.mark.asyncio
    async def test_invalidate_clears_memoized_entries(self) -> None:
        repo = _CountingRepo()
        cache = GalleryCache(repo)
        cache.invalidate()

        await cache.list_gallery_entries_for_tracklets({"t1"})
        cache.invalidate()
        await cache.list_gallery_entries_for_tracklets({"t1"})

        assert repo.call_count == 2


class TestGalleryCacheStalenessBackstop:
    @pytest.mark.asyncio
    async def test_stale_cache_self_invalidates_and_refetches(self) -> None:
        """After max_age_s elapses without an explicit invalidation, a read
        self-invalidates the cache and fetches fresh data from the repo."""
        repo = _CountingRepo()
        cache = GalleryCache(repo, max_age_s=0.01)
        cache.invalidate()

        await cache.list_gallery_entries_for_tracklets({"t1"})
        assert repo.call_count == 1

        await asyncio.sleep(0.02)

        await cache.list_gallery_entries_for_tracklets({"t1"})
        assert repo.call_count == 2
