"""Per-frame read-through cache for gallery queries.

Eliminates redundant Postgres round-trips when multiple pipeline stages
query the same tracklet's gallery entries within one ``_process_frame``
invocation.  Not thread-safe; intended for single-consumer async use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain import GalleryEmbedding
from ..storage.base import GalleryRepository


@dataclass
class GalleryCache:
    """Per-frame read-through cache for gallery queries.

    De-duplicates ``list_gallery_entries_for_tracklets`` and
    ``gallery_similarity`` calls within a single frame.  Call
    :meth:`invalidate` at the start of each frame to clear stale
    entries from the previous frame.
    """

    _repo: GalleryRepository
    _entries_by_tracklets: dict[frozenset[str], list[GalleryEmbedding]] = field(
        default_factory=dict
    )
    _similarity_cache: dict[tuple[frozenset[str], frozenset[str]], float] = field(
        default_factory=dict
    )

    def invalidate(self) -> None:
        """Clear all cached state.  Call at the top of each ``_process_frame``."""
        self._entries_by_tracklets.clear()
        self._similarity_cache.clear()

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
    ) -> list[GalleryEmbedding]:
        key = frozenset(tracklet_ids)
        if key not in self._entries_by_tracklets:
            self._entries_by_tracklets[key] = await self._repo.list_gallery_entries_for_tracklets(
                tracklet_ids, limit
            )
        return self._entries_by_tracklets[key]

    async def gallery_similarity(
        self,
        tracklet_ids_a: set[str],
        tracklet_ids_b: set[str],
        limit: int = 20,
    ) -> float:
        key_a = frozenset(tracklet_ids_a)
        key_b = frozenset(tracklet_ids_b)
        cache_key = (key_a, key_b) if hash(key_a) <= hash(key_b) else (key_b, key_a)
        if cache_key not in self._similarity_cache:
            entries_a = await self.list_gallery_entries_for_tracklets(tracklet_ids_a, limit)
            entries_b = await self.list_gallery_entries_for_tracklets(tracklet_ids_b, limit)
            if not entries_a and not entries_b:
                self._similarity_cache[cache_key] = 0.0
            elif not entries_a or not entries_b:
                self._similarity_cache[cache_key] = 0.5
            else:
                self._similarity_cache[cache_key] = GalleryRepository._cosine_between_centroids(
                    entries_a, entries_b
                )
        return self._similarity_cache[cache_key]
