"""Per-frame read-through cache for gallery queries.

Eliminates redundant Postgres round-trips when multiple pipeline stages
query the same tracklet's gallery entries within one ``_process_frame``
invocation.  Not thread-safe; intended for single-consumer async use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from structlog import get_logger

from ..domain import GalleryEmbedding
from ..storage.base import GalleryRepository

logger = get_logger(__name__)


@dataclass
class GalleryCache:
    """Per-frame read-through cache for gallery queries.

    De-duplicates ``list_gallery_entries_for_tracklets`` and
    ``gallery_similarity`` calls within a single frame.  Call
    :meth:`invalidate` via ``_begin_tracker_round`` at the start of each
    tracker round to clear stale entries from the previous round.

    ``max_age_s`` is a programming-error backstop: if more than
    ``max_age_s`` seconds elapse between invalidations the cache
    self-invalidates and logs a warning rather than silently serving
    stale gallery data.  It is not an operator-tunable knob.
    """

    _repo: GalleryRepository
    max_age_s: float = 5.0
    _entries_by_tracklets: dict[
        tuple[frozenset[str], int, frozenset[str], frozenset[str]],
        list[GalleryEmbedding],
    ] = field(default_factory=dict)
    _similarity_cache: dict[
        tuple[frozenset[str], frozenset[str], int, frozenset[str], frozenset[str]],
        float,
    ] = field(default_factory=dict)
    _invalidated_at: float = field(default_factory=time.monotonic, init=False)

    def invalidate(self) -> None:
        """Clear all cached state.  Call via ``_begin_tracker_round``."""
        self._entries_by_tracklets.clear()
        self._similarity_cache.clear()
        self._invalidated_at = time.monotonic()

    def _check_stale(self) -> None:
        if time.monotonic() - self._invalidated_at > self.max_age_s:
            self.invalidate()
            logger.warning("gallery_cache_stale_self_invalidated")

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> list[GalleryEmbedding]:
        self._check_stale()
        # Use frozen sets to ensure the cache key is immutable and hashable
        s_states = frozenset(allowed_states) if allowed_states else frozenset(["operator_verified"])
        s_models = frozenset(model_versions) if model_versions else frozenset()
        key = (frozenset(tracklet_ids), limit, s_states, s_models)

        if key not in self._entries_by_tracklets:
            self._entries_by_tracklets[key] = await self._repo.list_gallery_entries_for_tracklets(
                tracklet_ids,
                limit=limit,
                allowed_states=allowed_states or {"operator_verified"},
                model_versions=model_versions,
            )
        return self._entries_by_tracklets[key]

    async def gallery_similarity(
        self,
        tracklet_ids_a: set[str],
        tracklet_ids_b: set[str],
        limit: int = 20,
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> float:
        self._check_stale()
        key_a = frozenset(tracklet_ids_a)
        key_b = frozenset(tracklet_ids_b)
        s_states = frozenset(allowed_states) if allowed_states else frozenset(["operator_verified"])
        s_models = frozenset(model_versions) if model_versions else frozenset()

        cache_key = (
            (key_a, key_b, limit, s_states, s_models)
            if hash(key_a) <= hash(key_b)
            else (key_b, key_a, limit, s_states, s_models)
        )

        if cache_key not in self._similarity_cache:
            entries_a = await self.list_gallery_entries_for_tracklets(
                tracklet_ids_a, limit, allowed_states, model_versions
            )
            entries_b = await self.list_gallery_entries_for_tracklets(
                tracklet_ids_b, limit, allowed_states, model_versions
            )
            if not entries_a and not entries_b:
                self._similarity_cache[cache_key] = 0.0
            elif not entries_a or not entries_b:
                self._similarity_cache[cache_key] = 0.5
            else:
                self._similarity_cache[cache_key] = GalleryRepository._cosine_between_centroids(
                    entries_a, entries_b
                )
        return self._similarity_cache[cache_key]
