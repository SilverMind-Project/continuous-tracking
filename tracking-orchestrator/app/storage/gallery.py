"""Gallery embedding and identity storage."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

from ..domain import GalleryEmbedding, Identity


class GalleryRepository(ABC):
    """Persist identities and their gallery embeddings."""

    @abstractmethod
    async def upsert_identity(self, identity: Identity) -> str:
        """Store or update an identity. Returns the identity ID."""

    @abstractmethod
    async def get_identity(self, identity_id: str) -> Identity | None:
        """Retrieve an identity by ID."""

    @abstractmethod
    async def list_identities(self, active_only: bool = True) -> list[Identity]:
        """List all identities."""

    @abstractmethod
    async def upsert_gallery_entry(self, entry: GalleryEmbedding) -> str:
        """Store or update a gallery embedding. Returns the identity ID."""

    @abstractmethod
    async def get_gallery_entry(self, gallery_entry_id: str) -> GalleryEmbedding | None:
        """Retrieve a gallery embedding row by ID."""

    @abstractmethod
    async def list_gallery_entries(
        self, identity_id: str | None = None, active_only: bool = True
    ) -> list[GalleryEmbedding]:
        """List gallery embeddings."""

    @abstractmethod
    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
    ) -> list[tuple[GalleryEmbedding, float]]:
        """Nearest-neighbor search over gallery embeddings.

        Returns a list of (GalleryEmbedding, similarity_score) tuples,
        sorted by similarity descending. The similarity score is cosine
        similarity in [0, 1].

        Args:
            embedding: query embedding vector.
            limit: maximum number of results.
            camera_id: if provided, filter to gallery entries from this camera.
            max_age_seconds: if provided, filter to entries newer than now - max_age_seconds.
        """

    @abstractmethod
    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
    ) -> list[GalleryEmbedding]:
        """List gallery entries whose origin_tracklet_id is in *tracklet_ids*.

        Used by the identity resolver to build a real query embedding from
        a GlobalTrack's existing gallery entries.
        """

    @abstractmethod
    async def update_identity_for_tracklets(
        self,
        tracklet_ids: set[str],
        identity_id: str,
    ) -> int:
        """Backfill identity_id on all gallery entries for the given tracklets.

        Called after the identity resolver commits an identity so that
        future ReID gallery searches can use these entries as identity
        evidence.  Returns the number of rows updated.
        """

    @abstractmethod
    async def gallery_similarity(
        self,
        tracklet_ids_a: set[str],
        tracklet_ids_b: set[str],
        limit: int = 20,
    ) -> float:
        """Mean cosine similarity between two groups of gallery embeddings.

        Computes the centroid embedding for each group and returns their
        cosine similarity. Returns 0.0 when both groups have no gallery
        entries, and 0.5 when only one group has entries (conservative
        fallback that allows geometry to carry cross-camera pairs).
        """

    @staticmethod
    def _cosine_between_centroids(
        entries_a: list[GalleryEmbedding],
        entries_b: list[GalleryEmbedding],
    ) -> float:
        """Cosine similarity between the mean embeddings of two entry lists.

        Returns 0.0 when either list is empty or either centroid has zero norm.
        """
        import numpy as np

        if not entries_a or not entries_b:
            return 0.0
        emb_a = np.mean([e.embedding for e in entries_a], axis=0)
        emb_b = np.mean([e.embedding for e in entries_b], axis=0)
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b + 1e-9))


class InMemoryGalleryRepository(GalleryRepository):
    def __init__(self) -> None:
        self._identities: dict[str, Identity] = {}
        self._entries: dict[str, GalleryEmbedding] = {}

    async def upsert_identity(self, identity: Identity) -> str:
        self._identities[identity.identity_id] = identity
        return identity.identity_id

    async def get_identity(self, identity_id: str) -> Identity | None:
        return self._identities.get(identity_id)

    async def list_identities(self, active_only: bool = True) -> list[Identity]:
        identities = list(self._identities.values())
        if active_only:
            identities = [identity for identity in identities if identity.is_active]
        return identities

    async def upsert_gallery_entry(self, entry: GalleryEmbedding) -> str:
        self._entries[entry.gallery_entry_id] = entry
        return entry.identity_id

    async def get_gallery_entry(self, gallery_entry_id: str) -> GalleryEmbedding | None:
        return self._entries.get(gallery_entry_id)

    async def list_gallery_entries(
        self, identity_id: str | None = None, active_only: bool = True
    ) -> list[GalleryEmbedding]:
        entries = list(self._entries.values())
        if identity_id is not None:
            entries = [entry for entry in entries if entry.identity_id == identity_id]
        if active_only:
            active_ids = {
                identity.identity_id for identity in await self.list_identities(active_only=True)
            }
            entries = [
                entry
                for entry in entries
                if entry.identity_id in active_ids or entry.identity_id == ""
            ]
        return entries

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
    ) -> list[tuple[GalleryEmbedding, float]]:
        entries = await self.list_gallery_entries()
        if camera_id is not None:
            entries = [e for e in entries if e.camera_id == camera_id]
        if max_age_seconds is not None:
            cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
            entries = [e for e in entries if e.seen_at >= cutoff]
        scored = [(entry, _entry_cosine_sim(embedding, entry.embedding)) for entry in entries]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
    ) -> list[GalleryEmbedding]:
        if not tracklet_ids:
            return []
        entries = [
            entry for entry in self._entries.values() if entry.origin_tracklet_id in tracklet_ids
        ]
        entries.sort(key=lambda e: e.seen_at, reverse=True)
        return entries[:limit]

    async def update_identity_for_tracklets(
        self,
        tracklet_ids: set[str],
        identity_id: str,
    ) -> int:
        updated = 0
        for entry_id, entry in self._entries.items():
            if entry.origin_tracklet_id in tracklet_ids and not entry.identity_id:
                self._entries[entry_id] = GalleryEmbedding(
                    gallery_entry_id=entry.gallery_entry_id,
                    identity_id=identity_id,
                    embedding=entry.embedding,
                    seen_at=entry.seen_at,
                    quality=entry.quality,
                    origin_tracklet_id=entry.origin_tracklet_id,
                    face_confirmed=entry.face_confirmed,
                    camera_id=entry.camera_id,
                    orientation=entry.orientation,
                )
                updated += 1
        return updated

    async def gallery_similarity(
        self,
        tracklet_ids_a: set[str],
        tracklet_ids_b: set[str],
        limit: int = 20,
    ) -> float:
        entries_a = await self.list_gallery_entries_for_tracklets(tracklet_ids_a, limit)
        entries_b = await self.list_gallery_entries_for_tracklets(tracklet_ids_b, limit)
        if not entries_a and not entries_b:
            return 0.0
        if not entries_a or not entries_b:
            return 0.5
        return self._cosine_between_centroids(entries_a, entries_b)


def _entry_cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
