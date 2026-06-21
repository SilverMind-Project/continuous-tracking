"""Gallery embedding and identity storage."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

from ..domain import GalleryEmbedding, Identity, ReviewCandidate, ReviewEvent


class ReviewNotFoundError(Exception):
    """A review candidate id does not exist."""


class ReviewConflictError(Exception):
    """The candidate moved under the reviewer (stale audit version or already reviewed)."""


_REVIEW_ACTIONS = frozenset({"approve", "reject", "relabel"})


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
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
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
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> float:
        """Mean cosine similarity between two groups of gallery embeddings.

        Computes the centroid embedding for each group and returns their
        cosine similarity. Returns 0.0 when both groups have no gallery
        entries, and 0.5 when only one group has entries (conservative
        fallback that allows geometry to carry cross-camera pairs).
        """

    async def phs_with_pending_reid(self, ph_ids: list[str]) -> set[str]:
        """PHs in *ph_ids* that have a ``pending_review`` ReID candidate (M07).

        Only ``pending_review`` entries count: ``operator_verified`` candidates
        are already governed and ``rejected`` ones never resurface. Returns the
        subset of *ph_ids* awaiting review so the read model can flag cards.
        """
        return set()

    # -- M09 ReID review queue ------------------------------------------------
    #
    # Default no-op implementations keep non-gallery repository doubles working;
    # InMemory and Postgres override them with real behaviour and parity.

    async def list_review_candidates(
        self,
        *,
        state: str = "pending_review",
        identity_id: str | None = None,
        camera_id: str | None = None,
        model_version: str | None = None,
        source_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ReviewCandidate], int]:
        """Paginated review candidates plus the total matching count."""
        return ([], 0)

    async def get_review_candidate(self, candidate_id: str) -> ReviewCandidate | None:
        """One review candidate with full provenance, or ``None``."""
        return None

    async def list_review_events(self, candidate_id: str) -> list[ReviewEvent]:
        """Immutable review history for one candidate, oldest first."""
        return []

    async def count_review_queue(self) -> dict[str, int]:
        """Counts by lifecycle state (``pending_review``/``operator_verified``/``rejected``)."""
        return {}

    async def apply_review_action(
        self,
        candidate_id: str,
        *,
        action: str,
        actor: str,
        base_audit_version: int,
        reason: str | None = None,
        note: str | None = None,
        new_identity_id: str | None = None,
    ) -> ReviewCandidate:
        """Apply approve/reject/relabel under an optimistic ``audit_version`` guard.

        Rejection nulls the embedding and removes the dedicated crop object;
        audit metadata and fingerprint survive. Raises
        :class:`ReviewConflictError` when the candidate already moved and
        :class:`ReviewNotFoundError` when it does not exist.
        """
        raise NotImplementedError

    async def compensate_review(
        self,
        candidate_id: str,
        *,
        actor: str,
        base_audit_version: int,
    ) -> ReviewCandidate:
        """Un-verify an ``operator_verified`` candidate back to ``pending_review``.

        Records a compensating event and never restores a rejected vector or
        deletes a prior event.
        """
        raise NotImplementedError

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
        # M09 review queue: rich candidates + immutable history, independent of
        # the lean GalleryEmbedding voting rows above.
        self._candidates: dict[str, ReviewCandidate] = {}
        self._events: dict[str, list[ReviewEvent]] = {}

    def seed_review_candidate(self, candidate: ReviewCandidate) -> None:
        """Test seam: insert a review candidate (mirrors a Postgres candidate row)."""
        self._candidates[candidate.candidate_id] = candidate
        self._events.setdefault(candidate.candidate_id, [])

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

    async def phs_with_pending_reid(self, ph_ids: list[str]) -> set[str]:
        wanted = set(ph_ids)
        return {
            entry.origin_tracklet_id
            for entry in self._entries.values()
            if entry.state == "pending_review"
            and entry.origin_tracklet_id in wanted
        }

    # -- M09 ReID review queue ------------------------------------------------

    async def list_review_candidates(
        self,
        *,
        state: str = "pending_review",
        identity_id: str | None = None,
        camera_id: str | None = None,
        model_version: str | None = None,
        source_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ReviewCandidate], int]:
        rows = [c for c in self._candidates.values() if c.state == state]
        if identity_id is not None:
            rows = [c for c in rows if c.identity_id == identity_id]
        if camera_id is not None:
            rows = [c for c in rows if c.camera_id == camera_id]
        if model_version is not None:
            rows = [c for c in rows if c.model_version == model_version]
        if source_type is not None:
            rows = [c for c in rows if c.candidate_reason == source_type]
        if since is not None:
            rows = [c for c in rows if c.capture_time is not None and c.capture_time >= since]
        if until is not None:
            rows = [c for c in rows if c.capture_time is not None and c.capture_time <= until]
        # Oldest first: the queue surfaces the longest-waiting candidate at the top.
        rows.sort(key=lambda c: (c.created_at or c.seen_at or datetime.min.replace(tzinfo=UTC)))
        total = len(rows)
        return (rows[offset : offset + limit], total)

    async def get_review_candidate(self, candidate_id: str) -> ReviewCandidate | None:
        return self._candidates.get(candidate_id)

    async def list_review_events(self, candidate_id: str) -> list[ReviewEvent]:
        events = list(self._events.get(candidate_id, []))
        events.sort(key=lambda e: e.event_time)
        return events

    async def count_review_queue(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "pending_review": 0,
            "operator_verified": 0,
            "rejected": 0,
        }
        for c in self._candidates.values():
            counts[c.state] = counts.get(c.state, 0) + 1
        return counts

    async def apply_review_action(
        self,
        candidate_id: str,
        *,
        action: str,
        actor: str,
        base_audit_version: int,
        reason: str | None = None,
        note: str | None = None,
        new_identity_id: str | None = None,
    ) -> ReviewCandidate:
        if action not in _REVIEW_ACTIONS:
            raise ValueError(f"unknown review action: {action}")
        current = self._candidates.get(candidate_id)
        if current is None:
            raise ReviewNotFoundError(candidate_id)
        if current.state != "pending_review":
            raise ReviewConflictError(
                f"{candidate_id} already reviewed (state={current.state})"
            )
        if current.audit_version != base_audit_version:
            raise ReviewConflictError(
                f"{candidate_id} stale audit_version "
                f"(have {current.audit_version}, sent {base_audit_version})"
            )

        from dataclasses import replace

        now = datetime.now(UTC)
        if action == "reject":
            updated = replace(
                current,
                state="rejected",
                reviewed_actor=actor,
                reviewed_time=now,
                review_reason=reason,
                review_note=note,
                audit_version=current.audit_version + 1,
            )
        elif action == "relabel":
            updated = replace(
                current,
                state="operator_verified",
                identity_id=new_identity_id,
                reviewed_actor=actor,
                reviewed_time=now,
                review_reason=reason,
                review_note=note,
                audit_version=current.audit_version + 1,
            )
        else:  # approve
            updated = replace(
                current,
                state="operator_verified",
                reviewed_actor=actor,
                reviewed_time=now,
                review_reason=reason,
                review_note=note,
                audit_version=current.audit_version + 1,
            )

        self._candidates[candidate_id] = updated
        self._events.setdefault(candidate_id, []).append(
            ReviewEvent(
                event_id=f"evt-{candidate_id}-{updated.audit_version}",
                entry_id=candidate_id,
                previous_state=current.state,
                new_state=updated.state,
                actor=actor,
                reason=reason,
                note=note,
                event_time=now,
                audit_version=updated.audit_version,
            )
        )
        return updated

    async def compensate_review(
        self,
        candidate_id: str,
        *,
        actor: str,
        base_audit_version: int,
    ) -> ReviewCandidate:
        from dataclasses import replace

        current = self._candidates.get(candidate_id)
        if current is None:
            raise ReviewNotFoundError(candidate_id)
        if current.state != "operator_verified":
            raise ReviewConflictError(
                f"{candidate_id} is not operator_verified (state={current.state})"
            )
        if current.audit_version != base_audit_version:
            raise ReviewConflictError(f"{candidate_id} stale audit_version")
        now = datetime.now(UTC)
        updated = replace(
            current,
            state="pending_review",
            reviewed_actor=actor,
            reviewed_time=now,
            review_reason="compensated",
            audit_version=current.audit_version + 1,
        )
        self._candidates[candidate_id] = updated
        self._events.setdefault(candidate_id, []).append(
            ReviewEvent(
                event_id=f"evt-{candidate_id}-{updated.audit_version}",
                entry_id=candidate_id,
                previous_state=current.state,
                new_state="pending_review",
                actor=actor,
                reason="compensated",
                note=None,
                event_time=now,
                audit_version=updated.audit_version,
            )
        )
        return updated

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
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> list[GalleryEmbedding]:
        if not tracklet_ids:
            return []

        # Filter by allowed_states when the caller passes it (gallery_cache
        # passes {"operator_verified"}); otherwise leave entries unfiltered to
        # preserve existing test behaviour.
        if allowed_states is not None:
            entries = [
                entry for entry in self._entries.values()
                if entry.origin_tracklet_id in tracklet_ids
                and entry.state in allowed_states
            ]
        else:
            entries = [
                entry for entry in self._entries.values()
                if entry.origin_tracklet_id in tracklet_ids
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
        allowed_states: set[str] | None = None,
        model_versions: set[str] | None = None,
    ) -> float:
        entries_a = await self.list_gallery_entries_for_tracklets(
            tracklet_ids_a, limit, allowed_states, model_versions
        )
        entries_b = await self.list_gallery_entries_for_tracklets(
            tracklet_ids_b, limit, allowed_states, model_versions
        )
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
