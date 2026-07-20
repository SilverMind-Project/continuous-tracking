"""Gallery embedding and identity storage."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Final

from ..domain import GalleryEmbedding, Identity, NewReviewCandidate, ReviewCandidate, ReviewEvent


class ReviewNotFoundError(Exception):
    """A review candidate id does not exist."""


class ReviewConflictError(Exception):
    """The candidate moved under the reviewer (stale audit version or already reviewed)."""


_REVIEW_ACTIONS = frozenset({"approve", "reject", "relabel", "demote"})

# States apply_review_action's approve/reject/relabel may act on (M02): a
# pending row awaiting first review, or an auto_verified row an operator is
# still free to promote, correct, or reject. demote has its own, narrower
# guard (auto_verified only) enforced at the call site.
REVIEWABLE_STATES: Final[frozenset[str]] = frozenset({"pending_review", "auto_verified"})

# Governance-safe default for every state-sensitive gallery read: only rows an
# operator has verified vote or are administratively visible. `states=None`
# means "no filter" and is reserved for administrative/service callers, each
# of which must carry a one-line justification comment at the call site.
# Retained for admin/list surfaces (list_gallery_entries has no vote-path
# caller); vote-adjacent reads use VOTING_STATES instead (identity-continuity
# M02, decision D3).
VERIFIED_ONLY: Final[frozenset[str]] = frozenset({"operator_verified"})

# Vote-path default (identity-continuity M02, decision D3): both
# operator_verified and auto_verified rows vote in identity resolution, at
# their respective configured trust multipliers (see gallery_scoring.py).
# Every read whose result feeds identity resolution (search_similar, the
# gallery-similarity/tracklet-query paths that build or compare against a
# resolver query embedding) defaults to this set. Pending and rejected rows
# never vote under any default.
VOTING_STATES: Final[frozenset[str]] = frozenset({"operator_verified", "auto_verified"})

# Default state set for the per-(identity, orientation) creation cap (M04, F4;
# extended M02). Pending rows must count against the cap or the review queue
# floods; auto_verified rows must count too or the same-day-face-match
# population (which is the *dominant* state in practice) grows unbounded past
# the cap -- the identical F4 lesson applied to the new state. Never rejected.
PENDING_AND_VERIFIED: Final[frozenset[str]] = frozenset(
    {"pending_review", "operator_verified", "auto_verified"}
)


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
    async def create_review_candidate(self, candidate: NewReviewCandidate) -> str:
        """Create one governed ``pending_review`` gallery row (M04).

        The only pipeline write path into the gallery. ``candidate_id`` is
        caller-supplied and creation is idempotent on it (a retry after a
        partial MinIO/DB failure returns the same id without duplicating the
        row). Exposes the new row to both the lean voting read path
        (:meth:`list_gallery_entries` and friends) and the M09 review queue
        (:meth:`list_review_candidates`) — implementations must not create two
        divergent rows for one candidate.
        """

    @abstractmethod
    async def count_gallery_entries(
        self,
        identity_id: str,
        orientation: int,
        states: frozenset[str] | None = PENDING_AND_VERIFIED,
    ) -> int:
        """Count gallery rows for *(identity_id, orientation)* used by the creation cap.

        Defaults to counting ``pending_review`` and ``operator_verified``
        rows (never ``rejected``) so a flood of unreviewed candidates still
        engages the per-(identity, orientation) cap.
        """

    @abstractmethod
    async def list_gallery_entries(
        self,
        identity_id: str | None = None,
        active_only: bool = True,
        states: frozenset[str] | None = VERIFIED_ONLY,
    ) -> list[GalleryEmbedding]:
        """List gallery embeddings.

        ``states`` restricts by lifecycle state; the default excludes
        pending/rejected rows. ``states=None`` means no state filter and is
        reserved for administrative/service callers, each of which must carry
        a one-line justification comment at the call site.
        """

    @abstractmethod
    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
        states: frozenset[str] | None = VOTING_STATES,
    ) -> list[tuple[GalleryEmbedding, float]]:
        """Nearest-neighbor search over gallery embeddings.

        Returns a list of (GalleryEmbedding, similarity_score) tuples,
        sorted by similarity descending. The similarity score is cosine
        similarity in [0, 1].

        Args:
            embedding: query embedding vector.
            limit: maximum number of results.
            camera_id: if provided, filter to gallery entries from this camera.
            max_age_seconds: if provided, filter to entries with
                seen_at > now - max_age_seconds (strict; an entry exactly at
                the cutoff age is excluded, pinned identically on both the
                InMemory and Postgres peers).
            states: lifecycle-state filter; the default excludes pending/rejected
                rows so unverified vectors never vote (operator_verified and
                auto_verified both vote, at their configured trust
                multipliers). ``states=None`` means no filter and is reserved
                for administrative/service callers, each of which must carry
                a one-line justification comment.
        """

    @abstractmethod
    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
        allowed_states: frozenset[str] | None = VOTING_STATES,
        model_versions: set[str] | None = None,
    ) -> list[GalleryEmbedding]:
        """List gallery entries whose origin_tracklet_id is in *tracklet_ids*.

        Used by the identity resolver to build a real query embedding from
        a GlobalTrack's existing gallery entries. ``allowed_states`` defaults
        to the voting states (operator_verified and auto_verified); an
        entity whose only gallery rows are freshly auto_verified must still
        be able to build a query embedding from its own history.
        ``allowed_states=None`` means no filter and is reserved for
        administrative/service callers, each of which must carry a one-line
        justification comment at the call site.
        """

    @abstractmethod
    async def gallery_similarity(
        self,
        tracklet_ids_a: set[str],
        tracklet_ids_b: set[str],
        limit: int = 20,
        allowed_states: frozenset[str] | None = VOTING_STATES,
        model_versions: set[str] | None = None,
    ) -> float:
        """Mean cosine similarity between two groups of gallery embeddings.

        Computes the centroid embedding for each group and returns their
        cosine similarity. Returns 0.0 when both groups have no gallery
        entries, and 0.5 when only one group has entries (conservative
        fallback that allows geometry to carry cross-camera pairs).
        ``allowed_states`` defaults to the voting states (delegates to
        ``list_gallery_entries_for_tracklets``); ``allowed_states=None`` means
        no filter and is reserved for administrative/service callers, each of
        which must carry a one-line justification comment at the call site.
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
        """Counts by lifecycle state (``pending_review``/``auto_verified``/

        ``operator_verified``/``rejected``).
        """
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
        """Apply approve/reject/relabel/demote under an optimistic ``audit_version`` guard.

        Approve, reject, and relabel act on a row in ``pending_review`` or
        ``auto_verified`` (M02: an operator may still promote, correct, or
        reject a machine-trusted row). Demote acts only on ``auto_verified``
        and returns it to ``pending_review`` without touching the vector
        (an un-trust, not a rejection). Rejection nulls the embedding and
        removes the dedicated crop object; audit metadata and fingerprint
        survive. Raises :class:`ReviewConflictError` when the candidate
        already moved or is in the wrong state for *action*, and
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
        """Un-verify an ``operator_verified`` candidate back to its prior state.

        Restores whatever state the most recent review event's
        ``previous_state`` recorded (``auto_verified`` if this candidate was
        approved/relabelled from there, otherwise ``pending_review``); never
        guesses. Records a compensating event and never restores a rejected
        vector or deletes a prior event. Use :meth:`apply_review_action`'s
        ``demote`` action, not this method, to un-trust an ``auto_verified``
        row that was never promoted.
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

    async def create_review_candidate(self, candidate: NewReviewCandidate) -> str:
        # Idempotent on candidate_id: a retry after a partial MinIO/DB failure
        # must not duplicate the row (M04 recoverable-creation requirement).
        if candidate.candidate_id in self._candidates:
            return candidate.candidate_id

        now = datetime.now(UTC)
        self._candidates[candidate.candidate_id] = ReviewCandidate(
            candidate_id=candidate.candidate_id,
            identity_id=candidate.identity_id,
            proposed_identity_id=candidate.identity_id,
            effective_identity_id=None,
            state=candidate.state,
            label_source=None,
            candidate_reason=candidate.candidate_reason,
            model_version=candidate.model_version,
            preprocessing_version=candidate.preprocessing_version,
            dimension=candidate.dimensions[0] * candidate.dimensions[1],
            crop_key=candidate.crop_key,
            source_frame_key=candidate.source_frame_key,
            crop_hash=candidate.crop_hash,
            frame_hash=candidate.frame_hash,
            bbox=None,
            crop_width=candidate.dimensions[0],
            crop_height=candidate.dimensions[1],
            ph_id=candidate.ph_id,
            observation_id=candidate.observation_id,
            keyframe_id=candidate.keyframe_id,
            camera_id=candidate.camera_id,
            capture_time=candidate.capture_time,
            confidence=candidate.confidence,
            orientation=candidate.orientation,
            quality=candidate.quality,
            is_truncated=candidate.is_truncated,
            is_occluded=candidate.is_occluded,
            source_episode_id=candidate.source_episode_id,
            created_actor=candidate.created_actor,
            created_at=now,
            seen_at=candidate.capture_time,
            reviewed_actor=None,
            reviewed_time=None,
            review_reason=None,
            review_note=None,
            audit_version=1,
        )
        self._events.setdefault(candidate.candidate_id, [])
        # Mirror the same row into the lean voting dict so
        # list_gallery_entries/search_similar/list_gallery_entries_for_tracklets
        # see it once state transitions to operator_verified — Postgres exposes
        # one row to both readers; InMemory must match that single-row semantics.
        self._entries[candidate.candidate_id] = GalleryEmbedding(
            gallery_entry_id=candidate.candidate_id,
            identity_id=candidate.identity_id,
            embedding=candidate.embedding,
            seen_at=candidate.capture_time,
            quality=candidate.quality,
            origin_tracklet_id=candidate.origin_tracklet_id,
            # face_confirmed is a legacy authority boolean nothing reads
            # (M04 rationale); leave it at the domain default rather than
            # asserting authority through a deprecated field a second time.
            camera_id=candidate.camera_id,
            orientation=candidate.orientation,
            state=candidate.state,
            source_episode_id=candidate.source_episode_id,
            ph_id=candidate.ph_id,
        )
        return candidate.candidate_id

    async def count_gallery_entries(
        self,
        identity_id: str,
        orientation: int,
        states: frozenset[str] | None = PENDING_AND_VERIFIED,
    ) -> int:
        return sum(
            1
            for entry in self._entries.values()
            if entry.identity_id == identity_id
            and entry.orientation == orientation
            and (states is None or entry.state in states)
        )

    async def phs_with_pending_reid(self, ph_ids: list[str]) -> set[str]:
        wanted = set(ph_ids)
        matches: set[str] = set()
        for entry in self._entries.values():
            if entry.state != "pending_review":
                continue
            # Prefer the dedicated ph_id (matches Postgres, which stores it in
            # its own column); fall back to origin_tracklet_id for rows seeded
            # by older test/production paths that only set that field.
            key = entry.ph_id or entry.origin_tracklet_id
            if key in wanted:
                matches.add(key)
        return matches

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
        rows.sort(key=lambda c: c.created_at or c.seen_at or datetime.min.replace(tzinfo=UTC))
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
            "auto_verified": 0,
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
        if action == "demote":
            if current.state != "auto_verified":
                raise ReviewConflictError(
                    f"{candidate_id} is not auto_verified (state={current.state})"
                )
        elif current.state not in REVIEWABLE_STATES:
            raise ReviewConflictError(f"{candidate_id} already reviewed (state={current.state})")
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
        elif action == "demote":
            # An operator un-trusting a machine-minted row: back to
            # pending_review, vector kept intact (unlike reject).
            updated = replace(
                current,
                state="pending_review",
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
        self._mirror_entry_state(candidate_id, updated)
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

    def _mirror_entry_state(self, candidate_id: str, updated: ReviewCandidate) -> None:
        """Keep the lean voting row's state in sync with a review transition.

        Postgres has one ``reid_gallery`` row serving both readers, so a
        state change is atomically visible to both. InMemory splits the row
        across two dicts (:attr:`_entries` for voting, :attr:`_candidates`
        for the M09 audit trail); without this, approving a candidate would
        never make it eligible to vote in InMemory tests.
        """
        from dataclasses import replace

        entry = self._entries.get(candidate_id)
        if entry is None:
            return
        self._entries[candidate_id] = replace(
            entry,
            state=updated.state,
            identity_id=updated.identity_id if updated.identity_id else entry.identity_id,
            # Rejection deletes the vector (governance rule); voided here too
            # so a rejected row can never contribute to a similarity query,
            # even one that bypasses the state filter.
            embedding=[] if updated.state == "rejected" else entry.embedding,
        )

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
        # Restore the state the candidate was promoted from, per the most
        # recent review event, rather than assuming pending_review: undoing
        # an approve-from-auto_verified must land back on auto_verified, not
        # silently downgrade a machine-trusted row to pending (M02).
        prior_events = self._events.get(candidate_id, [])
        restore_state = prior_events[-1].previous_state if prior_events else "pending_review"
        now = datetime.now(UTC)
        updated = replace(
            current,
            state=restore_state,
            reviewed_actor=actor,
            reviewed_time=now,
            review_reason="compensated",
            audit_version=current.audit_version + 1,
        )
        self._candidates[candidate_id] = updated
        self._mirror_entry_state(candidate_id, updated)
        self._events.setdefault(candidate_id, []).append(
            ReviewEvent(
                event_id=f"evt-{candidate_id}-{updated.audit_version}",
                entry_id=candidate_id,
                previous_state=current.state,
                new_state=restore_state,
                actor=actor,
                reason="compensated",
                note=None,
                event_time=now,
                audit_version=updated.audit_version,
            )
        )
        return updated

    async def list_gallery_entries(
        self,
        identity_id: str | None = None,
        active_only: bool = True,
        states: frozenset[str] | None = VERIFIED_ONLY,
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
        if states is not None:
            entries = [entry for entry in entries if entry.state in states]
        return entries

    async def search_similar(
        self,
        embedding: list[float],
        limit: int = 10,
        camera_id: str | None = None,
        max_age_seconds: int | None = None,
        states: frozenset[str] | None = VOTING_STATES,
    ) -> list[tuple[GalleryEmbedding, float]]:
        entries = await self.list_gallery_entries(states=None)
        if camera_id is not None:
            entries = [e for e in entries if e.camera_id == camera_id]
        if max_age_seconds is not None:
            # Strict `>` matches the Postgres peer's `seen_at > now() - interval`
            # (identity-continuity M03): an entry exactly at the cutoff age is
            # excluded on both peers.
            cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
            entries = [e for e in entries if e.seen_at > cutoff]
        if states is not None:
            entries = [e for e in entries if e.state in states]
        scored = [(entry, _entry_cosine_sim(embedding, entry.embedding)) for entry in entries]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    async def list_gallery_entries_for_tracklets(
        self,
        tracklet_ids: set[str],
        limit: int = 20,
        allowed_states: frozenset[str] | None = VOTING_STATES,
        model_versions: set[str] | None = None,
    ) -> list[GalleryEmbedding]:
        if not tracklet_ids:
            return []

        if allowed_states is not None:
            entries = [
                entry
                for entry in self._entries.values()
                if entry.origin_tracklet_id in tracklet_ids and entry.state in allowed_states
            ]
        else:
            entries = [
                entry
                for entry in self._entries.values()
                if entry.origin_tracklet_id in tracklet_ids
            ]
        entries.sort(key=lambda e: e.seen_at, reverse=True)
        return entries[:limit]

    async def gallery_similarity(
        self,
        tracklet_ids_a: set[str],
        tracklet_ids_b: set[str],
        limit: int = 20,
        allowed_states: frozenset[str] | None = VOTING_STATES,
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
