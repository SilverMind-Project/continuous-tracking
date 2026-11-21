"""Wearable teaching revocation: undo everything a wearable association segment taught.

Revocation is scoped by association segment (``segment_id``), the provenance key recorded on
every artifact a wearable teaches:

1. ReID gallery ``auto_verified`` rows with ``seed_association_id = segment_id`` are demoted
   to ``pending_review`` through ``GalleryRepository.apply_review_action(action="demote")``
   (one audit event per row, vector preserved, never rejected or deleted; ``operator_verified``
   rows stay untouched because operator truth outranks automation).
2. Attested face captures in person-identification-service are marked revoked via
   ``POST /api/v1/calibration/attestations/revoke``; already-promoted face prototypes are
   returned for operator review and never removed automatically.
3. Live inferred backfill ranges with ``source_association_id = segment_id`` are compensated
   through ``IdentityCorrectionService.compensate_inferred_range`` using the same projection
   and ack path as operator compensation.
4. A teaching hold is opened on ``segment.tag_id`` in ``wearable_teaching_holds``, pausing
   further gallery seeding, face attestation, and wearable backfill for that tag until an
   operator explicitly resumes teaching.
5. The segment is stamped ``revoked_at``, ``revoked_by``, and ``revoke_reason``.

Every step is idempotent per segment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from structlog import get_logger

from app.domain import WearableTeachingHold
from app.inference.face_id_client import (
    FaceAttestationCounts,
    FaceRevocationResult,
    PromotedAttestationRef,
)
from app.services.identity_correction_service import IdentityCorrectionService
from app.storage.corrections import IdentityCorrectionRepositoryProtocol
from app.storage.gallery import GalleryRepository, ReviewConflictError
from app.storage.wearable_associations import WearableAssociationRepository
from app.storage.wearable_teaching_holds import WearableTeachingHoldRepository

logger = get_logger(__name__)

RevokeReason = Literal["swap_suspected", "operator"]


class SegmentNotFoundError(Exception):
    """Raised when revocation or taught-summary targets an unknown association segment."""


class FaceRevocationClient(Protocol):
    """Minimal person-identification-service contract for attestation revocation and counts."""

    async def revoke_attestations(
        self, *, association_id: str, revoked_by: str
    ) -> FaceRevocationResult | None: ...

    async def count_attestations(self, *, association_id: str) -> FaceAttestationCounts | None: ...


@dataclass(frozen=True)
class RevocationSummary:
    """Audit-visible result of revoking one wearable association segment."""

    segment_id: str
    tag_id: str
    reason: RevokeReason
    actor: str
    already_revoked: bool
    gallery_rows_demoted: int
    attested_faces_revoked: int | None
    already_promoted_faces: tuple[PromotedAttestationRef, ...]
    inferred_ranges_compensated: int
    hold: WearableTeachingHold


@dataclass(frozen=True)
class SegmentTaughtSummary:
    """Counts of every artifact one association segment taught, plus revocation/hold state."""

    segment_id: str
    tag_id: str
    gallery_rows_by_state: dict[str, int]
    gallery_rows_total: int
    attested_faces_total: int | None
    attested_faces_active: int | None
    attested_faces_revoked: int | None
    attested_faces_promoted: int | None
    inferred_ranges_live: int
    inferred_ranges_total: int
    revoked_at: datetime | None
    revoked_by: str | None
    revoke_reason: str | None
    tag_teaching_held: bool


class WearableTeachingRevocationService:
    """Coordinates idempotent per-segment revocation and per-tag teaching holds."""

    def __init__(
        self,
        *,
        association_repo: WearableAssociationRepository,
        teaching_holds: WearableTeachingHoldRepository,
        gallery_repo: GalleryRepository,
        correction_repo: IdentityCorrectionRepositoryProtocol | None = None,
        correction_service: IdentityCorrectionService | None = None,
        face_id_client: FaceRevocationClient | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._assoc = association_repo
        self._holds = teaching_holds
        self._gallery = gallery_repo
        self._correction_repo = correction_repo
        self._correction_service = correction_service
        self._face_client = face_id_client
        self._clock = clock

    def set_face_id_client(self, client: FaceRevocationClient | None) -> None:
        """Attach the person-identification client after pipeline initialization."""
        self._face_client = client

    async def revoke(
        self,
        segment_id: str,
        *,
        actor: str,
        reason: RevokeReason,
    ) -> RevocationSummary:
        """Undo everything *segment_id* taught and place a teaching hold on its tag."""
        segment = await self._assoc.get_segment(segment_id)
        if segment is None:
            raise SegmentNotFoundError(segment_id)

        already_revoked = segment.revoked_at is not None
        now = self._clock()

        # 1. Demote every auto_verified gallery row minted by this segment (never reject).
        demoted = 0
        demote_reason = f"wearable_revoked:{reason}"
        offset = 0
        batch_limit = 200
        while True:
            rows, _total = await self._gallery.list_review_candidates(
                state="auto_verified",
                seed_association_id=segment_id,
                limit=batch_limit,
                offset=offset,
            )
            if not rows:
                break
            progressed = False
            for candidate in rows:
                try:
                    await self._gallery.apply_review_action(
                        candidate.candidate_id,
                        action="demote",
                        actor=actor,
                        base_audit_version=candidate.audit_version,
                        reason=demote_reason,
                        note=f"association:{segment_id}",
                    )
                    demoted += 1
                    progressed = True
                except ReviewConflictError:
                    # Row moved concurrently (already demoted/approved/rejected); safe to skip.
                    continue
            if not progressed:
                # Remaining rows in this page all conflicted; advance offset to avoid a spin.
                offset += len(rows)

        # 2. Ask person-identification-service to revoke attested faces from this segment.
        faces_revoked: int | None = None
        already_promoted: tuple[PromotedAttestationRef, ...] = ()
        if self._face_client is not None:
            face_res = await self._face_client.revoke_attestations(
                association_id=segment_id,
                revoked_by=actor,
            )
            if face_res is not None:
                faces_revoked = face_res.revoked
                already_promoted = face_res.already_promoted

        # 3. Compensate live inferred backfill ranges sourced from this segment.
        compensated = 0
        if self._correction_repo is not None and self._correction_service is not None:
            live_ranges = await self._correction_repo.list_ranges_by_source_association(
                segment_id, live_only=True
            )
            for rng in live_ranges:
                res = await self._correction_service.compensate_inferred_range(
                    rng.range_id,
                    actor=actor,
                    reason=f"wearable_revoked:{reason}",
                )
                if res is not None:
                    compensated += 1

        # 4. Hold further teaching for this tag until an operator explicitly resumes it.
        hold = await self._holds.create_hold(
            tag_id=segment.tag_id,
            reason=reason,
            held_at=now,
        )

        # 5. Stamp the segment itself (no-op if already stamped).
        await self._assoc.mark_revoked(
            segment_id,
            revoked_at=now,
            revoked_by=actor,
            revoke_reason=reason,
        )

        logger.info(
            "wearable_teaching_revoked",
            segment_id=segment_id,
            tag_id=segment.tag_id,
            actor=actor,
            reason=reason,
            already_revoked=already_revoked,
            gallery_rows_demoted=demoted,
            attested_faces_revoked=faces_revoked,
            already_promoted_faces=len(already_promoted),
            inferred_ranges_compensated=compensated,
        )
        return RevocationSummary(
            segment_id=segment_id,
            tag_id=segment.tag_id,
            reason=reason,
            actor=actor,
            already_revoked=already_revoked,
            gallery_rows_demoted=demoted,
            attested_faces_revoked=faces_revoked,
            already_promoted_faces=already_promoted,
            inferred_ranges_compensated=compensated,
            hold=hold,
        )

    async def revoke_for_swap(
        self,
        segment_id: str,
        *,
        actor: str = "system:swap_guard",
        window_s: float = 600.0,
    ) -> list[RevocationSummary]:
        """Revoke *segment_id* and any unrevoked segments of its tag inside the conflict window.

        ``record_face_conflict`` closes an active segment on the first face conflict and only
        marks ``swap_suspected`` once ``swap_conflict_count`` conflicts occur within
        ``swap_conflict_window_s``. Sweeping unrevoked segments of the same tag overlapping
        that window ensures rows taught just before conflict 1 are demoted too.
        """
        trigger = await self._assoc.get_segment(segment_id)
        if trigger is None:
            raise SegmentNotFoundError(segment_id)
        until = trigger.ended_at or trigger.started_at
        since = until - timedelta(seconds=window_s)
        candidates = await self._assoc.list_unrevoked_overlapping(
            trigger.tag_id, since=since, until=until
        )
        target_ids: list[str] = [seg.segment_id for seg in candidates]
        if segment_id not in target_ids:
            target_ids.append(segment_id)
        summaries: list[RevocationSummary] = []
        for sid in target_ids:
            summaries.append(await self.revoke(sid, actor=actor, reason="swap_suspected"))
        return summaries

    async def resume_teaching(self, tag_id: str, *, actor: str) -> WearableTeachingHold | None:
        """Clear any open teaching hold for *tag_id* so it may teach again."""
        cleared = await self._holds.clear_hold(tag_id, cleared_by=actor, cleared_at=self._clock())
        if cleared is not None:
            logger.info("wearable_teaching_resumed", tag_id=tag_id, actor=actor)
        return cleared

    async def list_open_holds(self) -> list[WearableTeachingHold]:
        """Return every currently open teaching hold."""
        return await self._holds.list_open_holds()

    async def get_taught_summary(self, segment_id: str) -> SegmentTaughtSummary:
        """Report counts of gallery rows, attested faces, and inferred ranges for *segment_id*."""
        segment = await self._assoc.get_segment(segment_id)
        if segment is None:
            raise SegmentNotFoundError(segment_id)

        by_state = await self._gallery.count_by_seed_association(segment_id)
        gallery_total = sum(by_state.values())

        faces_total: int | None = None
        faces_active: int | None = None
        faces_revoked: int | None = None
        faces_promoted: int | None = None
        if self._face_client is not None:
            face_counts = await self._face_client.count_attestations(association_id=segment_id)
            if face_counts is not None:
                faces_total = face_counts.total
                faces_active = face_counts.active
                faces_revoked = face_counts.revoked
                faces_promoted = face_counts.promoted

        live_ranges = 0
        total_ranges = 0
        if self._correction_repo is not None:
            all_ranges = await self._correction_repo.list_ranges_by_source_association(
                segment_id, live_only=False
            )
            total_ranges = len(all_ranges)
            live_ranges = sum(
                1
                for r in all_ranges
                if r.superseded_by_range_id is None and r.compensated_by_revision_id is None
            )

        held = await self._holds.is_held(segment.tag_id)
        return SegmentTaughtSummary(
            segment_id=segment.segment_id,
            tag_id=segment.tag_id,
            gallery_rows_by_state=by_state,
            gallery_rows_total=gallery_total,
            attested_faces_total=faces_total,
            attested_faces_active=faces_active,
            attested_faces_revoked=faces_revoked,
            attested_faces_promoted=faces_promoted,
            inferred_ranges_live=live_ranges,
            inferred_ranges_total=total_ranges,
            revoked_at=segment.revoked_at,
            revoked_by=segment.revoked_by,
            revoke_reason=segment.revoke_reason,
            tag_teaching_held=held,
        )
