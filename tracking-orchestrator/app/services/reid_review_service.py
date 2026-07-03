"""ReIDReviewService: the single owner of the M09 gallery review queue.

The governed ReID gallery (M05) accumulates ``pending_review`` candidates with
full crop and frame provenance. This service is the one place that lists those
candidates, exposes per-candidate eligibility, and applies an operator decision
(approve, relabel, reject) under an optimistic ``audit_version`` guard.

Authority boundaries:

* Approval is never granted when current server eligibility is false (stale row,
  wrong lifecycle state, or an incompatible model/preprocessing version). The
  browser cannot override a failed gate; the check runs here on the live row.
* Rejection nulls the embedding (in the repository transaction) and then removes
  the dedicated crop object. The crop key, hashes, and audit history survive as a
  fingerprint, so rejected history renders without a broken image.
* Compensation un-verifies an approved or relabelled candidate back to
  ``pending_review`` with a new event. It never restores a rejected vector and
  never deletes a prior review event.

The router is a thin adapter over this service; it performs no query logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from structlog import get_logger

from ..domain import ReviewCandidate, ReviewEvent
from ..storage.gallery import (
    GalleryRepository,
    ReviewConflictError,
    ReviewNotFoundError,
)

logger = get_logger(__name__)


class ReviewIneligibleError(Exception):
    """The candidate cannot be approved/relabelled under current eligibility."""


@dataclass(frozen=True)
class Eligibility:
    """Server-computed approval gate for one candidate."""

    eligible: bool
    model_compatible: bool
    reasons: list[str]


@dataclass(frozen=True)
class BatchRejectOutcome:
    """Per-item result for a batch rejection."""

    candidate_id: str
    ok: bool
    error_code: str | None = None
    error_message: str | None = None


class ReIDReviewService:
    def __init__(
        self,
        gallery_repo: GalleryRepository,
        *,
        storage_client: object | None = None,
        active_model_versions: Iterable[str] | None = None,
        active_preprocessing_versions: Iterable[str] | None = None,
    ) -> None:
        self._repo = gallery_repo
        self._storage = storage_client
        # ``None`` means "no active-version constraint configured", so model
        # compatibility is not asserted. A configured set surfaces incompatibility
        # and blocks approval of stale-model candidates.
        self._active_models = (
            set(active_model_versions) if active_model_versions is not None else None
        )
        self._active_preproc = (
            set(active_preprocessing_versions)
            if active_preprocessing_versions is not None
            else None
        )

    # -- reads ----------------------------------------------------------------

    async def list_candidates(
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
        return await self._repo.list_review_candidates(
            state=state,
            identity_id=identity_id,
            camera_id=camera_id,
            model_version=model_version,
            source_type=source_type,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )

    async def get_detail(
        self, candidate_id: str
    ) -> tuple[ReviewCandidate, list[ReviewEvent], Eligibility] | None:
        candidate = await self._repo.get_review_candidate(candidate_id)
        if candidate is None:
            return None
        events = await self._repo.list_review_events(candidate_id)
        return (candidate, events, self.eligibility(candidate))

    async def list_events(self, candidate_id: str) -> list[ReviewEvent]:
        return await self._repo.list_review_events(candidate_id)

    async def counts(self) -> dict[str, int]:
        return await self._repo.count_review_queue()

    def eligibility(self, candidate: ReviewCandidate) -> Eligibility:
        reasons: list[str] = []
        if candidate.state != "pending_review":
            reasons.append(f"already_reviewed:{candidate.state}")
        if candidate.is_truncated:
            reasons.append("truncated")
        if candidate.is_occluded:
            reasons.append("occluded")
        model_compatible = True
        if self._active_models is not None and candidate.model_version not in self._active_models:
            model_compatible = False
            reasons.append(f"incompatible_model:{candidate.model_version}")
        if (
            self._active_preproc is not None
            and candidate.preprocessing_version not in self._active_preproc
        ):
            model_compatible = False
            reasons.append(f"incompatible_preprocessing:{candidate.preprocessing_version}")
        return Eligibility(
            eligible=not reasons,
            model_compatible=model_compatible,
            reasons=reasons,
        )

    # -- mutations ------------------------------------------------------------

    async def approve(
        self,
        candidate_id: str,
        *,
        actor: str,
        base_audit_version: int,
        reason: str | None = None,
        note: str | None = None,
    ) -> ReviewCandidate:
        await self._require_eligible(candidate_id)
        return await self._repo.apply_review_action(
            candidate_id,
            action="approve",
            actor=actor,
            base_audit_version=base_audit_version,
            reason=reason,
            note=note,
        )

    async def relabel(
        self,
        candidate_id: str,
        *,
        new_identity_id: str,
        actor: str,
        base_audit_version: int,
        reason: str | None = None,
        note: str | None = None,
    ) -> ReviewCandidate:
        if not new_identity_id:
            raise ReviewIneligibleError("relabel requires a target identity")
        await self._require_eligible(candidate_id)
        return await self._repo.apply_review_action(
            candidate_id,
            action="relabel",
            actor=actor,
            base_audit_version=base_audit_version,
            reason=reason,
            note=note,
            new_identity_id=new_identity_id,
        )

    async def reject(
        self,
        candidate_id: str,
        *,
        actor: str,
        base_audit_version: int,
        reason: str,
        note: str | None = None,
    ) -> ReviewCandidate:
        updated = await self._repo.apply_review_action(
            candidate_id,
            action="reject",
            actor=actor,
            base_audit_version=base_audit_version,
            reason=reason,
            note=note,
        )
        await self._delete_crop_object(updated)
        return updated

    async def reject_batch(
        self,
        items: list[dict[str, Any]],
        *,
        actor: str,
    ) -> list[BatchRejectOutcome]:
        """Reject several candidates; report per-item outcomes, never aborting the batch.

        Idempotent: a candidate already rejected (conflict) is reported as a
        per-item error rather than failing the whole request.
        """
        outcomes: list[BatchRejectOutcome] = []
        for item in items:
            candidate_id = item["candidate_id"]
            try:
                await self.reject(
                    candidate_id,
                    actor=actor,
                    base_audit_version=item["base_audit_version"],
                    reason=item.get("reason", "batch_reject"),
                    note=item.get("note"),
                )
                outcomes.append(BatchRejectOutcome(candidate_id=candidate_id, ok=True))
            except ReviewNotFoundError:
                outcomes.append(
                    BatchRejectOutcome(
                        candidate_id=candidate_id,
                        ok=False,
                        error_code="not_found",
                        error_message="candidate does not exist",
                    )
                )
            except ReviewConflictError as exc:
                outcomes.append(
                    BatchRejectOutcome(
                        candidate_id=candidate_id,
                        ok=False,
                        error_code="conflict",
                        error_message=str(exc),
                    )
                )
        return outcomes

    async def compensate(self, candidate_id: str, *, actor: str) -> ReviewCandidate:
        """Un-verify an approved/relabelled candidate back to ``pending_review``.

        Never restores a rejected vector. Adds a compensating event without
        deleting any prior event.
        """
        candidate = await self._repo.get_review_candidate(candidate_id)
        if candidate is None:
            raise ReviewNotFoundError(candidate_id)
        if candidate.state != "operator_verified":
            raise ReviewConflictError(
                f"only an operator_verified candidate can be compensated (state={candidate.state})"
            )
        return await self._repo.compensate_review(
            candidate_id,
            actor=actor,
            base_audit_version=candidate.audit_version,
        )

    # -- internals ------------------------------------------------------------

    async def _require_eligible(self, candidate_id: str) -> ReviewCandidate:
        candidate = await self._repo.get_review_candidate(candidate_id)
        if candidate is None:
            raise ReviewNotFoundError(candidate_id)
        elig = self.eligibility(candidate)
        if not elig.eligible:
            raise ReviewIneligibleError(",".join(elig.reasons) or "ineligible")
        return candidate

    async def _delete_crop_object(self, candidate: ReviewCandidate) -> None:
        if self._storage is None or not candidate.crop_key:
            return
        try:
            await self._storage.delete_object(candidate.crop_key)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a reconciliation job sweeps orphans
            logger.warning(
                "reid_review_crop_delete_failed",
                candidate_id=candidate.candidate_id,
                crop_key=candidate.crop_key,
            )
