"""ReID review-queue endpoints consumed by the CC BFF (M09).

A thin adapter over :class:`ReIDReviewService`, the single owner of the governed
gallery review queue. Routes never touch repositories directly. Approval and
relabel re-check live eligibility server-side and return 409 when the candidate
moved or became ineligible; bulk approval is intentionally absent.

Surface (all under ``/internal/reid-review``):

* ``GET  /candidates`` -- paginated pending list with filters.
* ``GET  /candidates/{id}`` -- detail, history, and eligibility.
* ``GET  /candidates/{id}/events`` -- review history.
* ``POST /candidates/{id}/approve`` -- individual approval (eligibility-gated).
* ``POST /candidates/{id}/relabel`` -- individual relabel to a target identity.
* ``POST /candidates/{id}/reject`` -- individual rejection.
* ``POST /reject-batch`` -- batch rejection with per-item results.
* ``POST /candidates/{id}/compensate`` -- un-verify an approved candidate.
* ``GET  /counts`` -- queue counts for Keyframe/PH indicators.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from structlog import get_logger

from ..domain import ReviewCandidate, ReviewEvent
from ..services.reid_review_service import Eligibility, ReIDReviewService, ReviewIneligibleError
from ..storage.gallery import ReviewConflictError, ReviewNotFoundError

logger = get_logger(__name__)

router = APIRouter(tags=["reid-review-internal"])

_HTTP_409 = status.HTTP_409_CONFLICT
_HTTP_404 = status.HTTP_404_NOT_FOUND


@dataclass
class _ReviewContext:
    service: ReIDReviewService | None


_ctx: _ReviewContext = _ReviewContext(service=None)


def get_context() -> _ReviewContext:
    return _ctx


def set_context(service: ReIDReviewService) -> None:
    """Wire the production review service at startup (called from lifespan)."""
    global _ctx
    _ctx = _ReviewContext(service=service)


def _require_service(ctx: _ReviewContext) -> ReIDReviewService:
    if ctx.service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "reid_review.unavailable", "message": "service not wired"},
        )
    return ctx.service


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EligibilityModel(BaseModel):
    eligible: bool
    model_compatible: bool
    reasons: list[str]


class CandidateModel(BaseModel):
    candidate_id: str
    identity_id: str | None
    proposed_identity_id: str | None
    effective_identity_id: str | None
    state: str
    label_source: str | None
    candidate_reason: str | None
    model_version: str | None
    preprocessing_version: str | None
    dimension: int | None
    crop_key: str | None
    source_frame_key: str | None
    crop_hash: str | None
    frame_hash: str | None
    bbox: dict[str, Any] | None
    crop_width: int | None
    crop_height: int | None
    ph_id: str | None
    observation_id: str | None
    keyframe_id: str | None
    camera_id: str | None
    capture_time: datetime | None
    confidence: float | None
    orientation: int
    quality: float
    is_truncated: bool
    is_occluded: bool
    source_episode_id: str | None
    created_actor: str | None
    created_at: datetime | None
    seen_at: datetime | None
    reviewed_actor: str | None
    reviewed_time: datetime | None
    review_reason: str | None
    review_note: str | None
    audit_version: int


class CandidateListResponse(BaseModel):
    candidates: list[CandidateModel]
    total: int
    limit: int
    offset: int


class EventModel(BaseModel):
    event_id: str
    entry_id: str
    previous_state: str
    new_state: str
    actor: str
    reason: str | None
    note: str | None
    event_time: datetime
    audit_version: int


class CandidateDetailResponse(BaseModel):
    candidate: CandidateModel
    events: list[EventModel]
    eligibility: EligibilityModel


class EventsResponse(BaseModel):
    events: list[EventModel]


class CountsResponse(BaseModel):
    pending_review: int
    auto_verified: int
    operator_verified: int
    rejected: int


class ApproveRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)
    base_audit_version: int
    reason: str | None = Field(default=None, max_length=512)
    note: str | None = Field(default=None, max_length=2048)


class RelabelRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)
    base_audit_version: int
    target_identity_id: str = Field(..., min_length=1, max_length=128)
    reason: str | None = Field(default=None, max_length=512)
    note: str | None = Field(default=None, max_length=2048)


class RejectRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)
    base_audit_version: int
    reason: str = Field(..., min_length=1, max_length=512)
    note: str | None = Field(default=None, max_length=2048)


class DemoteRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)
    base_audit_version: int
    reason: str | None = Field(default=None, max_length=512)
    note: str | None = Field(default=None, max_length=2048)


class CompensateRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)


class BatchRejectItem(BaseModel):
    candidate_id: str = Field(..., min_length=1, max_length=128)
    base_audit_version: int
    reason: str = Field(default="batch_reject", max_length=512)
    note: str | None = Field(default=None, max_length=2048)


class BatchRejectRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)
    items: list[BatchRejectItem] = Field(..., min_length=1, max_length=200)


class BatchRejectResultItem(BaseModel):
    candidate_id: str
    ok: bool
    error_code: str | None = None
    error_message: str | None = None


class BatchRejectResponse(BaseModel):
    results: list[BatchRejectResultItem]
    rejected: int
    failed: int


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def _candidate_model(c: ReviewCandidate) -> CandidateModel:
    return CandidateModel(**c.__dict__)


def _event_model(e: ReviewEvent) -> EventModel:
    return EventModel(**e.__dict__)


def _eligibility_model(e: Eligibility) -> EligibilityModel:
    return EligibilityModel(
        eligible=e.eligible, model_compatible=e.model_compatible, reasons=e.reasons
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/internal/reid-review/candidates", response_model=CandidateListResponse)
async def list_candidates(
    state: str = "pending_review",
    identity_id: str | None = None,
    camera_id: str | None = None,
    model_version: str | None = None,
    source_type: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
    ctx: _ReviewContext = Depends(get_context),
) -> CandidateListResponse:
    service = _require_service(ctx)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    rows, total = await service.list_candidates(
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
    return CandidateListResponse(
        candidates=[_candidate_model(c) for c in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/internal/reid-review/counts", response_model=CountsResponse)
async def review_counts(ctx: _ReviewContext = Depends(get_context)) -> CountsResponse:
    service = _require_service(ctx)
    counts = await service.counts()
    return CountsResponse(
        pending_review=counts.get("pending_review", 0),
        auto_verified=counts.get("auto_verified", 0),
        operator_verified=counts.get("operator_verified", 0),
        rejected=counts.get("rejected", 0),
    )


@router.get(
    "/internal/reid-review/candidates/{candidate_id}",
    response_model=CandidateDetailResponse,
)
async def get_candidate(
    candidate_id: str,
    ctx: _ReviewContext = Depends(get_context),
) -> CandidateDetailResponse:
    service = _require_service(ctx)
    detail = await service.get_detail(candidate_id)
    if detail is None:
        raise HTTPException(
            status_code=_HTTP_404,
            detail={"code": "reid_review.not_found", "message": candidate_id},
        )
    candidate, events, eligibility = detail
    return CandidateDetailResponse(
        candidate=_candidate_model(candidate),
        events=[_event_model(e) for e in events],
        eligibility=_eligibility_model(eligibility),
    )


@router.get(
    "/internal/reid-review/candidates/{candidate_id}/events",
    response_model=EventsResponse,
)
async def list_events(
    candidate_id: str,
    ctx: _ReviewContext = Depends(get_context),
) -> EventsResponse:
    service = _require_service(ctx)
    events = await service.list_events(candidate_id)
    return EventsResponse(events=[_event_model(e) for e in events])


@router.post(
    "/internal/reid-review/candidates/{candidate_id}/approve",
    response_model=CandidateModel,
)
async def approve_candidate(
    candidate_id: str,
    body: ApproveRequest,
    ctx: _ReviewContext = Depends(get_context),
) -> CandidateModel:
    service = _require_service(ctx)
    try:
        updated = await service.approve(
            candidate_id,
            actor=body.actor,
            base_audit_version=body.base_audit_version,
            reason=body.reason,
            note=body.note,
        )
    except ReviewNotFoundError as exc:
        _raise_not_found(candidate_id, exc)
    except ReviewIneligibleError as exc:
        _raise_ineligible(exc)
    except ReviewConflictError as exc:
        _raise_conflict(exc)
    return _candidate_model(updated)


@router.post(
    "/internal/reid-review/candidates/{candidate_id}/relabel",
    response_model=CandidateModel,
)
async def relabel_candidate(
    candidate_id: str,
    body: RelabelRequest,
    ctx: _ReviewContext = Depends(get_context),
) -> CandidateModel:
    service = _require_service(ctx)
    try:
        updated = await service.relabel(
            candidate_id,
            new_identity_id=body.target_identity_id,
            actor=body.actor,
            base_audit_version=body.base_audit_version,
            reason=body.reason,
            note=body.note,
        )
    except ReviewNotFoundError as exc:
        _raise_not_found(candidate_id, exc)
    except ReviewIneligibleError as exc:
        _raise_ineligible(exc)
    except ReviewConflictError as exc:
        _raise_conflict(exc)
    return _candidate_model(updated)


@router.post(
    "/internal/reid-review/candidates/{candidate_id}/demote",
    response_model=CandidateModel,
)
async def demote_candidate(
    candidate_id: str,
    body: DemoteRequest,
    ctx: _ReviewContext = Depends(get_context),
) -> CandidateModel:
    service = _require_service(ctx)
    try:
        updated = await service.demote(
            candidate_id,
            actor=body.actor,
            base_audit_version=body.base_audit_version,
            reason=body.reason,
            note=body.note,
        )
    except ReviewNotFoundError as exc:
        _raise_not_found(candidate_id, exc)
    except ReviewConflictError as exc:
        _raise_conflict(exc)
    return _candidate_model(updated)


@router.post(
    "/internal/reid-review/candidates/{candidate_id}/reject",
    response_model=CandidateModel,
)
async def reject_candidate(
    candidate_id: str,
    body: RejectRequest,
    ctx: _ReviewContext = Depends(get_context),
) -> CandidateModel:
    service = _require_service(ctx)
    try:
        updated = await service.reject(
            candidate_id,
            actor=body.actor,
            base_audit_version=body.base_audit_version,
            reason=body.reason,
            note=body.note,
        )
    except ReviewNotFoundError as exc:
        _raise_not_found(candidate_id, exc)
    except ReviewConflictError as exc:
        _raise_conflict(exc)
    return _candidate_model(updated)


@router.post("/internal/reid-review/reject-batch", response_model=BatchRejectResponse)
async def reject_batch(
    body: BatchRejectRequest,
    ctx: _ReviewContext = Depends(get_context),
) -> BatchRejectResponse:
    service = _require_service(ctx)
    outcomes = await service.reject_batch(
        [item.model_dump() for item in body.items], actor=body.actor
    )
    results = [
        BatchRejectResultItem(
            candidate_id=o.candidate_id,
            ok=o.ok,
            error_code=o.error_code,
            error_message=o.error_message,
        )
        for o in outcomes
    ]
    return BatchRejectResponse(
        results=results,
        rejected=sum(1 for o in outcomes if o.ok),
        failed=sum(1 for o in outcomes if not o.ok),
    )


@router.post(
    "/internal/reid-review/candidates/{candidate_id}/compensate",
    response_model=CandidateModel,
)
async def compensate_candidate(
    candidate_id: str,
    body: CompensateRequest,
    ctx: _ReviewContext = Depends(get_context),
) -> CandidateModel:
    service = _require_service(ctx)
    try:
        updated = await service.compensate(candidate_id, actor=body.actor)
    except ReviewNotFoundError as exc:
        _raise_not_found(candidate_id, exc)
    except ReviewConflictError as exc:
        _raise_conflict(exc)
    return _candidate_model(updated)


def _raise_not_found(candidate_id: str, exc: Exception) -> NoReturn:
    raise HTTPException(
        status_code=_HTTP_404,
        detail={"code": "reid_review.not_found", "message": candidate_id},
    ) from exc


def _raise_conflict(exc: Exception) -> NoReturn:
    raise HTTPException(
        status_code=_HTTP_409,
        detail={"code": "reid_review.stale", "message": str(exc)},
    ) from exc


def _raise_ineligible(exc: Exception) -> NoReturn:
    raise HTTPException(
        status_code=_HTTP_409,
        detail={"code": "reid_review.ineligible", "message": str(exc)},
    ) from exc
