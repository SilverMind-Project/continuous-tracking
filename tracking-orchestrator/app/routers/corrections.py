"""Identity-correction endpoints consumed by the CC BFF.

This router is a thin adapter over :class:`IdentityCorrectionService`, the single
owner of operator corrections (M06). Routes never mutate repositories directly.

Surface:

* ``POST /internal/corrections/propose`` -- advisory segment proposal.
* ``POST /internal/corrections/apply`` -- explicit frame-only/bounded correction
  or explicit Set-to-Unknown, guarded by an optimistic version token.
* ``POST /internal/corrections/{correction_id}/compensate`` -- undo via a
  compensating revision (never deletes the original).
* ``POST /internal/projection-acks`` -- downstream projection acknowledgement
  (CC posts here after applying a revision); completes the revision job.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from structlog import get_logger

from ..domain import ProjectionAck
from ..services.identity_correction_service import (
    CorrectionConflictError,
    CorrectionError,
    EmptyIdentityError,
    IdentityCorrectionService,
    PHNotFoundError,
    StaleVersionError,
)

_HTTP_422 = 422  # status.HTTP_422_UNPROCESSABLE_ENTITY is deprecated in Starlette

logger = get_logger(__name__)

router = APIRouter(tags=["corrections-internal"])


# ---------------------------------------------------------------------------
# Dependency wiring: module-level singleton overridden at startup.
# ---------------------------------------------------------------------------


@dataclass
class _CorrectionContext:
    service: IdentityCorrectionService | None


_ctx: _CorrectionContext = _CorrectionContext(service=None)


def get_context() -> _CorrectionContext:
    return _ctx


def set_context(service: IdentityCorrectionService) -> None:
    """Wire the production correction service at startup (called from lifespan)."""
    global _ctx
    _ctx = _CorrectionContext(service=service)


def _require_service(ctx: _CorrectionContext) -> IdentityCorrectionService:
    if ctx.service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "corrections.unavailable", "message": "service not wired"},
        )
    return ctx.service


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ProposeRequest(BaseModel):
    ph_id: str = Field(..., min_length=1, max_length=128)
    observation_id: str | None = Field(default=None, max_length=128)
    at: datetime | None = None


class SegmentBoundaryModel(BaseModel):
    observation_id: str
    captured_at: datetime
    reason: str


class ProposalResponse(BaseModel):
    ph_id: str
    observation_ids: list[str]
    start: SegmentBoundaryModel
    end: SegmentBoundaryModel
    ph_version: int
    effective_identity_id: str | None


class ApplyRequest(BaseModel):
    ph_id: str = Field(..., min_length=1, max_length=128)
    actor: str = Field(..., min_length=1, max_length=128)
    reason_code: str = Field(..., max_length=64)
    observation_start: datetime
    observation_end: datetime
    base_ph_version: int
    target_identity_id: str | None = Field(default=None, max_length=128)
    set_unknown: bool = False
    frame_only: bool = False
    note: str | None = Field(default=None, max_length=2048)
    source_view: str | None = Field(default=None, max_length=64)
    reviewed_frame_id: str | None = Field(default=None, max_length=128)
    reviewed_bbox: dict[str, Any] | None = None
    at_observation_id: str | None = Field(default=None, max_length=128)


class CorrectionResultResponse(BaseModel):
    revision_id: str
    correction_id: str
    ph_id: str
    previous_identity_id: str | None
    new_identity_id: str | None
    range_id: str
    new_ph_id: str | None
    job_status: str


class CompensateRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)


class ProjectionAckRequest(BaseModel):
    revision_id: str = Field(..., min_length=1, max_length=128)
    consumer: str = Field(..., min_length=1, max_length=64)
    schema_version: str = Field(..., min_length=1, max_length=32)
    status: str = Field(default="acked", max_length=16)
    counts: dict[str, int] = Field(default_factory=dict)


class ProjectionAckResponse(BaseModel):
    revision_id: str
    completed: bool


class JobStatusResponse(BaseModel):
    revision_id: str
    job_id: str
    status: str
    required_projections: list[str]
    row_counts: dict[str, int]
    attempts: int
    last_error: str | None


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _raise_for_correction_error(exc: CorrectionError) -> NoReturn:
    if isinstance(exc, PHNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "correction.not_found", "message": str(exc)},
        ) from exc
    if isinstance(exc, StaleVersionError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "correction.stale_version", "message": str(exc)},
        ) from exc
    if isinstance(exc, EmptyIdentityError):
        raise HTTPException(
            status_code=_HTTP_422,
            detail={"code": "correction.empty_identity", "message": str(exc)},
        ) from exc
    if isinstance(exc, CorrectionConflictError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "correction.conflict", "message": str(exc)},
        ) from exc
    raise HTTPException(
        status_code=_HTTP_422,
        detail={"code": "correction.rejected", "message": str(exc)},
    ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/internal/corrections/propose", response_model=ProposalResponse)
async def propose_segment(
    body: ProposeRequest,
    ctx: _CorrectionContext = Depends(get_context),
) -> ProposalResponse:
    service = _require_service(ctx)
    try:
        proposal = await service.propose_segment(
            body.ph_id, observation_id=body.observation_id, at=body.at
        )
    except CorrectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "correction.propose_failed", "message": str(exc)},
        ) from exc
    return ProposalResponse(
        ph_id=proposal.ph_id,
        observation_ids=proposal.observation_ids,
        start=SegmentBoundaryModel(
            observation_id=proposal.start.observation_id,
            captured_at=proposal.start.captured_at,
            reason=proposal.start.reason,
        ),
        end=SegmentBoundaryModel(
            observation_id=proposal.end.observation_id,
            captured_at=proposal.end.captured_at,
            reason=proposal.end.reason,
        ),
        ph_version=proposal.ph_version,
        effective_identity_id=proposal.effective_identity_id,
    )


@router.post("/internal/corrections/apply", response_model=CorrectionResultResponse)
async def apply_correction(
    body: ApplyRequest,
    ctx: _CorrectionContext = Depends(get_context),
) -> CorrectionResultResponse:
    service = _require_service(ctx)
    try:
        result = await service.apply_correction(
            ph_id=body.ph_id,
            actor=body.actor,
            reason_code=body.reason_code,  # type: ignore[arg-type]
            observation_start=body.observation_start,
            observation_end=body.observation_end,
            base_ph_version=body.base_ph_version,
            target_identity_id=body.target_identity_id,
            set_unknown=body.set_unknown,
            frame_only=body.frame_only,
            note=body.note,
            source_view=body.source_view,
            reviewed_frame_id=body.reviewed_frame_id,
            reviewed_bbox=body.reviewed_bbox,
            at_observation_id=body.at_observation_id,
        )
    except CorrectionError as exc:
        _raise_for_correction_error(exc)
    return _result_response(result)


@router.post(
    "/internal/corrections/{correction_id}/compensate",
    response_model=CorrectionResultResponse,
)
async def compensate_correction(
    correction_id: str,
    body: CompensateRequest,
    ctx: _CorrectionContext = Depends(get_context),
) -> CorrectionResultResponse:
    service = _require_service(ctx)
    try:
        result = await service.compensate(correction_id, actor=body.actor)
    except CorrectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "correction.compensate_failed", "message": str(exc)},
        ) from exc
    return _result_response(result)


@router.get("/internal/corrections/jobs/{revision_id}", response_model=JobStatusResponse)
async def get_job_status(
    revision_id: str,
    ctx: _CorrectionContext = Depends(get_context),
) -> JobStatusResponse:
    """Projection-job status for a revision (polled by the admin UI)."""
    service = _require_service(ctx)
    job = await service.get_job(revision_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "correction.job_not_found", "message": revision_id},
        )
    return JobStatusResponse(
        revision_id=job.revision_id,
        job_id=job.job_id,
        status=job.status,
        required_projections=list(job.required_projections),
        row_counts=dict(job.row_counts),
        attempts=job.attempts,
        last_error=job.last_error,
    )


@router.post("/internal/projection-acks", response_model=ProjectionAckResponse)
async def record_projection_ack(
    body: ProjectionAckRequest,
    ctx: _CorrectionContext = Depends(get_context),
) -> ProjectionAckResponse:
    service = _require_service(ctx)
    completed = await service.record_projection_ack(
        ProjectionAck(
            revision_id=body.revision_id,
            consumer=body.consumer,
            schema_version=body.schema_version,
            status=body.status,  # type: ignore[arg-type]
            counts=body.counts,
        )
    )
    return ProjectionAckResponse(revision_id=body.revision_id, completed=completed)


def _result_response(result: Any) -> CorrectionResultResponse:
    return CorrectionResultResponse(
        revision_id=result.revision_id,
        correction_id=result.correction_id,
        ph_id=result.ph_id,
        previous_identity_id=result.previous_identity_id,
        new_identity_id=result.new_identity_id,
        range_id=result.range_id,
        new_ph_id=result.new_ph_id,
        job_status=result.job.status,
    )
