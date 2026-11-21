"""Internal endpoints for inspecting and revoking wearable-to-PH association teaching."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from app.services.wearable_revocation import (
    SegmentNotFoundError,
    WearableTeachingRevocationService,
)
from app.storage.wearable_associations import WearableAssociationRepository

router = APIRouter(prefix="/internal/wearables", tags=["wearables-internal"])


def get_wearable_association_repo(request: Request) -> WearableAssociationRepository:
    repo: WearableAssociationRepository | None = getattr(
        request.app.state, "wearable_association_repo", None
    )
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "wearable_associations.unavailable", "message": "repo not wired"},
        )
    return repo


def get_wearable_revocation_service(request: Request) -> WearableTeachingRevocationService:
    service: WearableTeachingRevocationService | None = getattr(
        request.app.state, "wearable_revocation_service", None
    )
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "wearable_revocation.unavailable",
                "message": "revocation service not wired",
            },
        )
    return service


class WearableAssociationOut(BaseModel):
    segment_id: str
    ph_id: str
    tag_id: str
    identity_id: str
    started_at: datetime
    ended_at: datetime | None
    attestation_grade: str
    window_llr: float
    margin_llr: float
    matched_fixes: int
    mean_residual_m: float | None
    swap_suspected: bool
    close_reason: str | None
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    revoke_reason: str | None = None


class WearableTeachingHoldOut(BaseModel):
    hold_id: str
    tag_id: str
    reason: str
    held_at: datetime
    cleared_at: datetime | None = None
    cleared_by: str | None = None


class PromotedAttestationOut(BaseModel):
    candidate_id: str
    person_id: str | None = None
    pose_bin: str = "frontal"


class RevokeAssociationRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)
    reason: Literal["swap_suspected", "operator"] = "operator"


class RevokeAssociationResponse(BaseModel):
    segment_id: str
    tag_id: str
    reason: Literal["swap_suspected", "operator"]
    actor: str
    already_revoked: bool
    gallery_rows_demoted: int = Field(ge=0)
    attested_faces_revoked: int | None = Field(default=None, ge=0)
    already_promoted_faces: list[PromotedAttestationOut] = Field(default_factory=list)
    inferred_ranges_compensated: int = Field(ge=0)
    hold: WearableTeachingHoldOut


class ResumeTeachingRequest(BaseModel):
    actor: str = Field(..., min_length=1, max_length=128)


class ResumeTeachingResponse(BaseModel):
    tag_id: str
    resumed: bool
    hold: WearableTeachingHoldOut | None = None


class SegmentTaughtOut(BaseModel):
    segment_id: str
    tag_id: str
    gallery_rows_by_state: dict[str, int]
    gallery_rows_total: int = Field(ge=0)
    attested_faces_total: int | None = Field(default=None, ge=0)
    attested_faces_active: int | None = Field(default=None, ge=0)
    attested_faces_revoked: int | None = Field(default=None, ge=0)
    attested_faces_promoted: int | None = Field(default=None, ge=0)
    inferred_ranges_live: int = Field(ge=0)
    inferred_ranges_total: int = Field(ge=0)
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    revoke_reason: str | None = None
    tag_teaching_held: bool


@router.get("/associations", response_model=list[WearableAssociationOut])
async def list_wearable_associations(
    tag_id: str | None = Query(default=None),
    ph_id: str | None = Query(default=None),
    active_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    repo: WearableAssociationRepository = Depends(get_wearable_association_repo),
) -> list[WearableAssociationOut]:
    rows = await repo.list_recent(
        tag_id=tag_id,
        ph_id=ph_id,
        active_only=active_only,
        limit=limit,
    )
    return [
        WearableAssociationOut(
            segment_id=seg.segment_id,
            ph_id=seg.ph_id,
            tag_id=seg.tag_id,
            identity_id=seg.identity_id,
            started_at=seg.started_at,
            ended_at=seg.ended_at,
            attestation_grade=str(seg.attestation_grade),
            window_llr=seg.window_llr,
            margin_llr=seg.margin_llr,
            matched_fixes=seg.matched_fixes,
            mean_residual_m=seg.mean_residual_m,
            swap_suspected=seg.swap_suspected,
            close_reason=seg.close_reason,
            revoked_at=seg.revoked_at,
            revoked_by=seg.revoked_by,
            revoke_reason=seg.revoke_reason,
        )
        for seg in rows
    ]


@router.get("/associations/{segment_id}/taught", response_model=SegmentTaughtOut)
async def get_association_taught(
    segment_id: str,
    service: WearableTeachingRevocationService = Depends(get_wearable_revocation_service),
) -> SegmentTaughtOut:
    try:
        summary = await service.get_taught_summary(segment_id)
    except SegmentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "wearable_association.not_found",
                "message": f"association segment {segment_id!r} not found",
            },
        ) from exc
    return SegmentTaughtOut(
        segment_id=summary.segment_id,
        tag_id=summary.tag_id,
        gallery_rows_by_state=summary.gallery_rows_by_state,
        gallery_rows_total=summary.gallery_rows_total,
        attested_faces_total=summary.attested_faces_total,
        attested_faces_active=summary.attested_faces_active,
        attested_faces_revoked=summary.attested_faces_revoked,
        attested_faces_promoted=summary.attested_faces_promoted,
        inferred_ranges_live=summary.inferred_ranges_live,
        inferred_ranges_total=summary.inferred_ranges_total,
        revoked_at=summary.revoked_at,
        revoked_by=summary.revoked_by,
        revoke_reason=summary.revoke_reason,
        tag_teaching_held=summary.tag_teaching_held,
    )


@router.post("/associations/{segment_id}/revoke", response_model=RevokeAssociationResponse)
async def revoke_wearable_association(
    segment_id: str,
    body: RevokeAssociationRequest,
    service: WearableTeachingRevocationService = Depends(get_wearable_revocation_service),
) -> RevokeAssociationResponse:
    try:
        summary = await service.revoke(segment_id, actor=body.actor, reason=body.reason)
    except SegmentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "wearable_association.not_found",
                "message": f"association segment {segment_id!r} not found",
            },
        ) from exc
    return RevokeAssociationResponse(
        segment_id=summary.segment_id,
        tag_id=summary.tag_id,
        reason=summary.reason,
        actor=summary.actor,
        already_revoked=summary.already_revoked,
        gallery_rows_demoted=summary.gallery_rows_demoted,
        attested_faces_revoked=summary.attested_faces_revoked,
        already_promoted_faces=[
            PromotedAttestationOut(
                candidate_id=ref.candidate_id,
                person_id=ref.person_id,
                pose_bin=ref.pose_bin,
            )
            for ref in summary.already_promoted_faces
        ],
        inferred_ranges_compensated=summary.inferred_ranges_compensated,
        hold=WearableTeachingHoldOut(
            hold_id=summary.hold.hold_id,
            tag_id=summary.hold.tag_id,
            reason=summary.hold.reason,
            held_at=summary.hold.held_at,
            cleared_at=summary.hold.cleared_at,
            cleared_by=summary.hold.cleared_by,
        ),
    )


@router.post("/tags/{tag_id}/resume-teaching", response_model=ResumeTeachingResponse)
async def resume_wearable_tag_teaching(
    tag_id: str,
    body: ResumeTeachingRequest,
    service: WearableTeachingRevocationService = Depends(get_wearable_revocation_service),
) -> ResumeTeachingResponse:
    cleared = await service.resume_teaching(tag_id, actor=body.actor)
    if cleared is None:
        return ResumeTeachingResponse(tag_id=tag_id, resumed=False, hold=None)
    return ResumeTeachingResponse(
        tag_id=tag_id,
        resumed=True,
        hold=WearableTeachingHoldOut(
            hold_id=cleared.hold_id,
            tag_id=cleared.tag_id,
            reason=cleared.reason,
            held_at=cleared.held_at,
            cleared_at=cleared.cleared_at,
            cleared_by=cleared.cleared_by,
        ),
    )


@router.get("/holds", response_model=list[WearableTeachingHoldOut])
async def list_open_teaching_holds(
    service: WearableTeachingRevocationService = Depends(get_wearable_revocation_service),
) -> list[WearableTeachingHoldOut]:
    holds = await service.list_open_holds()
    return [
        WearableTeachingHoldOut(
            hold_id=h.hold_id,
            tag_id=h.tag_id,
            reason=h.reason,
            held_at=h.held_at,
            cleared_at=h.cleared_at,
            cleared_by=h.cleared_by,
        )
        for h in holds
    ]
