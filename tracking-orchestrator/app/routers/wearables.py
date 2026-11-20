"""Read-only internal endpoints for inspecting wearable-to-PH association segments."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

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
        )
        for seg in rows
    ]
