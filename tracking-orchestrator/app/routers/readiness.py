"""Identity readiness, read-only, for the CC identity setup page.

``GET /internal/identity/readiness`` reports the identity policies the pipeline runs with, the
governed ReID gallery by lifecycle state, which enabled cameras have a floor homography at
runtime, and how many overlap groups are in use. It carries counts and settings only: no identity
id, no person position.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from ..services.identity_readiness import IdentityReadinessService

router = APIRouter(prefix="/internal/identity", tags=["identity-readiness-internal"])


def get_readiness_service(request: Request) -> IdentityReadinessService:
    service: IdentityReadinessService | None = request.app.state.identity_readiness_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "identity_readiness.unavailable", "message": "service not wired"},
        )
    return service


class GalleryStateCountsOut(BaseModel):
    pending_review: int = Field(ge=0)
    auto_verified: int = Field(ge=0)
    operator_verified: int = Field(ge=0)
    rejected: int = Field(ge=0)


class CameraCalibrationOut(BaseModel):
    camera_id: str
    calibrated: bool = Field(description="A floor homography for this camera is in use")


class IdentityReadinessResponse(BaseModel):
    face_evidence_policy: str = Field(description="face_id.evidence_policy")
    face_templates_mode: str = Field(description="face_templates.mode")
    face_quality_mode: str = Field(description="resolver.face_quality_mode")
    cc_assertion_mode: str = Field(description="resolver.cc_assertion_mode")
    wearable_mode: str = Field(default="off", description="wearable.mode")
    expected_arcface_model_version: str | None = Field(
        description="ArcFace model version calibrated faces must carry; null accepts any"
    )
    expected_quality_version: str
    reid_gallery_counts: GalleryStateCountsOut | None = Field(
        description="ReID gallery rows by lifecycle state; null without a gallery repository"
    )
    cameras: list[CameraCalibrationOut] = Field(
        description="Enabled cameras from the last CC config sync"
    )
    overlap_group_count: int | None = Field(
        ge=0, description="Overlap groups in use; null before the world tracker is built"
    )
    identities_known: int | None = Field(
        ge=0, description="Active identities CTS knows; null without a gallery repository"
    )
    wearable_active_associations: int | None = Field(
        default=None,
        ge=0,
        description="Open wearable-to-PH associations; null before the world tracker is built",
    )


@router.get("/readiness", response_model=IdentityReadinessResponse)
async def get_identity_readiness(
    service: IdentityReadinessService = Depends(get_readiness_service),
) -> IdentityReadinessResponse:
    snap = await service.snapshot()
    counts = snap.reid_gallery_counts
    return IdentityReadinessResponse(
        face_evidence_policy=snap.policy.face_evidence_policy,
        face_templates_mode=snap.policy.face_templates_mode,
        face_quality_mode=snap.policy.face_quality_mode,
        cc_assertion_mode=snap.policy.cc_assertion_mode,
        wearable_mode=snap.policy.wearable_mode,
        expected_arcface_model_version=snap.policy.expected_arcface_model_version,
        expected_quality_version=snap.policy.expected_quality_version,
        reid_gallery_counts=None
        if counts is None
        else GalleryStateCountsOut(
            pending_review=counts.pending_review,
            auto_verified=counts.auto_verified,
            operator_verified=counts.operator_verified,
            rejected=counts.rejected,
        ),
        cameras=[
            CameraCalibrationOut(camera_id=c.camera_id, calibrated=c.calibrated)
            for c in snap.cameras
        ],
        overlap_group_count=snap.overlap_group_count,
        identities_known=snap.identities_known,
        wearable_active_associations=snap.wearable_active_associations,
    )
