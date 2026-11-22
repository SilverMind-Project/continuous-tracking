"""Identity readiness, read-only, for the CC identity setup page.

``GET /internal/identity/readiness`` reports the identity policies the pipeline runs with, the
governed ReID gallery by lifecycle state, which enabled cameras have a floor homography at
runtime, and how many overlap groups are in use. It carries counts and settings only: no identity
id, no person position.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.services.identity_readiness import (
    ClosedWorldReadiness,
    GalleryStateCounts,
    IdentityReadinessService,
    WearableLearnersReadiness,
)

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
    expired: int = Field(default=0, ge=0)


class CameraCalibrationOut(BaseModel):
    camera_id: str
    calibrated: bool = Field(description="A floor homography for this camera is in use")


class WearableLearnersReadinessOut(BaseModel):
    wearable_mode: str = Field(description="wearable.mode")
    wearable_seeding_enabled: bool = Field(description="reid_candidates.wearable_seeding_enabled")
    face_attestation_mode: str = Field(description="wearable.supervision.face_attestation")
    wearable_backfill_enabled: bool = Field(description="resolver.enable_wearable_backfill")
    auto_on_swap: bool = Field(description="wearable.revocation.auto_on_swap")
    evaluation_sample_rate: float = Field(ge=0.0, le=1.0)
    evaluation_days_14d: int = Field(ge=0)
    attribution_difference_14d: float | None = Field(
        default=None,
        description=(
            "Mean daily treatment-minus-control attribution difference over the last 14 UTC days; "
            "null until at least one paired day is recorded"
        ),
    )
    open_teaching_holds: int = Field(ge=0)
    revoked_segments_14d: int = Field(ge=0)


class ClosedWorldReadinessOut(BaseModel):
    mode: str = Field(description="closed_world.mode")
    active: bool = Field(description="Whether closed-world prior shaping is currently active")
    inactive_reason: str | None = Field(
        default=None,
        description="Why closed-world prior shaping is inactive or degraded, if applicable",
    )
    roster_complete: bool | None = Field(
        default=None,
        description="Latest roster_complete flag received from cc.household_presence",
    )
    visitors_present: bool | None = Field(
        default=None,
        description="Latest visitors_present flag received from cc.household_presence",
    )
    presence_age_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Age in seconds of the latest household presence snapshot",
    )


class IdentityReadinessResponse(BaseModel):
    face_evidence_policy: str = Field(description="face_id.evidence_policy")
    face_templates_mode: str = Field(description="face_templates.mode")
    face_quality_mode: str = Field(description="resolver.face_quality_mode")
    cc_assertion_mode: str = Field(description="resolver.cc_assertion_mode")
    wearable_mode: str = Field(default="off", description="wearable.mode")
    closed_world_mode: str = Field(default="off", description="closed_world.mode")
    expected_recognizer_version: str | None = Field(
        default=None,
        description="Face recognizer version calibrated faces must carry; null accepts any",
    )
    expected_arcface_model_version: str | None = Field(
        default=None,
        description="Deprecated one-release alias of expected_recognizer_version",
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
    wearable_gallery_counts: GalleryStateCountsOut | None = Field(
        default=None,
        description=(
            "ReID gallery rows a wearable taught (seed_source wearable_attested or "
            "wearable_room_attested), by state; null without a gallery repository"
        ),
    )
    wearable_learners: WearableLearnersReadinessOut | None = Field(
        default=None,
        description=(
            "Wearable-supervision learner flags, 14-day paired evaluation summary, and teaching "
            "revocation/hold counts"
        ),
    )
    closed_world: ClosedWorldReadinessOut | None = Field(
        default=None,
        description="Closed-world prior-shaping mode, gate status, and presence freshness",
    )


@router.get("/readiness", response_model=IdentityReadinessResponse)
async def get_identity_readiness(
    service: IdentityReadinessService = Depends(get_readiness_service),
) -> IdentityReadinessResponse:
    snap = await service.snapshot()
    return IdentityReadinessResponse(
        face_evidence_policy=snap.policy.face_evidence_policy,
        face_templates_mode=snap.policy.face_templates_mode,
        face_quality_mode=snap.policy.face_quality_mode,
        cc_assertion_mode=snap.policy.cc_assertion_mode,
        wearable_mode=snap.policy.wearable_mode,
        closed_world_mode=snap.policy.closed_world_mode,
        expected_recognizer_version=snap.policy.expected_recognizer_version,
        expected_arcface_model_version=snap.policy.expected_recognizer_version,
        expected_quality_version=snap.policy.expected_quality_version,
        reid_gallery_counts=_counts_out(snap.reid_gallery_counts),
        cameras=[
            CameraCalibrationOut(camera_id=c.camera_id, calibrated=c.calibrated)
            for c in snap.cameras
        ],
        overlap_group_count=snap.overlap_group_count,
        identities_known=snap.identities_known,
        wearable_active_associations=snap.wearable_active_associations,
        wearable_gallery_counts=_counts_out(snap.wearable_gallery_counts),
        wearable_learners=_learners_out(snap.wearable_learners),
        closed_world=_closed_world_out(snap.closed_world),
    )


def _counts_out(counts: GalleryStateCounts | None) -> GalleryStateCountsOut | None:
    if counts is None:
        return None
    return GalleryStateCountsOut(
        pending_review=counts.pending_review,
        auto_verified=counts.auto_verified,
        operator_verified=counts.operator_verified,
        rejected=counts.rejected,
        expired=counts.expired,
    )


def _learners_out(
    learners: WearableLearnersReadiness | None,
) -> WearableLearnersReadinessOut | None:
    if learners is None:
        return None
    return WearableLearnersReadinessOut(
        wearable_mode=learners.wearable_mode,
        wearable_seeding_enabled=learners.wearable_seeding_enabled,
        face_attestation_mode=learners.face_attestation_mode,
        wearable_backfill_enabled=learners.wearable_backfill_enabled,
        auto_on_swap=learners.auto_on_swap,
        evaluation_sample_rate=learners.evaluation_sample_rate,
        evaluation_days_14d=learners.evaluation_days_14d,
        attribution_difference_14d=learners.attribution_difference_14d,
        open_teaching_holds=learners.open_teaching_holds,
        revoked_segments_14d=learners.revoked_segments_14d,
    )


def _closed_world_out(
    closed_world: ClosedWorldReadiness | None,
) -> ClosedWorldReadinessOut | None:
    if closed_world is None:
        return None
    return ClosedWorldReadinessOut(
        mode=closed_world.mode,
        active=closed_world.active,
        inactive_reason=closed_world.inactive_reason,
        roster_complete=closed_world.roster_complete,
        visitors_present=closed_world.visitors_present,
        presence_age_s=closed_world.presence_age_s,
    )
