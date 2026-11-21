"""Unit tests for CTS identity readiness wearable_learners and expired counts."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.state import CalibrationState
from app.domain import (
    Identity,
    ImageOnlyDailyTotal,
    NewReviewCandidate,
    WearableAssociationSegment,
    WearableAttestationGrade,
)
from app.routers.readiness import router as readiness_router
from app.services.camera_room_map import CameraRoomMap
from app.services.identity_readiness import IdentityPolicySnapshot, IdentityReadinessService
from app.storage.gallery import InMemoryGalleryRepository
from app.storage.image_only_evaluation import InMemoryImageOnlyEvaluationRepository
from app.storage.wearable_associations import InMemoryWearableAssociationRepository
from app.storage.wearable_teaching_holds import InMemoryWearableTeachingHoldRepository
from scripts.image_only_evaluation_report import build_report

_NOW = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


class _StubOverlapAndWearable:
    def overlap_group_count(self) -> int | None:
        return 1

    def active_wearable_association_count(self) -> int | None:
        return 2


@pytest.mark.asyncio
async def test_readiness_wearable_learners_matches_report_formula_and_omits_pii() -> None:
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(
        Identity(identity_id="secret-resident-uuid", display_name="Alice", enrolled_at=_NOW)
    )
    # One expired row via retention
    await gallery.create_review_candidate(
        NewReviewCandidate(
            candidate_id="cand-exp",
            identity_id="secret-resident-uuid",
            embedding=[1.0, 0.0, 0.0, 0.0],
            quality=0.9,
            orientation=0,
            camera_id="cam-1",
            capture_time=_NOW - timedelta(days=40),
            ph_id="ph-1",
            observation_id="obs-1",
            origin_tracklet_id="obs-1",
            keyframe_id=None,
            crop_key="reid/cand-exp.jpg",
            source_frame_key=None,
            crop_hash="h",
            frame_hash=None,
            dimensions=(64, 128),
            is_truncated=False,
            is_occluded=False,
            candidate_reason="face_derived",
            source_episode_id="ep-1",
            created_actor="pipeline",
            model_version="reid",
            preprocessing_version="v1",
            confidence=0.95,
            state="pending_review",
        )
    )
    await gallery.expire_past_retention(
        state="pending_review", seen_before=_NOW - timedelta(days=30), limit=10
    )

    eval_repo = InMemoryImageOnlyEvaluationRepository()
    totals = [
        # Day 1 inside 14d window: control 0.2, treatment 0.7 -> diff +0.5
        ImageOnlyDailyTotal(
            day=date(2026, 6, 18),
            identity_id="secret-resident-uuid",
            arm="control",
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            correct_s=20.0,
            wrong_s=0.0,
            unknown_s=80.0,
        ),
        ImageOnlyDailyTotal(
            day=date(2026, 6, 18),
            identity_id="secret-resident-uuid",
            arm="treatment",
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            correct_s=70.0,
            wrong_s=0.0,
            unknown_s=30.0,
        ),
        # Day 2 inside 14d window: control 0.3, treatment 0.6 -> diff +0.3
        ImageOnlyDailyTotal(
            day=date(2026, 6, 19),
            identity_id="secret-resident-uuid",
            arm="control",
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            correct_s=30.0,
            wrong_s=0.0,
            unknown_s=70.0,
        ),
        ImageOnlyDailyTotal(
            day=date(2026, 6, 19),
            identity_id="secret-resident-uuid",
            arm="treatment",
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            correct_s=60.0,
            wrong_s=0.0,
            unknown_s=40.0,
        ),
        # Day outside 14d window: must be excluded
        ImageOnlyDailyTotal(
            day=date(2026, 5, 1),
            identity_id="secret-resident-uuid",
            arm="control",
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            correct_s=90.0,
            wrong_s=0.0,
            unknown_s=10.0,
        ),
        ImageOnlyDailyTotal(
            day=date(2026, 5, 1),
            identity_id="secret-resident-uuid",
            arm="treatment",
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            correct_s=10.0,
            wrong_s=0.0,
            unknown_s=90.0,
        ),
    ]
    await eval_repo.add_totals(totals)

    assoc_repo = InMemoryWearableAssociationRepository()
    await assoc_repo.save_segment(
        WearableAssociationSegment(
            segment_id="seg-rev",
            ph_id="ph-1",
            tag_id="tag-1",
            identity_id="secret-resident-uuid",
            started_at=_NOW - timedelta(days=2),
            ended_at=_NOW - timedelta(days=2),
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            window_llr=20.0,
            margin_llr=10.0,
            matched_fixes=10,
            mean_residual_m=0.2,
            swap_suspected=True,
            close_reason="face_conflict",
            revoked_at=_NOW - timedelta(days=1),
            revoked_by="system:swap_guard",
            revoke_reason="swap_suspected",
        )
    )
    holds_repo = InMemoryWearableTeachingHoldRepository()
    await holds_repo.create_hold(tag_id="tag-1", reason="swap_suspected", held_at=_NOW)

    svc = IdentityReadinessService(
        policy=IdentityPolicySnapshot(
            face_evidence_policy="calibrated_only",
            face_templates_mode="enabled",
            face_quality_mode="enabled",
            cc_assertion_mode="enabled",
            expected_arcface_model_version="buffalo_l",
            expected_quality_version="heuristic-v1",
            wearable_mode="enabled",
            wearable_seeding_enabled=True,
            wearable_face_attestation_mode="enabled",
            wearable_backfill_enabled=False,
            wearable_auto_on_swap=True,
            wearable_evaluation_sample_rate=0.2,
        ),
        calibration_state=CalibrationState(),
        camera_room_map=CameraRoomMap(),
        overlap_groups=_StubOverlapAndWearable(),
        gallery_repo=gallery,
        wearable_associations=_StubOverlapAndWearable(),
        image_only_eval_repo=eval_repo,
        wearable_assoc_repo=assoc_repo,
        teaching_holds_repo=holds_repo,
        clock=lambda: _NOW,
    )

    snap = await svc.snapshot()
    assert snap.reid_gallery_counts is not None
    assert snap.reid_gallery_counts.expired == 1
    assert snap.wearable_learners is not None
    assert snap.wearable_learners.evaluation_days_14d == 2
    assert snap.wearable_learners.open_teaching_holds == 1
    assert snap.wearable_learners.revoked_segments_14d == 1

    # Point estimate matches scripts/image_only_evaluation_report.build_report on the 14d window.
    in_window = [t for t in totals if t.day >= date(2026, 6, 7)]
    report = build_report(in_window, samples=100, seed=0)
    report_mean = report[0]["attribution_difference"][0]  # type: ignore[index]
    assert snap.wearable_learners.attribution_difference_14d == pytest.approx(report_mean)

    # HTTP response serializes cleanly and leaks zero identity IDs.
    app = FastAPI()
    app.include_router(readiness_router)
    app.state.identity_readiness_service = svc
    client = TestClient(app)
    resp = client.get("/internal/identity/readiness")
    assert resp.status_code == 200
    assert "secret-resident-uuid" not in resp.text
    payload = resp.json()
    assert payload["reid_gallery_counts"]["expired"] == 1
    assert payload["wearable_learners"]["evaluation_days_14d"] == 2
    assert payload["wearable_learners"]["open_teaching_holds"] == 1
    assert payload["wearable_learners"]["revoked_segments_14d"] == 1
    assert payload["wearable_learners"]["wearable_seeding_enabled"] is True
    assert asyncio.iscoroutinefunction(svc.snapshot)
