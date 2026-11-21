"""Unit tests for WearableTeachingRevocationService, holds, and internal routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from app.domain import (
    BoundingBox,
    Detection,
    FaceAnchor,
    FloorPoint,
    GallerySeedSource,
    Identity,
    NewReviewCandidate,
    OrientationBin,
    PersonHypothesis,
    WearableAssociationSegment,
    WearableAssociationTransition,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.inference.face_id_client import (
    FaceAttestationCounts,
    FaceAttestationResult,
    FaceRevocationResult,
    PromotedAttestationRef,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import (
    WearableFaceAttestationResult,
    WearableSeedOutcome,
    build_metrics,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.reid_candidates import ReIDCandidateStage
from app.routers.wearables import router as wearables_router
from app.services.identity_correction_service import IdentityCorrectionService
from app.services.identity_rewriter import InMemoryIdentityRewriter
from app.services.unknown_backfill import BackfillConfig, UnknownBackfillService
from app.services.wearable_face_attestation import WearableFaceAttestationService
from app.services.wearable_revocation import (
    SegmentNotFoundError,
    WearableTeachingRevocationService,
)
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryIdentityDecisionRepository,
    InMemoryPHRepository,
)
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from app.storage.wearable_associations import InMemoryWearableAssociationRepository
from app.storage.wearable_teaching_holds import InMemoryWearableTeachingHoldRepository
from app.tracking.identity.candidate_eligibility import CandidatePolicy
from app.transport.redis_streams import FrameReady

_T0 = datetime(2026, 6, 10, 10, 0, 0, tzinfo=UTC)


class _FakeFaceClient:
    def __init__(
        self,
        *,
        revoked: int = 2,
        promoted: tuple[PromotedAttestationRef, ...] = (),
    ) -> None:
        self.revoke_calls: list[tuple[str, str]] = []
        self.attest_calls: int = 0
        self._revoked = revoked
        self._promoted = promoted
        self._already_revoked: set[str] = set()

    async def attest_faces(self, **kwargs: object) -> FaceAttestationResult:
        self.attest_calls += 1
        return FaceAttestationResult(moved=1, missed=0)

    async def revoke_attestations(
        self, *, association_id: str, revoked_by: str
    ) -> FaceRevocationResult:
        self.revoke_calls.append((association_id, revoked_by))
        newly = 0 if association_id in self._already_revoked else self._revoked
        self._already_revoked.add(association_id)
        return FaceRevocationResult(
            association_id=association_id,
            revoked=newly,
            total_revoked=self._revoked,
            already_promoted=self._promoted,
        )

    async def count_attestations(self, *, association_id: str) -> FaceAttestationCounts:
        rev = self._revoked if association_id in self._already_revoked else 0
        return FaceAttestationCounts(
            association_id=association_id,
            total=self._revoked + len(self._promoted),
            active=max(0, self._revoked - rev),
            revoked=rev,
            promoted=len(self._promoted),
        )


class _NoopCrops:
    async def put_bytes(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        return None

    async def delete_object(self, key: str) -> None:
        return None


def _segment(
    segment_id: str,
    *,
    tag_id: str = "tag-1",
    started_at: datetime = _T0,
    ended_at: datetime | None = None,
    swap_suspected: bool = False,
) -> WearableAssociationSegment:
    return WearableAssociationSegment(
        segment_id=segment_id,
        ph_id="ph-1",
        tag_id=tag_id,
        identity_id="alice",
        started_at=started_at,
        ended_at=ended_at,
        attestation_grade=WearableAttestationGrade.TRAJECTORY,
        window_llr=25.0,
        margin_llr=12.0,
        matched_fixes=20,
        mean_residual_m=0.3,
        swap_suspected=swap_suspected,
        close_reason="face_conflict" if swap_suspected else None,
    )


def _wearable_candidate(
    candidate_id: str, *, association_id: str, state: str = "auto_verified"
) -> NewReviewCandidate:
    return NewReviewCandidate(
        candidate_id=candidate_id,
        identity_id="alice",
        embedding=[1.0, 0.0, 0.0, 0.0],
        quality=0.9,
        orientation=2,
        camera_id="cam-1",
        capture_time=_T0,
        ph_id="ph-1",
        observation_id=f"obs-{candidate_id}",
        origin_tracklet_id=f"obs-{candidate_id}",
        keyframe_id=None,
        crop_key=f"reid/{candidate_id}.jpg",
        source_frame_key=None,
        crop_hash="h",
        frame_hash=None,
        dimensions=(64, 128),
        is_truncated=False,
        is_occluded=False,
        candidate_reason="wearable_attested",
        source_episode_id="ph-1",
        created_actor="pipeline",
        model_version="reid",
        preprocessing_version="v1",
        confidence=None,
        state=state,
        seed_source=GallerySeedSource.WEARABLE_ATTESTED,
        seed_association_id=association_id,
    )


@pytest.fixture(autouse=True)
def _fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod, "metrics", build_metrics(CollectorRegistry()))


class _RecordingRewriter(InMemoryIdentityRewriter):
    def __init__(self) -> None:
        self.cleared_calls: list[tuple[str, str, str, datetime, datetime]] = []

    async def clear_identity_rows(
        self,
        revision_id: str,
        ph_id: str,
        old_identity_id: str,
        applies_from: datetime,
        applies_to: datetime,
    ) -> None:
        self.cleared_calls.append((revision_id, ph_id, old_identity_id, applies_from, applies_to))


@pytest.mark.asyncio
async def test_revoke_undoes_all_artifacts_holds_tag_and_is_idempotent() -> None:
    assoc_repo = InMemoryWearableAssociationRepository()
    holds_repo = InMemoryWearableTeachingHoldRepository()
    gallery_repo = InMemoryGalleryRepository()
    ph_repo = InMemoryPHRepository()
    correction_repo = InMemoryIdentityCorrectionRepository()
    rewriter = _RecordingRewriter()
    correction_service = IdentityCorrectionService(
        ph_repo=ph_repo,
        correction_repo=correction_repo,
        publisher=None,
        rewriter=rewriter,
    )
    promoted = (PromotedAttestationRef(candidate_id="fc-9", person_id="alice", pose_bin="frontal"),)
    face_client = _FakeFaceClient(revoked=3, promoted=promoted)

    await gallery_repo.upsert_identity(
        Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0)
    )
    await ph_repo.save(
        PersonHypothesis(
            ph_id="ph-1",
            state_mean=(1.0, 1.0, 0.0, 0.0),
            state_cov=(1.0,) * 16,
            born_at=_T0,
            last_seen_at=_T0 + timedelta(minutes=5),
            last_seen_camera="cam-1",
            observation_count=5,
        )
    )
    seg = _segment("seg-1", ended_at=_T0 + timedelta(minutes=5))
    await assoc_repo.save_segment(seg)

    # Two auto_verified rows + one operator_verified row from seg-1
    await gallery_repo.create_review_candidate(
        _wearable_candidate("g-auto-1", association_id="seg-1")
    )
    await gallery_repo.create_review_candidate(
        _wearable_candidate("g-auto-2", association_id="seg-1")
    )
    await gallery_repo.create_review_candidate(
        _wearable_candidate("g-op-1", association_id="seg-1")
    )
    await gallery_repo.apply_review_action(
        "g-op-1", action="approve", actor="op", base_audit_version=1
    )

    # One live inferred backfill range sourced from seg-1
    inferred_range = await correction_service.record_inferred_range(
        ph_id="ph-1",
        revision_id="rev-backfill-1",
        effective_identity_id="alice",
        start=_T0,
        end=_T0 + timedelta(minutes=4),
        source_association_id="seg-1",
    )
    assert inferred_range is not None

    svc = WearableTeachingRevocationService(
        association_repo=assoc_repo,
        teaching_holds=holds_repo,
        gallery_repo=gallery_repo,
        correction_repo=correction_repo,
        correction_service=correction_service,
        face_id_client=face_client,
        clock=lambda: _T0 + timedelta(minutes=10),
    )

    summary = await svc.revoke("seg-1", actor="admin:alice", reason="operator")
    assert summary.already_revoked is False
    assert summary.gallery_rows_demoted == 2
    assert summary.attested_faces_revoked == 3
    assert summary.already_promoted_faces == promoted
    assert summary.inferred_ranges_compensated == 1
    assert summary.hold.tag_id == "tag-1"
    assert summary.hold.cleared_at is None
    assert await holds_repo.is_held("tag-1") is True

    # Auto-verified rows demoted with vectors intact; operator_verified untouched.
    g1 = await gallery_repo.get_review_candidate("g-auto-1")
    g_op = await gallery_repo.get_review_candidate("g-op-1")
    assert g1 is not None and g1.state == "pending_review"
    assert g_op is not None and g_op.state == "operator_verified"
    entries = {
        e.gallery_entry_id: e
        for e in await gallery_repo.list_gallery_entries(active_only=False, states=None)
    }
    assert entries["g-auto-1"].embedding == [1.0, 0.0, 0.0, 0.0]
    events = await gallery_repo.list_review_events("g-auto-1")
    assert events[-1].new_state == "pending_review"
    assert events[-1].reason == "wearable_revoked:operator"

    # Inferred range compensated and cleared via rewriter.
    live_ranges = await correction_repo.list_ranges_by_source_association("seg-1", live_only=True)
    assert live_ranges == []
    assert len(rewriter.cleared_calls) == 1

    # Segment stamped revoked.
    stored_seg = await assoc_repo.get_segment("seg-1")
    assert stored_seg is not None
    assert stored_seg.revoked_at == _T0 + timedelta(minutes=10)
    assert stored_seg.revoked_by == "admin:alice"
    assert stored_seg.revoke_reason == "operator"

    # Idempotent re-call: 0 newly demoted, 0 newly revoked faces, 0 newly compensated.
    second = await svc.revoke("seg-1", actor="admin:alice", reason="operator")
    assert second.already_revoked is True
    assert second.gallery_rows_demoted == 0
    assert second.attested_faces_revoked == 0
    assert second.inferred_ranges_compensated == 0

    # Taught summary reflects current state.
    taught = await svc.get_taught_summary("seg-1")
    assert taught.gallery_rows_total == 3
    assert taught.gallery_rows_by_state["pending_review"] == 2
    assert taught.gallery_rows_by_state["operator_verified"] == 1
    assert taught.inferred_ranges_total == 1
    assert taught.inferred_ranges_live == 0
    assert taught.tag_teaching_held is True

    # Resume clears the hold.
    cleared = await svc.resume_teaching("tag-1", actor="admin:alice")
    assert cleared is not None and cleared.cleared_by == "admin:alice"
    assert await holds_repo.is_held("tag-1") is False


@pytest.mark.asyncio
async def test_revoke_for_swap_sweeps_prior_unrevoked_segments_in_conflict_window() -> None:
    assoc_repo = InMemoryWearableAssociationRepository()
    holds_repo = InMemoryWearableTeachingHoldRepository()
    gallery_repo = InMemoryGalleryRepository()
    await gallery_repo.upsert_identity(
        Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0)
    )
    # Segment A taught rows 2 minutes before conflict #3 flagged Segment B.
    seg_a = _segment("seg-a", started_at=_T0, ended_at=_T0 + timedelta(minutes=2))
    seg_b = _segment(
        "seg-b",
        started_at=_T0 + timedelta(minutes=3),
        ended_at=_T0 + timedelta(minutes=4),
        swap_suspected=True,
    )
    # Segment old is outside the 600s window and must stay untouched.
    seg_old = _segment(
        "seg-old",
        started_at=_T0 - timedelta(minutes=30),
        ended_at=_T0 - timedelta(minutes=20),
    )
    for seg in (seg_old, seg_a, seg_b):
        await assoc_repo.save_segment(seg)
    await gallery_repo.create_review_candidate(
        _wearable_candidate("g-old", association_id="seg-old")
    )
    await gallery_repo.create_review_candidate(_wearable_candidate("g-a", association_id="seg-a"))
    await gallery_repo.create_review_candidate(_wearable_candidate("g-b", association_id="seg-b"))

    svc = WearableTeachingRevocationService(
        association_repo=assoc_repo,
        teaching_holds=holds_repo,
        gallery_repo=gallery_repo,
        clock=lambda: _T0 + timedelta(minutes=5),
    )
    summaries = await svc.revoke_for_swap("seg-b", window_s=600.0)
    assert {s.segment_id for s in summaries} == {"seg-a", "seg-b"}
    assert (await gallery_repo.get_review_candidate("g-a")).state == "pending_review"  # type: ignore[union-attr]
    assert (await gallery_repo.get_review_candidate("g-b")).state == "pending_review"  # type: ignore[union-attr]
    assert (await gallery_repo.get_review_candidate("g-old")).state == "auto_verified"  # type: ignore[union-attr]
    assert await holds_repo.is_held("tag-1") is True


@pytest.mark.asyncio
async def test_open_teaching_hold_blocks_seeding_attestation_and_backfill() -> None:
    holds = InMemoryWearableTeachingHoldRepository()
    await holds.create_hold(tag_id="tag-1", reason="swap_suspected", held_at=_T0)

    # 1. ReIDCandidateStage skips with TEACHING_HELD
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(
        Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0)
    )
    stage = ReIDCandidateStage(
        gallery_repo=gallery,
        crop_storage=_NoopCrops(),
        policy=CandidatePolicy(model_version="reid", wearable_seeding_enabled=True),
        teaching_holds=holds,
    )
    ctx = FrameContext(
        frame=FrameReady(
            camera_id="cam-1",
            minio_key="k",
            width=640,
            height=480,
            frame_index=1,
            capture_time_unix_ns=int(_T0.timestamp() * 1e9),
        ),
        event_time=_T0,
        capture_time=_T0,
    )
    ctx.effective_width, ctx.effective_height = 640, 480
    ctx.domain_detections = [
        Detection(
            detection_id="det-1",
            camera_id="cam-1",
            bbox=BoundingBox(x_min=100, y_min=100, x_max=220, y_max=400),
            embedding=[1.0, 0.0, 0.0, 0.0],
            capture_time=_T0,
            event_time=_T0,
            confidence=0.95,
            floor_point=FloorPoint(x_mm=1000, y_mm=1000, calibrated=True),
            crop_quality=0.9,
        )
    ]
    ctx.crops = [np.zeros((256, 128, 3), dtype=np.uint8)]
    ctx.orientation_by_detection = {"det-1": (OrientationBin.BACK, 0.9)}
    ctx.det_to_ph = {"det-1": "ph-1"}
    ctx.det_to_observation_id = {"det-1": "obs-1"}
    ctx.committed_ids = {"ph-1": "alice"}
    ctx.wearable_evidence_by_ph = {
        "ph-1": WearableEvidence(
            ph_id="ph-1",
            tag_id="tag-1",
            identity_id="alice",
            evidence_class=WearableEvidenceClass.TRAJECTORY,
            segment_id="seg-new",
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            window_llr=30.0,
            margin_llr=15.0,
            matched_fixes=20,
            mean_residual_m=0.2,
            newest_fix_at=_T0,
        )
    }
    await stage.run(ctx)
    assert await gallery.list_gallery_entries(active_only=False, states=None) == []
    held_seeds = metrics_mod.metrics.cts_wearable_seed_outcomes_total.labels(
        outcome=WearableSeedOutcome.TEACHING_HELD
    )._value.get()
    assert held_seeds == 1.0

    # 2. WearableFaceAttestationService skips with TEACHING_HELD
    face_client = _FakeFaceClient()
    attester = WearableFaceAttestationService(face_client, teaching_holds=holds)
    ev_held = WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-1",
        identity_id="alice",
        evidence_class=WearableEvidenceClass.TRAJECTORY,
        segment_id="seg-2",
        attestation_grade=WearableAttestationGrade.TRAJECTORY,
        window_llr=30.0,
        margin_llr=15.0,
        matched_fixes=20,
        mean_residual_m=0.2,
        newest_fix_at=_T0,
    )
    await attester.offer_round(
        evidence_by_ph={"ph-1": ev_held},
        det_to_ph={"det-1": "ph-1"},
        emitted_faces=[
            FaceAnchor(
                person_id="alice",
                match_score=0.9,
                detection_id="det-1",
                captured_at=_T0,
            )
        ],
        shadow_faces=(),
        captured_at=_T0,
    )
    assert face_client.attest_calls == 0
    held_att = metrics_mod.metrics.cts_wearable_face_attestations_total.labels(
        result=WearableFaceAttestationResult.TEACHING_HELD
    )._value.get()
    assert held_att == 1.0

    # 3. UnknownBackfillService skips when held
    ph_repo = InMemoryPHRepository()
    await ph_repo.save(
        PersonHypothesis(
            ph_id="ph-1",
            state_mean=(1.0, 1.0, 0.0, 0.0),
            state_cov=(1.0,) * 16,
            born_at=_T0,
            last_seen_at=_T0 + timedelta(minutes=2),
            last_seen_camera="cam-1",
            observation_count=5,
        )
    )
    correction_repo = InMemoryIdentityCorrectionRepository()
    correction_svc = IdentityCorrectionService(
        ph_repo=ph_repo, correction_repo=correction_repo, publisher=None
    )
    backfill = UnknownBackfillService(
        ph_repo=ph_repo,
        identity_decision_repo=InMemoryIdentityDecisionRepository(),
        correction_service=correction_svc,
        identity_rewriter=InMemoryIdentityRewriter(),
        revision_publisher=None,
        config=BackfillConfig(enabled=True, shadow=False, wearable_enabled=True),
        teaching_holds=holds,
    )
    await backfill.process(
        outcome_decisions=[],
        ph_born_at_by_id={"ph-1": _T0},
        event_time=_T0 + timedelta(minutes=2),
        wearable_transitions=[
            WearableAssociationTransition(
                action="close",
                segment=_segment("seg-3", ended_at=_T0 + timedelta(minutes=2)),
            )
        ],
    )
    assert await correction_repo.list_ranges_by_source_association("seg-3") == []


@pytest.mark.asyncio
async def test_unknown_segment_raises_not_found() -> None:
    svc = WearableTeachingRevocationService(
        association_repo=InMemoryWearableAssociationRepository(),
        teaching_holds=InMemoryWearableTeachingHoldRepository(),
        gallery_repo=InMemoryGalleryRepository(),
    )
    with pytest.raises(SegmentNotFoundError):
        await svc.revoke("missing", actor="op", reason="operator")
    with pytest.raises(SegmentNotFoundError):
        await svc.get_taught_summary("missing")


def test_internal_wearable_revocation_routes() -> None:
    import asyncio

    assoc_repo = InMemoryWearableAssociationRepository()
    holds_repo = InMemoryWearableTeachingHoldRepository()
    gallery_repo = InMemoryGalleryRepository()
    asyncio.run(
        gallery_repo.upsert_identity(
            Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0)
        )
    )
    asyncio.run(assoc_repo.save_segment(_segment("seg-r1", ended_at=_T0 + timedelta(minutes=2))))
    asyncio.run(
        gallery_repo.create_review_candidate(_wearable_candidate("g-r1", association_id="seg-r1"))
    )
    svc = WearableTeachingRevocationService(
        association_repo=assoc_repo,
        teaching_holds=holds_repo,
        gallery_repo=gallery_repo,
        clock=lambda: _T0 + timedelta(minutes=3),
    )

    app = FastAPI()
    app.include_router(wearables_router)
    app.state.wearable_association_repo = assoc_repo
    app.state.wearable_revocation_service = svc

    client = TestClient(app)

    r_missing = client.get("/internal/wearables/associations/nope/taught")
    assert r_missing.status_code == 404

    r_taught = client.get("/internal/wearables/associations/seg-r1/taught")
    assert r_taught.status_code == 200
    assert r_taught.json()["gallery_rows_total"] == 1
    assert r_taught.json()["tag_teaching_held"] is False

    r_revoke = client.post(
        "/internal/wearables/associations/seg-r1/revoke",
        json={"actor": "admin:bob", "reason": "operator"},
    )
    assert r_revoke.status_code == 200
    body = r_revoke.json()
    assert body["gallery_rows_demoted"] == 1
    assert body["hold"]["tag_id"] == "tag-1"

    r_holds = client.get("/internal/wearables/holds")
    assert r_holds.status_code == 200
    assert len(r_holds.json()) == 1

    r_list = client.get("/internal/wearables/associations")
    assert r_list.status_code == 200
    assert r_list.json()[0]["revoked_by"] == "admin:bob"

    r_resume = client.post(
        "/internal/wearables/tags/tag-1/resume-teaching",
        json={"actor": "admin:bob"},
    )
    assert r_resume.status_code == 200
    assert r_resume.json()["resumed"] is True
    assert client.get("/internal/wearables/holds").json() == []
