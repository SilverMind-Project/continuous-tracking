"""Tests for working-set orientation cap (Task 2) and shadow room-grade seeding (Task 6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    BoundingBox,
    Detection,
    FloorPoint,
    GallerySeedSource,
    Identity,
    NewReviewCandidate,
    OrientationBin,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import WearableSeedOutcome, build_metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.reid_candidates import ReIDCandidateStage
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity.candidate_eligibility import CandidatePolicy
from app.transport.redis_streams import FrameReady

_T0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)


class _NoopCrops:
    async def put_bytes(self, key: str, data: bytes, content_type: str = "image/jpeg") -> None:
        return None

    async def delete_object(self, key: str) -> None:
        return None


def _make_ctx(
    at: datetime,
    *,
    committed_id: str | None,
    evidence: WearableEvidence | None = None,
    embedding: list[float] | None = None,
    det_id: str = "det-1",
) -> FrameContext:
    ctx = FrameContext(
        frame=FrameReady(
            camera_id="cam-1",
            minio_key="k",
            width=640,
            height=480,
            frame_index=1,
            capture_time_unix_ns=int(at.timestamp() * 1e9),
        ),
        event_time=at,
        capture_time=at,
    )
    ctx.effective_width, ctx.effective_height = 640, 480
    ctx.domain_detections = [
        Detection(
            detection_id=det_id,
            camera_id="cam-1",
            bbox=BoundingBox(x_min=100, y_min=100, x_max=220, y_max=400),
            embedding=embedding or [1.0, 0.0, 0.0, 0.0],
            capture_time=at,
            event_time=at,
            confidence=0.95,
            floor_point=FloorPoint(x_mm=1000, y_mm=1000, calibrated=True),
            crop_quality=0.9,
        )
    ]
    ctx.crops = [np.zeros((256, 128, 3), dtype=np.uint8)]
    ctx.orientation_by_detection = {det_id: (OrientationBin.BACK, 0.9)}
    ctx.det_to_ph = {det_id: "ph-1"}
    ctx.det_to_observation_id = {det_id: f"obs-{det_id}"}
    ctx.committed_ids = {"ph-1": committed_id}
    if evidence is not None:
        ctx.wearable_evidence_by_ph = {"ph-1": evidence}
    return ctx


async def _seed_existing(
    repo: InMemoryGalleryRepository, *, count: int, seen_at: datetime, orientation: int
) -> None:
    for i in range(count):
        await repo.create_review_candidate(
            NewReviewCandidate(
                candidate_id=f"existing-{int(seen_at.timestamp())}-{i}",
                identity_id="alice",
                embedding=[0.0, 1.0, 0.0, 0.0],
                quality=0.9,
                orientation=orientation,
                camera_id="cam-1",
                capture_time=seen_at,
                ph_id=f"ph-old-{i}",
                observation_id=f"obs-old-{i}",
                origin_tracklet_id=f"obs-old-{i}",
                keyframe_id=None,
                crop_key=f"reid/old-{i}.jpg",
                source_frame_key=None,
                crop_hash="h",
                frame_hash=None,
                dimensions=(64, 128),
                is_truncated=False,
                is_occluded=False,
                candidate_reason="face_derived",
                source_episode_id=f"ep-old-{i}",
                created_actor="pipeline",
                model_version="reid",
                preprocessing_version="v1",
                confidence=0.95,
                state="auto_verified",
            )
        )


@pytest.fixture(autouse=True)
def _fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod, "metrics", build_metrics(CollectorRegistry()))


@pytest.mark.asyncio
async def test_working_set_cap_resets_after_vote_window_and_blocks_inside() -> None:
    repo = InMemoryGalleryRepository()
    await repo.upsert_identity(Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0))
    # 10 rows from yesterday (24h ago, outside 12h vote window).
    await _seed_existing(
        repo, count=10, seen_at=_T0 - timedelta(hours=24), orientation=int(OrientationBin.BACK)
    )

    policy = CandidatePolicy(
        model_version="reid",
        max_per_identity_orientation=10,
        wearable_seeding_enabled=True,
        gallery_vote_max_age_s=43200.0,
        wearable_channel_live=True,
    )
    stage = ReIDCandidateStage(repo, _NoopCrops(), policy)
    evidence = WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-1",
        identity_id="alice",
        evidence_class=WearableEvidenceClass.TRAJECTORY,
        segment_id="seg-today",
        attestation_grade=WearableAttestationGrade.TRAJECTORY,
        window_llr=30.0,
        margin_llr=15.0,
        matched_fixes=20,
        mean_residual_m=0.2,
        newest_fix_at=_T0,
    )

    # Today's mint succeeds because yesterday's 10 rows are outside the 12h vote window.
    await stage.run(_make_ctx(_T0, committed_id="alice", evidence=evidence, det_id="det-today"))
    rows_today = await repo.count_gallery_entries(
        "alice", int(OrientationBin.BACK), seen_since=_T0 - timedelta(hours=12)
    )
    assert rows_today == 1

    # Now add 9 more rows from 1 hour ago (total 10 in the current 12h window).
    await _seed_existing(
        repo, count=9, seen_at=_T0 - timedelta(hours=1), orientation=int(OrientationBin.BACK)
    )
    assert (
        await repo.count_gallery_entries(
            "alice", int(OrientationBin.BACK), seen_since=_T0 - timedelta(hours=12)
        )
        == 10
    )

    # Next attempt 5 minutes later hits cap_reached because 10 rows exist in the current window.
    await stage.run(
        _make_ctx(
            _T0 + timedelta(minutes=5),
            committed_id="alice",
            evidence=evidence,
            embedding=[0.0, 0.0, 1.0, 0.0],
            det_id="det-capped",
        )
    )
    capped = metrics_mod.metrics.cts_wearable_seed_outcomes_total.labels(
        outcome=WearableSeedOutcome.CAP_REACHED
    )._value.get()
    assert capped == 1.0


@pytest.mark.asyncio
async def test_shadow_mode_allows_room_grade_pending_review_and_skips_trajectory() -> None:
    repo = InMemoryGalleryRepository()
    await repo.upsert_identity(Identity(identity_id="alice", display_name="Alice", enrolled_at=_T0))
    await repo.upsert_identity(Identity(identity_id="bob", display_name="Bob", enrolled_at=_T0))

    policy = CandidatePolicy(
        model_version="reid",
        wearable_seeding_enabled=True,
        wearable_channel_live=False,  # wearable.mode == "shadow"
    )
    stage = ReIDCandidateStage(repo, _NoopCrops(), policy)

    # 1. Room-grade with uncommitted PH mints pending_review for evidence.identity_id.
    room_ev = WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-1",
        identity_id="alice",
        evidence_class=WearableEvidenceClass.ROOM,
        segment_id="seg-room",
        attestation_grade=WearableAttestationGrade.ROOM,
        window_llr=15.0,
        margin_llr=8.0,
        matched_fixes=10,
        mean_residual_m=None,
        newest_fix_at=_T0,
    )
    await stage.run(_make_ctx(_T0, committed_id=None, evidence=room_ev, det_id="det-room-1"))
    cands, total = await repo.list_review_candidates(state="pending_review")
    assert total == 1
    assert cands[0].identity_id == "alice"
    assert cands[0].seed_source == GallerySeedSource.WEARABLE_ROOM_ATTESTED

    # 2. Room-grade when a DIFFERENT identity is committed still rejects identity_mismatch.
    await stage.run(
        _make_ctx(
            _T0 + timedelta(minutes=5),
            committed_id="bob",
            evidence=room_ev,
            embedding=[0.0, 1.0, 0.0, 0.0],
            det_id="det-room-mismatch",
        )
    )
    mismatch = metrics_mod.metrics.cts_wearable_seed_outcomes_total.labels(
        outcome=WearableSeedOutcome.IDENTITY_MISMATCH
    )._value.get()
    assert mismatch == 1.0

    # 3. Trajectory-grade in shadow skips as channel_not_live (never auto-verifies in shadow).
    traj_ev = WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-1",
        identity_id="alice",
        evidence_class=WearableEvidenceClass.TRAJECTORY,
        segment_id="seg-traj",
        attestation_grade=WearableAttestationGrade.TRAJECTORY,
        window_llr=30.0,
        margin_llr=15.0,
        matched_fixes=20,
        mean_residual_m=0.2,
        newest_fix_at=_T0 + timedelta(minutes=10),
    )
    await stage.run(
        _make_ctx(
            _T0 + timedelta(minutes=10),
            committed_id="alice",
            evidence=traj_ev,
            embedding=[0.0, 0.0, 1.0, 0.0],
            det_id="det-traj-shadow",
        )
    )
    not_live = metrics_mod.metrics.cts_wearable_seed_outcomes_total.labels(
        outcome=WearableSeedOutcome.CHANNEL_NOT_LIVE
    )._value.get()
    assert not_live == 1.0
    _auto_rows, auto_total = await repo.list_review_candidates(state="auto_verified")
    assert auto_total == 0
