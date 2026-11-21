"""ReIDCandidateStage wearable-attested seeding through the stage's public run().

The wearable path runs only when no face seed qualifies. Its spacing, diversity, and daily room
cap read the identity's own rows from the repository, so these tests drive several rounds
against one InMemory gallery and assert what was minted.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np

from app.domain import (
    BoundingBox,
    Detection,
    FaceAnchor,
    FloorPoint,
    GallerySeedSource,
    Identity,
    OrientationBin,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.observability import metrics as _metrics
from app.observability.metrics import WearableSeedOutcome
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.reid_candidates import ReIDCandidateStage
from app.storage.base import InMemoryGalleryRepository
from app.tracking.identity.candidate_eligibility import CandidatePolicy
from app.transport.redis_streams import FrameReady

_T0 = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
_SEGMENT = "22222222-2222-2222-2222-222222222222"


class _FakeCropStorage:
    def __init__(self) -> None:
        self.uploaded: dict[str, bytes] = {}

    async def put_bytes(
        self, minio_key: str, data: bytes, content_type: str = "image/jpeg"
    ) -> None:
        self.uploaded[minio_key] = data

    async def delete_object(self, minio_key: str) -> None:
        self.uploaded.pop(minio_key, None)


def _outcome(outcome: WearableSeedOutcome) -> float:
    return _metrics.metrics.cts_wearable_seed_outcomes_total.labels(outcome=outcome)._value.get()


def _evidence(grade: WearableAttestationGrade) -> WearableEvidence:
    return WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-1",
        identity_id="member-a",
        evidence_class=WearableEvidenceClass.TRAJECTORY,
        attestation_grade=grade,
        window_llr=20.0,
        margin_llr=8.0,
        matched_fixes=10,
        mean_residual_m=0.2,
        newest_fix_at=_T0,
        segment_id=_SEGMENT,
    )


def _ctx(
    at: datetime,
    *,
    embedding: list[float],
    grade: WearableAttestationGrade = WearableAttestationGrade.TRAJECTORY,
    face_anchors: list[FaceAnchor] | None = None,
    shadow_face_anchors: list[FaceAnchor] | None = None,
) -> FrameContext:
    frame = FrameReady(
        camera_id="cam01",
        minio_key="test/key",
        width=640,
        height=480,
        frame_index=1,
        capture_time_unix_ns=int(at.timestamp() * 1e9),
    )
    ctx = FrameContext(frame=frame, event_time=at, capture_time=at)
    ctx.effective_width = 640
    ctx.effective_height = 480
    ctx.domain_detections = [
        Detection(
            detection_id="d1",
            camera_id="cam01",
            bbox=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=300),
            embedding=embedding,
            capture_time=at,
            event_time=at,
            confidence=0.95,
            floor_point=FloorPoint(1000, 1000, calibrated=True),
            crop_quality=0.9,
        )
    ]
    ctx.crops = [np.zeros((256, 128, 3), dtype=np.uint8)]
    ctx.face_anchors = face_anchors or []
    ctx.shadow_face_anchors = shadow_face_anchors or []
    ctx.orientation_by_detection = {"d1": (OrientationBin.BACK, 0.9)}
    ctx.det_to_ph = {"d1": "ph-1"}
    ctx.committed_ids = {"ph-1": "member-a"}
    ctx.det_to_observation_id = {"d1": "00000000-0000-0000-0000-0000000000aa"}
    ctx.wearable_evidence_by_ph = {"ph-1": _evidence(grade)}
    return ctx


async def _stage(**policy: object) -> tuple[ReIDCandidateStage, InMemoryGalleryRepository]:
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(
        Identity(identity_id="member-a", display_name="A", enrolled_at=_T0)
    )
    cfg = replace(
        CandidatePolicy(model_version="reid", wearable_seeding_enabled=True),
        **policy,  # type: ignore[arg-type]
    )
    return ReIDCandidateStage(gallery, _FakeCropStorage(), cfg), gallery


async def _all_rows(gallery: InMemoryGalleryRepository) -> list[object]:
    return list(await gallery.list_gallery_entries(active_only=False, states=None))


async def test_trajectory_seed_mints_one_auto_verified_wearable_row_with_provenance() -> None:
    stage, gallery = await _stage()
    minted_before = _outcome(WearableSeedOutcome.MINTED_AUTO_VERIFIED)

    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0]))

    rows, total = await gallery.list_review_candidates(state="auto_verified")
    assert total == 1
    row = rows[0]
    assert row.seed_source == GallerySeedSource.WEARABLE_ATTESTED
    assert row.seed_association_id == _SEGMENT
    assert row.candidate_reason == "wearable_attested"
    assert row.confidence is None
    assert _outcome(WearableSeedOutcome.MINTED_AUTO_VERIFIED) == minted_before + 1


async def test_room_seed_mints_pending_review() -> None:
    stage, gallery = await _stage()
    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0], grade=WearableAttestationGrade.ROOM))
    rows, total = await gallery.list_review_candidates(state="pending_review")
    assert total == 1
    assert rows[0].seed_source == GallerySeedSource.WEARABLE_ROOM_ATTESTED


async def test_seeding_disabled_mints_nothing() -> None:
    gallery = InMemoryGalleryRepository()
    stage = ReIDCandidateStage(gallery, _FakeCropStorage(), CandidatePolicy(model_version="r"))
    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0]))
    assert await _all_rows(gallery) == []


async def test_a_qualifying_face_seed_wins_over_the_wearable() -> None:
    stage, gallery = await _stage()
    face = FaceAnchor(
        person_id="member-a",
        match_score=0.9,
        recognition_state="recognized",
        detection_id="d1",
        calibrated_confidence=0.95,
    )
    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0], face_anchors=[face]))
    rows, total = await gallery.list_review_candidates(state="auto_verified")
    assert total == 1
    assert rows[0].seed_source == GallerySeedSource.FACE


async def test_shadow_only_recognition_of_someone_else_vetoes_the_seed() -> None:
    stage, gallery = await _stage()
    weak_other = FaceAnchor(
        person_id="member-b",
        match_score=0.22,
        recognition_state="recognized",
        detection_id="d1",
        shadow_only=True,
    )
    before = _outcome(WearableSeedOutcome.FACE_CONFLICT)
    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0], shadow_face_anchors=[weak_other]))
    assert await _all_rows(gallery) == []
    assert _outcome(WearableSeedOutcome.FACE_CONFLICT) == before + 1


async def test_seed_spacing_then_a_later_distinct_pose_mints_again() -> None:
    stage, gallery = await _stage()
    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0]))
    before = _outcome(WearableSeedOutcome.SEED_INTERVAL)
    await stage.run(_ctx(_T0 + timedelta(seconds=60), embedding=[0.0, 1.0, 0.0, 0.0]))
    assert len(await _all_rows(gallery)) == 1
    assert _outcome(WearableSeedOutcome.SEED_INTERVAL) == before + 1

    await stage.run(_ctx(_T0 + timedelta(seconds=121), embedding=[0.0, 1.0, 0.0, 0.0]))
    assert len(await _all_rows(gallery)) == 2


async def test_near_duplicate_of_a_same_day_row_is_skipped() -> None:
    stage, gallery = await _stage()
    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0]))
    before = _outcome(WearableSeedOutcome.TOO_SIMILAR_TO_RECENT)
    await stage.run(_ctx(_T0 + timedelta(minutes=10), embedding=[0.99, 0.05, 0.0, 0.0]))
    assert len(await _all_rows(gallery)) == 1
    assert _outcome(WearableSeedOutcome.TOO_SIMILAR_TO_RECENT) == before + 1


async def test_room_grade_daily_cap() -> None:
    stage, gallery = await _stage(wearable_room_grade_max_per_day=1)
    room = WearableAttestationGrade.ROOM
    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0], grade=room))
    before = _outcome(WearableSeedOutcome.ROOM_DAILY_CAP)
    await stage.run(_ctx(_T0 + timedelta(minutes=5), embedding=[0.0, 1.0, 0.0, 0.0], grade=room))
    assert len(await _all_rows(gallery)) == 1
    assert _outcome(WearableSeedOutcome.ROOM_DAILY_CAP) == before + 1

    # The cap is per UTC day.
    await stage.run(_ctx(_T0 + timedelta(days=1), embedding=[0.0, 0.0, 1.0, 0.0], grade=room))
    assert len(await _all_rows(gallery)) == 2


async def test_lifetime_per_orientation_cap_applies_to_wearable_seeds() -> None:
    stage, gallery = await _stage(max_per_identity_orientation=1)
    await stage.run(_ctx(_T0, embedding=[1.0, 0.0, 0.0, 0.0]))
    before = _outcome(WearableSeedOutcome.CAP_REACHED)
    await stage.run(_ctx(_T0 + timedelta(minutes=5), embedding=[0.0, 1.0, 0.0, 0.0]))
    assert len(await _all_rows(gallery)) == 1
    assert _outcome(WearableSeedOutcome.CAP_REACHED) == before + 1
