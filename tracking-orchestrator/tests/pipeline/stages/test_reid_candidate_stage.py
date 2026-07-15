"""ReIDCandidateStage: the only pipeline write path into reid_gallery (M04).

Covers: eligible creation with full provenance, identity-mismatch rejection
(F3), the per-(identity, orientation) cap counting pending+verified (F4),
crop-upload/DB-failure cleanup, MinIO-unconfigured skip, and the fail-closed
uncalibrated default.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from app.domain import BoundingBox, Detection, FaceAnchor, FloorPoint, OrientationBin
from app.observability import metrics as _metrics
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.reid_candidates import ReIDCandidateStage
from app.storage.base import InMemoryGalleryRepository
from app.storage.gallery import PENDING_AND_VERIFIED
from app.tracking.identity.candidate_eligibility import CandidatePolicy
from app.transport.redis_streams import FrameReady

_T0 = datetime(2026, 7, 15, 12, 0, 0, tzinfo=UTC)


def _rejected_count(reason: str) -> float:
    return _metrics.metrics.reid_candidate_rejected_total.labels(reason=reason)._value.get()


class _FakeCropStorage:
    def __init__(self, *, fail_put: bool = False) -> None:
        self.uploaded: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self._fail_put = fail_put

    async def put_bytes(
        self, minio_key: str, data: bytes, content_type: str = "image/jpeg"
    ) -> None:
        if self._fail_put:
            raise RuntimeError("simulated MinIO failure")
        self.uploaded[minio_key] = data

    async def delete_object(self, minio_key: str) -> None:
        self.deleted.append(minio_key)


class _FailingCreateGalleryRepo(InMemoryGalleryRepository):
    async def create_review_candidate(self, candidate: object) -> str:  # type: ignore[override]
        raise RuntimeError("simulated DB failure")


def _crop() -> np.ndarray:
    return np.zeros((256, 128, 3), dtype=np.uint8)


def _ctx(
    *,
    committed_identity_id: str | None = "grandma",
    face_person_id: str = "grandma",
    calibrated_confidence: float | None = 0.90,
    orientation: OrientationBin = OrientationBin.FRONT,
    orientation_confidence: float = 0.9,
) -> FrameContext:
    frame = FrameReady(
        camera_id="cam01",
        minio_key="test/key",
        width=640,
        height=480,
        frame_index=1,
        capture_time_unix_ns=int(_T0.timestamp() * 1e9),
    )
    ctx = FrameContext(frame=frame, event_time=_T0, capture_time=_T0)
    ctx.effective_width = 640
    ctx.effective_height = 480
    detection = Detection(
        detection_id="d1",
        camera_id="cam01",
        bbox=BoundingBox(x_min=10, y_min=10, x_max=100, y_max=300),
        embedding=[1.0, 0.0, 0.0, 0.0],
        capture_time=_T0,
        event_time=_T0,
        confidence=0.95,
        floor_point=FloorPoint(1000, 1000, calibrated=True),
        crop_quality=0.9,
    )
    ctx.domain_detections = [detection]
    ctx.crops = [_crop()]
    ctx.face_anchors = [
        FaceAnchor(
            person_id=face_person_id,
            confidence=0.95,
            recognition_state="recognized",
            detection_id="d1",
            calibrated_confidence=calibrated_confidence,
        )
    ]
    ctx.orientation_by_detection = {"d1": (orientation, orientation_confidence)}
    ctx.det_to_ph = {"d1": "ph-1"}
    ctx.committed_ids = {"ph-1": committed_identity_id}
    ctx.det_to_observation_id = {"d1": "00000000-0000-0000-0000-0000000000aa"}
    return ctx


async def test_eligible_frame_creates_one_pending_candidate_with_full_provenance() -> None:
    gallery = InMemoryGalleryRepository()
    storage = _FakeCropStorage()
    stage = ReIDCandidateStage(
        gallery_repo=gallery,
        crop_storage=storage,
        policy=CandidatePolicy(model_version="reid-solider"),
    )

    await stage.run(_ctx())

    rows, total = await gallery.list_review_candidates(state="pending_review")
    assert total == 1
    row = rows[0]
    assert row.identity_id == "grandma"
    assert row.candidate_reason == "face_derived"
    assert row.ph_id == "ph-1"
    assert row.observation_id == "00000000-0000-0000-0000-0000000000aa"
    assert row.source_episode_id == "ph-1"
    assert row.model_version == "reid-solider"
    assert row.crop_key is not None
    assert row.crop_hash is not None
    assert storage.uploaded  # crop object uploaded

    entries = await gallery.list_gallery_entries(
        identity_id="grandma", active_only=False, states=None
    )
    assert len(entries) == 1
    assert entries[0].origin_tracklet_id == "00000000-0000-0000-0000-0000000000aa"
    assert entries[0].ph_id == "ph-1"


async def test_mismatched_face_identity_creates_no_row() -> None:
    gallery = InMemoryGalleryRepository()
    storage = _FakeCropStorage()
    stage = ReIDCandidateStage(gallery_repo=gallery, crop_storage=storage, policy=CandidatePolicy())
    before = _rejected_count("identity_mismatch")

    await stage.run(_ctx(committed_identity_id="grandma", face_person_id="amma"))

    _rows, total = await gallery.list_review_candidates(state="pending_review")
    assert total == 0
    assert _rejected_count("identity_mismatch") == pytest.approx(before + 1)


async def test_uncalibrated_face_anchor_rejected_fail_closed() -> None:
    gallery = InMemoryGalleryRepository()
    storage = _FakeCropStorage()
    stage = ReIDCandidateStage(gallery_repo=gallery, crop_storage=storage, policy=CandidatePolicy())
    before = _rejected_count("calibration_not_authoritative")

    await stage.run(_ctx(calibrated_confidence=None))

    _rows, total = await gallery.list_review_candidates(state="pending_review")
    assert total == 0
    assert _rejected_count("calibration_not_authoritative") == pytest.approx(before + 1)


async def test_cap_counts_pending_and_verified_rows() -> None:
    gallery = InMemoryGalleryRepository()
    storage = _FakeCropStorage()
    policy = CandidatePolicy(max_per_identity_orientation=10)
    stage = ReIDCandidateStage(gallery_repo=gallery, crop_storage=storage, policy=policy)

    # 6 pending + 4 verified = 10 already at the cap.
    for i in range(6):
        await gallery.upsert_gallery_entry(
            _entry(f"pending-{i}", state="pending_review", orientation=int(OrientationBin.FRONT))
        )
    for i in range(4):
        await gallery.upsert_gallery_entry(
            _entry(
                f"verified-{i}",
                state="operator_verified",
                orientation=int(OrientationBin.FRONT),
            )
        )
    assert (
        await gallery.count_gallery_entries(
            "grandma", int(OrientationBin.FRONT), states=PENDING_AND_VERIFIED
        )
        == 10
    )
    before = _rejected_count("cap_reached")

    await stage.run(_ctx())

    # The 11th (stage's own candidate) was skipped: the cap count stays at 10,
    # and no new pending_review row exists for the stage's own candidate_id.
    assert (
        await gallery.count_gallery_entries(
            "grandma", int(OrientationBin.FRONT), states=PENDING_AND_VERIFIED
        )
        == 10
    )
    _rows, total = await gallery.list_review_candidates(state="pending_review")
    assert total == 0  # pre-seeded rows bypass create_review_candidate's audit trail
    assert _rejected_count("cap_reached") == pytest.approx(before + 1)


def _entry(entry_id: str, *, state: str, orientation: int) -> object:
    from app.domain import GalleryEmbedding

    return GalleryEmbedding(
        gallery_entry_id=entry_id,
        identity_id="grandma",
        embedding=[1.0, 0.0, 0.0, 0.0],
        seen_at=_T0,
        quality=0.9,
        orientation=orientation,
        state=state,
    )


async def test_db_failure_after_minio_put_deletes_orphan_object() -> None:
    gallery = _FailingCreateGalleryRepo()
    storage = _FakeCropStorage()
    stage = ReIDCandidateStage(gallery_repo=gallery, crop_storage=storage, policy=CandidatePolicy())

    await stage.run(_ctx())

    assert storage.uploaded  # the crop was uploaded before the DB write failed
    uploaded_key = next(iter(storage.uploaded))
    assert storage.deleted == [uploaded_key]


async def test_minio_unconfigured_creates_no_row_and_warns_once() -> None:
    from unittest.mock import patch

    gallery = InMemoryGalleryRepository()
    stage = ReIDCandidateStage(gallery_repo=gallery, crop_storage=None, policy=CandidatePolicy())

    with patch("app.pipeline.stages.reid_candidates.logger.warning") as mock_warn:
        await stage.run(_ctx())
        await stage.run(_ctx())

    _rows, total = await gallery.list_review_candidates(state="pending_review")
    assert total == 0
    assert mock_warn.call_count == 1


async def test_create_review_candidate_idempotent_on_retry() -> None:
    """Repository-level proof for the M05 recoverable-object-creation requirement:
    a retry with the same candidate_id after a partial failure never duplicates."""
    from app.domain import NewReviewCandidate

    gallery = InMemoryGalleryRepository()
    candidate = NewReviewCandidate(
        candidate_id="cand-1",
        identity_id="grandma",
        embedding=[1.0, 0.0, 0.0, 0.0],
        quality=0.9,
        orientation=int(OrientationBin.FRONT),
        camera_id="cam01",
        capture_time=_T0,
        ph_id="ph-1",
        observation_id="obs-1",
        origin_tracklet_id="obs-1",
        keyframe_id=None,
        crop_key="reid-candidates/v1/cand-1.jpg",
        source_frame_key=None,
        crop_hash="deadbeef",
        frame_hash=None,
        dimensions=(128, 256),
        is_truncated=False,
        is_occluded=False,
        candidate_reason="face_derived",
        source_episode_id="ph-1",
        created_actor="pipeline",
        model_version="reid-solider",
        preprocessing_version="v1",
        confidence=0.9,
    )

    first_id = await gallery.create_review_candidate(candidate)
    second_id = await gallery.create_review_candidate(candidate)

    assert first_id == second_id == "cand-1"
    _rows, total = await gallery.list_review_candidates(state="pending_review")
    assert total == 1
