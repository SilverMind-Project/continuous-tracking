"""M07: physical-frame keyframe read model grouping and provenance."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.domain import (
    BboxAnnotation,
    IdentityProvenanceDecision,
    IdentityRevisionRange,
    TaggedKeyframe,
)
from app.services.keyframe_read_model import (
    KeyframeReadFilters,
    KeyframeReadModelService,
    KeyframeReadRepositoryBundle,
    physical_frame_id,
)
from app.storage.base import (
    InMemoryBboxAnnotationRepository,
    InMemoryIdentityDecisionRepository,
    InMemoryKeyframeRepository,
)
from app.storage.corrections import InMemoryIdentityCorrectionRepository
from app.storage.gallery import InMemoryGalleryRepository

_T0 = datetime(2026, 6, 19, 12, 0, 0, tzinfo=UTC)
_CAM = "camera-a"
_KEY = "frames/camera-a/2026/06/19/12/0001-0.jpg"


def _service(
    *,
    keyframe_repo: InMemoryKeyframeRepository,
    bbox_repo: InMemoryBboxAnnotationRepository,
    decision_repo: InMemoryIdentityDecisionRepository | None = None,
    correction_repo: InMemoryIdentityCorrectionRepository | None = None,
    gallery_repo: InMemoryGalleryRepository | None = None,
) -> KeyframeReadModelService:
    bundle = KeyframeReadRepositoryBundle(
        keyframe_repo=keyframe_repo,
        bbox_repo=bbox_repo,
        decision_repo=decision_repo or InMemoryIdentityDecisionRepository(),
        correction_repo=correction_repo or InMemoryIdentityCorrectionRepository(),
        gallery_repo=gallery_repo or InMemoryGalleryRepository(),
    )
    return KeyframeReadModelService(bundle)


def _keyframe(
    ph_id: str, *, captured_at: datetime = _T0, reason: str = "periodic"
) -> TaggedKeyframe:
    return TaggedKeyframe(
        keyframe_id=str(uuid.uuid4()),
        ph_id=ph_id,
        camera_id=_CAM,
        minio_key=_KEY,
        captured_at=captured_at,
        annotations={},
        tag_reason=reason,  # type: ignore[arg-type]
        expires_at=captured_at + timedelta(days=1),
    )


def _bbox(keyframe_id: str, ph_id: str, identity_id: str | None, x: float) -> BboxAnnotation:
    return BboxAnnotation(
        keyframe_id=keyframe_id,
        ph_id=ph_id,
        camera_id=_CAM,
        x1=x,
        y1=10,
        x2=x + 100,
        y2=210,
        detection_confidence=0.9,
        frame_width=1920,
        frame_height=1080,
        identity_id=identity_id,
    )


async def _persist(
    keyframe_repo: InMemoryKeyframeRepository,
    bbox_repo: InMemoryBboxAnnotationRepository,
    kf: TaggedKeyframe,
    bboxes: list[BboxAnnotation],
) -> None:
    await keyframe_repo.save_keyframe(kf)
    await bbox_repo.save_bbox_annotations(bboxes)


async def test_two_triggers_same_frame_yield_one_card() -> None:
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    # Same physical frame, two PH triggers. Each trigger persists the whole
    # frame's bbox set (both people).
    kf_a = _keyframe("ph-alpha", reason="identity_changed")
    kf_b = _keyframe("ph-beta", reason="periodic")
    boxes_a = [
        _bbox(kf_a.keyframe_id, "ph-alpha", "amma", 10),
        _bbox(kf_a.keyframe_id, "ph-beta", "grandma", 150),
    ]
    boxes_b = [
        _bbox(kf_b.keyframe_id, "ph-alpha", "amma", 10),
        _bbox(kf_b.keyframe_id, "ph-beta", "grandma", 150),
    ]
    await _persist(keyframe_repo, bbox_repo, kf_a, boxes_a)
    await _persist(keyframe_repo, bbox_repo, kf_b, boxes_b)

    svc = _service(keyframe_repo=keyframe_repo, bbox_repo=bbox_repo)
    page = await svc.list_physical_frames()

    assert page.total == 1
    card = page.frames[0]
    assert card.physical_frame_id == physical_frame_id(_CAM, _KEY, _T0)
    # Both trigger rows are retained as audit records.
    assert {t.tag_reason for t in card.triggers} == {"identity_changed", "periodic"}
    assert len(card.triggers) == 2
    # Bboxes deduplicate across triggers to one per person.
    assert len(card.bboxes) == 2
    assert {b.effective_identity_id for b in card.bboxes} == {"amma", "grandma"}


async def test_duplicate_identity_bboxes_aggregate_but_detail_retained() -> None:
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    kf = _keyframe("ph-alpha")
    # Two distinct PHs resolved to the same identity (e.g. a not-yet-deduped
    # crossing). They must not collapse: distinct PH -> distinct bbox.
    boxes = [
        _bbox(kf.keyframe_id, "ph-alpha", "amma", 10),
        _bbox(kf.keyframe_id, "ph-gamma", "amma", 300),
    ]
    await _persist(keyframe_repo, bbox_repo, kf, boxes)

    svc = _service(keyframe_repo=keyframe_repo, bbox_repo=bbox_repo)
    page = await svc.list_physical_frames()
    card = page.frames[0]

    assert len(card.bboxes) == 2  # detail keeps each PH's box
    assert all(b.effective_identity_id == "amma" for b in card.bboxes)


async def test_unknown_and_conflict_counts_are_explicit() -> None:
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    decision_repo = InMemoryIdentityDecisionRepository()
    kf = _keyframe("ph-alpha")
    boxes = [
        _bbox(kf.keyframe_id, "ph-alpha", None, 10),  # unknown (no identity)
        _bbox(kf.keyframe_id, "ph-beta", "grandma", 150),  # conflict via decision
    ]
    await _persist(keyframe_repo, bbox_repo, kf, boxes)
    await decision_repo.save(
        IdentityProvenanceDecision(
            decision_id=str(uuid.uuid4()),
            ph_id="ph-beta",
            captured_at=_T0,
            authority="none",
            decision_source="reid",
            diagnostics={},
            inferred_identity_id="grandma",
            effective_identity_id="grandma",
            conflict_kind="duplicate_active",
            top_probability=0.55,
        )
    )

    svc = _service(keyframe_repo=keyframe_repo, bbox_repo=bbox_repo, decision_repo=decision_repo)
    card = (await svc.list_physical_frames()).frames[0]

    assert card.unknown_count == 1
    assert card.conflict_count == 1
    beta = next(b for b in card.bboxes if b.ph_id == "ph-beta")
    assert beta.conflict is True
    assert beta.calibrated_confidence == pytest.approx(0.55)


async def test_operator_correction_shows_effective_keeps_inferred() -> None:
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    decision_repo = InMemoryIdentityDecisionRepository()
    correction_repo = InMemoryIdentityCorrectionRepository()
    kf = _keyframe("ph-alpha")
    await _persist(keyframe_repo, bbox_repo, kf, [_bbox(kf.keyframe_id, "ph-alpha", "amma", 10)])
    await decision_repo.save(
        IdentityProvenanceDecision(
            decision_id=str(uuid.uuid4()),
            ph_id="ph-alpha",
            captured_at=_T0,
            authority="arcface_authority",
            decision_source="face",
            diagnostics={},
            inferred_identity_id="amma",
            effective_identity_id="amma",
            top_probability=0.8,
        )
    )
    await correction_repo.save_range(
        IdentityRevisionRange(
            range_id=str(uuid.uuid4()),
            revision_id=str(uuid.uuid4()),
            ph_id="ph-alpha",
            authority="operator",
            range_start=_T0 - timedelta(minutes=1),
            range_end=_T0 + timedelta(minutes=1),
            effective_identity_id="grandma",
        )
    )

    svc = _service(
        keyframe_repo=keyframe_repo,
        bbox_repo=bbox_repo,
        decision_repo=decision_repo,
        correction_repo=correction_repo,
    )
    card = (await svc.list_physical_frames()).frames[0]
    box = card.bboxes[0]

    assert box.inferred_identity_id == "amma"
    assert box.effective_identity_id == "grandma"
    assert box.authority == "operator"
    assert box.calibrated_confidence is None  # operator -> Verified, no number


async def test_filters_apply_before_pagination_and_keep_context() -> None:
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    kf = _keyframe("ph-alpha")
    await _persist(
        keyframe_repo,
        bbox_repo,
        kf,
        [
            _bbox(kf.keyframe_id, "ph-alpha", "amma", 10),
            _bbox(kf.keyframe_id, "ph-beta", "grandma", 150),
        ],
    )
    # A second frame with neither identity.
    kf2 = _keyframe("ph-zeta", captured_at=_T0 + timedelta(seconds=30))
    kf2 = TaggedKeyframe(**{**kf2.__dict__, "minio_key": _KEY + "-other"})
    await _persist(keyframe_repo, bbox_repo, kf2, [_bbox(kf2.keyframe_id, "ph-zeta", "uncle", 10)])

    svc = _service(keyframe_repo=keyframe_repo, bbox_repo=bbox_repo)
    page = await svc.list_physical_frames(filters=KeyframeReadFilters(effective_identity_id="amma"))

    assert page.total == 1
    card = page.frames[0]
    # Frame matched on amma but still returns grandma for context.
    assert {b.effective_identity_id for b in card.bboxes} == {"amma", "grandma"}


async def test_decision_join_is_per_frame_not_page_max() -> None:
    # PH-alpha appears in two frames; its inferred identity changes between
    # them. The older frame must show the older decision, not the newest one.
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    decision_repo = InMemoryIdentityDecisionRepository()
    t1 = _T0
    t2 = _T0 + timedelta(minutes=5)
    for captured, key in ((t1, _KEY + "-1"), (t2, _KEY + "-2")):
        kf = _keyframe("ph-alpha", captured_at=captured)
        kf = TaggedKeyframe(**{**kf.__dict__, "minio_key": key})
        await _persist(
            keyframe_repo, bbox_repo, kf, [_bbox(kf.keyframe_id, "ph-alpha", "amma", 10)]
        )
    for captured, inferred in ((t1, "amma"), (t2, "bob")):
        await decision_repo.save(
            IdentityProvenanceDecision(
                decision_id=str(uuid.uuid4()),
                ph_id="ph-alpha",
                captured_at=captured,
                authority="arcface_authority",
                decision_source="face",
                diagnostics={},
                inferred_identity_id=inferred,
                effective_identity_id=inferred,
                top_probability=0.8,
            )
        )

    svc = _service(keyframe_repo=keyframe_repo, bbox_repo=bbox_repo, decision_repo=decision_repo)
    frames = (await svc.list_physical_frames()).frames
    by_time = {f.captured_at: f for f in frames}
    assert by_time[t1].bboxes[0].inferred_identity_id == "amma"
    assert by_time[t2].bboxes[0].inferred_identity_id == "bob"


async def test_stable_pagination_with_identical_timestamps() -> None:
    # Many frames sharing one capture timestamp must paginate deterministically
    # with no overlap or gap across pages.
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    for i in range(10):
        kf = _keyframe("ph-alpha", captured_at=_T0)
        kf = TaggedKeyframe(**{**kf.__dict__, "minio_key": f"{_KEY}-{i}"})
        await _persist(
            keyframe_repo, bbox_repo, kf, [_bbox(kf.keyframe_id, "ph-alpha", "amma", 10)]
        )

    svc = _service(keyframe_repo=keyframe_repo, bbox_repo=bbox_repo)
    page1 = await svc.list_physical_frames(limit=4, offset=0)
    page2 = await svc.list_physical_frames(limit=4, offset=4)
    page3 = await svc.list_physical_frames(limit=4, offset=8)

    assert page1.total == 10
    ids1 = [f.physical_frame_id for f in page1.frames]
    ids2 = [f.physical_frame_id for f in page2.frames]
    ids3 = [f.physical_frame_id for f in page3.frames]
    seen = ids1 + ids2 + ids3
    assert len(seen) == 10
    assert len(set(seen)) == 10  # no overlap, no gap


class _CountingBundle:
    """Wraps a real bundle and counts calls to each batch read."""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: dict[str, int] = {}

    def _wrap(self, name: str):
        async def _call(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return await getattr(self._inner, name)(*args, **kwargs)

        return _call

    def __getattr__(self, name: str):
        return self._wrap(name)


async def test_no_n_plus_one_query_count_is_bounded() -> None:
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    # Many physical frames, each with several PH bboxes and two triggers.
    for f in range(12):
        captured = _T0 + timedelta(seconds=f)
        key = f"{_KEY}-{f}"
        for reason in ("identity_changed", "periodic"):
            kf = TaggedKeyframe(
                keyframe_id=str(uuid.uuid4()),
                ph_id=f"ph-{f}-a",
                camera_id=_CAM,
                minio_key=key,
                captured_at=captured,
                annotations={},
                tag_reason=reason,  # type: ignore[arg-type]
                expires_at=captured + timedelta(days=1),
            )
            await keyframe_repo.save_keyframe(kf)
            await bbox_repo.save_bbox_annotations(
                [
                    _bbox(kf.keyframe_id, f"ph-{f}-a", "amma", 10),
                    _bbox(kf.keyframe_id, f"ph-{f}-b", "grandma", 150),
                    _bbox(kf.keyframe_id, f"ph-{f}-c", None, 300),
                ]
            )

    inner = KeyframeReadRepositoryBundle(
        keyframe_repo=keyframe_repo,
        bbox_repo=bbox_repo,
        decision_repo=InMemoryIdentityDecisionRepository(),
        correction_repo=InMemoryIdentityCorrectionRepository(),
        gallery_repo=InMemoryGalleryRepository(),
    )
    counting = _CountingBundle(inner)
    svc = KeyframeReadModelService(counting)  # type: ignore[arg-type]

    page = await svc.list_physical_frames(limit=50)

    assert page.total == 12
    # One call to each batch read regardless of frame/bbox cardinality.
    assert counting.calls == {
        "list_keyframes_for_read_model": 1,
        "get_bbox_annotations_for_keyframes": 1,
        "decisions_for_phs": 1,
        "live_ranges_for_phs": 1,
        "phs_with_pending_reid": 1,
    }


async def test_pending_review_joins_only_eligible() -> None:
    keyframe_repo = InMemoryKeyframeRepository()
    bbox_repo = InMemoryBboxAnnotationRepository()
    gallery_repo = InMemoryGalleryRepository()
    from app.domain import GalleryEmbedding

    kf = _keyframe("ph-alpha")
    await _persist(
        keyframe_repo,
        bbox_repo,
        kf,
        [
            _bbox(kf.keyframe_id, "ph-alpha", "amma", 10),
            _bbox(kf.keyframe_id, "ph-beta", "grandma", 150),
        ],
    )
    await gallery_repo.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id=str(uuid.uuid4()),
            identity_id="amma",
            embedding=[0.1, 0.2],
            seen_at=_T0,
            origin_tracklet_id="ph-alpha",
            state="pending_review",
        )
    )
    await gallery_repo.upsert_gallery_entry(
        GalleryEmbedding(
            gallery_entry_id=str(uuid.uuid4()),
            identity_id="grandma",
            embedding=[0.3, 0.4],
            seen_at=_T0,
            origin_tracklet_id="ph-beta",
            state="operator_verified",
        )
    )

    svc = _service(keyframe_repo=keyframe_repo, bbox_repo=bbox_repo, gallery_repo=gallery_repo)
    card = (await svc.list_physical_frames()).frames[0]

    assert card.pending_review_count == 1
    alpha = next(b for b in card.bboxes if b.ph_id == "ph-alpha")
    beta = next(b for b in card.bboxes if b.ph_id == "ph-beta")
    assert alpha.pending_review is True
    assert beta.pending_review is False
