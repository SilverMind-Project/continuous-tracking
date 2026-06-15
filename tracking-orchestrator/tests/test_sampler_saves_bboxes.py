"""Unit tests verifying that KeyframeSampler saves bbox annotations.

Tests cover:
- maybe_sample saves a bbox annotation when detection data is provided.
- trigger_sample saves a bbox annotation when detection data is provided.
- No bbox annotation is saved when the keyframe repo has no linked bbox repo.
- No bbox annotation is saved when detection_bbox is None.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.sampling.keyframe_sampler import FrameBbox, KeyframeSampler, SamplerConfig
from app.storage.base import InMemoryBboxAnnotationRepository, InMemoryKeyframeRepository

_T0 = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
_ANNS: dict[str, object] = {"bbox": [10, 20, 100, 200]}


@pytest.fixture()
def bbox_repo() -> InMemoryBboxAnnotationRepository:
    return InMemoryBboxAnnotationRepository()


@pytest.fixture()
def sampler(bbox_repo: InMemoryBboxAnnotationRepository) -> KeyframeSampler:
    config = SamplerConfig(keyframe_min_interval_s=30.0)
    return KeyframeSampler(
        repo=InMemoryKeyframeRepository(bbox_repo=bbox_repo),
        config=config,
    )


async def test_maybe_sample_saves_bbox_annotation(
    sampler: KeyframeSampler,
    bbox_repo: InMemoryBboxAnnotationRepository,
) -> None:
    kf = await sampler.maybe_sample(
        ph_id="ph-001",
        camera_id="cam-a",
        minio_key="frame.jpg",
        captured_at=_T0,
        annotations=_ANNS,
        detection_bbox=(10.0, 20.0, 100.0, 200.0),
        detection_confidence=0.95,
        detection_frame_width=1920,
        detection_frame_height=1080,
        detection_identity_id="alice",
    )
    assert kf is not None

    results = await bbox_repo.get_bbox_annotations_for_keyframe(kf.keyframe_id)
    assert len(results) == 1
    ann = results[0]
    assert ann.ph_id == "ph-001"
    assert ann.camera_id == "cam-a"
    assert ann.x1 == 10.0
    assert ann.y1 == 20.0
    assert ann.x2 == 100.0
    assert ann.y2 == 200.0
    assert ann.detection_confidence == 0.95
    assert ann.frame_width == 1920
    assert ann.frame_height == 1080
    assert ann.identity_id == "alice"


async def test_trigger_sample_saves_bbox_annotation(
    sampler: KeyframeSampler,
    bbox_repo: InMemoryBboxAnnotationRepository,
) -> None:
    kf = await sampler.trigger_sample(
        ph_id="ph-001",
        camera_id="cam-a",
        minio_key="frame.jpg",
        captured_at=_T0,
        annotations=_ANNS,
        tag_reason="identity_changed",
        detection_bbox=(50.0, 60.0, 150.0, 250.0),
        detection_confidence=0.88,
        detection_frame_width=1280,
        detection_frame_height=720,
        detection_identity_id=None,
    )
    assert kf is not None

    results = await bbox_repo.get_bbox_annotations_for_keyframe(kf.keyframe_id)
    assert len(results) == 1
    ann = results[0]
    assert ann.x1 == 50.0
    assert ann.frame_width == 1280
    assert ann.identity_id is None


async def test_frame_bboxes_persist_every_person(
    sampler: KeyframeSampler,
    bbox_repo: InMemoryBboxAnnotationRepository,
) -> None:
    """A keyframe with two people in the frame stores both bboxes under the
    same keyframe_id, so the annotation editor shows every identity."""
    kf = await sampler.maybe_sample(
        ph_id="ph-amma",
        camera_id="cam02",
        minio_key="frame.jpg",
        captured_at=_T0,
        annotations=_ANNS,
        detection_frame_width=1920,
        detection_frame_height=1080,
        frame_bboxes=[
            FrameBbox("ph-amma", (10.0, 20.0, 100.0, 200.0), 0.95, "amma"),
            FrameBbox("ph-grandma", (300.0, 40.0, 420.0, 260.0), 0.82, "grandma"),
        ],
    )
    assert kf is not None

    results = await bbox_repo.get_bbox_annotations_for_keyframe(kf.keyframe_id)
    assert len(results) == 2
    by_identity = {a.identity_id: a for a in results}
    assert set(by_identity) == {"amma", "grandma"}
    assert by_identity["grandma"].ph_id == "ph-grandma"
    assert by_identity["grandma"].x1 == 300.0
    assert all(a.frame_width == 1920 for a in results)


async def test_frame_bboxes_take_precedence_over_single_detection(
    sampler: KeyframeSampler,
    bbox_repo: InMemoryBboxAnnotationRepository,
) -> None:
    """When frame_bboxes is given, the single detection_* args are ignored."""
    kf = await sampler.maybe_sample(
        ph_id="ph-amma",
        camera_id="cam02",
        minio_key="frame.jpg",
        captured_at=_T0,
        annotations=_ANNS,
        detection_bbox=(1.0, 2.0, 3.0, 4.0),
        detection_identity_id="ignored",
        frame_bboxes=[FrameBbox("ph-amma", (10.0, 20.0, 100.0, 200.0), 0.95, "amma")],
    )
    assert kf is not None
    results = await bbox_repo.get_bbox_annotations_for_keyframe(kf.keyframe_id)
    assert len(results) == 1
    assert results[0].identity_id == "amma"
    assert results[0].x1 == 10.0


async def test_maybe_sample_no_bbox_when_within_interval(
    sampler: KeyframeSampler,
    bbox_repo: InMemoryBboxAnnotationRepository,
) -> None:
    await sampler.maybe_sample(
        ph_id="ph-001",
        camera_id="cam-a",
        minio_key="f1.jpg",
        captured_at=_T0,
        annotations=_ANNS,
        detection_bbox=(10.0, 20.0, 100.0, 200.0),
        detection_frame_width=1920,
        detection_frame_height=1080,
    )
    t2 = _T0 + timedelta(seconds=15)
    kf = await sampler.maybe_sample(
        ph_id="ph-001",
        camera_id="cam-a",
        minio_key="f2.jpg",
        captured_at=t2,
        annotations=_ANNS,
        detection_bbox=(10.0, 20.0, 100.0, 200.0),
        detection_frame_width=1920,
        detection_frame_height=1080,
    )
    assert kf is None

    all_anns = await bbox_repo.get_bbox_annotations_for_ph("ph-001")
    assert len(all_anns) == 1


async def test_no_bbox_saved_when_detection_bbox_is_none(
    sampler: KeyframeSampler,
    bbox_repo: InMemoryBboxAnnotationRepository,
) -> None:
    kf = await sampler.maybe_sample(
        ph_id="ph-001",
        camera_id="cam-a",
        minio_key="frame.jpg",
        captured_at=_T0,
        annotations=_ANNS,
        detection_bbox=None,
    )
    assert kf is not None
    results = await bbox_repo.get_bbox_annotations_for_keyframe(kf.keyframe_id)
    assert results == []


async def test_no_bbox_saved_when_keyframe_repo_has_no_bbox_repo() -> None:
    sampler_no_bbox = KeyframeSampler(
        repo=InMemoryKeyframeRepository(),
        config=SamplerConfig(keyframe_min_interval_s=30.0),
    )
    kf = await sampler_no_bbox.maybe_sample(
        ph_id="ph-001",
        camera_id="cam-a",
        minio_key="frame.jpg",
        captured_at=_T0,
        annotations=_ANNS,
        detection_bbox=(10.0, 20.0, 100.0, 200.0),
        detection_frame_width=1920,
        detection_frame_height=1080,
    )
    assert kf is not None
    # No bbox repo, so no annotations persisted — no crash.
