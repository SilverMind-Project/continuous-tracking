"""Unit tests for BboxAnnotationRepository (in-memory implementation).

Tests cover:
- save_bbox_annotations + retrieve by keyframe
- retrieve by tracklet
- update_identity_id propagates
- save_override_bbox persists
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import BboxAnnotation
from app.storage.base import InMemoryBboxAnnotationRepository


def _bbox(
    keyframe_id: str = "kf1",
    tracklet_id: str = "tr1",
    camera_id: str = "cam-a",
    identity_id: str | None = None,
) -> BboxAnnotation:
    return BboxAnnotation(
        keyframe_id=keyframe_id,
        tracklet_id=tracklet_id,
        camera_id=camera_id,
        x1=10.0,
        y1=20.0,
        x2=100.0,
        y2=200.0,
        detection_confidence=0.95,
        frame_width=1920,
        frame_height=1080,
        identity_id=identity_id,
        created_at=datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC),
    )


@pytest.fixture()
def repo() -> InMemoryBboxAnnotationRepository:
    return InMemoryBboxAnnotationRepository()


async def test_save_and_retrieve_by_keyframe(repo: InMemoryBboxAnnotationRepository) -> None:
    ann = _bbox()
    await repo.save_bbox_annotations([ann])
    results = await repo.get_bbox_annotations_for_keyframe("kf1")
    assert len(results) == 1
    assert results[0].x1 == ann.x1
    assert results[0].tracklet_id == ann.tracklet_id


async def test_save_and_retrieve_by_tracklet(repo: InMemoryBboxAnnotationRepository) -> None:
    ann = _bbox(tracklet_id="tr-alpha")
    await repo.save_bbox_annotations([ann])
    results = await repo.get_bbox_annotations_for_tracklet("tr-alpha")
    assert len(results) == 1
    assert results[0].keyframe_id == "kf1"


async def test_retrieve_empty_for_unknown_keyframe(repo: InMemoryBboxAnnotationRepository) -> None:
    results = await repo.get_bbox_annotations_for_keyframe("nonexistent")
    assert results == []


async def test_update_identity_id_propagates(repo: InMemoryBboxAnnotationRepository) -> None:
    ann = _bbox(identity_id=None)
    await repo.save_bbox_annotations([ann])
    await repo.update_identity_id("tr1", "alice")
    results = await repo.get_bbox_annotations_for_keyframe("kf1")
    assert results[0].identity_id == "alice"


async def test_update_identity_id_only_matches_target_tracklet(
    repo: InMemoryBboxAnnotationRepository,
) -> None:
    await repo.save_bbox_annotations(
        [
            _bbox(keyframe_id="kf1", tracklet_id="tr1"),
            _bbox(keyframe_id="kf2", tracklet_id="tr2"),
        ]
    )
    await repo.update_identity_id("tr1", "alice")
    kf1_results = await repo.get_bbox_annotations_for_keyframe("kf1")
    kf2_results = await repo.get_bbox_annotations_for_keyframe("kf2")
    assert kf1_results[0].identity_id == "alice"
    assert kf2_results[0].identity_id is None


async def test_override_bbox_persists(repo: InMemoryBboxAnnotationRepository) -> None:
    ann = _bbox()
    await repo.save_bbox_annotations([ann])

    # Retrieve the auto-generated ID.
    results = await repo.get_bbox_annotations_for_keyframe("kf1")
    assert len(results) == 1
    # The ID is generated internally in the repo.
    # For InMemory, we need to iterate to find the ID.
    annotation_id = next(iter(repo._rows.keys()))

    await repo.save_override_bbox(
        annotation_id,
        x1=50.0,
        y1=60.0,
        x2=150.0,
        y2=250.0,
        override_by="caregiver1",
    )

    results = await repo.get_bbox_annotations_for_keyframe("kf1")
    assert len(results) == 1
    assert results[0].override_x1 == 50.0
    assert results[0].override_y1 == 60.0
    assert results[0].override_x2 == 150.0
    assert results[0].override_y2 == 250.0
    assert results[0].override_by == "caregiver1"
    assert results[0].override_at is not None


async def test_save_empty_list_no_op(repo: InMemoryBboxAnnotationRepository) -> None:
    await repo.save_bbox_annotations([])
    results = await repo.get_bbox_annotations_for_keyframe("kf1")
    assert results == []


async def test_multiple_annotations_same_keyframe(repo: InMemoryBboxAnnotationRepository) -> None:
    await repo.save_bbox_annotations(
        [
            _bbox(keyframe_id="kf1", tracklet_id="tr1"),
            _bbox(keyframe_id="kf1", tracklet_id="tr2"),
        ]
    )
    results = await repo.get_bbox_annotations_for_keyframe("kf1")
    assert len(results) == 2
