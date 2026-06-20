"""M00 characterization for mismatched face and gallery seed labels."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest

from app.domain import BoundingBox, FaceAnchor, FloorPoint, OrientationBin, WorldObservation
from app.storage.base import InMemoryGalleryRepository
from app.tracking.world.tracker import WorldTracker

_FIXTURE = (
    Path(__file__).parents[1] / "fixtures/identity_integrity/gallery_seed_identity_mismatch.json"
)


class _GallerySeedHarness:
    def __init__(self, gallery_repo: InMemoryGalleryRepository) -> None:
        self._gallery_repo = gallery_repo
        self._identity_resolver = None


@pytest.mark.xfail(
    strict=True,
    reason="M05 removes this xfail when candidate identity must equal direct ArcFace identity",
)
async def test_recognized_face_must_match_gallery_seed_identity() -> None:
    data = json.loads(_FIXTURE.read_text())
    captured_at = datetime.fromisoformat(data["captured_at"])
    repo = InMemoryGalleryRepository()
    harness = _GallerySeedHarness(repo)
    observation = WorldObservation(
        camera_id=data["camera_id"],
        frame_index=1,
        captured_at=captured_at,
        floor_point=FloorPoint(1000, 1000, calibrated=True),
        bbox=BoundingBox(10, 20, 110, 220),
        embedding=data["body_embedding"],
        detection_confidence=0.95,
        face_anchor=FaceAnchor(
            person_id=data["direct_face_identity_id"],
            confidence=0.95,
            recognition_state=data["recognition_state"],
        ),
        quality=data["quality"],
        orientation=OrientationBin[data["orientation"]],
        orientation_confidence=data["orientation_confidence"],
    )

    await WorldTracker._seed_multiview_gallery(
        cast(WorldTracker, harness),
        data["resolved_identity_id"],
        observation,
    )

    seeded = await repo.list_gallery_entries(
        identity_id=data["resolved_identity_id"],
        active_only=False,
    )
    assert seeded == []
