"""Tests for WorldTracker._unpack_observations' association-vector construction.

Identity-continuity M09: external evidence (FaceAnchor.origin == "cc_assertion")
must never drive PH association or hard-conflict splits, only the resolver's
posterior. These are association vectors (face_person_ids/face_confidences),
distinct from the identity resolver's evidence pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import BoundingBox, FaceAnchor, FloorPoint, WorldObservation
from app.tracking.world.tracker import _unpack_observations


def _make_observation(face_anchor: FaceAnchor | None) -> WorldObservation:
    return WorldObservation(
        camera_id="cam-1",
        frame_index=1,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=[],
        detection_confidence=0.9,
        detection_id="det-1",
        face_anchor=face_anchor,
    )


def test_native_recognized_anchor_populates_association_vectors() -> None:
    anchor = FaceAnchor(
        person_id="alice", confidence=0.9, quality=0.9, recognition_state="recognized"
    )
    vecs = _unpack_observations([_make_observation(anchor)])
    assert vecs.face_person_ids == ["alice"]
    assert vecs.face_confidences == [0.9]


def test_cc_assertion_excluded_from_association_vectors() -> None:
    """A recognized cc_assertion anchor must not appear in face_person_ids."""
    anchor = FaceAnchor(
        person_id="alice",
        confidence=0.9,
        quality=0.9,
        recognition_state="recognized",
        origin="cc_assertion",
    )
    vecs = _unpack_observations([_make_observation(anchor)])
    assert vecs.face_person_ids == [None]
    assert vecs.face_confidences == [0.0]


def test_no_face_anchor_produces_none() -> None:
    vecs = _unpack_observations([_make_observation(None)])
    assert vecs.face_person_ids == [None]
    assert vecs.face_confidences == [0.0]
