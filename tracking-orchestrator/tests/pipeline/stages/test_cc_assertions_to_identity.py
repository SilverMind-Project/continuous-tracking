"""CC assertions to identity evidence tests.

Tests the spatial/temporal matching of CC identity assertions to
WorldObservations, producing FaceAnchors for the identity resolver.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain import BoundingBox, FloorPoint, WorldObservation
from app.tracking.world.assertion_matching import match_assertions_to_face_anchors


def _make_observation(
    camera_id: str = "cam-1",
    floor_x: float = 1.0,
    floor_y: float = 2.0,
    detection_id: str = "det-1",
) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=1,
        captured_at=datetime.now(UTC),
        floor_point=FloorPoint(x_mm=int(floor_x * 1000), y_mm=int(floor_y * 1000), calibrated=True),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        embedding=[],
        detection_confidence=0.9,
        detection_id=detection_id,
    )


def test_matching_assertion_produces_face_anchor():
    """A matching assertion (within time, distance, confidence) becomes a FaceAnchor."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "alice",
            "confidence": 0.85,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        }
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(assertions, observations, now)
    assert len(anchors) == 1
    assert anchors[0].person_id == "alice"
    assert anchors[0].detection_id == "det-1"


def test_far_away_assertion_is_ignored():
    """An assertion too far from the observation is not matched."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "bob",
            "confidence": 0.9,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 10.0,  # 8m away
            "floor_y_m": 10.0,  # 8m away
        }
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(
        assertions, observations, now, anchor_match_distance_m=5.0
    )
    assert len(anchors) == 0


def test_old_assertion_is_ignored():
    """An assertion outside the time window is not matched."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "carol",
            "confidence": 0.9,
            "camera_id": "cam-1",
            "captured_at": now - timedelta(seconds=120),  # 2 minutes old
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        }
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(
        assertions, observations, now, anchor_match_window_s=30.0
    )
    assert len(anchors) == 0


def test_low_confidence_assertion_is_ignored():
    """An assertion below min_confidence is not matched."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "dave",
            "confidence": 0.3,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        }
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(
        assertions, observations, now, anchor_min_confidence=0.5
    )
    assert len(anchors) == 0


def test_camera_id_mismatch_is_ignored():
    """An assertion with a different camera_id is not matched."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "eve",
            "confidence": 0.9,
            "camera_id": "cam-2",  # different camera
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        }
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(assertions, observations, now)
    assert len(anchors) == 0


def test_strongest_assertion_per_observation_wins():
    """When multiple assertions match, the highest-confidence one is used."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "alice",
            "confidence": 0.7,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        },
        {
            "person_id": "bob",
            "confidence": 0.95,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        },
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(assertions, observations, now)
    assert len(anchors) == 1
    assert anchors[0].person_id == "bob"
    assert anchors[0].confidence == 0.95


def test_empty_assertions_returns_empty():
    """No assertions produces no anchors."""
    observations = [_make_observation()]
    anchors = match_assertions_to_face_anchors([], observations, datetime.now(UTC))
    assert anchors == []
