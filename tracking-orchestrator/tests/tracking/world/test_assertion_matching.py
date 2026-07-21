"""Tests for match_assertions_to_face_anchors (identity-continuity M09).

Covers the spatial/temporal/confidence matching of CC identity assertions
to WorldObservations, producing FaceAnchors for the identity resolver.
"""

from __future__ import annotations

import math
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


# ---------------------------------------------------------------------------
# Floor-path regression gate (behavior unchanged for native-shaped assertions)
# ---------------------------------------------------------------------------


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
    assert anchors[0].confidence == 0.85
    assert anchors[0].origin == "cc_assertion"


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
    """A floor-point assertion with a different camera_id is not matched
    (the camera-id gate applies inside the floor branch)."""
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


# ---------------------------------------------------------------------------
# Room-level matching (identity-continuity M09)
# ---------------------------------------------------------------------------


def test_room_match_creates_anchor_with_scaled_confidence():
    """A coordinate-free assertion matches by room and its confidence is scaled."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "grace",
            "confidence": 0.9,
            "camera_id": "recamera_kitchen",
            "captured_at": now,
            "room_name": "Kitchen",
        }
    ]
    observations = [_make_observation("cts-cam-3", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(
        assertions,
        observations,
        now,
        camera_room_lookup={"cts-cam-3": "Kitchen"},
        room_match_confidence_scale=0.8,
    )
    assert len(anchors) == 1
    assert anchors[0].person_id == "grace"
    assert math.isclose(anchors[0].confidence, 0.9 * 0.8, rel_tol=1e-9)
    assert anchors[0].origin == "cc_assertion"


def test_room_mismatch_rejected():
    """A room-only assertion naming a different room than the observation's camera
    does not match, even though it has no floor point to disagree with."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "henry",
            "confidence": 0.9,
            "camera_id": "recamera_kitchen",
            "captured_at": now,
            "room_name": "Kitchen",
        }
    ]
    observations = [_make_observation("cts-cam-3", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(
        assertions,
        observations,
        now,
        camera_room_lookup={"cts-cam-3": "Living Room"},
    )
    assert len(anchors) == 0


def test_no_floor_no_room_rejected_never_housewide():
    """An assertion with neither camera id, floor point, nor room name matches
    nothing -- it must not anchor to every concurrent observation house-wide."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "ivan",
            "confidence": 0.95,
            "captured_at": now,
        }
    ]
    observations = [
        _make_observation("cam-1", 1.0, 2.0, detection_id="det-a"),
        _make_observation("cam-2", 50.0, 60.0, detection_id="det-b"),
    ]

    anchors = match_assertions_to_face_anchors(
        assertions,
        observations,
        now,
        camera_room_lookup={"cam-1": "Kitchen", "cam-2": "Bedroom"},
    )
    assert anchors == []


def test_zero_zero_without_flag_is_not_a_position():
    """An assertion dict with floor_x_m/floor_y_m explicitly None (the subscriber's
    representation of "no position") must fall through to the room gate, never
    match the (0, 0) origin as a real position."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "jill",
            "confidence": 0.9,
            "camera_id": "recamera_kitchen",
            "captured_at": now,
            "floor_x_m": None,
            "floor_y_m": None,
            "room_name": "Kitchen",
        }
    ]
    # Observation sits far from (0, 0); if the matcher treated the assertion's
    # absent coordinates as (0, 0) it would reject on distance instead of
    # falling through correctly to the room gate.
    observations = [_make_observation("cts-cam-9", 50.0, 60.0)]

    anchors = match_assertions_to_face_anchors(
        assertions,
        observations,
        now,
        camera_room_lookup={"cts-cam-9": "Kitchen"},
    )
    assert len(anchors) == 1
    assert anchors[0].person_id == "jill"


def test_uncalibrated_assertion_never_matches():
    """An assertion with no calibrated confidence (None) never matches, on
    either spatial gate."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "kim",
            "confidence": None,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        },
        {
            "person_id": "kim",
            "confidence": None,
            "camera_id": "recamera_kitchen",
            "captured_at": now,
            "room_name": "Kitchen",
        },
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(
        assertions,
        observations,
        now,
        camera_room_lookup={"cam-1": "Kitchen"},
    )
    assert anchors == []


def test_anchor_carries_origin_wire_yaw_quality_and_conservative_defaults():
    """Anchor construction carries wire yaw/quality/captured_at when present,
    and falls back to the configured conservative defaults when absent."""
    now = datetime.now(UTC)
    captured = now - timedelta(seconds=5)
    assertions = [
        {
            "person_id": "leo",
            "confidence": 0.9,
            "camera_id": "cam-1",
            "captured_at": captured,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
            "yaw_deg": 12.0,
            "quality": 0.75,
        }
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(assertions, observations, now)
    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.origin == "cc_assertion"
    assert anchor.calibrated_confidence is None
    assert anchor.recognition_state == "recognized"
    assert anchor.yaw_deg == 12.0
    assert anchor.quality == 0.75
    assert anchor.captured_at == captured

    # Absent yaw/quality: never 0.0 (perfectly frontal) or 1.0 (perfect crop).
    assertions_no_meta = [
        {
            "person_id": "leo",
            "confidence": 0.9,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        }
    ]
    anchors2 = match_assertions_to_face_anchors(
        assertions_no_meta,
        observations,
        now,
        default_quality=0.5,
        default_yaw_deg=60.0,
    )
    assert len(anchors2) == 1
    assert anchors2[0].yaw_deg == 60.0
    assert anchors2[0].quality == 0.5


def test_room_names_compared_case_normalized():
    """Room-name comparison is case-insensitive."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "maya",
            "confidence": 0.9,
            "camera_id": "recamera_kitchen",
            "captured_at": now,
            "room_name": "KITCHEN",
        }
    ]
    observations = [_make_observation("cts-cam-3", 1.0, 2.0)]

    anchors = match_assertions_to_face_anchors(
        assertions,
        observations,
        now,
        camera_room_lookup={"cts-cam-3": "kitchen"},
    )
    assert len(anchors) == 1
    assert anchors[0].person_id == "maya"


def test_diagnostics_records_matched_and_rejected_reasons():
    """The optional diagnostics dict is populated for matches and rejections."""
    now = datetime.now(UTC)
    assertions = [
        {
            "person_id": "nora",
            "confidence": 0.9,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        },
        {
            "person_id": "oscar",
            "confidence": None,
            "camera_id": "cam-1",
            "captured_at": now,
            "floor_x_m": 1.0,
            "floor_y_m": 2.0,
        },
    ]
    observations = [_make_observation("cam-1", 1.0, 2.0)]
    diagnostics: dict[str, int] = {}

    match_assertions_to_face_anchors(assertions, observations, now, diagnostics=diagnostics)
    assert diagnostics.get("matched_floor") == 1
    assert diagnostics.get("uncalibrated") == 1
