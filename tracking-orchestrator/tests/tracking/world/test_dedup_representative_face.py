"""Which face anchor a cross-camera dedup cluster keeps for its representative observation.

The representative carries one face into association. Ranking by raw ``match_score`` alone
compared a face detection score (unrecognized markers) with a cosine similarity (recognized
faces), so a confidently detected stranger face could displace a recognized resident.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain import BoundingBox, FaceAnchor, FloorPoint, WorldObservation
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.dedup import dedup_observations

_NOW = datetime(2026, 11, 11, 10, 0, 0, tzinfo=UTC)
_CFG = WorldTrackerConfig(dedup_enabled=True, dedup_max_distance_m=0.6)


def _face(
    camera_id: str,
    state: str,
    match_score: float,
    *,
    person_id: str = "alice",
    calibrated_confidence: float | None = None,
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id if state != "unrecognized" else "unknown",
        match_score=match_score,
        camera_id=camera_id,
        detection_id=f"det-{camera_id}",
        recognition_state=state,
        similarity=match_score if state != "unrecognized" else 0.1,
        calibrated_confidence=calibrated_confidence,
    )


def _obs(camera_id: str, x_m: float, quality: float, face: FaceAnchor | None) -> WorldObservation:
    return WorldObservation(
        camera_id=camera_id,
        frame_index=1,
        captured_at=_NOW,
        floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=5000, calibrated=True),
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
        embedding=[0.9, 0.1, 0.0],
        detection_confidence=0.92,
        detection_id=f"det-{camera_id}",
        face_anchor=face,
        quality=quality,
    )


def _representative_face(
    best_face: FaceAnchor | None, other_face: FaceAnchor | None
) -> FaceAnchor | None:
    # cam-1 has the higher crop quality, so its observation is the cluster's best observation.
    best = _obs("cam-1", 5.0, 0.8, best_face)
    other = _obs("cam-2", 5.1, 0.4, other_face)
    deduped, _ = dedup_observations([best, other], _CFG)
    assert len(deduped) == 1, "the two observations must merge for the test to mean anything"
    assert deduped[0].camera_id == "cam-1"
    return deduped[0].face_anchor


def test_recognized_face_beats_unrecognized_marker_with_higher_detection_score() -> None:
    unrecognized = _face("cam-1", "unrecognized", 0.95)
    recognized = _face("cam-2", "recognized", 0.55)

    assert _representative_face(unrecognized, recognized) is recognized


def test_recognized_face_beats_candidate_with_higher_similarity() -> None:
    candidate = _face("cam-1", "candidate", 0.58)
    recognized = _face("cam-2", "recognized", 0.52)

    assert _representative_face(candidate, recognized) is recognized


def test_candidate_beats_unrecognized_marker() -> None:
    unrecognized = _face("cam-1", "unrecognized", 0.99)
    candidate = _face("cam-2", "candidate", 0.41)

    assert _representative_face(unrecognized, candidate) is candidate


def test_calibrated_face_beats_higher_raw_similarity_without_calibration() -> None:
    uncalibrated = _face("cam-1", "recognized", 0.80)
    calibrated = _face("cam-2", "recognized", 0.62, calibrated_confidence=0.85)

    assert _representative_face(uncalibrated, calibrated) is calibrated


def test_calibrated_confidence_of_zero_still_outranks_no_calibration() -> None:
    uncalibrated = _face("cam-1", "recognized", 0.70)
    calibrated_zero = _face("cam-2", "recognized", 0.61, calibrated_confidence=0.0)

    assert _representative_face(uncalibrated, calibrated_zero) is calibrated_zero


def test_higher_calibrated_confidence_wins_between_calibrated_faces() -> None:
    lower = _face("cam-1", "recognized", 0.90, calibrated_confidence=0.81)
    higher = _face("cam-2", "recognized", 0.70, calibrated_confidence=0.93)

    assert _representative_face(lower, higher) is higher


def test_exact_tie_keeps_the_best_observations_own_face() -> None:
    own = _face("cam-1", "recognized", 0.75)
    other = _face("cam-2", "recognized", 0.75)

    assert _representative_face(own, other) is own


def test_face_from_another_observation_fills_a_faceless_best_observation() -> None:
    other = _face("cam-2", "candidate", 0.45)

    assert _representative_face(None, other) is other
    assert _representative_face(None, None) is None
