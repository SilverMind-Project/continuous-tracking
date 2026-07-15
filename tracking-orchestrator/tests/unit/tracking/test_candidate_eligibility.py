"""M04: candidate_eligibility.evaluate_candidate gate coverage.

Every typed rejection reason, boundary values at the calibrated-confidence
and quality thresholds, and the unconditional identity-mismatch gate (the
direct fix for F3, the confirmed gallery seed-identity mismatch bug).
"""

from __future__ import annotations

from app.domain import FaceAnchor, OrientationBin
from app.tracking.identity.candidate_eligibility import (
    CandidateEligibility,
    CandidatePolicy,
    evaluate_candidate,
)

_CFG = CandidatePolicy()
_EMB = [1.0, 0.0, 0.0, 0.0]


def _face(
    *,
    person_id: str = "grandma",
    recognition_state: str = "recognized",
    calibrated_confidence: float | None = 0.90,
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        confidence=0.95,
        recognition_state=recognition_state,
        calibrated_confidence=calibrated_confidence,
    )


def _eval(**overrides: object) -> CandidateEligibility:
    kwargs: dict[str, object] = {
        "committed_identity_id": "grandma",
        "face_anchor": _face(),
        "embedding": _EMB,
        "quality": 0.9,
        "orientation": OrientationBin.FRONT,
        "orientation_confidence": 0.9,
        "cfg": _CFG,
    }
    kwargs.update(overrides)
    return evaluate_candidate(**kwargs)  # type: ignore[arg-type]


def test_eligible_baseline() -> None:
    result = _eval()
    assert result.eligible is True
    assert result.reason == ""


def test_no_identity_when_committed_identity_missing() -> None:
    for missing in (None, ""):
        result = _eval(committed_identity_id=missing)
        assert result.eligible is False
        assert result.reason == "no_identity"


def test_no_face_anchor() -> None:
    result = _eval(face_anchor=None)
    assert result.eligible is False
    assert result.reason == "no_face_anchor"


def test_face_not_recognized_candidate_state() -> None:
    result = _eval(face_anchor=_face(recognition_state="candidate"))
    assert result.eligible is False
    assert result.reason == "face_not_recognized"


def test_face_not_recognized_unrecognized_state() -> None:
    result = _eval(face_anchor=_face(recognition_state="unrecognized"))
    assert result.eligible is False
    assert result.reason == "face_not_recognized"


def test_identity_mismatch_unconditional_gate() -> None:
    """F3: a recognized face naming a different person than the committed
    identity must never be eligible, regardless of any other evidence quality."""
    result = _eval(
        committed_identity_id="grandma",
        face_anchor=_face(person_id="amma"),
    )
    assert result.eligible is False
    assert result.reason == "identity_mismatch"


def test_calibration_not_authoritative_when_missing() -> None:
    result = _eval(face_anchor=_face(calibrated_confidence=None))
    assert result.eligible is False
    assert result.reason == "calibration_not_authoritative"


def test_calibration_boundary_just_below_threshold() -> None:
    result = _eval(face_anchor=_face(calibrated_confidence=0.79))
    assert result.eligible is False
    assert result.reason == "calibration_not_authoritative"


def test_calibration_boundary_at_threshold_is_eligible() -> None:
    result = _eval(face_anchor=_face(calibrated_confidence=0.80))
    assert result.eligible is True


def test_calibration_gate_disabled_by_policy() -> None:
    cfg = CandidatePolicy(require_calibrated_face=False)
    result = _eval(face_anchor=_face(calibrated_confidence=None), cfg=cfg)
    assert result.eligible is True


def test_unknown_orientation_rejected() -> None:
    result = _eval(orientation=OrientationBin.UNKNOWN, orientation_confidence=0.9)
    assert result.eligible is False
    assert result.reason == "unknown_orientation"


def test_low_orientation_confidence() -> None:
    result = _eval(orientation=OrientationBin.FRONT, orientation_confidence=0.1)
    assert result.eligible is False
    assert result.reason == "low_orientation_confidence"


def test_orientation_confidence_boundary_at_threshold_is_eligible() -> None:
    result = _eval(orientation_confidence=0.5)
    assert result.eligible is True


def test_no_embedding() -> None:
    for empty in (None, []):
        result = _eval(embedding=empty)
        assert result.eligible is False
        assert result.reason == "no_embedding"


def test_non_finite_embedding() -> None:
    result = _eval(embedding=[float("nan"), 0.0, 0.0, 0.0])
    assert result.eligible is False
    assert result.reason == "non_finite_embedding"


def test_zero_vector_embedding_rejected() -> None:
    result = _eval(embedding=[0.0, 0.0, 0.0, 0.0])
    assert result.eligible is False
    assert result.reason == "non_finite_embedding"


def test_low_quality() -> None:
    result = _eval(quality=0.1)
    assert result.eligible is False
    assert result.reason == "low_quality"


def test_quality_boundary_just_below_threshold() -> None:
    result = _eval(quality=0.349)
    assert result.eligible is False
    assert result.reason == "low_quality"


def test_quality_boundary_at_threshold_is_eligible() -> None:
    result = _eval(quality=0.35)
    assert result.eligible is True
