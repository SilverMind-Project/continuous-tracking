"""Identity Continuity M02: auto_verified minting rule coverage.

``evaluate_candidate``'s ``mint_state`` field decides whether an eligible
candidate lands ``pending_review`` or ``auto_verified`` (decision D3). Every
case here assumes the candidate is already eligible (calibrated >= 0.80,
the ArcFace authority bar); the mint-state boundary sits at 0.90 and is
fail-closed against raw (uncalibrated) confidence.
"""

from __future__ import annotations

from app.domain import FaceAnchor, OrientationBin
from app.tracking.identity.candidate_eligibility import (
    CandidateEligibility,
    CandidatePolicy,
    evaluate_candidate,
)

_EMB = [1.0, 0.0, 0.0, 0.0]


def _face(
    *,
    person_id: str = "grandma",
    confidence: float = 0.95,
    calibrated_confidence: float | None = 0.90,
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        confidence=confidence,
        recognition_state="recognized",
        calibrated_confidence=calibrated_confidence,
    )


def _eval(*, cfg: CandidatePolicy, face_anchor: FaceAnchor) -> CandidateEligibility:
    return evaluate_candidate(
        committed_identity_id="grandma",
        face_anchor=face_anchor,
        embedding=_EMB,
        quality=0.9,
        orientation=OrientationBin.FRONT,
        orientation_confidence=0.9,
        cfg=cfg,
    )


def test_calibrated_090_mints_auto_verified() -> None:
    result = _eval(cfg=CandidatePolicy(), face_anchor=_face(calibrated_confidence=0.90))
    assert result.eligible is True
    assert result.mint_state == "auto_verified"


def test_calibrated_089_mints_pending() -> None:
    result = _eval(cfg=CandidatePolicy(), face_anchor=_face(calibrated_confidence=0.89))
    assert result.eligible is True
    assert result.mint_state == "pending_review"


def test_uncalibrated_high_raw_confidence_mints_pending() -> None:
    """Fail-closed: a high raw ArcFace similarity never substitutes for a
    calibrated confidence in the auto-verify decision, even when the
    eligibility gate itself was relaxed to allow uncalibrated candidates."""
    cfg = CandidatePolicy(require_calibrated_face=False)
    result = _eval(
        cfg=cfg,
        face_anchor=_face(confidence=0.99, calibrated_confidence=None),
    )
    assert result.eligible is True
    assert result.mint_state == "pending_review"


def test_disabled_threshold_none_never_auto() -> None:
    """Config sanity / rollback path: setting the threshold above 1.0 (the
    documented disable knob) means nothing can ever auto-verify, even a
    perfect calibrated confidence of 1.0."""
    cfg = CandidatePolicy(auto_verify_min_confidence=1.1)
    result = _eval(cfg=cfg, face_anchor=_face(calibrated_confidence=1.0))
    assert result.eligible is True
    assert result.mint_state == "pending_review"
