"""Pure eligibility gate for governed ReID gallery candidate creation (M04).

Decides whether one detection's evidence may become a ``pending_review``
gallery row. No I/O, no repository access: :class:`ReIDCandidateStage`
(``app/pipeline/stages/reid_candidates.py``) calls this and then persists via
``GalleryRepository.create_review_candidate``. See the cts-identity-governance
skill for the authority rules this encodes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ...domain import FaceAnchor, OrientationBin


@dataclass(frozen=True)
class CandidatePolicy:
    """Tunables for :func:`evaluate_candidate`. Wired from ``settings.yaml``'s
    ``reid_candidates:`` section in ``app/main.py`` — never a silent call-site default.
    """

    enabled: bool = True
    require_calibrated_face: bool = True
    calibrated_confidence_min: float = 0.80
    seed_orientation_min_confidence: float = 0.5
    min_quality: float = 0.35
    max_per_identity_orientation: int = 10
    # Identity-continuity M02, decision D3: a calibrated direct-face
    # confidence at or above this bar mints the candidate straight into
    # auto_verified instead of pending_review. Raw (uncalibrated) confidence
    # never qualifies, matching the ArcFace-authority fail-closed posture.
    auto_verify_min_confidence: float = 0.90
    # Provenance stamped on every created row. model_version reuses the live
    # Triton ReID model name (settings.yaml `triton.reid_model`) rather than a
    # hardcoded "v1" (the M09 lesson: a wrong static version silently makes
    # cross-version compatibility partitioning meaningless).
    model_version: str = ""
    preprocessing_version: str = "v1"


@dataclass(frozen=True)
class CandidateEligibility:
    """Result of :func:`evaluate_candidate`. ``reason`` is ``""`` iff eligible.

    ``mint_state`` is only meaningful when ``eligible`` is True: it is the
    ``reid_gallery`` lifecycle state the candidate should be created in,
    either ``"pending_review"`` or ``"auto_verified"`` (M02, decision D3).
    """

    eligible: bool
    reason: str
    mint_state: str = "pending_review"


def _ineligible(reason: str) -> CandidateEligibility:
    return CandidateEligibility(eligible=False, reason=reason)


def evaluate_candidate(
    *,
    committed_identity_id: str | None,
    face_anchor: FaceAnchor | None,
    embedding: Sequence[float] | None,
    quality: float,
    orientation: OrientationBin,
    orientation_confidence: float,
    cfg: CandidatePolicy,
) -> CandidateEligibility:
    """Evaluate whether one detection may seed a governed gallery candidate.

    Every gate below is fail-closed: an ineligible detection produces nothing,
    never a degraded or unlabeled row. The identity-equality gate
    (``identity_mismatch``) is unconditional and is the direct fix for the
    confirmed gallery seed-identity mismatch bug (F3): a face naming a
    different person than the PH's committed identity must never seed under
    the held label.
    """
    if not committed_identity_id:
        return _ineligible("no_identity")

    if face_anchor is None:
        return _ineligible("no_face_anchor")

    if face_anchor.recognition_state != "recognized":
        return _ineligible("face_not_recognized")

    if face_anchor.person_id != committed_identity_id:
        return _ineligible("identity_mismatch")

    if cfg.require_calibrated_face:
        calibrated = face_anchor.calibrated_confidence
        if calibrated is None or calibrated < cfg.calibrated_confidence_min:
            return _ineligible("calibration_not_authoritative")

    if orientation == OrientationBin.UNKNOWN:
        return _ineligible("unknown_orientation")

    if orientation_confidence < cfg.seed_orientation_min_confidence:
        return _ineligible("low_orientation_confidence")

    if not embedding:
        return _ineligible("no_embedding")

    if not all(math.isfinite(v) for v in embedding) or all(v == 0.0 for v in embedding):
        return _ineligible("non_finite_embedding")

    if quality < cfg.min_quality:
        return _ineligible("low_quality")

    # Auto-verify mint rule (M02, D3): only a *calibrated* confidence at or
    # above the bar mints auto_verified. Raw ArcFace similarity is never
    # substituted here, even when calibration is unavailable and the
    # eligibility gate above already fell back to raw confidence -- that
    # would let an uncalibrated match auto-verify, breaking the fail-closed
    # posture the ArcFace authority gate depends on.
    mint_state = "pending_review"
    if (
        face_anchor.calibrated_confidence is not None
        and face_anchor.calibrated_confidence >= cfg.auto_verify_min_confidence
    ):
        mint_state = "auto_verified"

    return CandidateEligibility(eligible=True, reason="", mint_state=mint_state)
