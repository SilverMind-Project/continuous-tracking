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
    # Provenance stamped on every created row. model_version reuses the live
    # Triton ReID model name (settings.yaml `triton.reid_model`) rather than a
    # hardcoded "v1" (the M09 lesson: a wrong static version silently makes
    # cross-version compatibility partitioning meaningless).
    model_version: str = ""
    preprocessing_version: str = "v1"


@dataclass(frozen=True)
class CandidateEligibility:
    """Result of :func:`evaluate_candidate`. ``reason`` is ``""`` iff eligible."""

    eligible: bool
    reason: str


def _ineligible(reason: str) -> CandidateEligibility:
    return CandidateEligibility(eligible=False, reason=reason)


_ELIGIBLE = CandidateEligibility(eligible=True, reason="")


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

    return _ELIGIBLE
