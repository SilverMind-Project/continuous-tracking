"""Pure eligibility gate for governed ReID gallery candidate creation.

Decides whether one detection's evidence may become a gallery row, and in which state. No I/O,
no repository access: :class:`ReIDCandidateStage` (``app/pipeline/stages/reid_candidates.py``)
calls this and then persists via ``GalleryRepository.create_review_candidate``. A candidate is
seeded either by a recognized face on the detection (:class:`FaceSeed`) or by an attested
wearable association on its PH (:class:`WearableSeed`); see the cts-identity-governance skill
for the authority and sensor-attested label rules this encodes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.domain import (
    FaceAnchor,
    GallerySeedSource,
    OrientationBin,
    WearableAttestationGrade,
    WearableEvidence,
)


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
    # Minimum service-measured face image quality of the seeding face. 0.0 disables the gate;
    # above 0.0 a face without a measured quality fails closed.
    min_face_quality: float = 0.0
    max_per_identity_orientation: int = 10
    # A calibrated direct-face
    # confidence at or above this bar mints the candidate straight into
    # auto_verified instead of pending_review. Raw (uncalibrated) confidence
    # never qualifies, matching the ArcFace-authority fail-closed posture.
    auto_verify_min_confidence: float = 0.90
    # Provenance stamped on every created row. model_version reuses the live
    # Triton ReID model name (settings.yaml `triton.reid_model`) rather than a
    # hardcoded "v1" (the lesson: a wrong static version silently makes
    # cross-version compatibility partitioning meaningless).
    model_version: str = ""
    preprocessing_version: str = "v1"
    # Wearable-attested seeding. Off by default; startup refuses it unless the wearable channel
    # itself is live, because an auto_verified row votes at once.
    wearable_seeding_enabled: bool = False
    # At most one wearable seed per (identity, orientation) per interval, so a sedentary person
    # does not fill the gallery with near-duplicates of one pose.
    wearable_min_seed_interval_s: float = 120.0
    # Skip a wearable seed whose cosine to any same-day row of the identity and orientation is
    # above this, keeping the gallery diverse.
    wearable_max_similarity_to_recent: float = 0.92
    # Room-attested seeds are reviewed by hand; this caps them per identity per UTC day so the
    # ReID Review Queue stays reviewable.
    wearable_room_grade_max_per_day: int = 10
    # Working-set creation cap window in seconds, wired from resolver.gallery_vote_max_age_s in
    # app/main.py. When set, max_per_identity_orientation counts only rows seen within this window
    # so the gallery keeps learning today's outfit instead of locking after lifetime rows fill.
    # None disables the time bound (counts all non-rejected/non-expired rows).
    gallery_vote_max_age_s: float | None = None
    # True when wearable.mode == "enabled". Under "shadow", room-grade seeds still mint
    # pending_review rows for operator review while trajectory seeds skip with channel_not_live.
    wearable_channel_live: bool = True

    def __post_init__(self) -> None:
        if self.wearable_min_seed_interval_s < 0.0:
            raise ValueError("reid_candidates.wearable_min_seed_interval_s must be >= 0")
        if not 0.0 < self.wearable_max_similarity_to_recent <= 1.0:
            raise ValueError("reid_candidates.wearable_max_similarity_to_recent must be in (0, 1]")
        if self.wearable_room_grade_max_per_day < 0:
            raise ValueError("reid_candidates.wearable_room_grade_max_per_day must be >= 0")
        if self.gallery_vote_max_age_s is not None and self.gallery_vote_max_age_s <= 0.0:
            raise ValueError("gallery_vote_max_age_s must be positive or None")


@dataclass(frozen=True)
class FaceSeed:
    """A recognized face on the detection, proposing its own identity."""

    anchor: FaceAnchor


@dataclass(frozen=True)
class WearableSeed:
    """An attested wearable association on the detection's PH, proposing the tag's member.

    ``detection_faces`` are every face the face stage produced for this detection (live and
    shadow-only), so the conflict gate sees a recognition that never reached live evidence.
    """

    evidence: WearableEvidence
    detection_faces: tuple[FaceAnchor, ...] = ()


Seed = FaceSeed | WearableSeed


@dataclass(frozen=True)
class CandidateEligibility:
    """Result of :func:`evaluate_candidate`. ``reason`` is ``""`` iff eligible.

    ``mint_state`` and ``seed_source`` are only meaningful when ``eligible`` is True:
    ``mint_state`` is the ``reid_gallery`` lifecycle state the candidate is created in
    (``"pending_review"`` or ``"auto_verified"``), ``seed_source`` its provenance.
    """

    eligible: bool
    reason: str
    mint_state: str = "pending_review"
    seed_source: GallerySeedSource = GallerySeedSource.FACE


def _ineligible(reason: str) -> CandidateEligibility:
    return CandidateEligibility(eligible=False, reason=reason)


def evaluate_candidate(
    *,
    committed_identity_id: str | None,
    seed: Seed | None,
    embedding: Sequence[float] | None,
    quality: float,
    orientation: OrientationBin,
    orientation_confidence: float,
    cfg: CandidatePolicy,
) -> CandidateEligibility:
    """Evaluate whether one detection may seed a governed gallery candidate.

    Every gate is fail-closed: an ineligible detection produces nothing, never a degraded or
    unlabeled row. The identity-equality gate (``identity_mismatch``) is unconditional on both
    paths: neither a face nor a tag naming a different person than the PH's committed identity
    may seed under the held label.
    """
    if isinstance(seed, WearableSeed):
        return _evaluate_wearable_seed(
            committed_identity_id=committed_identity_id,
            seed=seed,
            embedding=embedding,
            quality=quality,
            orientation=orientation,
            orientation_confidence=orientation_confidence,
            cfg=cfg,
        )

    if not committed_identity_id:
        return _ineligible("no_identity")

    if seed is None:
        return _ineligible("no_face_anchor")
    face_anchor = seed.anchor

    if face_anchor.recognition_state != "recognized":
        return _ineligible("face_not_recognized")

    if face_anchor.person_id != committed_identity_id:
        return _ineligible("identity_mismatch")

    if cfg.require_calibrated_face:
        calibrated = face_anchor.calibrated_confidence
        if calibrated is None or calibrated < cfg.calibrated_confidence_min:
            return _ineligible("calibration_not_authoritative")

    if cfg.min_face_quality > 0.0 and (
        face_anchor.face_quality is None or face_anchor.face_quality < cfg.min_face_quality
    ):
        return _ineligible("face_quality_too_low")

    shared = _shared_gates(embedding, quality, orientation, orientation_confidence, cfg)
    if shared is not None:
        return shared

    # Auto-verify mint rule: only a *calibrated* confidence at or
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


def _evaluate_wearable_seed(
    *,
    committed_identity_id: str | None,
    seed: WearableSeed,
    embedding: Sequence[float] | None,
    quality: float,
    orientation: OrientationBin,
    orientation_confidence: float,
    cfg: CandidatePolicy,
) -> CandidateEligibility:
    """Gate a wearable-attested seed. The grade decides the mint state, never the operator.

    Trajectory grade mints ``auto_verified`` (the wearable placed the person on this track)
    only when the wearable channel is live (``wearable_channel_live=True``); in shadow mode
    trajectory seeds skip with ``channel_not_live``. Room grade mints ``pending_review`` and
    votes only after an operator approves it, so it is accepted in shadow even when the PH has
    no live committed identity yet (still refusing any mismatch against a committed identity).
    """
    evidence = seed.evidence
    if not cfg.wearable_seeding_enabled:
        return _ineligible("wearable_seeding_disabled")

    if not evidence.attested:
        return _ineligible("wearable_not_attested")

    is_trajectory = evidence.attestation_grade == WearableAttestationGrade.TRAJECTORY
    if is_trajectory and not cfg.wearable_channel_live:
        return _ineligible("channel_not_live")

    if committed_identity_id is None:
        if cfg.wearable_channel_live or is_trajectory:
            return _ineligible("no_identity")
        target_identity_id = evidence.identity_id
    else:
        if evidence.identity_id != committed_identity_id:
            return _ineligible("identity_mismatch")
        target_identity_id = committed_identity_id

    # Stricter than the face path: the wearable cannot see the face, so any recognition of
    # someone else on this detection, at any similarity, vetoes the seed.
    if any(
        face.recognition_state == "recognized"
        and face.person_id is not None
        and face.person_id != target_identity_id
        for face in seed.detection_faces
    ):
        return _ineligible("face_conflict")

    shared = _shared_gates(embedding, quality, orientation, orientation_confidence, cfg)
    if shared is not None:
        return shared

    if is_trajectory:
        return CandidateEligibility(
            eligible=True,
            reason="",
            mint_state="auto_verified",
            seed_source=GallerySeedSource.WEARABLE_ATTESTED,
        )
    return CandidateEligibility(
        eligible=True,
        reason="",
        mint_state="pending_review",
        seed_source=GallerySeedSource.WEARABLE_ROOM_ATTESTED,
    )


def _shared_gates(
    embedding: Sequence[float] | None,
    quality: float,
    orientation: OrientationBin,
    orientation_confidence: float,
    cfg: CandidatePolicy,
) -> CandidateEligibility | None:
    """Orientation, embedding, and crop-quality gates every seed path applies, in this order."""
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
    return None
