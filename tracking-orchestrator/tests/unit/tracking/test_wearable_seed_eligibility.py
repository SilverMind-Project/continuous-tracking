"""Wearable-attested gallery seeds: the pure gate and the scorer's trust for seeded rows.

A wearable seed needs an attested association naming the committed identity, no recognized
face of anyone else on the detection at any similarity, and the same orientation, embedding,
and quality gates as a face seed. The attestation grade alone decides the mint state.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.domain import (
    FaceAnchor,
    GalleryEmbedding,
    GallerySeedSource,
    OrientationBin,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.tracking.identity.candidate_eligibility import (
    CandidateEligibility,
    CandidatePolicy,
    FaceSeed,
    WearableSeed,
    evaluate_candidate,
)
from app.tracking.identity.gallery_scoring import GalleryScoringConfig, score_hits

_T0 = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
_EMB = [1.0, 0.0, 0.0, 0.0]
_ON = CandidatePolicy(wearable_seeding_enabled=True)


def _evidence(
    *,
    identity_id: str = "member-a",
    grade: WearableAttestationGrade = WearableAttestationGrade.TRAJECTORY,
) -> WearableEvidence:
    return WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-1",
        identity_id=identity_id,
        evidence_class=WearableEvidenceClass.TRAJECTORY,
        attestation_grade=grade,
        window_llr=20.0,
        margin_llr=8.0,
        matched_fixes=10,
        mean_residual_m=0.2,
        newest_fix_at=_T0,
        segment_id="11111111-1111-1111-1111-111111111111",
    )


def _face(person_id: str, *, state: str = "recognized", similarity: float = 0.31) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        match_score=similarity,
        recognition_state=state,
        detection_id="d1",
    )


def _eval(
    seed: FaceSeed | WearableSeed | None,
    *,
    committed: str | None = "member-a",
    cfg: CandidatePolicy = _ON,
    orientation: OrientationBin = OrientationBin.FRONT,
    quality: float = 0.9,
) -> CandidateEligibility:
    return evaluate_candidate(
        committed_identity_id=committed,
        seed=seed,
        embedding=_EMB,
        quality=quality,
        orientation=orientation,
        orientation_confidence=0.9,
        cfg=cfg,
    )


def test_trajectory_grade_mints_auto_verified_as_wearable_attested() -> None:
    result = _eval(WearableSeed(evidence=_evidence()))
    assert result.eligible is True
    assert result.mint_state == "auto_verified"
    assert result.seed_source == GallerySeedSource.WEARABLE_ATTESTED


def test_room_grade_mints_pending_review_as_wearable_room_attested() -> None:
    result = _eval(WearableSeed(evidence=_evidence(grade=WearableAttestationGrade.ROOM)))
    assert result.eligible is True
    assert result.mint_state == "pending_review"
    assert result.seed_source == GallerySeedSource.WEARABLE_ROOM_ATTESTED


def test_unattested_evidence_never_seeds() -> None:
    result = _eval(WearableSeed(evidence=_evidence(grade=WearableAttestationGrade.NONE)))
    assert (result.eligible, result.reason) == (False, "wearable_not_attested")


def test_seeding_disabled_never_seeds() -> None:
    result = _eval(WearableSeed(evidence=_evidence()), cfg=CandidatePolicy())
    assert (result.eligible, result.reason) == (False, "wearable_seeding_disabled")


def test_no_committed_identity_is_checked_first() -> None:
    result = _eval(WearableSeed(evidence=_evidence()), committed=None)
    assert (result.eligible, result.reason) == (False, "no_identity")


def test_tag_naming_another_member_never_seeds_under_the_held_label() -> None:
    result = _eval(WearableSeed(evidence=_evidence(identity_id="member-b")))
    assert (result.eligible, result.reason) == (False, "identity_mismatch")


@pytest.mark.parametrize("similarity", [0.05, 0.31, 0.99])
def test_any_recognized_face_of_someone_else_vetoes_at_any_similarity(similarity: float) -> None:
    seed = WearableSeed(
        evidence=_evidence(),
        detection_faces=(_face("member-b", similarity=similarity),),
    )
    result = _eval(seed)
    assert (result.eligible, result.reason) == (False, "face_conflict")


def test_unrecognized_or_same_identity_faces_do_not_conflict() -> None:
    seed = WearableSeed(
        evidence=_evidence(),
        detection_faces=(
            _face("member-b", state="unrecognized"),
            _face("member-b", state="candidate"),
            _face("member-a", similarity=0.7),
        ),
    )
    assert _eval(seed).eligible is True


def test_shared_gates_apply_to_wearable_seeds() -> None:
    seed = WearableSeed(evidence=_evidence())
    assert _eval(seed, orientation=OrientationBin.UNKNOWN).reason == "unknown_orientation"
    assert _eval(seed, quality=0.1).reason == "low_quality"


def test_face_path_unchanged_without_a_seed() -> None:
    assert _eval(None, cfg=CandidatePolicy()).reason == "no_face_anchor"


def test_policy_rejects_out_of_range_wearable_knobs() -> None:
    with pytest.raises(ValueError, match="wearable_max_similarity_to_recent"):
        CandidatePolicy(wearable_max_similarity_to_recent=0.0)
    with pytest.raises(ValueError, match="wearable_min_seed_interval_s"):
        CandidatePolicy(wearable_min_seed_interval_s=-1.0)
    with pytest.raises(ValueError, match="wearable_room_grade_max_per_day"):
        CandidatePolicy(wearable_room_grade_max_per_day=-1)


def _row(state: str, seed_source: GallerySeedSource) -> GalleryEmbedding:
    return GalleryEmbedding(
        gallery_entry_id="e1",
        identity_id="member-a",
        embedding=_EMB,
        seen_at=_T0,
        state=state,
        seed_source=seed_source,
    )


@pytest.mark.parametrize(
    ("state", "seed_source", "expected"),
    [
        ("auto_verified", GallerySeedSource.FACE, 1.5),
        ("auto_verified", GallerySeedSource.WEARABLE_ATTESTED, 1.25),
        # An operator approval lifts a wearable row to full trust like any other row.
        ("operator_verified", GallerySeedSource.WEARABLE_ATTESTED, 2.0),
        ("operator_verified", GallerySeedSource.WEARABLE_ROOM_ATTESTED, 2.0),
    ],
)
def test_trust_multiplier_by_state_and_seed_source(
    state: str, seed_source: GallerySeedSource, expected: float
) -> None:
    (hit,) = score_hits(
        [(_row(state, seed_source), 0.8)],
        now=_T0,
        cfg=GalleryScoringConfig(),
        logistic=lambda s: s,
        on_nonvoting_state=lambda: None,
    )
    assert hit.trust_multiplier == pytest.approx(expected)


def test_wearable_trust_is_configurable() -> None:
    cfg = replace(GalleryScoringConfig(), wearable_trust_multiplier=1.1)
    (hit,) = score_hits(
        [(_row("auto_verified", GallerySeedSource.WEARABLE_ATTESTED), 0.8)],
        now=_T0,
        cfg=cfg,
        logistic=lambda s: s,
        on_nonvoting_state=lambda: None,
    )
    assert hit.trust_multiplier == pytest.approx(1.1)
