"""Focused public-API governance tests for EvidenceGrade and grade_of().

Verifies that CC_ASSERTION is excluded from ArcFace authority, the independent-
evidence clock, duplicate-active-identity ranking/bypass, and association/dedup
vectors through the public resolver and tracker APIs, and that unknown origin
or aggregation values fail loudly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.domain import (
    AnchorAggregation,
    AnchorOrigin,
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    Identity,
    PersonHypothesis,
    WorldObservation,
)
from app.storage.base import InMemoryGalleryRepository
from app.tracking.identity.evidence_grades import (
    CC_ASSERTION_FRAME,
    EVIDENCE_GRADES,
    NATIVE_FRAME,
    NATIVE_TEMPLATE,
    grade_of,
)
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.dedup import _faces_conflict
from app.tracking.world.tracker import _unpack_observations

_NOW = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)


def _ph(
    ph_id: str,
    *,
    identity_id: str | None = None,
    evidence_at: datetime | None = None,
) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(1.0, 2.0, 0.0, 0.0),
        state_cov=(0.1, 0.0, 0.0, 0.0) * 4,
        born_at=_NOW,
        last_seen_at=_NOW,
        last_seen_camera="cam-1",
        observation_count=3,
        current_identity_id=identity_id,
        last_independent_identity_evidence_at=evidence_at,
    )


async def _resolver() -> IdentityResolver:
    repo = InMemoryGalleryRepository()
    for ident_id, name in (("alice", "Alice"), ("bob", "Bob")):
        await repo.upsert_identity(
            Identity(identity_id=ident_id, display_name=name, enrolled_at=_NOW)
        )
    return IdentityResolver(
        gallery_repo=repo,
        config=ResolverConfig(arcface_authority_calibrated_confidence=0.80),
    )


def test_grade_registry_maps_all_registered_pairs_and_rejects_unknown() -> None:
    native = FaceAnchor(person_id="alice", match_score=0.9)
    template = FaceAnchor(
        person_id="alice",
        match_score=0.9,
        aggregation=AnchorAggregation.TEMPLATE,
    )
    cc = FaceAnchor(
        person_id="alice",
        match_score=0.9,
        origin=AnchorOrigin.CC_ASSERTION,
    )

    assert grade_of(native) is NATIVE_FRAME
    assert grade_of(template) is NATIVE_TEMPLATE
    assert grade_of(cc) is CC_ASSERTION_FRAME
    assert len(EVIDENCE_GRADES) == 3

    unknown_origin = FaceAnchor(person_id="alice", match_score=0.9, origin="mystery_vendor")
    with pytest.raises(KeyError, match="mystery_vendor"):
        grade_of(unknown_origin)

    unknown_agg = FaceAnchor(person_id="alice", match_score=0.9, aggregation="mystery_agg")
    with pytest.raises(KeyError, match="mystery_agg"):
        grade_of(unknown_agg)


@pytest.mark.asyncio
async def test_cc_assertion_never_grants_arcface_authority() -> None:
    """Even if a CC_ASSERTION anchor somehow carried high calibrated_confidence, it never
    grants direct_face authority."""
    resolver = await _resolver()
    ph = _ph("ph-1")
    anchor = FaceAnchor(
        person_id="alice",
        match_score=0.95,
        similarity=0.95,
        calibrated_confidence=0.99,
        detection_id="det-1",
        tracklet_id="ph-1",
        camera_id="cam-1",
        captured_at=_NOW,
        recognition_state="recognized",
        origin=AnchorOrigin.CC_ASSERTION,
    )

    outcome = await resolver.resolve([ph], [anchor], captured_at=_NOW)
    assert len(outcome.decisions) == 1
    assert outcome.decisions[0].authority != "direct_face"
    assert outcome.decisions[0].decision_source != "arcface_authority"


@pytest.mark.asyncio
async def test_cc_assertion_never_refreshes_evidence_clock() -> None:
    """A commit backed solely by CC_ASSERTION does not set evidence_backed or evidence_at."""
    resolver = await _resolver()
    ph = _ph("ph-1")
    anchor = FaceAnchor(
        person_id="alice",
        match_score=0.95,
        similarity=0.95,
        quality=0.9,
        detection_id="det-1",
        tracklet_id="ph-1",
        camera_id="cam-1",
        captured_at=_NOW,
        recognition_state="recognized",
        origin=AnchorOrigin.CC_ASSERTION,
    )

    outcome = await resolver.resolve([ph], [anchor], captured_at=_NOW)
    decision = outcome.decisions[0]
    assert decision.evidence_backed is False
    assert decision.evidence_at is None


@pytest.mark.asyncio
async def test_cc_assertion_never_ranks_in_duplicate_guard() -> None:
    """A CC_ASSERTION anchor reports 0.0 direct_face_confidence and never wins duplicate bypass."""
    resolver = await _resolver()
    ph_a = _ph("ph-a")
    ph_b = _ph("ph-b")
    cc_anchor = FaceAnchor(
        person_id="alice",
        match_score=0.95,
        similarity=0.95,
        quality=0.95,
        detection_id="det-a",
        tracklet_id="ph-a",
        camera_id="cam-1",
        captured_at=_NOW,
        recognition_state="recognized",
        origin=AnchorOrigin.CC_ASSERTION,
    )
    native_anchor = FaceAnchor(
        person_id="alice",
        match_score=0.82,
        similarity=0.82,
        quality=1.0,
        detection_id="det-b",
        tracklet_id="ph-b",
        camera_id="cam-1",
        captured_at=_NOW,
        recognition_state="recognized",
        origin=AnchorOrigin.NATIVE,
    )

    outcome = await resolver.resolve([ph_a, ph_b], [cc_anchor, native_anchor], captured_at=_NOW)
    by_ph = {d.ph_id: d for d in outcome.decisions}
    assert by_ph["ph-a"].evidence["direct_face_confidence"] == 0.0
    assert by_ph["ph-b"].evidence["direct_face_confidence"] == pytest.approx(0.82)
    assert by_ph["ph-b"].identity_id == "alice"
    assert by_ph["ph-a"].identity_id is None


def test_cc_assertion_and_template_excluded_from_association_and_dedup_conflict() -> None:
    """Only grades with enters_association=True enter association vectors and dedup conflict."""
    native_a = FaceAnchor(
        person_id="alice",
        match_score=0.85,
        detection_id="det-1",
        camera_id="cam-1",
        captured_at=_NOW,
        recognition_state="recognized",
        origin=AnchorOrigin.NATIVE,
        aggregation=AnchorAggregation.FRAME,
    )
    cc_b = FaceAnchor(
        person_id="bob",
        match_score=0.90,
        detection_id="det-2",
        camera_id="cam-2",
        captured_at=_NOW,
        recognition_state="recognized",
        origin=AnchorOrigin.CC_ASSERTION,
        aggregation=AnchorAggregation.FRAME,
    )
    template_b = FaceAnchor(
        person_id="bob",
        match_score=0.90,
        detection_id="det-3",
        camera_id="cam-2",
        captured_at=_NOW,
        recognition_state="recognized",
        origin=AnchorOrigin.NATIVE,
        aggregation=AnchorAggregation.TEMPLATE,
    )

    def _obs(det_id: str, fa: FaceAnchor) -> WorldObservation:
        return WorldObservation(
            camera_id=fa.camera_id,
            frame_index=1,
            captured_at=_NOW,
            floor_point=FloorPoint(x_mm=1000, y_mm=2000, calibrated=True),
            bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=200),
            embedding=[],
            detection_confidence=0.9,
            face_anchor=fa,
            detection_id=det_id,
        )

    vectors = _unpack_observations(
        [_obs("det-1", native_a), _obs("det-2", cc_b), _obs("det-3", template_b)]
    )
    assert vectors.face_person_ids == ["alice", None, None]
    assert vectors.face_confidences == [0.85, 0.0, 0.0]

    cfg = WorldTrackerConfig(face_conflict_min_similarity=0.70)
    assert _faces_conflict(native_a, cc_b, cfg) is False
    assert _faces_conflict(native_a, template_b, cfg) is False
