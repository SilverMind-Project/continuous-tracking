"""Calibrated face weighting and the calibrated-policy shadow in IdentityResolver.

Under the "calibrated" face evidence policy a recognized face weighs by its calibrated confidence;
a face without one weighs at most ``uncalibrated_face_max_likelihood`` and only corroborates:
it never qualifies as commit or clock evidence, so it cannot establish an identity alone. Under
"shadow" the live decision stays legacy while a calibrated decision is computed per PH with a
recognized face and its outcome counted; shadow-only anchors never reach a live decision.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    FaceAnchor,
    FaceEvidencePolicy,
    GalleryEmbedding,
    GlobalTrack,
    Identity,
    IdentityDecision,
    PosteriorDist,
)
from app.observability import metrics as metrics_pkg
from app.observability.metrics import Metrics, build_metrics
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity.commit_policy import collect_evidence_identity_ids
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

# The in-memory gallery's recency weighting reads the wall clock, so votes are seeded "now".
_NOW = datetime.now(UTC)
_TRACKLET = "t1"


@pytest.fixture(autouse=True)
def fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> Metrics:
    fresh = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh)
    return fresh


def _anchor(
    similarity: float,
    *,
    person_id: str = "alice",
    calibrated: float | None = None,
    state: str = "recognized",
    shadow_only: bool = False,
    yaw_deg: float = 0.0,
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        match_score=similarity,
        tracklet_id=_TRACKLET,
        detection_id=f"det-{person_id}-{similarity}",
        recognition_state=state,
        similarity=similarity,
        yaw_deg=yaw_deg,
        calibrated_confidence=calibrated,
        shadow_only=shadow_only,
    )


def _entity(
    identity_id: str | None = None, evidence_age_s: float | None = None, entity_id: str = "ph-1"
) -> GlobalTrack:
    evidence_at = None if evidence_age_s is None else _NOW - timedelta(seconds=evidence_age_s)
    return GlobalTrack(
        global_track_id=entity_id,
        camera_ids=["cam-a"],
        tracklet_ids=[_TRACKLET] if entity_id == "ph-1" else [f"{entity_id}-t"],
        started_at=_NOW - timedelta(minutes=5),
        last_seen_at=_NOW,
        current_identity_id=identity_id,
        current_identity_committed_at=None if identity_id is None else _NOW - timedelta(minutes=2),
        last_independent_identity_evidence_at=evidence_at,
    )


async def _resolver(
    policy: FaceEvidencePolicy,
    identities: tuple[str, ...] = ("alice", "bob"),
    gallery_vote_for: str | None = None,
) -> IdentityResolver:
    gallery = InMemoryGalleryRepository()
    for identity_id in identities:
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=_NOW)
        )
    if gallery_vote_for is not None:
        # The PH's own recent gallery entry, minted auto_verified for this identity: the query
        # that finds it is the PH's own appearance, so it votes for that identity.
        await gallery.upsert_gallery_entry(
            GalleryEmbedding(
                gallery_entry_id="g-1",
                identity_id=gallery_vote_for,
                embedding=[1.0] + [0.0] * 7,
                seen_at=_NOW,
                origin_tracklet_id=_TRACKLET,
                state="auto_verified",
            )
        )
    return IdentityResolver(
        gallery_repo=gallery, config=ResolverConfig(face_evidence_policy=policy)
    )


async def _decide(
    resolver: IdentityResolver,
    anchors: list[FaceAnchor],
    entity: GlobalTrack | None = None,
) -> IdentityDecision:
    outcome = await resolver.resolve(
        hypotheses=[entity or _entity()],
        new_face_anchors=anchors,
        captured_at=_NOW,
        ph_qualities={"ph-1": 0.9},
    )
    [decision] = outcome.decisions
    return decision


def _outcomes(metrics: Metrics) -> dict[str, float]:
    return {
        sample.labels["outcome"]: sample.value
        for family in metrics.cts_face_shadow_outcomes_total.collect()
        for sample in family.samples
        if sample.name.endswith("_total")
    }


def _calibrated_mismatches(metrics: Metrics) -> float:
    return sum(
        sample.value
        for family in metrics.identity_shadow_mismatch_total.collect()
        for sample in family.samples
        if sample.name.endswith("_total") and sample.labels.get("feature") == "calibrated_face"
    )


# ---------------------------------------------------------------------------
# The likelihood helper
# ---------------------------------------------------------------------------


async def test_calibrated_confidence_is_the_likelihood_with_no_quality_term() -> None:
    resolver = await _resolver("calibrated")
    anchor = replace(_anchor(0.62, calibrated=0.92), face_quality=0.2)

    for quality_mode in ("off", "enabled"):
        likelihood = resolver._recognized_face_likelihood(
            anchor, source="direct", face_quality_mode=quality_mode, calibrated=True
        )
        assert likelihood == pytest.approx(0.92)


async def test_frontality_still_scales_a_calibrated_face() -> None:
    resolver = await _resolver("calibrated")
    profile = _anchor(0.62, calibrated=0.92, yaw_deg=60.0)

    likelihood = resolver._recognized_face_likelihood(
        profile, source="direct", face_quality_mode="off", calibrated=True
    )

    assert likelihood == pytest.approx(0.92 * 0.3)


async def test_uncalibrated_face_is_capped_but_never_raised() -> None:
    resolver = await _resolver("calibrated")

    def weigh(anchor: FaceAnchor, calibrated: bool) -> float:
        return resolver._recognized_face_likelihood(
            anchor, source="direct", face_quality_mode="off", calibrated=calibrated
        )

    strong = _anchor(0.95)
    weak = _anchor(0.10)
    assert weigh(strong, calibrated=False) > 0.9
    assert weigh(strong, calibrated=True) == pytest.approx(0.55)
    assert weigh(weak, calibrated=False) < 0.55
    assert weigh(weak, calibrated=True) == pytest.approx(weigh(weak, calibrated=False))


async def test_propagated_and_external_multipliers_apply_under_calibrated_weighting() -> None:
    resolver = await _resolver("calibrated")
    anchor = _anchor(0.95, calibrated=0.9)

    direct = resolver._recognized_face_likelihood(
        anchor, source="direct", face_quality_mode="off", calibrated=True
    )
    propagated = resolver._recognized_face_likelihood(
        anchor, source="propagated", face_quality_mode="off", calibrated=True
    )

    assert propagated == pytest.approx(direct * 0.5)


# ---------------------------------------------------------------------------
# Live calibrated decisions
# ---------------------------------------------------------------------------


async def test_calibrated_face_commits_an_unknown_ph_with_direct_face_authority() -> None:
    resolver = await _resolver("calibrated")

    decision = await _decide(resolver, [_anchor(0.62, calibrated=0.92)])

    assert decision.identity_id == "alice"
    assert decision.authority == "direct_face"
    assert decision.evidence_backed


@pytest.mark.parametrize(
    "identities",
    [("alice",), ("alice", "bob"), ("alice", "bob", "carol")],
    ids=["one_resident", "two_residents", "three_residents"],
)
async def test_uncalibrated_face_alone_never_commits_an_unknown_ph(
    identities: tuple[str, ...],
) -> None:
    resolver = await _resolver("calibrated", identities=identities)

    decision = await _decide(resolver, [_anchor(0.47)])

    # The posterior favors alice (the face still shapes it) but nothing qualifies as evidence.
    top_id, _ = decision.posterior.top_identity()
    assert top_id == "alice"
    assert decision.identity_id is None


async def test_uncalibrated_face_tips_an_auto_verified_gallery_vote_into_a_commit() -> None:
    identities = ("alice", "bob", "carol")
    gallery_only = await _decide(
        await _resolver("calibrated", identities, gallery_vote_for="alice"), []
    )
    face_only = await _decide(await _resolver("calibrated", identities), [_anchor(0.47)])
    both = await _decide(
        await _resolver("calibrated", identities, gallery_vote_for="alice"), [_anchor(0.47)]
    )

    assert gallery_only.identity_id is None
    assert face_only.identity_id is None
    assert both.identity_id == "alice"
    assert both.authority == "posterior"


async def test_uncalibrated_face_remainder_goes_to_unknown_not_other_residents() -> None:
    resolver = await _resolver("calibrated", identities=("alice", "bob", "carol"))
    entity = _entity()
    await _decide(resolver, [], entity)  # loads the enrolled roster

    uncalibrated, _ = resolver._from_face_anchors(entity, [_anchor(0.47)], calibrated=True)
    calibrated, _ = resolver._from_face_anchors(
        entity, [_anchor(0.47, calibrated=0.7)], calibrated=True
    )

    assert set(uncalibrated.distribution) == {"alice", "UNKNOWN"}
    assert set(calibrated.distribution) == {"alice", "bob", "carol", "UNKNOWN"}


async def test_candidate_face_corroborates_only_under_calibrated() -> None:
    candidate = _anchor(0.35, state="candidate")

    legacy = await _decide(await _resolver("legacy", identities=("alice",)), [candidate])
    calibrated = await _decide(await _resolver("calibrated", identities=("alice",)), [candidate])

    # Legacy lets a grey-zone face commit a one-resident household on its own; calibrated does
    # not let any face without a calibrated confidence establish an identity.
    assert legacy.identity_id == "alice"
    assert calibrated.identity_id is None


async def test_uncalibrated_face_never_advances_the_evidence_clock() -> None:
    resolver = await _resolver("calibrated")
    holding_alice = _entity("alice", evidence_age_s=10.0)

    uncalibrated = await _decide(resolver, [_anchor(0.75)], holding_alice)
    calibrated = await _decide(resolver, [_anchor(0.75, calibrated=0.85)], holding_alice)

    assert uncalibrated.identity_id == "alice"
    assert not uncalibrated.evidence_backed
    assert calibrated.identity_id == "alice"
    assert calibrated.evidence_backed


def test_evidence_set_under_calibrated_faces_only() -> None:
    anchors = [
        _anchor(0.45, person_id="alice"),
        _anchor(0.35, person_id="bob", state="candidate"),
        _anchor(0.62, person_id="carol", calibrated=0.88),
    ]
    reid = PosteriorDist({"dave": 0.8, "UNKNOWN": 0.2})

    def ids(calibrated_faces_only: bool, recognized_only: bool = False) -> frozenset[str]:
        return collect_evidence_identity_ids(
            entity_obs_ids=frozenset({_TRACKLET}),
            entity_id="ph-1",
            face_anchors=anchors,
            face_evidence=[],
            reid_likelihood=reid,
            recognized_only=recognized_only,
            calibrated_faces_only=calibrated_faces_only,
        )

    assert ids(False) == {"alice", "bob", "carol", "dave"}
    assert ids(False, recognized_only=True) == {"alice", "carol", "dave"}
    assert ids(True) == ids(True, recognized_only=True) == {"carol", "dave"}


# ---------------------------------------------------------------------------
# Shadow: live is legacy, shadow-only anchors never reach a live decision
# ---------------------------------------------------------------------------


async def test_shadow_live_decisions_equal_legacy_with_shadow_only_anchors_present() -> None:
    live = [_anchor(0.75, person_id="bob")]
    shadow_only = [_anchor(0.55, calibrated=0.95, shadow_only=True)]
    for entity in (_entity(), _entity("bob", evidence_age_s=5.0)):
        legacy = await _decide(await _resolver("legacy"), live, entity)
        shadow = await _decide(await _resolver("shadow"), live + shadow_only, entity)
        for field in (
            "identity_id",
            "authority",
            "evidence_backed",
            "evidence",
            "decision_source",
            "inferred_identity_id",
        ):
            assert getattr(shadow, field) == getattr(legacy, field), field
        assert shadow.posterior.distribution == legacy.posterior.distribution


async def test_shadow_only_anchor_never_grants_authority(fresh_metrics: Metrics) -> None:
    resolver = await _resolver("shadow")

    decision = await _decide(resolver, [_anchor(0.55, calibrated=0.95, shadow_only=True)])

    assert decision.identity_id is None
    assert decision.authority != "direct_face"
    # The calibrated policy would have committed alice on that face.
    assert _outcomes(fresh_metrics) == {"would_commit_new": 1.0}


async def test_shadow_only_anchor_never_advances_the_evidence_clock() -> None:
    resolver = await _resolver("shadow")
    holding_alice = _entity("alice", evidence_age_s=10.0)

    decision = await _decide(
        resolver, [_anchor(0.55, calibrated=0.95, shadow_only=True)], holding_alice
    )

    assert decision.identity_id == "alice"  # held by the open maintenance window
    assert not decision.evidence_backed


async def test_shadow_only_anchor_never_enters_duplicate_ranking() -> None:
    resolver = await _resolver("shadow")

    decision = await _decide(resolver, [_anchor(0.55, calibrated=0.95, shadow_only=True)])

    assert decision.evidence["direct_face_confidence"] == 0.0


async def test_shadow_counts_each_outcome(fresh_metrics: Metrics) -> None:
    resolver = await _resolver("shadow")

    # Both policies commit alice on a calibrated face.
    await _decide(resolver, [_anchor(0.75, calibrated=0.9)])
    # Legacy commits alice on an uncalibrated 0.75 face; calibrated leaves the PH Unknown.
    await _decide(resolver, [_anchor(0.75)])
    # Legacy commits bob; a calibrated shadow-only face for alice takes authority instead.
    await _decide(
        resolver,
        [_anchor(0.75, person_id="bob"), _anchor(0.55, calibrated=0.9, shadow_only=True)],
    )

    assert _outcomes(fresh_metrics) == {
        "agrees": 1.0,
        "would_become_unknown": 1.0,
        "would_change_identity": 1.0,
    }
    assert _calibrated_mismatches(fresh_metrics) == 2.0


async def test_no_shadow_evaluation_outside_the_shadow_policy(fresh_metrics: Metrics) -> None:
    for policy in ("legacy", "calibrated"):
        await _decide(await _resolver(policy), [_anchor(0.75)])

    assert _outcomes(fresh_metrics) == {}


async def test_no_shadow_evaluation_without_a_recognized_face(fresh_metrics: Metrics) -> None:
    await _decide(await _resolver("shadow"), [_anchor(0.35, state="candidate")])

    assert _outcomes(fresh_metrics) == {}


def test_config_rejects_an_unknown_policy_and_an_out_of_range_cap() -> None:
    with pytest.raises(ValueError, match="face evidence policy"):
        ResolverConfig(face_evidence_policy="strict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="uncalibrated_face_max_likelihood"):
        ResolverConfig(uncalibrated_face_max_likelihood=1.5)
