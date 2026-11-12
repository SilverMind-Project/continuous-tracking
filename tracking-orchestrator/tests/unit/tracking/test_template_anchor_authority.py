"""Face template anchors in IdentityResolver.

A template is several faces of one PH matched together. It grants ArcFace authority only with a
template-profile calibrated confidence above the threshold and member faces that agree; it is
always weighed and gated as calibrated evidence, whatever the face evidence policy; it never
feeds the raw-threshold contradiction check, the duplicate-guard bypass, or propagation; and in
shadow mode it changes no live decision.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from prometheus_client import CollectorRegistry

from app.domain import FaceAnchor, FaceEvidencePolicy, GlobalTrack, Identity, IdentityDecision
from app.inference.evidence import FaceEvidence
from app.observability import metrics as metrics_pkg
from app.observability.metrics import Metrics, build_metrics
from app.storage.gallery import InMemoryGalleryRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig

_NOW = datetime.now(UTC)
_PH = "ph-1"


@pytest.fixture(autouse=True)
def fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> Metrics:
    fresh = build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_pkg, "metrics", fresh)
    return fresh


def _template(
    *,
    calibrated: float | None = 0.85,
    raw: float = 0.62,
    eligible: bool = True,
    shadow_only: bool = False,
    person_id: str = "alice",
) -> FaceAnchor:
    return FaceAnchor(
        person_id=person_id,
        match_score=raw,
        tracklet_id=_PH,
        detection_id="det-newest",
        recognition_state="recognized",
        similarity=raw,
        yaw_deg=10.0,
        calibrated_confidence=calibrated,
        face_quality=0.5,
        aggregation="template",
        authority_eligible=eligible,
        shadow_only=shadow_only,
    )


def _evidence(anchor: FaceAnchor) -> FaceEvidence:
    return FaceEvidence(
        person_id=anchor.person_id,
        match_score=anchor.match_score,
        tracklet_id=_PH,
        detection_id=anchor.detection_id,
        source="template",
        recognition_state="recognized",
        similarity=anchor.similarity,
        calibrated_confidence=anchor.calibrated_confidence,
        captured_at=_NOW,
    )


async def _decide(
    anchors: list[FaceAnchor],
    *,
    policy: FaceEvidencePolicy = "legacy",
    evidence: list[FaceEvidence] | None = None,
    held: str | None = None,
) -> IdentityDecision:
    gallery = InMemoryGalleryRepository()
    for identity_id in ("alice", "bob"):
        await gallery.upsert_identity(
            Identity(identity_id=identity_id, display_name=identity_id, enrolled_at=_NOW)
        )
    resolver = IdentityResolver(
        gallery_repo=gallery, config=ResolverConfig(face_evidence_policy=policy)
    )
    entity = GlobalTrack(
        global_track_id=_PH,
        camera_ids=["cam-1"],
        tracklet_ids=["t1"],
        started_at=_NOW,
        last_seen_at=_NOW,
        current_identity_id=held,
        last_independent_identity_evidence_at=_NOW if held else None,
    )
    outcome = await resolver.resolve(
        hypotheses=[entity],
        new_face_anchors=anchors,
        captured_at=_NOW,
        ph_qualities={_PH: 0.9},
        face_evidence=evidence,
    )
    [decision] = outcome.decisions
    return decision


def _count(metric: object, **labels: str) -> float:
    return sum(
        sample.value
        for family in metric.collect()  # type: ignore[attr-defined]
        for sample in family.samples
        if sample.name.endswith("_total")
        and all(sample.labels.get(k) == v for k, v in labels.items())
    )


async def test_consistent_calibrated_template_grants_direct_face_authority() -> None:
    template = _template(calibrated=0.85)
    decision = await _decide([template], evidence=[_evidence(template)])

    assert decision.identity_id == "alice"
    assert decision.authority == "direct_face"


async def test_a_disagreeing_member_blocks_authority_but_the_template_still_votes() -> None:
    template = _template(calibrated=0.85, eligible=False)
    decision = await _decide([template], evidence=[_evidence(template)])

    assert decision.authority != "direct_face"
    # It still carries its calibrated weight and counts as evidence through the posterior.
    assert decision.posterior.top_identity()[0] == "alice"
    assert decision.identity_id == "alice"
    assert decision.authority == "posterior"


@pytest.mark.parametrize("policy", ["legacy", "calibrated"])
async def test_an_uncalibrated_template_corroborates_only_under_every_policy(
    policy: FaceEvidencePolicy,
) -> None:
    template = _template(calibrated=None, raw=0.85)
    native = FaceAnchor(
        person_id="alice",
        match_score=0.85,
        tracklet_id=_PH,
        detection_id="det-native",
        recognition_state="recognized",
        similarity=0.85,
        calibrated_confidence=None,
    )

    with_template = await _decide([template], policy=policy, evidence=[_evidence(template)])
    with_native_face = await _decide([native], policy="legacy")

    # A raw 0.85 single face commits under legacy; the same raw score on a template does not,
    # because averaging inflates a template's similarity and only calibration makes it evidence.
    assert with_native_face.identity_id == "alice"
    assert with_template.identity_id is None
    assert with_template.posterior.top_identity()[0] == "alice"


async def test_a_template_never_feeds_the_duplicate_guard_bypass() -> None:
    template = _template(calibrated=0.95, raw=0.97)
    decision = await _decide([template], evidence=[_evidence(template)])

    assert decision.identity_id == "alice"
    assert decision.evidence["direct_face_confidence"] == 0.0


async def test_a_template_reports_no_raw_score_to_the_contradiction_check() -> None:
    gallery = InMemoryGalleryRepository()
    resolver = IdentityResolver(gallery_repo=gallery)
    entity = GlobalTrack(
        global_track_id=_PH,
        camera_ids=["cam-1"],
        tracklet_ids=["t1"],
        started_at=_NOW,
        last_seen_at=_NOW,
    )

    likelihood, best_raw = resolver._from_face_anchors(entity, [_template(raw=0.95)])

    assert likelihood.top_identity()[0] == "alice"
    assert best_raw is None


async def test_a_template_does_not_propagate_to_neighbouring_phs() -> None:
    resolver = IdentityResolver(gallery_repo=InMemoryGalleryRepository())
    entities = [
        GlobalTrack(
            global_track_id=pid,
            camera_ids=["cam-1"],
            tracklet_ids=[f"{pid}-t"],
            started_at=_NOW,
            last_seen_at=_NOW,
        )
        for pid in (_PH, "ph-2")
    ]
    anchors = [_template(calibrated=0.95, raw=0.97)]

    assert await resolver._propagate_face_anchors(entities, anchors) == anchors


async def test_the_ledger_records_the_template_as_its_own_source() -> None:
    template = _template(calibrated=0.85)
    decision = await _decide([template], evidence=[_evidence(template)])
    assert decision.evidence["sources"].get("face_template") == 1


async def test_a_shadow_template_changes_no_live_decision_and_is_counted(
    fresh_metrics: Metrics,
) -> None:
    shadow = _template(calibrated=0.9, shadow_only=True)

    live_without = await _decide([])
    with_shadow = await _decide([shadow])

    assert with_shadow.identity_id == live_without.identity_id
    assert with_shadow.identity_id is None
    assert with_shadow.posterior.distribution == live_without.posterior.distribution
    assert (
        _count(fresh_metrics.cts_face_template_shadow_outcomes_total, outcome="would_commit_new")
        == 1
    )
    assert (
        _count(fresh_metrics.cts_face_shadow_outcomes_total) == 0
    )  # the calibrated shadow is separate
