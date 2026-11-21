"""Image-only evaluation: the resolver's two arms, the service's time accounting, and storage.

The falsifiable claim is that wearable-taught gallery rows let the cameras name a person the
untaught cameras cannot. These tests pin the measurement itself: the control arm never sees a
wearable-taught row, the treatment arm never sees a row from the current stretch, the wearable
channel enters neither, and time is counted from capture-time gaps.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    GallerySeedSource,
    Identity,
    ImageOnlyDailyTotal,
    ImageOnlyEvaluation,
    NewReviewCandidate,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import build_metrics
from app.services.image_only_evaluator import ImageOnlyEvaluationService, classify
from app.storage.base import InMemoryGalleryRepository
from app.storage.image_only_evaluation import InMemoryImageOnlyEvaluationRepository
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.wearable_config import WearableConfig

# The gallery vote-age cutoff runs on wall clock, so the scenario sits just before now.
T0 = datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=1)
_EMB = [1.0, 0.0, 0.0, 0.0]


class _Entity:
    def __init__(self, entity_id: str, *, current_identity_id: str | None = None) -> None:
        self.entity_id = entity_id
        self.observation_ids = [f"obs-{entity_id}"]
        self.camera_ids = ["cam-1"]
        self.current_identity_id = current_identity_id
        self.current_identity_committed_at = T0 - timedelta(seconds=5)
        self.last_independent_identity_evidence_at = T0 - timedelta(seconds=5)
        self.last_seen_at = T0
        self.started_at = T0 - timedelta(minutes=5)
        self.view_prototypes = ()


def _evidence(
    grade: WearableAttestationGrade = WearableAttestationGrade.TRAJECTORY,
) -> WearableEvidence:
    return WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-a",
        identity_id="alice",
        evidence_class=WearableEvidenceClass.TRAJECTORY,
        attestation_grade=grade,
        window_llr=18.0,
        margin_llr=12.0,
        matched_fixes=10,
        mean_residual_m=0.1,
        newest_fix_at=T0,
    )


def _row(
    candidate_id: str, *, tracklet: str, seen_at: datetime, source: GallerySeedSource
) -> NewReviewCandidate:
    return NewReviewCandidate(
        candidate_id=candidate_id,
        identity_id="alice",
        embedding=_EMB,
        quality=0.9,
        orientation=1,
        camera_id="cam-1",
        capture_time=seen_at,
        ph_id=tracklet,
        observation_id=tracklet,
        origin_tracklet_id=tracklet,
        keyframe_id=None,
        crop_key=None,
        source_frame_key=None,
        crop_hash=None,
        frame_hash=None,
        dimensions=(64, 128),
        is_truncated=False,
        is_occluded=False,
        candidate_reason=str(source),
        source_episode_id=candidate_id,
        created_actor="pipeline",
        model_version="reid",
        preprocessing_version="v1",
        confidence=None,
        state="auto_verified",
        seed_source=source,
    )


async def _gallery(*, earlier_source: GallerySeedSource) -> InMemoryGalleryRepository:
    """Alice was taught earlier today (``earlier_source``); this PH's own row is from now.

    One enrolled resident, as in this household: a single gallery vote clears the commit bar
    only when nothing competes for the posterior mass.
    """
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(Identity(identity_id="alice", display_name="a", enrolled_at=T0))
    await gallery.create_review_candidate(
        _row(
            "earlier",
            tracklet="obs-earlier",
            seen_at=T0 - timedelta(hours=2),
            source=earlier_source,
        )
    )
    await gallery.create_review_candidate(
        _row(
            "this-stretch",
            tracklet="obs-ph-1",
            seen_at=T0 - timedelta(seconds=60),
            source=GallerySeedSource.WEARABLE_ATTESTED,
        )
    )
    return gallery


def _resolver(gallery: InMemoryGalleryRepository, *, mode: str, rate: float) -> IdentityResolver:
    return IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(),
        wearable_config=WearableConfig(mode=mode, evaluation_sample_rate=rate),  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod, "metrics", build_metrics(CollectorRegistry()))


async def test_treatment_uses_earlier_wearable_rows_and_control_cannot() -> None:
    gallery = await _gallery(earlier_source=GallerySeedSource.WEARABLE_ATTESTED)
    out = await _resolver(gallery, mode="shadow", rate=1.0).resolve(
        hypotheses=[_Entity("ph-1")],
        new_face_anchors=[],
        captured_at=T0,
        wearable_evidence={"ph-1": _evidence()},
    )
    (evaluation,) = out.image_only
    assert evaluation.truth_identity_id == "alice"
    assert evaluation.attestation_grade == WearableAttestationGrade.TRAJECTORY
    assert evaluation.treatment_identity_id == "alice"
    # Without wearable-taught rows this PH has no query and no votes: Unknown.
    assert evaluation.control_identity_id is None


async def test_face_taught_rows_count_in_both_arms() -> None:
    gallery = await _gallery(earlier_source=GallerySeedSource.FACE)
    # A face-seeded row of this PH gives the control arm a query embedding of its own.
    await gallery.create_review_candidate(
        _row(
            "this-ph-face",
            tracklet="obs-ph-1",
            seen_at=T0 - timedelta(seconds=30),
            source=GallerySeedSource.FACE,
        )
    )
    out = await _resolver(gallery, mode="shadow", rate=1.0).resolve(
        hypotheses=[_Entity("ph-1")],
        new_face_anchors=[],
        captured_at=T0,
        wearable_evidence={"ph-1": _evidence()},
    )
    (evaluation,) = out.image_only
    assert evaluation.control_identity_id == "alice"
    assert evaluation.treatment_identity_id == "alice"


async def test_rows_from_the_current_stretch_never_vote_for_themselves() -> None:
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(Identity(identity_id="alice", display_name="a", enrolled_at=T0))
    await gallery.create_review_candidate(
        _row(
            "this-stretch",
            tracklet="obs-ph-1",
            seen_at=T0 - timedelta(seconds=60),
            source=GallerySeedSource.WEARABLE_ATTESTED,
        )
    )
    out = await _resolver(gallery, mode="enabled", rate=1.0).resolve(
        hypotheses=[_Entity("ph-1")],
        new_face_anchors=[],
        captured_at=T0,
        wearable_evidence={"ph-1": _evidence()},
    )
    (evaluation,) = out.image_only
    assert evaluation.treatment_identity_id is None
    assert evaluation.control_identity_id is None


async def test_a_wearable_held_identity_does_not_leak_through_the_prior() -> None:
    """The live decision holds alice (from the wearable); both image-only arms still see none."""
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(Identity(identity_id="alice", display_name="a", enrolled_at=T0))
    out = await _resolver(gallery, mode="enabled", rate=1.0).resolve(
        hypotheses=[_Entity("ph-1", current_identity_id="alice")],
        new_face_anchors=[],
        captured_at=T0,
        wearable_evidence={"ph-1": _evidence()},
    )
    assert out.decisions[0].identity_id == "alice"
    (evaluation,) = out.image_only
    assert (evaluation.control_identity_id, evaluation.treatment_identity_id) == (None, None)


async def test_sampling_off_or_unattested_evaluates_nothing() -> None:
    gallery = await _gallery(earlier_source=GallerySeedSource.WEARABLE_ATTESTED)
    off = await _resolver(gallery, mode="shadow", rate=0.0).resolve(
        hypotheses=[_Entity("ph-1")],
        new_face_anchors=[],
        captured_at=T0,
        wearable_evidence={"ph-1": _evidence()},
    )
    assert off.image_only == []
    unattested = await _resolver(gallery, mode="shadow", rate=1.0).resolve(
        hypotheses=[_Entity("ph-1")],
        new_face_anchors=[],
        captured_at=T0,
        wearable_evidence={"ph-1": _evidence(WearableAttestationGrade.NONE)},
    )
    assert unattested.image_only == []


def _evaluation(
    at: datetime,
    *,
    control: str | None,
    treatment: str | None,
    ph_id: str = "ph-1",
    grade: WearableAttestationGrade = WearableAttestationGrade.TRAJECTORY,
) -> ImageOnlyEvaluation:
    return ImageOnlyEvaluation(
        ph_id=ph_id,
        truth_identity_id="alice",
        attestation_grade=grade,
        control_identity_id=control,
        treatment_identity_id=treatment,
        captured_at=at,
    )


def _seconds(arm: str, outcome: str, grade: str = "trajectory") -> float:
    counter = metrics_mod.metrics.cts_image_only_eval_seconds_total
    return float(counter.labels(arm=arm, outcome=outcome, grade=grade)._value.get())


def test_classify() -> None:
    assert classify("alice", "alice") == "correct"
    assert classify("bob", "alice") == "wrong"
    assert classify(None, "alice") == "unknown"


async def test_seconds_are_capped_capture_time_gaps_shared_by_both_arms() -> None:
    repo = InMemoryImageOnlyEvaluationRepository()
    service = ImageOnlyEvaluationService(repo, max_gap_s=30.0, flush_interval_s=3600.0)
    # The first evaluation of a PH opens an interval and counts nothing.
    await service.record_round([_evaluation(T0, control=None, treatment="alice")], captured_at=T0)
    t1 = T0 + timedelta(seconds=5)
    await service.record_round([_evaluation(t1, control=None, treatment="alice")], captured_at=t1)
    t2 = t1 + timedelta(seconds=4)
    await service.record_round([_evaluation(t2, control="bob", treatment="alice")], captured_at=t2)
    # A gap over max_gap_s counts nothing and restarts the interval.
    t3 = t2 + timedelta(seconds=45)
    await service.record_round([_evaluation(t3, control="bob", treatment="bob")], captured_at=t3)

    assert _seconds("control", "unknown") == pytest.approx(5.0)
    assert _seconds("control", "wrong") == pytest.approx(4.0)
    assert _seconds("treatment", "correct") == pytest.approx(9.0)
    assert _seconds("treatment", "wrong") == pytest.approx(0.0)

    await service.flush()
    totals = {t.arm: t for t in await repo.list_totals()}
    assert (totals["control"].correct_s, totals["control"].wrong_s) == (0.0, 4.0)
    assert totals["control"].unknown_s == pytest.approx(5.0)
    assert totals["treatment"].correct_s == pytest.approx(9.0)
    assert totals["treatment"].day == T0.date()


async def test_flushes_on_capture_time_and_purges_an_identity() -> None:
    repo = InMemoryImageOnlyEvaluationRepository()
    service = ImageOnlyEvaluationService(repo, max_gap_s=30.0, flush_interval_s=60.0)
    at = T0
    for _ in range(14):  # 13 gaps of 5 s
        await service.record_round(
            [_evaluation(at, control=None, treatment="alice")], captured_at=at
        )
        at += timedelta(seconds=5)
    stored = await repo.list_totals()
    assert {t.arm for t in stored} == {"control", "treatment"}
    assert sum(t.unknown_s for t in stored if t.arm == "control") == pytest.approx(60.0)

    removed = await service.purge_identity("alice")
    assert removed >= 2
    assert await repo.list_totals() == []
    await service.flush()
    assert await repo.list_totals() == []


class _FlakyRepo(InMemoryImageOnlyEvaluationRepository):
    """Fails the first write, then behaves."""

    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 1

    async def add_totals(self, totals: Sequence[ImageOnlyDailyTotal]) -> None:
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("database down")
        await super().add_totals(totals)


async def test_a_failed_flush_keeps_the_totals_for_the_next_one() -> None:
    repo = _FlakyRepo()
    service = ImageOnlyEvaluationService(repo, max_gap_s=30.0, flush_interval_s=3600.0)
    t1 = T0 + timedelta(seconds=2)
    await service.record_round([_evaluation(T0, control=None, treatment=None)], captured_at=T0)
    await service.record_round([_evaluation(t1, control=None, treatment=None)], captured_at=t1)
    await service.flush()  # logged, not raised
    assert await repo.list_totals() == []
    await service.flush()
    assert sum(t.unknown_s for t in await repo.list_totals()) == pytest.approx(4.0)


async def test_repository_adds_into_one_row_per_key() -> None:
    repo = InMemoryImageOnlyEvaluationRepository()
    total = ImageOnlyDailyTotal(
        day=T0.date(),
        identity_id="alice",
        arm="control",
        attestation_grade=WearableAttestationGrade.ROOM,
        correct_s=1.0,
        wrong_s=2.0,
        unknown_s=3.0,
    )
    await repo.add_totals([total, total])
    (row,) = await repo.list_totals(since=T0.date(), until=T0.date())
    assert (row.correct_s, row.wrong_s, row.unknown_s) == (2.0, 4.0, 6.0)
    assert await repo.list_totals(since=T0.date() + timedelta(days=1)) == []
