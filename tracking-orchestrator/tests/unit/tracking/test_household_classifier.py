"""Unit and integration tests for HouseholdClassifierEvaluator, channel, and Part E arm."""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from app.domain import (
    AppearanceFeature,
    AppearanceLabelSource,
    AppearanceTrainingSample,
    HouseholdClassifierHealthState,
    Identity,
    OrientationBin,
    ViewPrototype,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import build_metrics
from app.services.identity_roster_sync import IdentityRosterSyncService
from app.services.image_only_evaluator import ImageOnlyEvaluationService
from app.services.wearable_revocation import WearableTeachingRevocationService
from app.storage.appearance_samples import InMemoryAppearanceSampleRepository
from app.storage.base import InMemoryGalleryRepository
from app.storage.image_only_evaluation import InMemoryImageOnlyEvaluationRepository
from app.storage.wearable_associations import InMemoryWearableAssociationRepository
from app.storage.wearable_teaching_holds import InMemoryWearableTeachingHoldRepository
from app.tracking.identity.channels import ResolverInputs
from app.tracking.identity.evidence_grades import HOUSEHOLD_CLASSIFIER, EvidenceGradeName
from app.tracking.identity.household_classifier import (
    HouseholdClassifierArtifact,
    HouseholdClassifierArtifactMetrics,
    HouseholdClassifierChannel,
    HouseholdClassifierConfig,
    HouseholdClassifierEvaluator,
    SoliderFeatureExtractor,
)
from app.tracking.identity.types import IdentityAuthority
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.wearable_config import WearableConfig

T0 = datetime(2026, 11, 21, 12, 0, 0, tzinfo=UTC)
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


class _Entity:
    def __init__(self, entity_id: str, *, current_identity_id: str | None = None) -> None:
        self.entity_id = entity_id
        self.observation_ids = [f"obs-{entity_id}"]
        self.camera_ids = ["cam-1"]
        self.current_identity_id = current_identity_id
        self.current_identity_committed_at = None
        self.last_independent_identity_evidence_at = None
        self.last_seen_at = T0
        self.started_at = T0 - timedelta(minutes=2)
        self.view_prototypes = ()


def _artifact_dict(
    *,
    feature_version: str = "reid-solider",
    weights: list[list[float]] | None = None,
    bias: list[float] | None = None,
    temperature: float = 1.0,
    classes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_id": "hc-v1-test",
        "feature_version": feature_version,
        "classes": classes if classes is not None else ["alice", "bob"],
        "weights": weights if weights is not None else [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]],
        "bias": bias if bias is not None else [0.0, 0.0],
        "temperature": temperature,
        "trained_days": ["2026-11-01", "2026-11-02", "2026-11-03"],
        "eval_days": ["2026-11-04", "2026-11-05", "2026-11-06"],
        "metrics": {
            "macro_top1_at_0_8": 0.94,
            "per_class_recall_at_0_8": {"alice": 0.92, "bob": 0.95},
            "ece": 0.02,
            "open_set_false_match_rate_at_0_8": 0.01,
            "selected_c": 1.0,
            "cv_log_loss": 0.12,
            "eval_positive_samples": 200,
            "eval_negative_samples": 100,
        },
        "tool_version": "household-classifier-trainer-v1",
        "seed": 42,
        "created_at": "2026-11-21T00:00:00+00:00",
    }


def _ready_evaluator(*, temperature: float = 1.0) -> HouseholdClassifierEvaluator:
    art = HouseholdClassifierArtifact.model_validate(_artifact_dict(temperature=temperature))
    ev = HouseholdClassifierEvaluator.from_artifact(
        art, expected_feature_version="reid-solider", expected_embedding_dim=4
    )
    assert ev.is_ready
    return ev


@pytest.fixture(autouse=True)
def _fresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod, "metrics", build_metrics(CollectorRegistry()))


def test_loader_rejects_missing_incompatible_and_invalid_artifacts(tmp_path: Path) -> None:
    # Mode off -> OFF state.
    ev_off = HouseholdClassifierEvaluator.from_artifact_path(
        "", expected_feature_version="reid-solider", mode="off", expected_embedding_dim=4
    )
    assert ev_off.health_state == HouseholdClassifierHealthState.OFF
    assert not ev_off.is_ready

    # Missing file -> DEGRADED_MISSING.
    ev_missing = HouseholdClassifierEvaluator.from_artifact_path(
        tmp_path / "does-not-exist.json",
        expected_feature_version="reid-solider",
        mode="shadow",
        expected_embedding_dim=4,
    )
    assert ev_missing.health_state == HouseholdClassifierHealthState.DEGRADED_MISSING

    # Wrong feature_version -> DEGRADED_INCOMPATIBLE.
    bad_ver = tmp_path / "bad_ver.json"
    bad_ver.write_text(json.dumps(_artifact_dict(feature_version="dinov2-v1")), encoding="utf-8")
    ev_incompat = HouseholdClassifierEvaluator.from_artifact_path(
        bad_ver,
        expected_feature_version="reid-solider",
        mode="shadow",
        expected_embedding_dim=4,
    )
    assert ev_incompat.health_state == HouseholdClassifierHealthState.DEGRADED_INCOMPATIBLE

    # Wrong weight shape -> DEGRADED_INVALID.
    bad_shape = tmp_path / "bad_shape.json"
    bad_shape.write_text(
        json.dumps(_artifact_dict(weights=[[1.0, 0.0], [0.0, 1.0]])), encoding="utf-8"
    )
    ev_shape = HouseholdClassifierEvaluator.from_artifact_path(
        bad_shape,
        expected_feature_version="reid-solider",
        mode="shadow",
        expected_embedding_dim=4,
    )
    assert ev_shape.health_state == HouseholdClassifierHealthState.DEGRADED_INVALID

    # Non-finite weight (built directly via model_construct to bypass JSON strictness) -> INVALID.
    art_nan = HouseholdClassifierArtifact.model_construct(
        schema_version=1,
        artifact_id="nan",
        feature_version="reid-solider",
        classes=["alice", "bob"],
        weights=[[float("nan"), 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        bias=[0.0, 0.0],
        temperature=1.0,
        trained_days=["2026-11-01"],
        eval_days=["2026-11-02"],
        metrics=HouseholdClassifierArtifactMetrics(
            macro_top1_at_0_8=0.9,
            per_class_recall_at_0_8={"alice": 0.9, "bob": 0.9},
            ece=0.01,
            open_set_false_match_rate_at_0_8=0.0,
            selected_c=1.0,
            cv_log_loss=0.1,
            eval_positive_samples=10,
            eval_negative_samples=10,
        ),
        tool_version="v1",
        seed=1,
    )
    ev_nan = HouseholdClassifierEvaluator.from_artifact(
        art_nan, expected_feature_version="reid-solider", expected_embedding_dim=4
    )
    assert ev_nan.health_state == HouseholdClassifierHealthState.DEGRADED_INVALID

    # Valid file -> READY, gauge updated, health_summary leaks no identity IDs.
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_artifact_dict()), encoding="utf-8")
    ev_ready = HouseholdClassifierEvaluator.from_artifact_path(
        good,
        expected_feature_version="reid-solider",
        mode="shadow",
        expected_embedding_dim=4,
    )
    assert ev_ready.health_state == HouseholdClassifierHealthState.READY
    assert (
        metrics_mod.metrics.cts_household_classifier_health.labels(state="ready")._value.get()
        == 1.0
    )
    assert (
        metrics_mod.metrics.cts_household_classifier_health.labels(
            state="degraded_missing"
        )._value.get()
        == 0.0
    )
    summary = ev_ready.health_summary()
    assert summary["status"] == "ok"
    assert summary["class_count"] == 2
    assert "alice" not in json.dumps(summary)


def test_softmax_temperature_weight_multiplier_unknown_floor_and_grade() -> None:
    assert HOUSEHOLD_CLASSIFIER.name == EvidenceGradeName.HOUSEHOLD_CLASSIFIER
    assert not HOUSEHOLD_CLASSIFIER.grants_authority
    assert not HOUSEHOLD_CLASSIFIER.refreshes_evidence_clock
    assert not HOUSEHOLD_CLASSIFIER.ranks_in_duplicate_guard
    assert not HOUSEHOLD_CLASSIFIER.enters_association
    assert HOUSEHOLD_CLASSIFIER.counts_as_commit_evidence

    ev_t2 = _ready_evaluator(temperature=2.0)
    # Embedding [1, 0, 0, 0] -> logits [4, 0] -> / T=2 -> [2, 0] -> softmax = [e^2/(e^2+1), ...]
    expected_p_alice = math.exp(2.0) / (math.exp(2.0) + 1.0)
    pred = ev_t2.predict_proba([1.0, 0.0, 0.0, 0.0], weight_multiplier=1.0)
    assert pred is not None
    cal_probs, _ = pred
    assert cal_probs["alice"] == pytest.approx(expected_p_alice)

    # UNKNOWN floor = max(0.2, 1 - max_prob); weight_multiplier scales pre-softmax logits.
    ch_w1 = HouseholdClassifierChannel(
        config=HouseholdClassifierConfig(mode="enabled", weight_multiplier=1.0, unknown_floor=0.2),
        evaluator=ev_t2,
    )
    ch_w2 = HouseholdClassifierChannel(
        config=HouseholdClassifierConfig(mode="enabled", weight_multiplier=2.0, unknown_floor=0.2),
        evaluator=ev_t2,
    )
    inputs = ResolverInputs(
        captured_at=T0,
        face_anchors=(),
        face_evidence=(),
        ph_heights_m={},
        ph_qualities={},
        ph_appearance_features={
            "ph-1": AppearanceFeature(
                embedding=(1.0, 0.0, 0.0, 0.0),
                orientation=OrientationBin.FRONT,
                feature_version="reid-solider",
                quality=0.8,
            )
        },
    )
    res_w1 = ch_w1.evaluate_classifier(_Entity("ph-1"), inputs)
    res_w2 = ch_w2.evaluate_classifier(_Entity("ph-1"), inputs)
    assert res_w1.active and res_w2.active
    assert res_w1.likelihood.support_multiplier == 1.0
    assert res_w1.likelihood.evidence_at is None
    assert res_w1.likelihood.clock_evidence_ids == frozenset()
    assert res_w1.likelihood.distribution.distribution["UNKNOWN"] == pytest.approx(0.2)
    # Higher weight_multiplier sharpens alice vs bob (does not cancel on normalization).
    assert (
        res_w2.likelihood.distribution.distribution["alice"]
        > res_w1.likelihood.distribution.distribution["alice"]
    )
    # Uncertain embedding [0.707, 0.707, 0, 0] -> p_cal=[0.5, 0.5] -> UNKNOWN=0.5 > 0.2.
    uncertain_inputs = ResolverInputs(
        captured_at=T0,
        face_anchors=(),
        face_evidence=(),
        ph_heights_m={},
        ph_qualities={},
        ph_appearance_features={
            "ph-1": AppearanceFeature(
                embedding=(1.0, 1.0, 0.0, 0.0),
                orientation=OrientationBin.FRONT,
                feature_version="reid-solider",
                quality=0.8,
            )
        },
    )
    res_unc = ch_w1.evaluate_classifier(_Entity("ph-1"), uncertain_inputs)
    assert res_unc.likelihood.distribution.distribution["UNKNOWN"] == pytest.approx(0.5)
    assert res_unc.likelihood.commit_evidence_ids == frozenset()


def test_solider_feature_extractor_prefers_matching_orientation_prototype() -> None:
    extractor = SoliderFeatureExtractor(feature_version="reid-solider", embedding_dim=4)
    proto_front = ViewPrototype(
        orientation=OrientationBin.FRONT, embedding=(1.0, 0.0, 0.0, 0.0), count=5
    )
    proto_back = ViewPrototype(
        orientation=OrientationBin.BACK, embedding=(0.0, 1.0, 0.0, 0.0), count=2
    )
    feat = extractor.extract(
        view_prototypes=(proto_front, proto_back),
        current_orientation=OrientationBin.BACK,
        observation_embedding=[0.0, 0.0, 1.0, 0.0],
        gallery_mean=[0.0, 0.0, 0.0, 1.0],
        quality=0.8,
    )
    assert feat is not None
    assert feat.embedding == pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert feat.orientation == OrientationBin.BACK

    # Missing matching prototype falls back to current observation embedding.
    feat_obs = extractor.extract(
        view_prototypes=(proto_front,),
        current_orientation=OrientationBin.LEFT,
        observation_embedding=[0.0, 0.0, 3.0, 4.0],
        gallery_mean=[0.0, 0.0, 0.0, 1.0],
        quality=0.7,
    )
    assert feat_obs is not None
    assert feat_obs.embedding == pytest.approx((0.0, 0.0, 0.6, 0.8))


async def test_commit_evidence_threshold_and_never_refreshes_clock() -> None:
    gallery = InMemoryGalleryRepository()
    for uid in ("alice", "bob"):
        await gallery.upsert_identity(Identity(identity_id=uid, display_name=uid, enrolled_at=T0))
    evaluator = _ready_evaluator(temperature=1.0)
    resolver = IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(),
        household_classifier_config=HouseholdClassifierConfig(
            mode="enabled",
            commit_evidence_min_probability=0.8,
        ),
        household_classifier_evaluator=evaluator,
    )

    # Strong alice vector -> p_cal(alice) ~ 0.982 >= 0.8 -> commits via posterior, NO clock refresh.
    strong = await resolver.resolve(
        hypotheses=[_Entity("ph-1")],
        new_face_anchors=[],
        captured_at=T0,
        ph_appearance_features={
            "ph-1": AppearanceFeature(
                embedding=(1.0, 0.0, 0.0, 0.0),
                orientation=OrientationBin.FRONT,
                feature_version="reid-solider",
                quality=0.9,
            )
        },
    )
    d_strong = strong.decisions[0]
    assert d_strong.identity_id == "alice"
    assert d_strong.authority == str(IdentityAuthority.POSTERIOR)
    assert d_strong.evidence_at is None
    assert d_strong.evidence_backed is False

    # Weak alice vector ([0.6, 0.5, 0, 0] normalized -> prob < 0.8) -> stays Unknown.
    weak = await resolver.resolve(
        hypotheses=[_Entity("ph-2")],
        new_face_anchors=[],
        captured_at=T0,
        ph_appearance_features={
            "ph-2": AppearanceFeature(
                embedding=(0.6, 0.5, 0.0, 0.0),
                orientation=OrientationBin.FRONT,
                feature_version="reid-solider",
                quality=0.9,
            )
        },
    )
    assert weak.decisions[0].identity_id is None


async def test_shadow_mode_and_part_e_classifier_arm_accumulate_outcomes() -> None:
    gallery = InMemoryGalleryRepository()
    for uid in ("alice", "bob"):
        await gallery.upsert_identity(Identity(identity_id=uid, display_name=uid, enrolled_at=T0))
    evaluator = _ready_evaluator(temperature=1.0)
    resolver = IdentityResolver(
        gallery_repo=gallery,
        config=ResolverConfig(),
        wearable_config=WearableConfig(mode="shadow", evaluation_sample_rate=1.0),
        household_classifier_config=HouseholdClassifierConfig(mode="shadow"),
        household_classifier_evaluator=evaluator,
    )

    eval_repo = InMemoryImageOnlyEvaluationRepository()
    eval_svc = ImageOnlyEvaluationService(eval_repo, max_gap_s=30.0, flush_interval_s=3600.0)

    wearable_ev = WearableEvidence(
        ph_id="ph-1",
        tag_id="tag-a",
        identity_id="alice",
        evidence_class=WearableEvidenceClass.TRAJECTORY,
        attestation_grade=WearableAttestationGrade.TRAJECTORY,
        window_llr=18.0,
        margin_llr=10.0,
        matched_fixes=10,
        mean_residual_m=0.1,
        newest_fix_at=T0,
    )
    feat = {
        "ph-1": AppearanceFeature(
            embedding=(1.0, 0.0, 0.0, 0.0),
            orientation=OrientationBin.FRONT,
            feature_version="reid-solider",
            quality=0.9,
        )
    }

    for step_s in (0, 5):
        at = T0 + timedelta(seconds=step_s)
        out = await resolver.resolve(
            hypotheses=[_Entity("ph-1")],
            new_face_anchors=[],
            captured_at=at,
            wearable_evidence={"ph-1": wearable_ev},
            ph_appearance_features=feat,
        )
        # Shadow mode does NOT change the live Unknown decision.
        assert out.decisions[0].identity_id is None
        assert len(out.image_only) == 1
        ev_row = out.image_only[0]
        assert ev_row.control_identity_id is None
        assert ev_row.treatment_identity_id is None
        assert ev_row.classifier_evaluated is True
        assert ev_row.classifier_identity_id == "alice"
        await eval_svc.record_round(out.image_only, captured_at=at)

    assert (
        metrics_mod.metrics.cts_household_classifier_shadow_outcomes_total.labels(
            outcome="would_name_unknown"
        )._value.get()
        == 2.0
    )
    await eval_svc.flush()
    totals = {t.arm: t for t in await eval_repo.list_totals()}
    assert set(totals.keys()) == {"control", "treatment", "classifier"}
    assert totals["control"].unknown_s == pytest.approx(5.0)
    assert totals["treatment"].unknown_s == pytest.approx(5.0)
    assert totals["classifier"].correct_s == pytest.approx(5.0)

    # Verify scripts/image_only_evaluation_report.py summarizes classifier vs treatment.
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from image_only_evaluation_report import build_report, render_table

    report = build_report(list(totals.values()), samples=50, seed=0)
    assert len(report) == 1
    diff = report[0]["classifier_attribution_difference"]
    assert isinstance(diff, tuple)
    assert diff[0] == pytest.approx(1.0)
    table = render_table(report)
    assert "classifier attr" in table


async def test_revocation_and_roster_purge_clean_appearance_samples() -> None:
    gallery = InMemoryGalleryRepository()
    await gallery.upsert_identity(
        Identity(identity_id="alice", display_name="Alice", enrolled_at=T0)
    )
    assoc_repo = InMemoryWearableAssociationRepository()
    holds_repo = InMemoryWearableTeachingHoldRepository()
    sample_repo = InMemoryAppearanceSampleRepository()

    seg_id = "22222222-2222-4222-8222-222222222222"
    from app.domain import WearableAssociationSegment

    await assoc_repo.save_segment(
        WearableAssociationSegment(
            segment_id=seg_id,
            ph_id="ph-1",
            tag_id="tag-1",
            identity_id="alice",
            started_at=T0 - timedelta(minutes=2),
            ended_at=T0,
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            window_llr=18.0,
            margin_llr=10.0,
            matched_fixes=10,
            mean_residual_m=0.1,
        )
    )
    await sample_repo.append_many(
        [
            AppearanceTrainingSample(
                sample_id="s-wear",
                observed_at=T0,
                day=T0.date(),
                identity_id="alice",
                label_source=AppearanceLabelSource.WEARABLE_ATTESTED,
                camera_id="cam-1",
                orientation=OrientationBin.FRONT,
                quality=0.8,
                feature_version="reid-solider",
                embedding=(1.0, 0.0, 0.0, 0.0),
                seed_association_id=seg_id,
            ),
            AppearanceTrainingSample(
                sample_id="s-face",
                observed_at=T0 + timedelta(seconds=1),
                day=T0.date(),
                identity_id="alice",
                label_source=AppearanceLabelSource.DIRECT_FACE,
                camera_id="cam-1",
                orientation=OrientationBin.FRONT,
                quality=0.8,
                feature_version="reid-solider",
                embedding=(1.0, 0.0, 0.0, 0.0),
            ),
        ]
    )

    revoker = WearableTeachingRevocationService(
        association_repo=assoc_repo,
        teaching_holds=holds_repo,
        gallery_repo=gallery,
        appearance_sample_repo=sample_repo,
    )
    await revoker.revoke(seg_id, actor="admin", reason="operator")
    active_after_revoke = await sample_repo.list_samples()
    assert [s.sample_id for s in active_after_revoke] == ["s-face"]

    sync = IdentityRosterSyncService(
        gallery_repo=gallery,
        appearance_sample_repo=sample_repo,
    )
    await sync.purge_identity("alice")
    assert await sample_repo.list_samples(include_revoked=True) == []
