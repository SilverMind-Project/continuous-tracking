"""Unit tests for wearable fix ingest, buffer, WearableChannel, grade, and resolver rules."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from cts_contracts import FixKind
from prometheus_client import CollectorRegistry

from app.domain import (
    AnchorAggregation,
    AnchorOrigin,
    FaceAnchor,
    Identity,
    WearableAssociationSegment,
    WearableAssociationTransition,
    WearableAttestationGrade,
    WearableBinding,
    WearableEvidence,
    WearableEvidenceClass,
    WearableFix,
    WearableSiteConfig,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import (
    DeadLetterStream,
    WearableFixIngestResult,
    WearableFixKindLabel,
    WearableShadowOutcome,
    build_metrics,
)
from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.wearable_associations import WearableAssociationPersistStage
from app.proto.continuoustracking.v1 import wearable_pb2
from app.services.wearable_fix_buffer import WearableFixBuffer
from app.services.wearable_fix_subscriber import WearableFixSubscriber, decode_tag_fix
from app.services.wearable_site_config_map import WearableSiteConfigMap
from app.storage.base import InMemoryGalleryRepository
from app.storage.wearable_associations import InMemoryWearableAssociationRepository
from app.tracking.identity.evidence_grades import WEARABLE, EvidenceGradeName
from app.tracking.identity.wearable_channel import WearableChannel, wearable_evidence_probability
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.wearable_config import WearableConfig
from app.transport.redis_streams import FrameReady

T0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)


def _unix_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


class _SimpleEntity:
    def __init__(
        self,
        entity_id: str,
        *,
        current_identity_id: str | None = None,
        last_evidence_at: datetime | None = None,
    ) -> None:
        self.entity_id = entity_id
        self.observation_ids = [f"obs-{entity_id}"]
        self.camera_ids = ["cam-1"]
        self.current_identity_id = current_identity_id
        self.current_identity_committed_at = last_evidence_at
        self.last_independent_identity_evidence_at = last_evidence_at
        self.last_seen_at = T0
        self.started_at = T0 - timedelta(seconds=10)
        self.view_prototypes = ()


class TestWearableFixBufferAndSubscriber:
    def test_buffer_orders_by_measured_at_deduplicates_and_purges_identity(self) -> None:
        buf = WearableFixBuffer(max_age_s=30.0, max_per_tag=10)
        f2 = WearableFix(
            "tag-1", "alice", T0 + timedelta(seconds=2), FixKind.ROOM, room_id="r1", epoch_seq=2
        )
        f1 = WearableFix(
            "tag-1", "alice", T0 + timedelta(seconds=1), FixKind.ROOM, room_id="r1", epoch_seq=1
        )
        f3 = WearableFix(
            "tag-2", "bob", T0 + timedelta(seconds=3), FixKind.ROOM, room_id="r2", epoch_seq=1
        )
        buf.add(f2)
        buf.add(f1)
        buf.add(f1)  # duplicate ignored
        buf.add(f3)

        recent = buf.recent(now=T0 + timedelta(seconds=4), max_age_s=10.0)
        assert [f.measured_at for f in recent] == [
            T0 + timedelta(seconds=1),
            T0 + timedelta(seconds=2),
            T0 + timedelta(seconds=3),
        ]

        removed = buf.purge_identity("alice")
        assert removed == 2
        remaining = buf.recent(now=T0 + timedelta(seconds=4), max_age_s=10.0)
        assert [f.member_id for f in remaining] == ["bob"]

    def test_decode_enforces_presence_flags_off_body_stale_unbound_and_dead_letters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = CollectorRegistry()
        monkeypatch.setattr(metrics_mod, "metrics", build_metrics(reg))

        site = WearableSiteConfig(
            bindings=(
                WearableBinding(
                    tag_id="tag-1",
                    member_id="alice",
                    valid_from=T0 - timedelta(hours=1),
                ),
            )
        )

        # 1. Valid position fix with presence flags.
        msg = wearable_pb2.TagFix(
            tag_id="tag-1",
            measured_at_unix_ns=_unix_ns(T0),
            kind=wearable_pb2.FIX_KIND_POSITION,
            has_position=True,
            x_mm=1500,
            y_mm=2500,
            cov_xx_m2=0.04,
            cov_xy_m2=0.0,
            cov_yy_m2=0.04,
            wear_state=wearable_pb2.WEAR_STATE_ON_BODY,
            motion_state=wearable_pb2.MOTION_STATE_MOVING,
            # Unset presence flag with nonzero scalar must decode as None.
            has_quality=False,
            quality=0.99,
        )
        fix, kind, result = decode_tag_fix(
            msg.SerializeToString(),
            site_config=site,
            now=T0 + timedelta(seconds=2),
            max_fix_age_s=15.0,
        )
        assert result == WearableFixIngestResult.ACCEPTED
        assert kind == WearableFixKindLabel.POSITION
        assert fix is not None
        assert math.isclose(fix.x_m or 0.0, 1.5)
        assert fix.quality is None  # presence flag enforced!

        # 2. OFF_BODY dropped.
        msg_off = wearable_pb2.TagFix(
            tag_id="tag-1",
            measured_at_unix_ns=_unix_ns(T0),
            kind=wearable_pb2.FIX_KIND_ROOM,
            room_id="r1",
            wear_state=wearable_pb2.WEAR_STATE_OFF_BODY,
        )
        fix_off, _, res_off = decode_tag_fix(
            msg_off.SerializeToString(), site_config=site, now=T0, max_fix_age_s=15.0
        )
        assert fix_off is None
        assert res_off == WearableFixIngestResult.DROPPED_OFF_BODY

        # 3. Stale fix (> 15 s) dropped.
        fix_stale, _, res_stale = decode_tag_fix(
            msg.SerializeToString(),
            site_config=site,
            now=T0 + timedelta(seconds=16),
            max_fix_age_s=15.0,
        )
        assert fix_stale is None
        assert res_stale == WearableFixIngestResult.DROPPED_STALE

        # 4. Unbound tag dropped.
        msg_unbound = wearable_pb2.TagFix(
            tag_id="tag-unknown",
            measured_at_unix_ns=_unix_ns(T0),
            kind=wearable_pb2.FIX_KIND_ROOM,
            room_id="r1",
        )
        fix_u, _, res_u = decode_tag_fix(
            msg_unbound.SerializeToString(), site_config=site, now=T0, max_fix_age_s=15.0
        )
        assert fix_u is None
        assert res_u == WearableFixIngestResult.DROPPED_UNBOUND

        # 5. Malformed covariance / missing payload dead-lettered.
        site_map = WearableSiteConfigMap(site)
        buf = WearableFixBuffer()
        sub = WearableFixSubscriber(redis_client=None, buffer=buf, site_config_map=site_map)
        msg_bad = wearable_pb2.TagFix(
            tag_id="tag-1",
            measured_at_unix_ns=_unix_ns(T0),
            kind=wearable_pb2.FIX_KIND_POSITION,
            has_position=True,
            cov_xx_m2=-1.0,
            cov_yy_m2=0.1,
        )
        assert (
            sub.handle_fields({b"fix": msg_bad.SerializeToString()}, now=T0)
            == WearableFixIngestResult.DROPPED_MALFORMED
        )
        assert sub.handle_fields({}, now=T0) == WearableFixIngestResult.DROPPED_MALFORMED
        dlq = metrics_mod.metrics.cts_messages_dead_lettered_total.labels(
            stream=str(DeadLetterStream.WEARABLE_FIXES)
        )._value.get()
        assert dlq == 2.0


class TestWearableGradeChannelAndResolver:
    def test_wearable_grade_flags(self) -> None:
        assert WEARABLE.name == EvidenceGradeName.WEARABLE
        assert WEARABLE.grants_authority is False
        assert WEARABLE.refreshes_evidence_clock is True
        assert WEARABLE.ranks_in_duplicate_guard is True
        assert WEARABLE.grants_duplicate_bypass is False
        assert WEARABLE.enters_association is False
        assert WEARABLE.uses_own_capture_time is True

    def test_wearable_channel_caps_power_weighting_and_evidence_time(self) -> None:
        cfg_w2 = WearableConfig(mode="enabled", weight_multiplier=2.0, p_max=0.97, p_max_room=0.90)
        cfg_w1 = WearableConfig(mode="enabled", weight_multiplier=1.0, p_max=0.97, p_max_room=0.90)
        identities = {
            "alice": Identity(identity_id="alice", display_name="Alice", enrolled_at=T0),
            "bob": Identity(identity_id="bob", display_name="Bob", enrolled_at=T0),
        }
        ev_traj = WearableEvidence(
            ph_id="ph-1",
            tag_id="tag-a",
            identity_id="alice",
            evidence_class=WearableEvidenceClass.TRAJECTORY,
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            window_llr=40.0,
            margin_llr=20.0,
            matched_fixes=10,
            mean_residual_m=0.1,
            newest_fix_at=T0 - timedelta(seconds=2),
        )
        ev_room = WearableEvidence(
            ph_id="ph-1",
            tag_id="tag-a",
            identity_id="alice",
            evidence_class=WearableEvidenceClass.ROOM,
            attestation_grade=WearableAttestationGrade.NONE,
            window_llr=40.0,
            margin_llr=20.0,
            matched_fixes=10,
            mean_residual_m=None,
            newest_fix_at=T0 - timedelta(seconds=1),
        )
        assert math.isclose(wearable_evidence_probability(ev_traj, cfg_w2), 0.97)
        assert math.isclose(wearable_evidence_probability(ev_room, cfg_w2), 0.90)

        ch2 = WearableChannel(config=cfg_w2, identities_getter=lambda: identities)
        ch1 = WearableChannel(config=cfg_w1, identities_getter=lambda: identities)
        ent = _SimpleEntity("ph-1")
        lik2, p2 = ch2.evaluate_wearable(ent, ev_traj)
        lik1, _ = ch1.evaluate_wearable(ent, ev_traj)

        assert lik2.support_multiplier == 1.0
        assert lik2.evidence_at == ev_traj.newest_fix_at
        assert lik2.clock_evidence_ids == frozenset({"alice"})
        assert p2 == pytest.approx(0.97)
        # Power weighting (w=2.0) sharpens posterior mass over w=1.0.
        assert lik2.distribution.distribution["alice"] > lik1.distribution.distribution["alice"]

    @pytest.mark.asyncio
    async def test_resolver_wearable_commit_conflict_uncovered_face_and_shadow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = CollectorRegistry()
        monkeypatch.setattr(metrics_mod, "metrics", build_metrics(reg))

        gallery = InMemoryGalleryRepository()
        for uid in ("alice", "bob"):
            await gallery.upsert_identity(
                Identity(identity_id=uid, display_name=uid.title(), enrolled_at=T0)
            )

        ev_alice = WearableEvidence(
            ph_id="ph-1",
            tag_id="tag-a",
            identity_id="alice",
            evidence_class=WearableEvidenceClass.TRAJECTORY,
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            window_llr=18.0,
            margin_llr=12.0,
            matched_fixes=8,
            mean_residual_m=0.12,
            newest_fix_at=T0 - timedelta(seconds=1),
        )

        # 1. Enabled mode: wearable alone commits unknown PH and uses newest_fix_at.
        resolver = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(arcface_authority_calibrated_confidence=0.80),
            wearable_config=WearableConfig(mode="enabled"),
        )
        out = await resolver.resolve(
            hypotheses=[_SimpleEntity("ph-1")],
            new_face_anchors=[],
            captured_at=T0,
            wearable_evidence={"ph-1": ev_alice},
        )
        assert len(out.decisions) == 1
        dec = out.decisions[0]
        assert dec.identity_id == "alice"
        assert dec.decision_source == "wearable"
        assert dec.evidence_backed is True
        assert dec.evidence_at == ev_alice.newest_fix_at

        # 2. Calibrated direct face (bob) vs wearable (alice) forces Unknown + conflict.
        cal_face_bob = FaceAnchor(
            person_id="bob",
            match_score=0.85,
            calibrated_confidence=0.92,
            tracklet_id="ph-1",
            detection_id="obs-ph-1",
            origin=AnchorOrigin.NATIVE,
            aggregation=AnchorAggregation.FRAME,
            captured_at=T0,
        )
        out_conflict = await resolver.resolve(
            hypotheses=[
                _SimpleEntity(
                    "ph-1", current_identity_id="alice", last_evidence_at=T0 - timedelta(seconds=2)
                )
            ],
            new_face_anchors=[cal_face_bob],
            captured_at=T0,
            wearable_evidence={"ph-1": ev_alice},
        )
        dec_c = out_conflict.decisions[0]
        assert dec_c.identity_id is None
        assert dec_c.conflict == "wearable_face"
        assert dec_c.decision_source == "wearable_face_conflict"
        assert metrics_mod.metrics.cts_wearable_face_conflict_total._value.get() == 1.0

        # 3. Uncovered resident face (calibrated_confidence=None) does not trigger conflict.
        uncal_face_bob = FaceAnchor(
            person_id="bob",
            match_score=0.72,
            calibrated_confidence=None,
            tracklet_id="ph-1",
            detection_id="obs-ph-1",
            origin=AnchorOrigin.NATIVE,
            aggregation=AnchorAggregation.FRAME,
            captured_at=T0,
        )
        out_uncal = await resolver.resolve(
            hypotheses=[_SimpleEntity("ph-1")],
            new_face_anchors=[uncal_face_bob],
            captured_at=T0,
            wearable_evidence={"ph-1": ev_alice},
        )
        assert out_uncal.decisions[0].conflict == ""
        assert metrics_mod.metrics.cts_wearable_face_conflict_total._value.get() == 1.0

        # 4. Shadow mode: live decision unchanged; shadow counts counterfactual outcomes.
        shadow_resolver = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(arcface_authority_calibrated_confidence=0.80),
            wearable_config=WearableConfig(mode="shadow"),
        )
        out_shadow = await shadow_resolver.resolve(
            hypotheses=[_SimpleEntity("ph-1")],
            new_face_anchors=[],
            captured_at=T0,
            wearable_evidence={"ph-1": ev_alice},
        )
        assert out_shadow.decisions[0].identity_id is None
        assert (
            metrics_mod.metrics.cts_wearable_shadow_outcomes_total.labels(
                outcome=str(WearableShadowOutcome.WOULD_NAME_UNKNOWN)
            )._value.get()
            == 1.0
        )

        out_shadow_conflict = await shadow_resolver.resolve(
            hypotheses=[_SimpleEntity("ph-1")],
            new_face_anchors=[cal_face_bob],
            captured_at=T0,
            wearable_evidence={"ph-1": ev_alice},
        )
        # Live decision still honors calibrated direct face (bob) in shadow mode!
        assert out_shadow_conflict.decisions[0].identity_id == "bob"
        assert (
            metrics_mod.metrics.cts_wearable_shadow_outcomes_total.labels(
                outcome=str(WearableShadowOutcome.WOULD_CONFLICT)
            )._value.get()
            == 1.0
        )

    @pytest.mark.asyncio
    async def test_duplicate_guard_ranks_wearable_without_bypassing_incumbent(self) -> None:
        gallery = InMemoryGalleryRepository()
        for uid in ("alice", "bob"):
            await gallery.upsert_identity(
                Identity(identity_id=uid, display_name=uid.title(), enrolled_at=T0)
            )

        resolver = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(enable_duplicate_active_identity_guard=True),
            wearable_config=WearableConfig(mode="enabled", duplicate_guard_min_probability=0.90),
        )
        ev_strong = WearableEvidence(
            ph_id="ph-2",
            tag_id="tag-a",
            identity_id="alice",
            evidence_class=WearableEvidenceClass.TRAJECTORY,
            attestation_grade=WearableAttestationGrade.NONE,
            window_llr=22.0,
            margin_llr=10.0,
            matched_fixes=8,
            mean_residual_m=0.1,
            newest_fix_at=T0,
        )
        # Incumbent ph-1 holds alice; ph-2 with strong wearable cannot bypass incumbent.
        out = await resolver.resolve(
            hypotheses=[_SimpleEntity("ph-2")],
            new_face_anchors=[],
            captured_at=T0,
            open_ph_identities={"ph-1": "alice"},
            wearable_evidence={"ph-2": ev_strong},
        )
        assert out.decisions[0].identity_id is None
        assert "duplicate_active_identity_blocked" in out.decisions[0].reason


class TestWearableAssociationPersistStage:
    @pytest.mark.asyncio
    async def test_persists_transitions_from_context(self) -> None:
        repo = InMemoryWearableAssociationRepository()
        stage = WearableAssociationPersistStage(repo=repo)
        seg = WearableAssociationSegment(
            segment_id="00000000-0000-0000-0000-000000000001",
            ph_id="00000000-0000-0000-0000-000000000010",
            tag_id="tag-1",
            identity_id="alice",
            started_at=T0,
            ended_at=None,
            attestation_grade=WearableAttestationGrade.TRAJECTORY,
            window_llr=18.5,
            margin_llr=10.0,
            matched_fixes=9,
            mean_residual_m=0.11,
        )
        ctx = FrameContext(
            frame=FrameReady(
                camera_id="cam-1",
                frame_index=1,
                capture_time_unix_ns=_unix_ns(T0),
                minio_key="frames/cam-1/1.jpg",
                width=640,
                height=480,
            ),
            event_time=T0,
            capture_time=T0,
            wearable_association_transitions=[
                WearableAssociationTransition(action="open", segment=seg)
            ],
        )
        await stage.run(ctx)
        stored = await repo.get_segment(seg.segment_id)
        assert stored == seg

    @pytest.mark.asyncio
    async def test_roster_sync_purge_clears_wearable_associations_and_buffer(self) -> None:
        from app.services.identity_roster_sync import IdentityRosterSyncService

        gallery = InMemoryGalleryRepository()
        await gallery.upsert_identity(
            Identity(identity_id="alice", display_name="Alice", enrolled_at=T0)
        )
        assoc_repo = InMemoryWearableAssociationRepository()
        await assoc_repo.save_segment(
            WearableAssociationSegment(
                segment_id="00000000-0000-0000-0000-000000000002",
                ph_id="00000000-0000-0000-0000-000000000010",
                tag_id="tag-1",
                identity_id="alice",
                started_at=T0,
                ended_at=None,
                attestation_grade=WearableAttestationGrade.TRAJECTORY,
                window_llr=18.0,
                margin_llr=10.0,
                matched_fixes=9,
                mean_residual_m=0.1,
            )
        )
        buf = WearableFixBuffer()
        buf.add(WearableFix("tag-1", "alice", T0, FixKind.ROOM, room_id="r1", epoch_seq=1))
        site_map = WearableSiteConfigMap(
            WearableSiteConfig(
                bindings=(WearableBinding(tag_id="tag-1", member_id="alice", valid_from=T0),)
            )
        )

        roster_sync = IdentityRosterSyncService(
            gallery_repo=gallery,
            wearable_site_config=site_map,
            wearable_fix_buffer=buf,
            wearable_association_repo=assoc_repo,
        )
        assert await roster_sync.purge_identity("alice") is True
        assert await assoc_repo.list_recent() == []
        assert buf.recent(now=T0 + timedelta(seconds=1), max_age_s=10.0) == ()
        assert site_map.get_sync().member_at("tag-1", T0) is None

    def test_settings_yaml_builds_wearable_config_default_off(self) -> None:
        from app.config import settings
        from app.main import _build_wearable_config

        cfg = _build_wearable_config(settings)
        assert cfg.mode == "off"
        assert cfg.timing_sigma_s == 0.5
        assert cfg.max_fix_age_s == 15.0
        assert cfg.trajectory_max_sigma_m == 0.75
        assert cfg.parked_after_s == 600.0
        assert cfg.trajectory_attest_min_duration_s == 10.0
        assert cfg.trajectory_attest_min_fixes == 8
        assert cfg.room_attest_min_duration_s == 60.0

    @pytest.mark.asyncio
    async def test_wearables_internal_router_and_readiness_expose_state(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.calibration.state import CalibrationState
        from app.routers.readiness import router as readiness_router
        from app.routers.wearables import router as wearables_router
        from app.services.camera_room_map import CameraRoomMap
        from app.services.identity_readiness import IdentityPolicySnapshot, IdentityReadinessService

        class _OverlapAndWearableStub:
            def overlap_group_count(self) -> int:
                return 1

            def active_wearable_association_count(self) -> int:
                return 2

        repo = InMemoryWearableAssociationRepository()
        await repo.save_segment(
            WearableAssociationSegment(
                segment_id="00000000-0000-0000-0000-000000000003",
                ph_id="00000000-0000-0000-0000-000000000011",
                tag_id="tag-1",
                identity_id="alice",
                started_at=T0,
                ended_at=None,
                attestation_grade=WearableAttestationGrade.ROOM,
                window_llr=12.0,
                margin_llr=8.0,
                matched_fixes=7,
                mean_residual_m=None,
            )
        )

        app = FastAPI()
        app.include_router(readiness_router)
        app.include_router(wearables_router)
        app.state.wearable_association_repo = repo
        app.state.identity_readiness_service = IdentityReadinessService(
            policy=IdentityPolicySnapshot(
                face_evidence_policy="shadow",
                face_templates_mode="off",
                face_quality_mode="shadow",
                cc_assertion_mode="shadow",
                expected_arcface_model_version="buffalo_l",
                expected_quality_version="heuristic-v1",
                wearable_mode="shadow",
            ),
            calibration_state=CalibrationState(),
            camera_room_map=CameraRoomMap(),
            overlap_groups=_OverlapAndWearableStub(),
            wearable_associations=_OverlapAndWearableStub(),
        )

        client = TestClient(app)
        r_ready = client.get("/internal/identity/readiness")
        assert r_ready.status_code == 200
        assert r_ready.json()["wearable_mode"] == "shadow"
        assert r_ready.json()["wearable_active_associations"] == 2

        r_assoc = client.get("/internal/wearables/associations", params={"active_only": "true"})
        assert r_assoc.status_code == 200
        items = r_assoc.json()
        assert len(items) == 1
        assert items[0]["tag_id"] == "tag-1"
        assert items[0]["attestation_grade"] == "room"
