"""Guardrail and replay proofs for closed-world identity prior shaping.

Proves three load-bearing invariants:
1. Closed-world prior shaping NEVER commits an identity on its own, even when
   only one enrolled candidate remains after away/elimination.
2. Adding a weak face that alone cannot commit DOES commit when combined with
   closed-world prior shaping (and stays Unknown without it, or in shadow mode
   where it counts ``would_name_unknown``).
3. In a multi-person WorldTracker replay with two enrolled residents and one
   visitor, elimination never assigns the visitor a resident identity and causes
   zero authoritative swaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from cts_contracts import FixKind, MotionState, WearState
from cts_contracts.household import PresenceStatus
from prometheus_client import CollectorRegistry

from app.domain import (
    BoundingBox,
    FaceAnchor,
    FloorPoint,
    HouseholdPresenceSnapshot,
    Identity,
    MemberPresenceSnapshot,
    OpenPHOccupancy,
    ViewPrototype,
    WearableFix,
    WorldObservation,
)
from app.observability import metrics as metrics_module
from app.services.household_presence_subscriber import HouseholdPresenceCache
from app.storage.base import (
    InMemoryGalleryRepository,
    InMemoryPHRepository,
    InMemoryWorldObservationRepository,
)
from app.tracking.identity.closed_world_channel import ClosedWorldConfig
from app.tracking.identity_resolver import IdentityResolver, ResolverConfig
from app.tracking.world.config import WorldTrackerConfig
from app.tracking.world.tracker import WorldTracker
from app.tracking.world.wearable_config import WearableConfig


@dataclass(frozen=True)
class _FakeEntity:
    entity_id: str
    observation_ids: list[str] = field(default_factory=list)
    camera_ids: list[str] = field(default_factory=list)
    current_identity_id: str | None = None
    current_identity_committed_at: datetime | None = None
    last_independent_identity_evidence_at: datetime | None = None
    last_seen_at: datetime = field(
        default_factory=lambda: datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
    )
    started_at: datetime = field(default_factory=lambda: datetime(2026, 11, 21, 11, 59, tzinfo=UTC))
    view_prototypes: tuple[ViewPrototype, ...] = ()


async def _gallery_with(*names: str) -> InMemoryGalleryRepository:
    repo = InMemoryGalleryRepository()
    now = datetime(2026, 11, 21, 10, 0, tzinfo=UTC)
    for name in names:
        await repo.upsert_identity(
            Identity(identity_id=name, display_name=name.title(), enrolled_at=now, is_active=True)
        )
    return repo


def _full_presence(
    now: datetime,
    statuses: dict[str, tuple[PresenceStatus, float]],
    *,
    visitors_present: bool = False,
) -> HouseholdPresenceSnapshot:
    members = tuple(
        MemberPresenceSnapshot(
            member_id=mid,
            status=st,
            confidence=conf,
            room_id=None,
            inferred_at=now,
        )
        for mid, (st, conf) in statuses.items()
    )
    return HouseholdPresenceSnapshot(
        members=members,
        roster_complete=True,
        visitors_present=visitors_present,
        published_at=now,
        received_at=now,
    )


@pytest.fixture()
def isolated_metrics(monkeypatch: pytest.MonkeyPatch) -> metrics_module.Metrics:
    fresh = metrics_module.build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_module, "metrics", fresh)
    return fresh


class TestClosedWorldNeverCommitsAlone:
    @pytest.mark.asyncio
    async def test_unknown_ph_with_only_closed_world_stays_unknown(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        gallery = await _gallery_with("alice", "grandma")
        resolver = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(height_mode="off"),
            closed_world_config=ClosedWorldConfig(mode="enabled"),
        )
        entity = _FakeEntity(entity_id="ph-target", observation_ids=["det-1"], last_seen_at=now)
        pres = _full_presence(
            now,
            {
                "alice": (PresenceStatus.AWAY, 0.95),
                "grandma": (PresenceStatus.PRESENT_HOME, 0.95),
            },
        )
        occ = {
            "ph-target": OpenPHOccupancy(
                ph_id="ph-target",
                identity_id=None,
                last_independent_evidence_at=None,
                last_seen_at=now,
                room_id="bedroom",
            ),
            "ph-alice": OpenPHOccupancy(
                ph_id="ph-alice",
                identity_id="alice",
                last_independent_evidence_at=now - timedelta(seconds=5),
                last_seen_at=now,
                room_id="kitchen",
            ),
        }

        outcome = await resolver.resolve(
            hypotheses=[entity],
            new_face_anchors=[],
            captured_at=now,
            open_ph_identities={"ph-alice": "alice"},
            open_ph_occupancies=occ,
            household_presence=pres,
        )
        assert len(outcome.decisions) == 1
        decision = outcome.decisions[0]
        # Posterior strongly favors grandma, yet commit guard keeps PH Unknown.
        top_id, top_prob = decision.posterior.top_identity()
        assert top_id == "grandma"
        assert top_prob >= 0.70
        assert decision.identity_id is None
        assert decision.evidence_backed is False

    @pytest.mark.asyncio
    async def test_weak_face_commits_with_closed_world_and_not_without(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        gallery = await _gallery_with("alice", "bob", "grandma")

        entity = _FakeEntity(entity_id="ph-target", observation_ids=["det-1"], last_seen_at=now)
        # Weak recognized face for grandma: standalone posterior is ~0.51 (< 0.65),
        # so it stays Unknown without closed-world and commits only when closed-world
        # down-weights the away household members (alice and bob).
        weak_face = FaceAnchor(
            person_id="grandma",
            match_score=0.45,
            quality=0.5,
            detection_id="det-1",
            tracklet_id="ph-target",
            camera_id="cam-bedroom",
            captured_at=now,
            recognition_state="recognized",
            similarity=0.45,
        )
        pres = _full_presence(
            now,
            {
                "alice": (PresenceStatus.AWAY, 0.95),
                "bob": (PresenceStatus.AWAY, 0.95),
                "grandma": (PresenceStatus.PRESENT_HOME, 0.95),
            },
        )
        occ = {
            "ph-target": OpenPHOccupancy(
                ph_id="ph-target",
                identity_id=None,
                last_independent_evidence_at=None,
                last_seen_at=now,
                room_id="bedroom",
            )
        }

        # 1. Without closed-world: weak face stays Unknown.
        resolver_off = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(height_mode="off"),
            closed_world_config=ClosedWorldConfig(mode="off"),
        )
        out_off = await resolver_off.resolve(
            hypotheses=[entity],
            new_face_anchors=[weak_face],
            captured_at=now,
            open_ph_occupancies=occ,
            household_presence=pres,
        )
        assert out_off.decisions[0].identity_id is None

        # 2. In shadow mode: live stays Unknown, shadow records would_name_unknown + mismatch.
        resolver_shadow = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(height_mode="off"),
            closed_world_config=ClosedWorldConfig(mode="shadow"),
        )
        out_shadow = await resolver_shadow.resolve(
            hypotheses=[entity],
            new_face_anchors=[weak_face],
            captured_at=now,
            open_ph_occupancies=occ,
            household_presence=pres,
        )
        assert out_shadow.decisions[0].identity_id is None
        assert (
            isolated_metrics.cts_closed_world_shadow_outcomes_total.labels(
                outcome="would_name_unknown"
            )._value.get()
            == 1.0
        )
        assert (
            isolated_metrics.identity_shadow_mismatch_total.labels(
                feature="closed_world"
            )._value.get()
            == 1.0
        )

        # 3. In enabled mode: weak face + closed-world prior commits grandma.
        resolver_enabled = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(height_mode="off"),
            closed_world_config=ClosedWorldConfig(mode="enabled"),
        )
        out_enabled = await resolver_enabled.resolve(
            hypotheses=[entity],
            new_face_anchors=[weak_face],
            captured_at=now,
            open_ph_occupancies=occ,
            household_presence=pres,
        )
        assert out_enabled.decisions[0].identity_id == "grandma"


class TestClosedWorldReplayAndWearableIntegration:
    @pytest.mark.asyncio
    async def test_replay_two_residents_and_one_visitor_never_assigns_visitor(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        """Two residents occupy kitchen/bedroom; visitor in hallway stays Unknown."""
        t0 = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        gallery = await _gallery_with("alice", "bob")
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()
        presence_cache = HouseholdPresenceCache()
        presence_cache.update(
            _full_presence(
                t0,
                {
                    "alice": (PresenceStatus.PRESENT_ROOM, 0.95),
                    "bob": (PresenceStatus.PRESENT_ROOM, 0.95),
                },
                visitors_present=True,
            )
        )
        cw_cfg = ClosedWorldConfig(mode="enabled")
        resolver = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(height_mode="off"),
            closed_world_config=cw_cfg,
            household_presence_source=presence_cache,
        )
        tracker = WorldTracker(
            ph_repo=ph_repo,
            obs_repo=obs_repo,
            config=WorldTrackerConfig(),
            identity_resolver=resolver,
            gallery_repo=gallery,
            closed_world_config=cw_cfg,
        )
        cam_rooms = {
            "cam-kitchen": "kitchen",
            "cam-bedroom": "bedroom",
            "cam-hall": "hallway",
        }
        bbox = BoundingBox(x_min=100, y_min=100, x_max=220, y_max=400)

        def _obs(
            det_id: str,
            cam: str,
            x_m: float,
            y_m: float,
            ts: datetime,
            face: FaceAnchor | None = None,
        ) -> WorldObservation:
            return WorldObservation(
                camera_id=cam,
                frame_index=1,
                captured_at=ts,
                floor_point=FloorPoint(x_mm=int(x_m * 1000), y_mm=int(y_m * 1000), calibrated=True),
                bbox=bbox,
                embedding=[],
                detection_confidence=0.95,
                face_anchor=face,
                detection_id=det_id,
                quality=0.8,
            )

        # Round 1: Alice and Bob are both directly identified in separate rooms.
        fa_alice = FaceAnchor(
            person_id="alice",
            match_score=0.92,
            detection_id="det-a-1",
            tracklet_id="",
            camera_id="cam-kitchen",
            captured_at=t0,
            recognition_state="recognized",
            similarity=0.92,
            calibrated_confidence=0.96,
        )
        fa_bob = FaceAnchor(
            person_id="bob",
            match_score=0.91,
            detection_id="det-b-1",
            tracklet_id="",
            camera_id="cam-bedroom",
            captured_at=t0,
            recognition_state="recognized",
            similarity=0.91,
            calibrated_confidence=0.95,
        )
        r1 = await tracker.step(
            observations=[
                _obs("det-a-1", "cam-kitchen", 1.0, 1.0, t0, fa_alice),
                _obs("det-b-1", "cam-bedroom", 8.0, 1.0, t0, fa_bob),
            ],
            now=t0,
            room_polygons={},
            camera_room_map=cam_rooms,
            face_anchors=[fa_alice, fa_bob],
        )
        ph_alice = r1.det_to_ph["det-a-1"]
        ph_bob = r1.det_to_ph["det-b-1"]
        dec1 = {d.ph_id: d.identity_id for d in r1.identity_decisions}
        assert dec1[ph_alice] == "alice"
        assert dec1[ph_bob] == "bob"

        # Round 2 (t0 + 1 s): Alice and Bob stay observed; a visitor appears in hallway
        # with an unrecognized face marker. Elimination + guest floor must keep visitor Unknown
        # with zero swaps on Alice or Bob.
        t1 = t0 + timedelta(seconds=1)
        fa_visitor_unknown = FaceAnchor(
            person_id="",
            match_score=0.90,
            detection_id="det-v-2",
            tracklet_id="",
            camera_id="cam-hall",
            captured_at=t1,
            recognition_state="unrecognized",
            similarity=0.0,
        )
        r2 = await tracker.step(
            observations=[
                _obs("det-a-2", "cam-kitchen", 1.05, 1.0, t1),
                _obs("det-b-2", "cam-bedroom", 8.05, 1.0, t1),
                _obs("det-v-2", "cam-hall", 4.5, 5.0, t1, fa_visitor_unknown),
            ],
            now=t1,
            room_polygons={},
            camera_room_map=cam_rooms,
            face_anchors=[fa_visitor_unknown],
        )
        ph_visitor = r2.det_to_ph["det-v-2"]
        dec2 = {d.ph_id: d for d in r2.identity_decisions}
        assert dec2[ph_alice].identity_id == "alice"
        assert dec2[ph_bob].identity_id == "bob"
        assert dec2[ph_visitor].identity_id is None
        assert dec2[ph_visitor].posterior.top_identity()[0] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_world_tracker_wearable_10min_non_parked_override(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        """A non-parked tag fix keeps member home for 10 min even after fix leaves 30 s buffer."""
        t0 = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        gallery = await _gallery_with("alice", "bob")
        ph_repo = InMemoryPHRepository()
        obs_repo = InMemoryWorldObservationRepository()
        presence_cache = HouseholdPresenceCache()
        cw_cfg = ClosedWorldConfig(mode="enabled", wearable_home_window_s=600.0)
        wearable_cfg = WearableConfig(mode="shadow")
        resolver = IdentityResolver(
            gallery_repo=gallery,
            config=ResolverConfig(height_mode="off"),
            wearable_config=wearable_cfg,
            closed_world_config=cw_cfg,
            household_presence_source=presence_cache,
        )
        tracker = WorldTracker(
            ph_repo=ph_repo,
            obs_repo=obs_repo,
            config=WorldTrackerConfig(),
            identity_resolver=resolver,
            gallery_repo=gallery,
            wearable_config=wearable_cfg,
            closed_world_config=cw_cfg,
        )
        moving_fix = WearableFix(
            tag_id="tag-alice",
            member_id="alice",
            measured_at=t0,
            kind=FixKind.ROOM,
            room_id="living_room",
            room_probability=0.95,
            wear_state=WearState.UNSPECIFIED,
            motion_state=MotionState.MOVING,
            epoch_seq=1,
            received_at=t0,
        )
        # Round at t0 feeds moving fix into WearableAssociationTracker.
        await tracker.step(
            observations=[],
            now=t0,
            room_polygons={},
            camera_room_map={},
            wearable_fixes=[moving_fix],
        )
        # 5 minutes later (well past 30 s fix buffer, within 600 s wearable home window),
        # CC reports alice away; tracker still reports alice as wearable-present.
        t5 = t0 + timedelta(seconds=300)
        assert tracker._wearable_tracker is not None
        assert "alice" in tracker._wearable_tracker.members_with_fresh_non_parked_fix(
            now=t5, window_s=600.0
        )
        # 11 minutes later (beyond 600 s window), override expires.
        t11 = t0 + timedelta(seconds=660)
        assert "alice" not in tracker._wearable_tracker.members_with_fresh_non_parked_fix(
            now=t11, window_s=600.0
        )
