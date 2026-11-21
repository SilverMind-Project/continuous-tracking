"""Unit tests for ClosedWorldChannel, spatial separation, grade flags, and presence stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from cts_contracts.household import PresenceStatus
from prometheus_client import CollectorRegistry

from app.domain import (
    HouseholdPresenceSnapshot,
    Identity,
    MemberPresenceSnapshot,
    OpenPHOccupancy,
    OverlapGroup,
    ViewPrototype,
)
from app.observability import metrics as metrics_module
from app.proto.continuoustracking.v1 import household_pb2
from app.services.household_presence_subscriber import (
    FIELD_PRESENCE,
    HouseholdPresenceCache,
    HouseholdPresenceSubscriber,
    decode_household_presence,
)
from app.tracking.identity.channels import ResolverInputs
from app.tracking.identity.closed_world_channel import (
    ClosedWorldChannel,
    ClosedWorldConfig,
    canonical_ph_pair,
)
from app.tracking.identity.evidence_grades import (
    CC_ASSERTION,
    CLOSED_WORLD,
    NATIVE_FRAME,
    NATIVE_TEMPLATE,
    WEARABLE,
    EvidenceGradeName,
)


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


def _identities(*names: str) -> dict[str, Identity]:
    now = datetime(2026, 11, 21, 10, 0, tzinfo=UTC)
    return {
        n: Identity(identity_id=n, display_name=n.title(), enrolled_at=now, is_active=True)
        for n in names
    }


def _presence(
    now: datetime,
    statuses: dict[str, tuple[PresenceStatus, float]],
    *,
    roster_complete: bool = True,
    visitors_present: bool = False,
    age_s: float = 5.0,
) -> HouseholdPresenceSnapshot:
    published_at = now - timedelta(seconds=age_s)
    members = tuple(
        MemberPresenceSnapshot(
            member_id=mid,
            status=st,
            confidence=conf,
            room_id=None,
            inferred_at=published_at,
        )
        for mid, (st, conf) in statuses.items()
    )
    return HouseholdPresenceSnapshot(
        members=members,
        roster_complete=roster_complete,
        visitors_present=visitors_present,
        published_at=published_at,
        received_at=now,
    )


def _inputs(
    now: datetime,
    *,
    presence: HouseholdPresenceSnapshot | None,
    occupancies: dict[str, OpenPHOccupancy] | None = None,
    copresence_pairs: frozenset[tuple[str, str]] = frozenset(),
    overlap_groups: tuple[OverlapGroup, ...] = (),
    camera_room_map: dict[str, str] | None = None,
    wearable_present_identities: frozenset[str] = frozenset(),
) -> ResolverInputs:
    return ResolverInputs(
        captured_at=now,
        face_anchors=(),
        face_evidence=(),
        ph_heights_m={},
        ph_qualities={},
        open_ph_occupancies=occupancies or {},
        copresence_pairs=copresence_pairs,
        overlap_groups=overlap_groups,
        camera_room_map=camera_room_map or {},
        household_presence=presence,
        wearable_present_identities=wearable_present_identities,
    )


@pytest.fixture()
def isolated_metrics(monkeypatch: pytest.MonkeyPatch) -> metrics_module.Metrics:
    fresh = metrics_module.build_metrics(registry=CollectorRegistry())
    monkeypatch.setattr(metrics_module, "metrics", fresh)
    return fresh


class TestClosedWorldGrade:
    def test_grade_flags_forbid_commit_evidence_and_authority(self) -> None:
        assert CLOSED_WORLD.name == EvidenceGradeName.CLOSED_WORLD
        assert CLOSED_WORLD.counts_as_commit_evidence is False
        assert CLOSED_WORLD.grants_authority is False
        assert CLOSED_WORLD.refreshes_evidence_clock is False
        assert CLOSED_WORLD.ranks_in_duplicate_guard is False
        assert CLOSED_WORLD.grants_duplicate_bypass is False
        assert CLOSED_WORLD.enters_association is False
        assert CLOSED_WORLD.enters_contradiction is False
        assert CLOSED_WORLD.propagates is False
        # Existing grades default to counting as commit evidence.
        assert NATIVE_FRAME.counts_as_commit_evidence is True
        assert NATIVE_TEMPLATE.counts_as_commit_evidence is True
        assert CC_ASSERTION.counts_as_commit_evidence is True
        assert WEARABLE.counts_as_commit_evidence is True


class TestClosedWorldHomeFactor:
    def test_away_member_downweighted_unknown_and_low_confidence_neutral(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        idents = _identities("alice", "bob", "charlie", "dana")
        ch = ClosedWorldChannel(
            config=ClosedWorldConfig(mode="enabled"),
            identities_getter=lambda: idents,
        )
        pres = _presence(
            now,
            {
                "alice": (PresenceStatus.AWAY, 0.90),  # penalized -> 0.3
                "bob": (PresenceStatus.AWAY, 0.50),  # below 0.8 -> neutral 1.0
                "charlie": (PresenceStatus.UNKNOWN, 0.95),  # neutral 1.0
                "dana": (PresenceStatus.PRESENT_HOME, 0.95),  # neutral 1.0
            },
        )
        res = ch.evaluate_closed_world(_FakeEntity("ph-1"), _inputs(now, presence=pres))
        assert res.active is True
        assert res.likelihood.counts_as_commit_evidence is False
        assert res.likelihood.clock_evidence_ids == frozenset()
        assert res.likelihood.evidence_at is None
        dist = res.likelihood.distribution.distribution
        assert dist["alice"] == pytest.approx(dist["dana"] * 0.3, rel=1e-6)
        assert dist["bob"] == pytest.approx(dist["dana"], rel=1e-6)
        assert dist["charlie"] == pytest.approx(dist["dana"], rel=1e-6)

    def test_non_parked_wearable_overrides_cc_away(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        idents = _identities("alice", "bob")
        ch = ClosedWorldChannel(
            config=ClosedWorldConfig(mode="enabled"),
            identities_getter=lambda: idents,
        )
        pres = _presence(
            now,
            {
                "alice": (PresenceStatus.AWAY, 0.95),
                "bob": (PresenceStatus.PRESENT_HOME, 0.95),
            },
        )
        res = ch.evaluate_closed_world(
            _FakeEntity("ph-1"),
            _inputs(now, presence=pres, wearable_present_identities=frozenset({"alice"})),
        )
        dist = res.likelihood.distribution.distribution
        # Alice's fresh non-parked wearable raises home_alice back to 1.0 without lowering bob.
        assert dist["alice"] == pytest.approx(dist["bob"], rel=1e-6)

    def test_stale_presence_treats_everyone_as_unknown(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        idents = _identities("alice", "bob")
        ch = ClosedWorldChannel(
            config=ClosedWorldConfig(mode="enabled", presence_stale_after_s=180.0),
            identities_getter=lambda: idents,
        )
        pres = _presence(
            now,
            {
                "alice": (PresenceStatus.AWAY, 0.95),
                "bob": (PresenceStatus.PRESENT_HOME, 0.95),
            },
            age_s=240.0,
        )
        res = ch.evaluate_closed_world(_FakeEntity("ph-1"), _inputs(now, presence=pres))
        assert res.active is True
        assert res.presence_stale is True
        dist = res.likelihood.distribution.distribution
        assert dist["alice"] == pytest.approx(dist["bob"], rel=1e-6)


class TestClosedWorldEliminationGuards:
    def test_elimination_requires_fresh_evidence_on_other_ph(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        idents = _identities("alice", "bob")
        ch = ClosedWorldChannel(
            config=ClosedWorldConfig(mode="enabled"),
            identities_getter=lambda: idents,
        )
        pres = _presence(now, {})
        occ_j = OpenPHOccupancy(
            ph_id="ph-j",
            identity_id=None,
            last_independent_evidence_at=None,
            last_seen_at=now,
            room_id="bedroom",
        )
        occ_k_fresh = OpenPHOccupancy(
            ph_id="ph-k",
            identity_id="alice",
            last_independent_evidence_at=now - timedelta(seconds=30),
            last_seen_at=now,
            room_id="kitchen",
        )
        res_fresh = ch.evaluate_closed_world(
            _FakeEntity("ph-j"),
            _inputs(now, presence=pres, occupancies={"ph-j": occ_j, "ph-k": occ_k_fresh}),
        )
        d_fresh = res_fresh.likelihood.distribution.distribution
        assert d_fresh["alice"] == pytest.approx(d_fresh["bob"] * 0.2, rel=1e-6)

        occ_k_stale = OpenPHOccupancy(
            ph_id="ph-k",
            identity_id="alice",
            last_independent_evidence_at=now - timedelta(seconds=90),
            last_seen_at=now,
            room_id="kitchen",
        )
        res_stale = ch.evaluate_closed_world(
            _FakeEntity("ph-j"),
            _inputs(now, presence=pres, occupancies={"ph-j": occ_j, "ph-k": occ_k_stale}),
        )
        d_stale = res_stale.likelihood.distribution.distribution
        assert d_stale["alice"] == pytest.approx(d_stale["bob"], rel=1e-6)

    def test_elimination_guards_uncalibrated_same_room_and_overlapping_rooms(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        idents = _identities("alice", "bob")
        ch = ClosedWorldChannel(
            config=ClosedWorldConfig(mode="enabled"),
            identities_getter=lambda: idents,
        )
        pres = _presence(now, {})

        # Uncalibrated PHs in the same room -> must NOT eliminate.
        occ_j_same = OpenPHOccupancy(
            ph_id="ph-j",
            identity_id=None,
            last_independent_evidence_at=None,
            last_seen_at=now,
            floor_calibrated=False,
            room_id="living_room",
            last_seen_camera="cam-lr-1",
        )
        occ_k_same = OpenPHOccupancy(
            ph_id="ph-k",
            identity_id="alice",
            last_independent_evidence_at=now - timedelta(seconds=5),
            last_seen_at=now,
            floor_calibrated=False,
            room_id="living_room",
            last_seen_camera="cam-lr-2",
        )
        res_same = ch.evaluate_closed_world(
            _FakeEntity("ph-j"),
            _inputs(now, presence=pres, occupancies={"ph-j": occ_j_same, "ph-k": occ_k_same}),
        )
        d_same = res_same.likelihood.distribution.distribution
        assert d_same["alice"] == pytest.approx(d_same["bob"], rel=1e-6)

        # Open-plan living room + kitchen cameras in the same overlap group -> must NOT eliminate.
        occ_k_kitchen = OpenPHOccupancy(
            ph_id="ph-k",
            identity_id="alice",
            last_independent_evidence_at=now - timedelta(seconds=5),
            last_seen_at=now,
            floor_calibrated=False,
            room_id="kitchen",
            last_seen_camera="cam-kitchen",
        )
        overlap = (
            OverlapGroup(
                group_id="open-plan",
                name="Living + Kitchen",
                camera_ids=("cam-lr-1", "cam-kitchen"),
            ),
        )
        cam_rooms = {"cam-lr-1": "living_room", "cam-kitchen": "kitchen"}
        res_overlap = ch.evaluate_closed_world(
            _FakeEntity("ph-j"),
            _inputs(
                now,
                presence=pres,
                occupancies={"ph-j": occ_j_same, "ph-k": occ_k_kitchen},
                overlap_groups=overlap,
                camera_room_map=cam_rooms,
            ),
        )
        d_overlap = res_overlap.likelihood.distribution.distribution
        assert d_overlap["alice"] == pytest.approx(d_overlap["bob"], rel=1e-6)

    def test_elimination_guards_coasting_ph_and_skew_and_copresence_link(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        idents = _identities("alice", "bob")
        ch = ClosedWorldChannel(
            config=ClosedWorldConfig(mode="enabled"),
            identities_getter=lambda: idents,
        )
        pres = _presence(now, {})

        occ_j = OpenPHOccupancy(
            ph_id="ph-j",
            identity_id=None,
            last_independent_evidence_at=None,
            last_seen_at=now,
            floor_calibrated=True,
            floor_x_m=0.0,
            floor_y_m=0.0,
            room_id="living_room",
        )
        # Coasting during ph_close_grace_s -> must NOT eliminate.
        occ_k_coasting = OpenPHOccupancy(
            ph_id="ph-k",
            identity_id="alice",
            last_independent_evidence_at=now - timedelta(seconds=5),
            last_seen_at=now - timedelta(seconds=2.5),
            coasting=True,
            floor_calibrated=True,
            floor_x_m=5.0,
            floor_y_m=0.0,
            room_id="bedroom",
        )
        res_coast = ch.evaluate_closed_world(
            _FakeEntity("ph-j"),
            _inputs(now, presence=pres, occupancies={"ph-j": occ_j, "ph-k": occ_k_coasting}),
        )
        d_coast = res_coast.likelihood.distribution.distribution
        assert d_coast["alice"] == pytest.approx(d_coast["bob"], rel=1e-6)

        # Non-coasting calibrated PHs 3m apart with NO CoPresenceLink -> eliminates.
        occ_k_far = OpenPHOccupancy(
            ph_id="ph-k",
            identity_id="alice",
            last_independent_evidence_at=now - timedelta(seconds=5),
            last_seen_at=now - timedelta(seconds=0.4),
            floor_calibrated=True,
            floor_x_m=3.0,
            floor_y_m=0.0,
            room_id="living_room",
        )
        res_far = ch.evaluate_closed_world(
            _FakeEntity("ph-j"),
            _inputs(now, presence=pres, occupancies={"ph-j": occ_j, "ph-k": occ_k_far}),
        )
        d_far = res_far.likelihood.distribution.distribution
        assert d_far["alice"] == pytest.approx(d_far["bob"] * 0.2, rel=1e-6)

        # Same pair WITH an active CoPresenceLink -> must NOT eliminate.
        res_linked = ch.evaluate_closed_world(
            _FakeEntity("ph-j"),
            _inputs(
                now,
                presence=pres,
                occupancies={"ph-j": occ_j, "ph-k": occ_k_far},
                copresence_pairs=frozenset({canonical_ph_pair("ph-j", "ph-k")}),
            ),
        )
        d_linked = res_linked.likelihood.distribution.distribution
        assert d_linked["alice"] == pytest.approx(d_linked["bob"], rel=1e-6)

        # Room-class wearable-only PH (elim_eligible=False) -> must NOT eliminate.
        occ_k_room_wearable = OpenPHOccupancy(
            ph_id="ph-k",
            identity_id="alice",
            last_independent_evidence_at=now - timedelta(seconds=5),
            last_seen_at=now,
            floor_calibrated=True,
            floor_x_m=5.0,
            floor_y_m=0.0,
            room_id="bedroom",
            elim_eligible=False,
        )
        res_rw = ch.evaluate_closed_world(
            _FakeEntity("ph-j"),
            _inputs(now, presence=pres, occupancies={"ph-j": occ_j, "ph-k": occ_k_room_wearable}),
        )
        d_rw = res_rw.likelihood.distribution.distribution
        assert d_rw["alice"] == pytest.approx(d_rw["bob"], rel=1e-6)


class TestClosedWorldGatesAndGuestFloor:
    def test_guest_floor_enforced_and_rises_with_visitors(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        # 12 enrolled residents -> raw UNKNOWN share 1/13 = 0.0769 < 0.10.
        idents = _identities(*(f"member-{i:02d}" for i in range(12)))
        ch = ClosedWorldChannel(
            config=ClosedWorldConfig(mode="enabled"),
            identities_getter=lambda: idents,
        )
        pres_normal = _presence(now, {}, visitors_present=False)
        res_normal = ch.evaluate_closed_world(
            _FakeEntity("ph-1"), _inputs(now, presence=pres_normal)
        )
        assert res_normal.likelihood.distribution.distribution["UNKNOWN"] == pytest.approx(
            0.10, rel=1e-6
        )

        pres_visitors = _presence(now, {}, visitors_present=True)
        res_visitors = ch.evaluate_closed_world(
            _FakeEntity("ph-1"), _inputs(now, presence=pres_visitors)
        )
        assert res_visitors.likelihood.distribution.distribution["UNKNOWN"] == pytest.approx(
            0.30, rel=1e-6
        )

    def test_roster_incomplete_returns_empty_and_increments_metric(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        idents = _identities("alice", "bob")
        ch = ClosedWorldChannel(
            config=ClosedWorldConfig(mode="enabled"),
            identities_getter=lambda: idents,
        )
        pres = _presence(now, {"alice": (PresenceStatus.AWAY, 0.95)}, roster_complete=False)
        res = ch.evaluate_closed_world(_FakeEntity("ph-1"), _inputs(now, presence=pres))
        assert res.active is False
        assert res.likelihood.distribution.distribution == {}
        assert (
            isolated_metrics.cts_closed_world_inactive_total.labels(
                reason="roster_incomplete"
            )._value.get()
            == 1.0
        )


class TestHouseholdPresenceRoundTrip:
    def test_proto_round_trip_and_subscriber_cache(
        self, isolated_metrics: metrics_module.Metrics
    ) -> None:
        now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        pub_ns = int(now.timestamp() * 1e9)
        msg = household_pb2.HouseholdPresence(
            members=[
                household_pb2.MemberPresence(
                    member_id="alice",
                    status=household_pb2.PRESENCE_STATUS_AWAY,
                    confidence=0.9,
                    room_id="",
                    inferred_at_unix_ns=pub_ns,
                ),
                household_pb2.MemberPresence(
                    member_id="grandma",
                    status=household_pb2.PRESENCE_STATUS_PRESENT_ROOM,
                    confidence=0.95,
                    room_id="living_room",
                    inferred_at_unix_ns=pub_ns,
                ),
            ],
            roster_complete=True,
            visitors_present=True,
            published_at_unix_ns=pub_ns,
        )
        raw = msg.SerializeToString()
        decoded = decode_household_presence(raw, received_at=now)
        assert decoded is not None
        assert decoded.roster_complete is True
        assert decoded.visitors_present is True
        assert decoded.member("alice") is not None
        assert decoded.member("alice").status is PresenceStatus.AWAY  # type: ignore[union-attr]
        assert decoded.member("grandma").room_id == "living_room"  # type: ignore[union-attr]

        cache = HouseholdPresenceCache()
        sub = HouseholdPresenceSubscriber(redis_client=None, cache=cache)
        assert sub.handle_fields({FIELD_PRESENCE: raw}, now=now) is True
        assert cache.latest() is not None
        assert cache.purge_identity("alice") == 1
        assert cache.latest().member("alice") is None  # type: ignore[union-attr]

        # Malformed payload dead-letters cleanly without overwriting the cache.
        assert sub.handle_fields({FIELD_PRESENCE: b"\xff\xff\x00"}, now=now) is False
        assert (
            isolated_metrics.cts_messages_dead_lettered_total.labels(
                stream="household_presence"
            )._value.get()
            == 1.0
        )
        assert cache.latest() is not None
