"""Dual-mode watch continuity, physical-device grouping, segment hold, and room namespace.

A dual-mode watch is two logical tags (a 1 Hz BLE room beacon and duty-cycled UWB position
bursts) that share one ``physical_device_id``. These tests pin how one physical device keeps
one association segment, how its trajectory grade is earned and latched, how separate
wearables of one member stay separate, and that PH rooms are resolved in the room-id
namespace wearable fixes use.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from cts_contracts import FixKind, MotionState, WearState

from app.domain import (
    PersonHypothesis,
    WearableAssociationTransition,
    WearableAttestationGrade,
    WearableEvidenceClass,
    WearableFix,
)
from app.observability.metrics import WearableAssociationOutcome
from app.tracking.world.ph_floor_history import PHFloorHistory
from app.tracking.world.wearable_association import (
    AssociationRoundResult,
    WearableAssociationTracker,
)
from app.tracking.world.wearable_config import WearableConfig

T0 = datetime(2026, 11, 25, 9, 0, 0, tzinfo=UTC)

# Room ids as Cognitive Companion sends them: numeric strings, never display names.
ROOMS: dict[str, list[tuple[float, float]]] = {
    "1": [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)],
    "2": [(6.0, 0.0), (12.0, 0.0), (12.0, 5.0), (6.0, 5.0)],
}
CAMERA_ROOM_IDS = {"cam-1": "1"}
WATCH = "watch-amma"


def _ph(
    ph_id: str,
    x_m: float,
    y_m: float,
    *,
    last_seen_at: datetime,
    born_at: datetime = T0,
    camera: str = "cam-1",
    room_name: str = "Living Room",
) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(x_m, y_m, 0.0, 0.0),
        state_cov=(0.04, 0, 0, 0, 0, 0.04, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.25),
        born_at=born_at,
        last_seen_at=last_seen_at,
        last_seen_camera=camera,
        observation_count=5,
        # The tracker's room metadata holds a display name when no polygon contains the PH.
        metadata={"last_room_id": room_name, "last_room_name": room_name},
    )


def _ble(
    t: datetime,
    *,
    tag_id: str = "watch-ble",
    member: str = "amma",
    room_id: str = "1",
    physical: str | None = WATCH,
    motion: MotionState = MotionState.MOVING,
) -> WearableFix:
    return WearableFix(
        tag_id=tag_id,
        member_id=member,
        measured_at=t,
        kind=FixKind.ROOM,
        room_id=room_id,
        room_probability=0.95,
        wear_state=WearState.ON_BODY,
        motion_state=motion,
        physical_device_id=physical,
    )


def _uwb(
    t: datetime,
    x_m: float,
    y_m: float,
    *,
    tag_id: str = "watch-uwb",
    member: str = "amma",
    sigma_m: float = 0.15,
    physical: str | None = WATCH,
    room_id: str | None = None,
) -> WearableFix:
    var = sigma_m * sigma_m
    return WearableFix(
        tag_id=tag_id,
        member_id=member,
        measured_at=t,
        kind=FixKind.POSITION,
        x_m=x_m,
        y_m=y_m,
        cov_m2=(var, 0.0, 0.0, var),
        room_id=room_id,
        wear_state=WearState.ON_BODY,
        motion_state=MotionState.MOVING,
        physical_device_id=physical,
    )


class _Replay:
    """Runs one association round per second over a rolling 30 s fix buffer."""

    def __init__(self, cfg: WearableConfig | None = None) -> None:
        self.cfg = cfg or WearableConfig(mode="shadow")
        self.tracker = WearableAssociationTracker(self.cfg)
        self.history = PHFloorHistory(max_span_s=45.0)
        self.fixes: list[WearableFix] = []
        self.transitions: list[WearableAssociationTransition] = []

    def record(self, ph_id: str, t: datetime, x_m: float, y_m: float) -> None:
        self.history.record(
            ph_id, captured_at=t, x_m=x_m, y_m=y_m, cov_m2=(0.04, 0, 0, 0.04), speed_m_s=0.5
        )

    def step(
        self,
        t: datetime,
        phs: Sequence[PersonHypothesis],
        new_fixes: Sequence[WearableFix],
        *,
        rooms: dict[str, list[tuple[float, float]]] = ROOMS,
        camera_room_ids: dict[str, str] = CAMERA_ROOM_IDS,
    ) -> AssociationRoundResult:
        self.fixes.extend(new_fixes)
        self.fixes = [f for f in self.fixes if (t - f.measured_at).total_seconds() <= 30.0]
        res = self.tracker.step(
            fixes=self.fixes,
            open_phs=phs,
            closed_ph_ids=[],
            ph_history=self.history,
            room_polygons=rooms,
            camera_room_ids=camera_room_ids,
            now=t,
        )
        self.transitions.extend(res.transitions)
        return res

    def closes(self) -> list[WearableAssociationTransition]:
        return [tr for tr in self.transitions if tr.action == "close"]


def _walking_x(second: int) -> float:
    return 1.0 + 0.3 * min(second, 14)


class TestDualModeContinuity:
    def test_burst_then_ble_cooldown_keeps_one_segment_and_latched_trajectory_grade(self) -> None:
        r = _Replay()
        grades: dict[int, WearableAttestationGrade] = {}
        segment_ids: set[str] = set()
        for s in range(46):
            t = T0 + timedelta(seconds=s)
            x = _walking_x(s)
            r.record("ph-a", t, x, 2.0)
            new = [_ble(t)]
            if 1 <= s <= 12:
                new.append(_uwb(t, x, 2.0))
            res = r.step(t, [_ph("ph-a", x, 2.0, last_seen_at=t)], new)
            ev = res.evidence_by_ph.get("ph-a")
            if ev is not None:
                grades[s] = ev.attestation_grade
                segment_ids.add(ev.segment_id or "")

        assert len(segment_ids) == 1
        assert r.closes() == []
        assert grades[12] == WearableAttestationGrade.TRAJECTORY
        # The burst ended at 12 s; the BLE beacon carries the association through the
        # cooldown and the trajectory grade stays latched.
        assert all(grades[s] == WearableAttestationGrade.TRAJECTORY for s in range(12, 46))
        final = r.step(
            T0 + timedelta(seconds=46),
            [_ph("ph-a", _walking_x(46), 2.0, last_seen_at=T0 + timedelta(seconds=46))],
            [_ble(T0 + timedelta(seconds=46))],
        )
        ev = final.evidence_by_ph["ph-a"]
        assert ev.evidence_class == WearableEvidenceClass.ROOM
        assert ev.tag_id == "watch-ble"
        opened = [tr for tr in r.transitions if tr.action == "open"]
        # The segment keeps the tag that opened it as its provenance tag.
        assert [tr.segment.tag_id for tr in opened] == ["watch-ble"]

    def test_ble_pre_history_does_not_count_toward_trajectory_span(self) -> None:
        r = _Replay()
        grade_at: dict[int, WearableAttestationGrade] = {}
        for s in range(42):
            t = T0 + timedelta(seconds=s)
            x = 2.0 if s < 31 else 2.0 + 0.3 * (s - 31)
            r.record("ph-a", t, x, 2.0)
            new = [_ble(t)]
            if s >= 31:
                new.append(_uwb(t, x, 2.0))
            res = r.step(t, [_ph("ph-a", x, 2.0, last_seen_at=t)], new)
            if "ph-a" in res.evidence_by_ph:
                grade_at[s] = res.evidence_by_ph["ph-a"].attestation_grade
        # Nine UWB fixes over 8 s: enough fixes, too short a trajectory streak, even
        # though the segment itself has been open since the BLE beacon started.
        assert grade_at[39] == WearableAttestationGrade.NONE
        assert grade_at[41] == WearableAttestationGrade.TRAJECTORY

    def test_two_short_bursts_never_sum_to_a_trajectory_grade(self) -> None:
        r = _Replay()
        grades: set[WearableAttestationGrade] = set()
        for s in range(85):
            t = T0 + timedelta(seconds=s)
            x = 1.0 + 0.3 * (s % 10)
            r.record("ph-a", t, x, 2.0)
            new = [_ble(t)]
            if s <= 6 or 70 <= s <= 76:
                new.append(_uwb(t, x, 2.0))
            res = r.step(t, [_ph("ph-a", x, 2.0, last_seen_at=t)], new)
            if "ph-a" in res.evidence_by_ph:
                grades.add(res.evidence_by_ph["ph-a"].attestation_grade)
        assert WearableAttestationGrade.TRAJECTORY not in grades
        assert r.closes() == []

    def test_separate_device_of_same_member_closes_segment_and_drops_grade(self) -> None:
        r = _Replay()
        for s in range(30):
            t = T0 + timedelta(seconds=s)
            x = _walking_x(s)
            r.record("ph-a", t, x, 2.0)
            new: list[WearableFix] = []
            if s <= 13:
                new.append(_uwb(t, x, 2.0))
            else:
                # Only the pendant (its own physical device) reports from now on.
                new.append(_ble(t, tag_id="pendant", physical=None))
            res = r.step(t, [_ph("ph-a", x, 2.0, last_seen_at=t)], new)
        closes = r.closes()
        assert [c.segment.close_reason for c in closes] == ["reassigned"]
        assert closes[0].segment.attestation_grade == WearableAttestationGrade.TRAJECTORY
        ev = res.evidence_by_ph["ph-a"]
        assert ev.tag_id == "pendant"
        assert ev.attestation_grade == WearableAttestationGrade.NONE


class TestPhysicalDeviceDisagreement:
    def _run(
        self, fixes_at: dict[int, list[WearableFix]], x_m: float
    ) -> list[WearableAssociationOutcome]:
        r = _Replay()
        outcomes: list[WearableAssociationOutcome] = []
        for s in range(16):
            t = T0 + timedelta(seconds=s)
            r.record("ph-a", t, x_m, 2.0)
            res = r.step(t, [_ph("ph-a", x_m, 2.0, last_seen_at=t)], fixes_at.get(s, []))
            outcomes.extend(res.outcomes)
        return outcomes

    def test_dual_mode_ble_and_uwb_rooms_never_disagree(self) -> None:
        # BLE nearest receiver says room 1; UWB solves just across the doorway in room 2.
        fixes = {
            s: [_ble(T0 + timedelta(seconds=s)), _uwb(T0 + timedelta(seconds=s), 6.4, 2.0)]
            for s in range(16)
        }
        outcomes = self._run(fixes, 6.4)
        assert WearableAssociationOutcome.TAG_DISAGREEMENT not in outcomes
        assert WearableAssociationOutcome.ACCEPTED in outcomes

    def test_watch_and_pendant_in_different_rooms_still_disagree(self) -> None:
        fixes = {
            s: [
                _ble(T0 + timedelta(seconds=s)),
                _ble(T0 + timedelta(seconds=s), tag_id="pendant", room_id="2", physical=None),
            ]
            for s in range(16)
        }
        assert WearableAssociationOutcome.TAG_DISAGREEMENT in self._run(fixes, 2.0)

    def test_device_with_split_room_mass_does_not_trip_disagreement(self) -> None:
        # A wide pendant fix centred on the room boundary puts about half its mass in each
        # room: not confident enough to call a disagreement with the watch.
        fixes = {
            s: [
                _ble(T0 + timedelta(seconds=s)),
                _uwb(
                    T0 + timedelta(seconds=s),
                    6.0,
                    2.5,
                    tag_id="pendant",
                    sigma_m=1.5,
                    physical=None,
                ),
            ]
            for s in range(16)
        }
        assert WearableAssociationOutcome.TAG_DISAGREEMENT not in self._run(fixes, 2.0)

    def test_face_conflicts_count_per_physical_device(self) -> None:
        r = _Replay()
        t = T0 + timedelta(seconds=1)
        r.step(t, [], [_ble(t), _uwb(t, 2.0, 2.0)])
        for i, tag in enumerate(("watch-ble", "watch-uwb")):
            r.tracker.record_face_conflict(ph_id="ph-x", tag_id=tag, at=t + timedelta(seconds=i))
        assert r.tracker._is_swap_suspected("watch-ble", t + timedelta(seconds=2)) is False
        r.tracker.record_face_conflict(
            ph_id="ph-x", tag_id="watch-uwb", at=t + timedelta(seconds=2)
        )
        # Three conflicts on one radio, split across its two logical tags.
        assert r.tracker._is_swap_suspected("watch-ble", t + timedelta(seconds=3)) is True


class TestSegmentHold:
    def _burst_in_shared_room(self, r: _Replay, start: int, end: int) -> dict[int, object]:
        evidence: dict[int, object] = {}
        for s in range(start, end):
            t = T0 + timedelta(seconds=s)
            x = 1.0 + 0.3 * ((s - start) % 15)
            r.record("ph-a", t, x, 2.0)
            r.record("ph-b", t, 5.0, 4.5)
            new = [_ble(t)]
            if s - start <= 12:
                new.append(_uwb(t, x, 2.0))
            res = r.step(
                t,
                [_ph("ph-a", x, 2.0, last_seen_at=t), _ph("ph-b", 5.0, 4.5, last_seen_at=t)],
                new,
            )
            evidence[s] = res.evidence_by_ph.get("ph-a")
        return evidence

    def _ble_only(self, r: _Replay, start: int, end: int) -> int | None:
        """Run BLE-only rounds; return the last second whose round still accepted ph-a."""
        last_accepted: int | None = None
        for s in range(start, end):
            t = T0 + timedelta(seconds=s)
            r.record("ph-a", t, 2.0, 2.0)
            r.record("ph-b", t, 5.0, 4.5)
            res = r.step(
                t,
                [_ph("ph-a", 2.0, 2.0, last_seen_at=t), _ph("ph-b", 5.0, 4.5, last_seen_at=t)],
                [_ble(t)],
            )
            if "ph-a" in res.evidence_by_ph:
                last_accepted = s
            if s >= 20:
                # The burst's fixes have left the window, and a shared room gives BLE room
                # evidence to nobody.
                assert res.evidence_by_ph == {}
        return last_accepted

    def test_shared_room_cooldown_holds_segment_and_next_burst_resumes_latched_grade(
        self,
    ) -> None:
        r = _Replay()
        first = self._burst_in_shared_room(r, 0, 18)
        seg = first[12]
        assert seg is not None
        assert seg.attestation_grade == WearableAttestationGrade.TRAJECTORY  # type: ignore[attr-defined]
        self._ble_only(r, 18, 72)
        assert r.closes() == []
        assert r.tracker.active_segment_count == 1

        second = self._burst_in_shared_room(r, 72, 78)
        resumed = [ev for ev in second.values() if ev is not None]
        assert resumed
        # Same segment, grade still latched on the first accepted round of the next burst.
        assert resumed[0].segment_id == seg.segment_id  # type: ignore[attr-defined]
        assert resumed[0].attestation_grade == WearableAttestationGrade.TRAJECTORY  # type: ignore[attr-defined]

    def test_hold_expires_and_segment_ends_at_its_last_accepted_round(self) -> None:
        r = _Replay(WearableConfig(mode="shadow", segment_hold_s=20.0))
        self._burst_in_shared_room(r, 0, 18)
        last_accepted = self._ble_only(r, 18, 60)
        assert last_accepted is not None
        closes = r.closes()
        assert [c.segment.close_reason for c in closes] == ["association_lost"]
        assert closes[0].segment.ended_at == T0 + timedelta(seconds=last_accepted)

    def test_member_accepted_on_another_ph_closes_held_segment_at_once(self) -> None:
        r = _Replay()
        self._burst_in_shared_room(r, 0, 18)
        # The resident is now tracked as a new PH alone in room 2 (camera handoff).
        for s in range(18, 30):
            t = T0 + timedelta(seconds=s)
            r.record("ph-c", t, 9.0, 2.0)
            res = r.step(
                t,
                [_ph("ph-a", 2.0, 2.0, last_seen_at=t), _ph("ph-c", 9.0, 2.0, last_seen_at=t)],
                [_ble(t, room_id="2")],
            )
        assert "ph-c" in res.evidence_by_ph
        assert [c.segment.close_reason for c in r.closes()] == ["member_moved"]


class TestRoomIdNamespace:
    def _uncalibrated_round(
        self, camera_room_ids: dict[str, str], fixes: list[WearableFix]
    ) -> AssociationRoundResult:
        r = _Replay()
        res = None
        for s in range(10):
            t = T0 + timedelta(seconds=s)
            # A PH on an uncalibrated camera sits on its virtual tile, far outside every
            # polygon, and has no calibrated floor history.
            ph = _ph("ph-u", 40_000.0, 512.0, last_seen_at=t)
            res = r.step(
                t,
                [ph],
                [f for f in fixes if f.measured_at == t],
                camera_room_ids=camera_room_ids,
            )
        assert res is not None
        return res

    def test_uwb_position_fixes_give_room_evidence_on_uncalibrated_camera(self) -> None:
        fixes = [_uwb(T0 + timedelta(seconds=s), 2.0, 2.0, room_id="1") for s in range(10)]
        res = self._uncalibrated_round(CAMERA_ROOM_IDS, fixes)
        ev = res.evidence_by_ph["ph-u"]
        assert ev.evidence_class == WearableEvidenceClass.ROOM
        assert ev.window_llr > 0.0
        assert ev.identity_id == "amma"

    def test_ble_room_fixes_give_room_evidence_on_uncalibrated_camera(self) -> None:
        fixes = [_ble(T0 + timedelta(seconds=s)) for s in range(10)]
        res = self._uncalibrated_round(CAMERA_ROOM_IDS, fixes)
        assert res.evidence_by_ph["ph-u"].evidence_class == WearableEvidenceClass.ROOM

    def test_room_names_never_match_fix_room_ids(self) -> None:
        # The tracker's camera-to-room-name map is the wrong namespace: no evidence at all.
        fixes = [_ble(T0 + timedelta(seconds=s)) for s in range(10)]
        res = self._uncalibrated_round({"cam-1": "Living Room"}, fixes)
        assert res.evidence_by_ph == {}

    def test_room_ids_match_without_polygons(self) -> None:
        # CTS drops room polygons when the floor-plan scale is missing; fix and camera room
        # ids still meet.
        r = _Replay()
        res = None
        for s in range(10):
            t = T0 + timedelta(seconds=s)
            res = r.step(
                t,
                [_ph("ph-u", 40_000.0, 512.0, last_seen_at=t)],
                [_uwb(t, 2.0, 2.0, room_id="1")],
                rooms={},
            )
        assert res is not None
        assert res.evidence_by_ph["ph-u"].evidence_class == WearableEvidenceClass.ROOM

    def test_respawned_ph_waits_for_motion_when_watch_already_in_room(self) -> None:
        # PH churn: a seated resident's PH closes and respawns while the watch sits still in
        # the room. The pre-existing-tag guard withholds room evidence until the watch moves.
        r = _Replay()
        for s in range(0, 60, 5):
            t = T0 + timedelta(seconds=s)
            r.step(t, [], [_ble(t, motion=MotionState.STILL)])
        born = T0 + timedelta(seconds=60)
        res = None
        for s in range(60, 70):
            t = T0 + timedelta(seconds=s)
            ph = _ph("ph-new", 40_000.0, 512.0, last_seen_at=t, born_at=born)
            res = r.step(t, [ph], [_ble(t, motion=MotionState.STILL)])
        assert res is not None
        assert res.evidence_by_ph == {}
        for s in range(70, 78):
            t = T0 + timedelta(seconds=s)
            ph = _ph("ph-new", 40_000.0, 512.0, last_seen_at=t, born_at=born)
            res = r.step(t, [ph], [_ble(t)])
        assert "ph-new" in res.evidence_by_ph
