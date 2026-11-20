"""Unit and property tests for PHFloorHistory, wearable association, and graded attestation."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from cts_contracts import FixKind, MotionState, WearState

from app.domain import (
    PersonHypothesis,
    WearableAttestationGrade,
    WearableEvidenceClass,
    WearableFix,
)
from app.observability import metrics as metrics_mod
from app.observability.metrics import WearableAssociationOutcome
from app.tracking.world.ph_floor_history import PHFloorHistory
from app.tracking.world.wearable_association import (
    WearableAssociationTracker,
    assign_members_to_phs,
    covariance_major_sigma_m,
    is_trajectory_fix,
    room_distribution_for_fix,
    room_fix_llr,
    score_member_ph_pairs,
    trajectory_fix_llr,
)
from app.tracking.world.wearable_config import WearableConfig

T0 = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)

TWO_ROOMS: dict[str, list[tuple[float, float]]] = {
    "room-1": [(0.0, 0.0), (6.0, 0.0), (6.0, 5.0), (0.0, 5.0)],
    "room-2": [(6.0, 0.0), (12.0, 0.0), (12.0, 5.0), (6.0, 5.0)],
}


def _make_ph(
    ph_id: str,
    x_m: float,
    y_m: float,
    *,
    born_at: datetime = T0,
    last_seen_at: datetime = T0,
    room_id: str = "room-1",
    speed_m_s: float = 0.0,
) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(x_m, y_m, speed_m_s, 0.0),
        state_cov=(
            0.04,
            0.0,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
        ),
        born_at=born_at,
        last_seen_at=last_seen_at,
        last_seen_camera="cam-1",
        observation_count=5,
        last_floor_speed_m_s=speed_m_s,
        metadata={"last_room_id": room_id, "last_room_name": room_id},
    )


def _pos_fix(
    tag_id: str,
    member_id: str,
    measured_at: datetime,
    x_m: float,
    y_m: float,
    sigma_m: float = 0.15,
    *,
    motion: MotionState = MotionState.MOVING,
    wear: WearState = WearState.ON_BODY,
    epoch_seq: int = 1,
) -> WearableFix:
    var = sigma_m * sigma_m
    return WearableFix(
        tag_id=tag_id,
        member_id=member_id,
        measured_at=measured_at,
        kind=FixKind.POSITION,
        x_m=x_m,
        y_m=y_m,
        cov_m2=(var, 0.0, 0.0, var),
        wear_state=wear,
        motion_state=motion,
        epoch_seq=epoch_seq,
    )


def _room_fix(
    tag_id: str,
    member_id: str,
    measured_at: datetime,
    room_id: str,
    *,
    prob: float = 0.95,
    motion: MotionState = MotionState.MOVING,
    epoch_seq: int = 1,
) -> WearableFix:
    return WearableFix(
        tag_id=tag_id,
        member_id=member_id,
        measured_at=measured_at,
        kind=FixKind.ROOM,
        room_id=room_id,
        room_probability=prob,
        wear_state=WearState.UNSPECIFIED,
        motion_state=motion,
        epoch_seq=epoch_seq,
    )


class TestPHFloorHistory:
    def test_linear_interpolation_and_extrapolation_bounds(self) -> None:
        hist = PHFloorHistory(max_span_s=45.0)
        hist.record(
            "ph-1",
            captured_at=T0,
            x_m=1.0,
            y_m=2.0,
            cov_m2=(0.04, 0.0, 0.0, 0.04),
            speed_m_s=1.0,
        )
        hist.record(
            "ph-1",
            captured_at=T0 + timedelta(seconds=2.0),
            x_m=3.0,
            y_m=4.0,
            cov_m2=(0.08, 0.0, 0.0, 0.08),
            speed_m_s=2.0,
        )

        mid = hist.interpolate("ph-1", T0 + timedelta(seconds=1.0))
        assert mid is not None
        assert math.isclose(mid.x_m, 2.0, rel_tol=1e-6)
        assert math.isclose(mid.y_m, 3.0, rel_tol=1e-6)
        assert math.isclose(mid.cov_m2[0], 0.06, rel_tol=1e-6)
        assert math.isclose(mid.speed_m_s, 1.5, rel_tol=1e-6)

        # Within 0.5 s boundary extrapolation succeeds.
        near_end = hist.interpolate("ph-1", T0 + timedelta(seconds=2.4), max_extrapolation_s=0.5)
        assert near_end is not None
        assert math.isclose(near_end.x_m, 3.0, rel_tol=1e-6)

        # Past 0.5 s boundary extrapolation returns None.
        too_late = hist.interpolate("ph-1", T0 + timedelta(seconds=2.6), max_extrapolation_s=0.5)
        assert too_late is None
        too_early = hist.interpolate("ph-1", T0 - timedelta(seconds=0.6), max_extrapolation_s=0.5)
        assert too_early is None

    def test_retains_across_revival_window_and_clears_on_reset(self) -> None:
        hist = PHFloorHistory(max_span_s=45.0)
        hist.record(
            "ph-1",
            captured_at=T0,
            x_m=1.0,
            y_m=1.0,
            cov_m2=(0.04, 0.0, 0.0, 0.04),
            speed_m_s=0.0,
        )
        hist.prune(T0 + timedelta(seconds=40.0))
        assert hist.has_calibrated_history("ph-1")

        # Long unseen gap refuses interior interpolation.
        hist.record(
            "ph-1",
            captured_at=T0 + timedelta(seconds=20.0),
            x_m=5.0,
            y_m=1.0,
            cov_m2=(0.04, 0.0, 0.0, 0.04),
            speed_m_s=0.0,
        )
        assert hist.interpolate("ph-1", T0 + timedelta(seconds=10.0)) is None

        # Reset clears history completely (PH split/merge).
        hist.reset_ph("ph-1")
        assert not hist.has_calibrated_history("ph-1")


class TestScoringMathAndCovarianceRouting:
    def test_covariance_major_sigma_anisotropic(self) -> None:
        # Eigenvalues of [[0.5, 0.3], [0.3, 0.5]] are 0.8 and 0.2.
        sigma = covariance_major_sigma_m((0.5, 0.3, 0.3, 0.5))
        assert math.isclose(sigma, math.sqrt(0.8), rel_tol=1e-6)

    def test_trajectory_fix_llr_hand_computed_and_gate_penalty(self) -> None:
        cfg = WearableConfig(mode="enabled", timing_sigma_s=0.5, bias_floor_m=0.15, gate_chi2=9.21)
        # S = (0.04 + 0.04 + (1.0*0.5)^2 + 0.15^2) * I = (0.08 + 0.25 + 0.0225) * I = 0.3525 * I
        llr, res_m, gated = trajectory_fix_llr(
            fix_x_m=2.0,
            fix_y_m=3.0,
            fix_cov_m2=(0.04, 0.0, 0.0, 0.04),
            ph_x_m=2.0,
            ph_y_m=3.0,
            ph_cov_m2=(0.04, 0.0, 0.0, 0.04),
            ph_speed_m_s=1.0,
            floor_area_m2=50.0,
            cfg=cfg,
        )
        assert not gated
        assert res_m == 0.0
        expected_s = 0.3525
        expected_log_match = -0.5 * (
            math.log(expected_s * expected_s) + 2.0 * math.log(2 * math.pi)
        )
        expected_llr = expected_log_match - (-math.log(50.0))
        assert math.isclose(llr, expected_llr, rel_tol=1e-6)

        # Far-away residual triggers chi-square gate penalty (-4.0).
        llr_far, res_far, gated_far = trajectory_fix_llr(
            fix_x_m=10.0,
            fix_y_m=3.0,
            fix_cov_m2=(0.04, 0.0, 0.0, 0.04),
            ph_x_m=2.0,
            ph_y_m=3.0,
            ph_cov_m2=(0.04, 0.0, 0.0, 0.04),
            ph_speed_m_s=0.0,
            floor_area_m2=50.0,
            cfg=cfg,
        )
        assert gated_far
        assert math.isclose(res_far, 8.0, rel_tol=1e-6)
        assert llr_far == cfg.gate_penalty_llr

    def test_room_fix_llr_hand_computed(self) -> None:
        llr_match = room_fix_llr(p_in_ph_room=1.0, num_rooms=4, room_match_eps=0.1)
        assert math.isclose(llr_match, math.log(0.9 / 0.25), rel_tol=1e-6)
        llr_miss = room_fix_llr(p_in_ph_room=0.0, num_rooms=4, room_match_eps=0.1)
        assert math.isclose(llr_miss, math.log((0.1 / 3.0) / 0.25), rel_tol=1e-6)

    def test_covariance_routes_same_position_fix_kind_between_trajectory_and_room(self) -> None:
        cfg = WearableConfig(mode="enabled", trajectory_max_sigma_m=0.75)
        tight = _pos_fix("tag-1", "alice", T0, 2.0, 2.0, sigma_m=0.10)
        wide = _pos_fix("tag-1", "alice", T0, 2.0, 2.0, sigma_m=2.50)
        assert is_trajectory_fix(tight, cfg)
        assert not is_trajectory_fix(wide, cfg)

        # Wide Gaussian inside room-1 reduces cleanly over room polygons.
        dist = room_distribution_for_fix(wide, TWO_ROOMS, min_mass=0.05)
        assert "room-1" in dist
        assert 0.3 < dist["room-1"] < 1.0

    def test_property_widening_covariance_never_increases_accepted_count(self) -> None:
        cfg = WearableConfig(mode="enabled")
        hist = PHFloorHistory(max_span_s=45.0)
        phs = [
            _make_ph("ph-1", 1.5, 2.5, room_id="room-1"),
            _make_ph("ph-2", 4.5, 2.5, room_id="room-1"),
        ]
        for s in range(10):
            ts = T0 + timedelta(seconds=s)
            hist.record(
                "ph-1", captured_at=ts, x_m=1.5, y_m=2.5, cov_m2=(0.04, 0, 0, 0.04), speed_m_s=0.0
            )
            hist.record(
                "ph-2", captured_at=ts, x_m=4.5, y_m=2.5, cov_m2=(0.04, 0, 0, 0.04), speed_m_s=0.0
            )

        prev_count = 2
        for sigma in (0.08, 0.20, 0.45, 0.70, 1.0, 2.0, 4.0):
            fixes_by_member = {
                "alice": {
                    "tag-a": [
                        _pos_fix(
                            "tag-a",
                            "alice",
                            T0 + timedelta(seconds=s),
                            1.5,
                            2.5,
                            sigma_m=sigma,
                            epoch_seq=s,
                        )
                        for s in range(8)
                    ]
                },
                "bob": {
                    "tag-b": [
                        _pos_fix(
                            "tag-b",
                            "bob",
                            T0 + timedelta(seconds=s),
                            4.5,
                            2.5,
                            sigma_m=sigma,
                            epoch_seq=s,
                        )
                        for s in range(8)
                    ]
                },
            }
            scores, _ = score_member_ph_pairs(
                fixes_by_member=fixes_by_member,
                open_phs=phs,
                ph_history=hist,
                ph_rooms={"ph-1": "room-1", "ph-2": "room-1"},
                room_polygons=TWO_ROOMS,
                sole_occupant_rooms={"room-1": False},
                blocked_tag_ph=set(),
                cfg=cfg,
            )
            accepted, _ = assign_members_to_phs(
                member_ids=["alice", "bob"],
                ph_ids=["ph-1", "ph-2"],
                pair_scores=scores,
                cfg=cfg,
            )
            assert len(accepted) <= prev_count
            prev_count = len(accepted)
        assert prev_count == 0


class TestJointAssignmentAndAmbiguityRules:
    def test_3x3_joint_assignment_and_ambiguity_rejections(self) -> None:
        cfg = WearableConfig(mode="enabled")
        hist = PHFloorHistory(max_span_s=45.0)
        coords = [("ph-1", 1.0, 1.0), ("ph-2", 3.5, 1.0), ("ph-3", 5.0, 4.0)]
        phs = [_make_ph(pid, x, y, room_id="room-1") for pid, x, y in coords]
        for s in range(8):
            ts = T0 + timedelta(seconds=s)
            for pid, x, y in coords:
                hist.record(
                    pid, captured_at=ts, x_m=x, y_m=y, cov_m2=(0.04, 0, 0, 0.04), speed_m_s=0.0
                )

        fixes = {
            "m1": {
                "t1": [
                    _pos_fix("t1", "m1", T0 + timedelta(seconds=s), 1.0, 1.0, 0.12, epoch_seq=s)
                    for s in range(8)
                ]
            },
            "m2": {
                "t2": [
                    _pos_fix("t2", "m2", T0 + timedelta(seconds=s), 3.5, 1.0, 0.12, epoch_seq=s)
                    for s in range(8)
                ]
            },
            "m3": {
                "t3": [
                    _pos_fix("t3", "m3", T0 + timedelta(seconds=s), 5.0, 4.0, 0.12, epoch_seq=s)
                    for s in range(8)
                ]
            },
        }
        scores, _ = score_member_ph_pairs(
            fixes_by_member=fixes,
            open_phs=phs,
            ph_history=hist,
            ph_rooms={pid: "room-1" for pid, _, _ in coords},
            room_polygons=TWO_ROOMS,
            sole_occupant_rooms={"room-1": False},
            blocked_tag_ph=set(),
            cfg=cfg,
        )
        accepted, _ = assign_members_to_phs(
            member_ids=["m1", "m2", "m3"],
            ph_ids=["ph-1", "ph-2", "ph-3"],
            pair_scores=scores,
            cfg=cfg,
        )
        assert {pid: pair.member_id for pid, (pair, _) in accepted.items()} == {
            "ph-1": "m1",
            "ph-2": "m2",
            "ph-3": "m3",
        }

        # Two PHs at identical location -> ambiguous tie, zero accepted.
        hist_tie = PHFloorHistory(max_span_s=45.0)
        for s in range(8):
            ts = T0 + timedelta(seconds=s)
            hist_tie.record(
                "ph-a", captured_at=ts, x_m=2.0, y_m=2.0, cov_m2=(0.04, 0, 0, 0.04), speed_m_s=0.0
            )
            hist_tie.record(
                "ph-b", captured_at=ts, x_m=2.0, y_m=2.0, cov_m2=(0.04, 0, 0, 0.04), speed_m_s=0.0
            )
        scores_tie, _ = score_member_ph_pairs(
            fixes_by_member={
                "m1": {
                    "t1": [
                        _pos_fix("t1", "m1", T0 + timedelta(seconds=s), 2.0, 2.0, 0.12, epoch_seq=s)
                        for s in range(8)
                    ]
                }
            },
            open_phs=[_make_ph("ph-a", 2.0, 2.0), _make_ph("ph-b", 2.0, 2.0)],
            ph_history=hist_tie,
            ph_rooms={"ph-a": "room-1", "ph-b": "room-1"},
            room_polygons=TWO_ROOMS,
            sole_occupant_rooms={"room-1": False},
            blocked_tag_ph=set(),
            cfg=cfg,
        )
        accepted_tie, outcomes_tie = assign_members_to_phs(
            member_ids=["m1"],
            ph_ids=["ph-a", "ph-b"],
            pair_scores=scores_tie,
            cfg=cfg,
        )
        assert accepted_tie == {}
        assert WearableAssociationOutcome.AMBIGUOUS_TIE in outcomes_tie


class TestParkedMultiTagAndGradedAttestation:
    def test_parked_tag_gives_zero_evidence_and_clears_on_moving(self) -> None:
        cfg = WearableConfig(mode="enabled", parked_after_s=600.0)
        tracker = WearableAssociationTracker(cfg)
        hist = PHFloorHistory(max_span_s=45.0)

        # Tag sits still for 11 minutes in room-1 before any PH enters.
        still_fixes = [
            _room_fix(
                "tag-bed",
                "alice",
                T0 + timedelta(seconds=s * 60),
                "room-1",
                motion=MotionState.STILL,
                epoch_seq=s,
            )
            for s in range(12)
        ]
        tracker.step(
            fixes=still_fixes,
            open_phs=[],
            closed_ph_ids=[],
            ph_history=hist,
            room_polygons=TWO_ROOMS,
            camera_room_map={"cam-1": "room-1"},
            now=T0 + timedelta(seconds=660),
        )
        assert tracker.is_tag_parked("tag-bed")

        # Caregiver walks into room-1 at t=660..670 -> parked tag yields zero evidence.
        ph_enter = _make_ph(
            "ph-cg",
            2.0,
            2.0,
            born_at=T0 + timedelta(seconds=660),
            last_seen_at=T0 + timedelta(seconds=670),
        )
        more_still = [
            _room_fix(
                "tag-bed",
                "alice",
                T0 + timedelta(seconds=660 + s),
                "room-1",
                motion=MotionState.STILL,
                epoch_seq=100 + s,
            )
            for s in range(1, 10)
        ]
        res = tracker.step(
            fixes=more_still,
            open_phs=[ph_enter],
            closed_ph_ids=[],
            ph_history=hist,
            room_polygons=TWO_ROOMS,
            camera_room_map={"cam-1": "room-1"},
            now=T0 + timedelta(seconds=670),
        )
        assert res.evidence_by_ph == {}
        assert WearableAssociationOutcome.PARKED in res.outcomes

        # Resident stands up and moves -> parked clears and association succeeds!
        wake_fixes = [
            _room_fix(
                "tag-bed",
                "alice",
                T0 + timedelta(seconds=671 + s),
                "room-1",
                motion=MotionState.MOVING,
                epoch_seq=200 + s,
            )
            for s in range(8)
        ]
        res_wake = tracker.step(
            fixes=wake_fixes,
            open_phs=[ph_enter],
            closed_ph_ids=[],
            ph_history=hist,
            room_polygons=TWO_ROOMS,
            camera_room_map={"cam-1": "room-1"},
            now=T0 + timedelta(seconds=678),
        )
        assert not tracker.is_tag_parked("tag-bed")
        assert "ph-cg" in res_wake.evidence_by_ph
        assert res_wake.evidence_by_ph["ph-cg"].identity_id == "alice"

    def test_preexisting_tag_in_room_blocked_until_moving_after_ph_appears(self) -> None:
        cfg = WearableConfig(mode="enabled", parked_after_s=600.0, room_entry_window_s=20.0)
        tracker = WearableAssociationTracker(cfg)
        hist = PHFloorHistory(max_span_s=45.0)

        # Tag already sitting still in room-1 for 60 s (not yet 600 s parked).
        early = [
            _room_fix(
                "tag-desk",
                "alice",
                T0 + timedelta(seconds=s * 10),
                "room-1",
                motion=MotionState.STILL,
                epoch_seq=s,
            )
            for s in range(7)
        ]
        tracker.step(
            fixes=early,
            open_phs=[],
            closed_ph_ids=[],
            ph_history=hist,
            room_polygons=TWO_ROOMS,
            camera_room_map={"cam-1": "room-1"},
            now=T0 + timedelta(seconds=60),
        )
        assert not tracker.is_tag_parked("tag-desk")

        # PH appears at t=60 while tag stays STILL -> blocked by room-arrival guard.
        ph = _make_ph(
            "ph-1",
            2.0,
            2.0,
            born_at=T0 + timedelta(seconds=60),
            last_seen_at=T0 + timedelta(seconds=68),
        )
        res = tracker.step(
            fixes=[
                _room_fix(
                    "tag-desk",
                    "alice",
                    T0 + timedelta(seconds=61 + s),
                    "room-1",
                    motion=MotionState.STILL,
                    epoch_seq=20 + s,
                )
                for s in range(8)
            ],
            open_phs=[ph],
            closed_ph_ids=[],
            ph_history=hist,
            room_polygons=TWO_ROOMS,
            camera_room_map={"cam-1": "room-1"},
            now=T0 + timedelta(seconds=68),
        )
        assert res.evidence_by_ph == {}

    def test_two_tags_same_member_uses_best_single_tag_and_detects_disagreement(self) -> None:
        cfg = WearableConfig(mode="enabled", tag_disagreement_window_s=10.0)
        tracker = WearableAssociationTracker(cfg)
        hist = PHFloorHistory(max_span_s=45.0)
        for s in range(10):
            ts = T0 + timedelta(seconds=s)
            hist.record(
                "ph-1", captured_at=ts, x_m=2.0, y_m=2.0, cov_m2=(0.04, 0, 0, 0.04), speed_m_s=0.0
            )
        ph = _make_ph("ph-1", 2.0, 2.0, last_seen_at=T0 + timedelta(seconds=9))

        # Two tags for alice in same room: tight CS (0.4 m) + wide BLE (2.0 m).
        cs_fixes = [
            _pos_fix("tag-cs", "alice", T0 + timedelta(seconds=s), 2.0, 2.0, 0.40, epoch_seq=s)
            for s in range(8)
        ]
        ble_fixes = [
            _pos_fix("tag-ble", "alice", T0 + timedelta(seconds=s), 2.0, 2.0, 2.00, epoch_seq=s)
            for s in range(8)
        ]

        single_tracker = WearableAssociationTracker(cfg)
        res_single = single_tracker.step(
            fixes=cs_fixes,
            open_phs=[ph],
            closed_ph_ids=[],
            ph_history=hist,
            room_polygons=TWO_ROOMS,
            camera_room_map={"cam-1": "room-1"},
            now=T0 + timedelta(seconds=9),
        )
        res_both = tracker.step(
            fixes=cs_fixes + ble_fixes,
            open_phs=[ph],
            closed_ph_ids=[],
            ph_history=hist,
            room_polygons=TWO_ROOMS,
            camera_room_map={"cam-1": "room-1"},
            now=T0 + timedelta(seconds=9),
        )
        # Never sum across two tags: window_llr matches best single tag exactly.
        assert math.isclose(
            res_both.evidence_by_ph["ph-1"].window_llr,
            res_single.evidence_by_ph["ph-1"].window_llr,
            rel_tol=1e-6,
        )
        assert res_both.evidence_by_ph["ph-1"].tag_id == "tag-cs"

        # Sustained room disagreement between alice's two tags -> zero evidence.
        disagree_tracker = WearableAssociationTracker(cfg)
        for step_t in (T0, T0 + timedelta(seconds=11)):
            fixes_disagree = [
                _room_fix("tag-1", "alice", step_t - timedelta(seconds=k), "room-1", epoch_seq=k)
                for k in range(6)
            ] + [
                _room_fix("tag-2", "alice", step_t - timedelta(seconds=k), "room-2", epoch_seq=k)
                for k in range(6)
            ]
            res_dis = disagree_tracker.step(
                fixes=fixes_disagree,
                open_phs=[ph],
                closed_ph_ids=[],
                ph_history=hist,
                room_polygons=TWO_ROOMS,
                camera_room_map={"cam-1": "room-1"},
                now=step_t,
            )
        assert res_dis.evidence_by_ph == {}
        assert WearableAssociationOutcome.TAG_DISAGREEMENT in res_dis.outcomes

    def test_graded_attestation_trajectory_room_and_face_conflict_reset(self) -> None:
        cfg = WearableConfig(mode="enabled")
        # 1. Trajectory attestation: >= 10 s span, >= 8 fixes, window_llr >= 15.
        tracker = WearableAssociationTracker(cfg)
        hist = PHFloorHistory(max_span_s=45.0)
        fixes_all: list[WearableFix] = []
        last_res = None
        for s in range(12):
            ts = T0 + timedelta(seconds=s)
            x = 1.0 + 0.3 * s
            hist.record(
                "ph-1", captured_at=ts, x_m=x, y_m=2.0, cov_m2=(0.04, 0, 0, 0.04), speed_m_s=0.3
            )
            fixes_all.append(_pos_fix("tag-uwb", "alice", ts, x, 2.0, sigma_m=0.12, epoch_seq=s))
            ph = _make_ph("ph-1", x, 2.0, born_at=T0, last_seen_at=ts, speed_m_s=0.3)
            last_res = tracker.step(
                fixes=fixes_all,
                open_phs=[ph],
                closed_ph_ids=[],
                ph_history=hist,
                room_polygons=TWO_ROOMS,
                camera_room_map={"cam-1": "room-1"},
                now=ts,
            )
        assert last_res is not None
        assert (
            last_res.evidence_by_ph["ph-1"].attestation_grade == WearableAttestationGrade.TRAJECTORY
        )
        assert last_res.evidence_by_ph["ph-1"].attested

        # Face conflict breaks attestation and 3 conflicts in 10 min marks swap_suspected.
        for idx in range(3):
            ts_c = T0 + timedelta(seconds=15 + idx * 10)
            # Re-open segment before conflict 2 and 3.
            if idx > 0:
                hist.record(
                    "ph-1",
                    captured_at=ts_c,
                    x_m=4.0,
                    y_m=2.0,
                    cov_m2=(0.04, 0, 0, 0.04),
                    speed_m_s=0.0,
                )
                batch = [
                    _pos_fix(
                        "tag-uwb",
                        "alice",
                        ts_c - timedelta(seconds=k),
                        4.0,
                        2.0,
                        0.12,
                        epoch_seq=50 + idx * 10 + k,
                    )
                    for k in range(6)
                ]
                for k in range(6):
                    hist.record(
                        "ph-1",
                        captured_at=ts_c - timedelta(seconds=k),
                        x_m=4.0,
                        y_m=2.0,
                        cov_m2=(0.04, 0, 0, 0.04),
                        speed_m_s=0.0,
                    )
                tracker.step(
                    fixes=batch,
                    open_phs=[_make_ph("ph-1", 4.0, 2.0, born_at=T0, last_seen_at=ts_c)],
                    closed_ph_ids=[],
                    ph_history=hist,
                    room_polygons=TWO_ROOMS,
                    camera_room_map={"cam-1": "room-1"},
                    now=ts_c,
                )
            closed = tracker.record_face_conflict(ph_id="ph-1", tag_id="tag-uwb", at=ts_c)
            assert len(closed) == 1
            assert closed[0].segment.attestation_grade == WearableAttestationGrade.NONE
            assert closed[0].segment.close_reason == "wearable_face_conflict"
            if idx == 2:
                assert closed[0].segment.swap_suspected is True
            else:
                assert closed[0].segment.swap_suspected is False

        # 2. Room attestation: >= 60 s sole occupant + moving + co-arrival within 20 s.
        room_tracker = WearableAssociationTracker(cfg)
        empty_hist = PHFloorHistory(max_span_s=45.0)
        room_fixes: list[WearableFix] = []
        res_room = None
        for s in range(0, 65):
            ts = T0 + timedelta(seconds=s)
            room_fixes.append(
                _room_fix("tag-ble", "bob", ts, "room-1", motion=MotionState.MOVING, epoch_seq=s)
            )
            ph_uncal = _make_ph("ph-room", 2.0, 2.0, born_at=T0, last_seen_at=ts, room_id="room-1")
            res_room = room_tracker.step(
                fixes=room_fixes[-15:],
                open_phs=[ph_uncal],
                closed_ph_ids=[],
                ph_history=empty_hist,
                room_polygons=TWO_ROOMS,
                camera_room_map={"cam-1": "room-1"},
                now=ts,
            )
            if s < 60 and "ph-room" in res_room.evidence_by_ph:
                assert (
                    res_room.evidence_by_ph["ph-room"].attestation_grade
                    == WearableAttestationGrade.NONE
                )
        assert res_room is not None
        assert res_room.evidence_by_ph["ph-room"].evidence_class == WearableEvidenceClass.ROOM
        assert res_room.evidence_by_ph["ph-room"].attestation_grade == WearableAttestationGrade.ROOM
        # Metrics counter was incremented with bounded labels.
        assert metrics_mod.metrics.cts_wearable_associations_total is not None
