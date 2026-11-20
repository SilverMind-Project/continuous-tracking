"""Pure wearable-to-PH association, covariance routing, and graded attestation.

No I/O, no database calls, no wall-clock ``datetime.now()`` reads, and no
gallery/appearance inputs: association and attestation are derived solely from
wearable localization fixes, calibrated PH floor histories, room occupancy,
and explicit face-conflict notifications.

Evidence class (``trajectory`` vs ``room``) is selected solely from fix kind
and the major-axis standard deviation of the 2x2 position covariance; this
module never branches on hardware vendor, adapter kind, or radio technology.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np
from cts_contracts import FixKind, MotionState
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon
from structlog import get_logger

from app.domain import (
    PersonHypothesis,
    WearableAssociationSegment,
    WearableAssociationTransition,
    WearableAttestationGrade,
    WearableEvidence,
    WearableEvidenceClass,
    WearableFix,
)
from app.observability import metrics as _metrics
from app.observability.metrics import WearableAssociationOutcome
from app.tracking.world.helpers import resolve_room
from app.tracking.world.ph_floor_history import PHFloorHistory
from app.tracking.world.wearable_config import WearableConfig

logger = get_logger(__name__)

# Precomputed 7x7 Gauss-Hermite quadrature grid for deterministic 2D Gaussian
# integration over arbitrary room polygons (no Monte Carlo randomness).
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(7)
_GH_POINTS_2D: np.ndarray = np.array(
    [(math.sqrt(2.0) * xi, math.sqrt(2.0) * yj) for xi in _GH_X for yj in _GH_X],
    dtype=np.float64,
)
_GH_WEIGHTS_2D: np.ndarray = np.array(
    [(wi * wj) / math.pi for wi in _GH_W for wj in _GH_W],
    dtype=np.float64,
)
_GH_WEIGHTS_2D /= float(np.sum(_GH_WEIGHTS_2D))

_INFEASIBLE_COST: float = 1e6
_DEFAULT_FLOOR_AREA_M2: float = 50.0
_MIN_EFFECTIVE_ROOMS: int = 4
_STILL_COVARIANCE_CHI2_95: float = 5.991
_TIE_TOL: float = 1e-6


def covariance_major_sigma_m(cov_m2: tuple[float, float, float, float]) -> float:
    """Return ``sqrt(lambda_max(Sigma))`` in metres for a 2x2 symmetric covariance."""
    a, b, _, c = cov_m2
    tr = a + c
    disc = max(0.0, (a - c) * (a - c) + 4.0 * b * b)
    lam_max = 0.5 * (tr + math.sqrt(disc))
    return math.sqrt(max(lam_max, 0.0))


def is_trajectory_fix(fix: WearableFix, cfg: WearableConfig) -> bool:
    """Return True when *fix* qualifies for trajectory-class scoring by covariance."""
    return (
        fix.kind == FixKind.POSITION
        and fix.x_m is not None
        and fix.y_m is not None
        and fix.cov_m2 is not None
        and covariance_major_sigma_m(fix.cov_m2) <= cfg.trajectory_max_sigma_m
    )


def room_distribution_for_fix(
    fix: WearableFix,
    room_polygons: Mapping[str, list[tuple[float, float]]],
    *,
    min_mass: float = 0.05,
) -> dict[str, float]:
    """Reduce a wearable fix to a room probability distribution.

    - ``FIX_KIND_ROOM`` and ``FIX_KIND_PROXIMITY`` with ``room_id`` map directly
      to ``{room_id: room_probability or 1.0}``.
    - ``FIX_KIND_POSITION`` integrates the 2D Gaussian ``N(mu, Sigma)`` over
      ``room_polygons`` using deterministic 7x7 Gauss-Hermite quadrature,
      dropping rooms below *min_mass*. When ``room_polygons`` is empty and
      ``fix.room_id`` is populated, falls back to ``fix.room_id``.
    """
    if fix.kind in (FixKind.ROOM, FixKind.PROXIMITY):
        if not fix.room_id:
            return {}
        prob = 1.0 if fix.room_probability is None else max(0.0, min(1.0, fix.room_probability))
        return {fix.room_id: prob} if prob >= min_mass else {}

    if fix.x_m is None or fix.y_m is None or fix.cov_m2 is None:
        if fix.room_id:
            prob = 1.0 if fix.room_probability is None else max(0.0, min(1.0, fix.room_probability))
            return {fix.room_id: prob} if prob >= min_mass else {}
        return {}

    if not room_polygons:
        if fix.room_id:
            prob = 1.0 if fix.room_probability is None else max(0.0, min(1.0, fix.room_probability))
            return {fix.room_id: prob} if prob >= min_mass else {}
        return {}

    cov = np.array(
        [[fix.cov_m2[0], fix.cov_m2[1]], [fix.cov_m2[2], fix.cov_m2[3]]],
        dtype=np.float64,
    )
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Symmetrize and clamp tiny numerical negative eigenvalues if any.
        eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
        eigvals = np.maximum(eigvals, 1e-9)
        chol = eigvecs @ np.diag(np.sqrt(eigvals))

    pts = np.array([fix.x_m, fix.y_m], dtype=np.float64) + (_GH_POINTS_2D @ chol.T)
    masses: dict[str, float] = {}
    for room_id, vertices in room_polygons.items():
        if len(vertices) < 3:
            continue
        poly = Polygon(vertices)
        if poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        mass = 0.0
        for idx in range(pts.shape[0]):
            px = float(pts[idx, 0])
            py = float(pts[idx, 1])
            if minx <= px <= maxx and miny <= py <= maxy and poly.contains(Point(px, py)):
                mass += float(_GH_WEIGHTS_2D[idx])
        if mass >= min_mass:
            masses[room_id] = min(1.0, mass)
    return masses


def trajectory_fix_llr(
    *,
    fix_x_m: float,
    fix_y_m: float,
    fix_cov_m2: tuple[float, float, float, float],
    ph_x_m: float,
    ph_y_m: float,
    ph_cov_m2: tuple[float, float, float, float],
    ph_speed_m_s: float,
    floor_area_m2: float,
    cfg: WearableConfig,
) -> tuple[float, float, bool]:
    """Compute per-fix trajectory LLR, Euclidean residual (m), and gated flag.

    Innovation covariance combines fix covariance, PH position covariance,
    velocity-scaled timing uncertainty, and the bias floor:
    ``S = Sigma_f + Sigma_j + (v_j * sigma_t)^2 I + sigma_bias^2 I``.
    """
    timing_var = (ph_speed_m_s * cfg.timing_sigma_s) ** 2
    bias_var = cfg.bias_floor_m**2
    s00 = fix_cov_m2[0] + ph_cov_m2[0] + timing_var + bias_var
    s01 = fix_cov_m2[1] + ph_cov_m2[1]
    s11 = fix_cov_m2[3] + ph_cov_m2[3] + timing_var + bias_var

    rx = fix_x_m - ph_x_m
    ry = fix_y_m - ph_y_m
    residual_m = math.hypot(rx, ry)

    det = s00 * s11 - s01 * s01
    if det <= 1e-12 or not math.isfinite(det):
        return cfg.gate_penalty_llr, residual_m, True

    inv_det = 1.0 / det
    d2 = (s11 * rx * rx - 2.0 * s01 * rx * ry + s00 * ry * ry) * inv_det
    if not math.isfinite(d2) or d2 > cfg.gate_chi2:
        return cfg.gate_penalty_llr, residual_m, True

    log_p_match = -0.5 * (d2 + math.log(det) + 2.0 * math.log(2.0 * math.pi))
    log_p_bg = -math.log(max(floor_area_m2, 1.0))
    return log_p_match - log_p_bg, residual_m, False


def room_fix_llr(
    *,
    p_in_ph_room: float,
    num_rooms: int,
    room_match_eps: float,
) -> float:
    """Compute per-fix room-class LLR given room mass in the PH's room."""
    k = max(num_rooms, 2)
    p = max(0.0, min(1.0, p_in_ph_room))
    eps = room_match_eps
    p_match = p * (1.0 - eps) + (1.0 - p) * (eps / (k - 1))
    p_bg = 1.0 / k
    return math.log(max(p_match, 1e-9)) - math.log(p_bg)


def _floor_area_m2(room_polygons: Mapping[str, list[tuple[float, float]]]) -> float:
    area = 0.0
    for vertices in room_polygons.values():
        if len(vertices) >= 3:
            area += float(Polygon(vertices).area)
    return area if area >= 1.0 else _DEFAULT_FLOOR_AREA_M2


def _ph_room_id(
    ph: PersonHypothesis,
    room_polygons: Mapping[str, list[tuple[float, float]]],
    camera_room_map: Mapping[str, str],
) -> str:
    meta_room = ph.metadata.get("last_room_id")
    if isinstance(meta_room, str) and meta_room:
        return meta_room
    rid, _ = resolve_room(
        ph.state_mean[0],
        ph.state_mean[1],
        ph.last_seen_camera,
        dict(room_polygons),
        dict(camera_room_map),
    )
    return rid


@dataclass(frozen=True)
class PairScore:
    """Scored candidate pairing between one member's best tag and one open PH."""

    member_id: str
    tag_id: str
    ph_id: str
    evidence_class: WearableEvidenceClass
    window_llr: float
    matched_fixes: int
    mean_residual_m: float | None
    earliest_fix_at: datetime
    newest_fix_at: datetime
    fix_timestamps: tuple[datetime, ...]
    eligible: bool
    reject_outcome: WearableAssociationOutcome | None = None


@dataclass(frozen=True)
class AssociationRoundResult:
    """Output of one tracker-round wearable association evaluation."""

    evidence_by_ph: Mapping[str, WearableEvidence]
    transitions: tuple[WearableAssociationTransition, ...]
    outcomes: tuple[WearableAssociationOutcome, ...]


def score_member_ph_pairs(
    *,
    fixes_by_member: Mapping[str, Mapping[str, Sequence[WearableFix]]],
    open_phs: Sequence[PersonHypothesis],
    ph_history: PHFloorHistory,
    ph_rooms: Mapping[str, str],
    room_polygons: Mapping[str, list[tuple[float, float]]],
    sole_occupant_rooms: Mapping[str, bool],
    blocked_tag_ph: set[tuple[str, str]],
    cfg: WearableConfig,
) -> tuple[dict[tuple[str, str], PairScore], list[WearableAssociationOutcome]]:
    """Score every ``(member_id, ph_id)`` pair using each member's best single tag.

    Never sums across tags of the same member. Trajectory-class scoring is
    preferred whenever a member has at least ``cfg.min_fixes`` trajectory fixes
    and the PH has calibrated floor history; otherwise falls back to room-class
    scoring where room-only rules apply.
    """
    floor_area = _floor_area_m2(room_polygons)
    known_room_ids: set[str] = set(room_polygons.keys()) | {r for r in ph_rooms.values() if r}
    num_rooms = max(len(known_room_ids), _MIN_EFFECTIVE_ROOMS)

    scores: dict[tuple[str, str], PairScore] = {}
    extra_outcomes: list[WearableAssociationOutcome] = []

    for member_id, by_tag in fixes_by_member.items():
        for ph in open_phs:
            ph_id = ph.ph_id
            ph_has_cal = ph_history.has_calibrated_history(ph_id)
            ph_room = ph_rooms.get(ph_id, "")

            best_traj: PairScore | None = None
            best_room: PairScore | None = None
            saw_shared_room_block = False
            saw_gate_reject = False
            saw_insufficient = False

            for tag_id, tag_fixes in by_tag.items():
                # 1. Trajectory-class scoring (when PH is calibrated and tag has >= min_fixes
                #    tight-covariance fixes).
                traj_fixes = [f for f in tag_fixes if is_trajectory_fix(f, cfg)]
                if ph_has_cal and len(traj_fixes) >= cfg.min_fixes:
                    llr_sum = 0.0
                    residuals: list[float] = []
                    matched_ts: list[datetime] = []
                    gated_count = 0
                    for fix in traj_fixes:
                        sample = ph_history.interpolate(
                            ph_id,
                            fix.measured_at,
                            max_extrapolation_s=cfg.max_extrapolation_s,
                        )
                        if sample is None:
                            continue
                        assert fix.x_m is not None
                        assert fix.y_m is not None
                        assert fix.cov_m2 is not None
                        llr, res_m, gated = trajectory_fix_llr(
                            fix_x_m=fix.x_m,
                            fix_y_m=fix.y_m,
                            fix_cov_m2=fix.cov_m2,
                            ph_x_m=sample.x_m,
                            ph_y_m=sample.y_m,
                            ph_cov_m2=sample.cov_m2,
                            ph_speed_m_s=sample.speed_m_s,
                            floor_area_m2=floor_area,
                            cfg=cfg,
                        )
                        llr_sum += llr
                        residuals.append(res_m)
                        matched_ts.append(fix.measured_at)
                        if gated:
                            gated_count += 1

                    if len(matched_ts) >= cfg.min_fixes:
                        eligible = llr_sum >= cfg.min_window_llr
                        reject_reason = (
                            None
                            if eligible
                            else (
                                WearableAssociationOutcome.REJECTED_GATE
                                if gated_count > 0
                                else WearableAssociationOutcome.INSUFFICIENT_FIXES
                            )
                        )
                        if not eligible and gated_count > 0:
                            saw_gate_reject = True
                        cand = PairScore(
                            member_id=member_id,
                            tag_id=tag_id,
                            ph_id=ph_id,
                            evidence_class=WearableEvidenceClass.TRAJECTORY,
                            window_llr=llr_sum,
                            matched_fixes=len(matched_ts),
                            mean_residual_m=sum(residuals) / len(residuals),
                            earliest_fix_at=min(matched_ts),
                            newest_fix_at=max(matched_ts),
                            fix_timestamps=tuple(sorted(matched_ts)),
                            eligible=eligible,
                            reject_outcome=reject_reason,
                        )
                        if best_traj is None or cand.window_llr > best_traj.window_llr:
                            best_traj = cand
                    else:
                        saw_insufficient = True

                # 2. Room-class scoring (for wide-covariance position fixes, room fixes,
                #    or uncalibrated PHs). Never used if trajectory already succeeded for
                #    this (member, ph).
                if ph_room and len(tag_fixes) >= cfg.min_fixes:
                    if (tag_id, ph_id) in blocked_tag_ph:
                        continue
                    total_mass = 0.0
                    llr_sum = 0.0
                    room_ts: list[datetime] = []
                    for fix in tag_fixes:
                        dist = room_distribution_for_fix(
                            fix, room_polygons, min_mass=cfg.room_polygon_min_mass
                        )
                        p_in_room = dist.get(ph_room, 0.0)
                        total_mass += p_in_room
                        llr_sum += room_fix_llr(
                            p_in_ph_room=p_in_room,
                            num_rooms=num_rooms,
                            room_match_eps=cfg.room_match_eps,
                        )
                        room_ts.append(fix.measured_at)

                    min_required_mass = cfg.room_min_mass * len(room_ts)
                    meets_mass = total_mass >= min_required_mass
                    sole_occ = sole_occupant_rooms.get(ph_room, False)
                    if meets_mass and not sole_occ:
                        saw_shared_room_block = True
                        cand_room = PairScore(
                            member_id=member_id,
                            tag_id=tag_id,
                            ph_id=ph_id,
                            evidence_class=WearableEvidenceClass.ROOM,
                            window_llr=llr_sum,
                            matched_fixes=len(room_ts),
                            mean_residual_m=None,
                            earliest_fix_at=min(room_ts),
                            newest_fix_at=max(room_ts),
                            fix_timestamps=tuple(sorted(room_ts)),
                            eligible=False,
                            reject_outcome=WearableAssociationOutcome.UNCALIBRATED_SHARED_ROOM,
                        )
                    else:
                        eligible = meets_mass and sole_occ and llr_sum >= cfg.min_window_llr
                        cand_room = PairScore(
                            member_id=member_id,
                            tag_id=tag_id,
                            ph_id=ph_id,
                            evidence_class=WearableEvidenceClass.ROOM,
                            window_llr=llr_sum,
                            matched_fixes=len(room_ts),
                            mean_residual_m=None,
                            earliest_fix_at=min(room_ts),
                            newest_fix_at=max(room_ts),
                            fix_timestamps=tuple(sorted(room_ts)),
                            eligible=eligible,
                            reject_outcome=(
                                None if eligible else WearableAssociationOutcome.REJECTED_GATE
                            ),
                        )
                    if best_room is None or cand_room.window_llr > best_room.window_llr:
                        best_room = cand_room
                elif len(tag_fixes) < cfg.min_fixes:
                    saw_insufficient = True

            # Prefer trajectory class when available for this (member, ph).
            chosen = best_traj if best_traj is not None else best_room
            if chosen is not None:
                scores[(member_id, ph_id)] = chosen
            elif saw_shared_room_block:
                extra_outcomes.append(WearableAssociationOutcome.UNCALIBRATED_SHARED_ROOM)
            elif saw_gate_reject:
                extra_outcomes.append(WearableAssociationOutcome.REJECTED_GATE)
            elif saw_insufficient:
                extra_outcomes.append(WearableAssociationOutcome.INSUFFICIENT_FIXES)

    return scores, extra_outcomes


def assign_members_to_phs(
    *,
    member_ids: Sequence[str],
    ph_ids: Sequence[str],
    pair_scores: Mapping[tuple[str, str], PairScore],
    cfg: WearableConfig,
) -> tuple[dict[str, tuple[PairScore, float]], list[WearableAssociationOutcome]]:
    """Run joint 1-to-1 linear-sum assignment with dummy columns and both-sided margins.

    Returns ``({ph_id: (winning_pair_score, min_margin_llr)}, outcomes)``.
    """
    if not member_ids or not ph_ids:
        return {}, []

    n_m = len(member_ids)
    n_p = len(ph_ids)
    cost = np.full((n_m, n_p + n_m), _INFEASIBLE_COST, dtype=np.float64)
    for i, mid in enumerate(member_ids):
        cost[i, n_p + i] = 0.0  # unmatched dummy column
        for j, pid in enumerate(ph_ids):
            ps = pair_scores.get((mid, pid))
            if ps is not None and ps.eligible:
                cost[i, j] = -ps.window_llr

    row_ind, col_ind = linear_sum_assignment(cost)

    accepted: dict[str, tuple[PairScore, float]] = {}
    outcomes: list[WearableAssociationOutcome] = []

    for i, j in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
        mid = member_ids[i]
        if j >= n_p:
            # Member stayed unmatched: record diagnostic outcome from its best pair if any.
            member_pairs = [pair_scores[(mid, pid)] for pid in ph_ids if (mid, pid) in pair_scores]
            if member_pairs:
                best_pair = max(member_pairs, key=lambda p: p.window_llr)
                if best_pair.reject_outcome is not None:
                    outcomes.append(best_pair.reject_outcome)
            continue

        pid = ph_ids[j]
        winner = pair_scores.get((mid, pid))
        if winner is None or not winner.eligible:
            continue

        # Tag-side margin: winner vs best competing PH for this member (floor at 0.0 dummy).
        other_ph_llrs = [
            pair_scores[(mid, other_pid)].window_llr
            for other_pid in ph_ids
            if other_pid != pid and (mid, other_pid) in pair_scores
        ]
        second_ph_llr = max(other_ph_llrs, default=0.0)
        tag_margin = winner.window_llr - max(second_ph_llr, 0.0)

        # PH-side margin: winner vs best competing member for this PH (floor at 0.0 dummy).
        other_member_llrs = [
            pair_scores[(other_mid, pid)].window_llr
            for other_mid in member_ids
            if other_mid != mid and (other_mid, pid) in pair_scores
        ]
        second_member_llr = max(other_member_llrs, default=0.0)
        ph_margin = winner.window_llr - max(second_member_llr, 0.0)

        # Check exact tie or insufficient margin on either side.
        if (other_ph_llrs and abs(winner.window_llr - second_ph_llr) <= _TIE_TOL) or (
            other_member_llrs and abs(winner.window_llr - second_member_llr) <= _TIE_TOL
        ):
            outcomes.append(WearableAssociationOutcome.AMBIGUOUS_TIE)
            continue
        if tag_margin < cfg.min_margin_llr:
            outcomes.append(WearableAssociationOutcome.AMBIGUOUS_TWO_PHS)
            continue
        if ph_margin < cfg.min_margin_llr:
            outcomes.append(WearableAssociationOutcome.AMBIGUOUS_TWO_TAGS)
            continue

        min_margin = min(tag_margin, ph_margin)
        accepted[pid] = (winner, min_margin)
        outcomes.append(WearableAssociationOutcome.ACCEPTED)

    return accepted, outcomes


@dataclass
class _TagMotionTracker:
    """Per-tag motion, stillness, room-entry, and parked-state tracker."""

    last_room_id: str | None = None
    room_entered_at: dict[str, datetime] = None  # type: ignore[assignment]
    still_since: datetime | None = None
    last_moving_at: datetime | None = None
    last_motion_state: MotionState = MotionState.UNSPECIFIED
    parked: bool = False
    still_mean_x: float | None = None
    still_mean_y: float | None = None
    still_pos_count: int = 0
    last_processed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.room_entered_at is None:
            self.room_entered_at = {}


@dataclass
class _ActiveSegmentState:
    """In-memory state of an open wearable-to-PH association segment."""

    segment: WearableAssociationSegment
    evidence_class: WearableEvidenceClass
    continuous_first_fix_at: datetime
    distinct_traj_fix_times: set[datetime]
    had_face_conflict: bool = False
    room_id: str = ""
    sole_occupant_throughout: bool = True


class WearableAssociationTracker:
    """Stateful per-tracker coordinator for wearable association and attestation.

    One instance per :class:`WorldTracker`. Maintains parked-tag detection,
    multi-tag disagreement windows, room occupancy history, continuous
    attestation streaks, face-conflict swap detection, and segment transitions.
    """

    def __init__(self, config: WearableConfig) -> None:
        self._cfg = config
        self._tag_motion: dict[str, _TagMotionTracker] = {}
        self._tag_disagreement_since: dict[str, datetime] = {}
        self._ph_room_entered_at: dict[tuple[str, str], datetime] = {}
        self._room_multi_occupant_at: dict[str, datetime] = {}
        self._active_by_ph: dict[str, _ActiveSegmentState] = {}
        self._tag_conflicts: dict[str, deque[datetime]] = {}
        self._last_evidence_by_ph: dict[str, WearableEvidence] = {}

    @property
    def active_segment_count(self) -> int:
        """Return count of currently open wearable association segments."""
        return len(self._active_by_ph)

    def reset_ph(
        self, ph_id: str, *, now: datetime, reason: str = "ph_reset"
    ) -> list[WearableAssociationTransition]:
        """Close any active segment on *ph_id* (for PH split, merge, or close)."""
        state = self._active_by_ph.pop(ph_id, None)
        self._last_evidence_by_ph.pop(ph_id, None)
        if state is None:
            return []
        closed = replace(state.segment, ended_at=now, close_reason=reason)
        return [WearableAssociationTransition(action="close", segment=closed)]

    def record_face_conflict(
        self,
        *,
        ph_id: str,
        tag_id: str,
        at: datetime,
    ) -> list[WearableAssociationTransition]:
        """Record a calibrated-direct-face vs. wearable conflict.

        Breaks attestation on *ph_id*, closes the active segment, and flags
        ``swap_suspected=True`` when ``>= swap_conflict_count`` conflicts occur
        for *tag_id* within ``swap_conflict_window_s``.
        """
        dq = self._tag_conflicts.setdefault(tag_id, deque())
        dq.append(at)
        cutoff = at - timedelta(seconds=self._cfg.swap_conflict_window_s)
        while dq and dq[0] < cutoff:
            dq.popleft()

        swap_suspected = len(dq) >= self._cfg.swap_conflict_count
        if swap_suspected:
            logger.warning(
                "wearable_swap_suspected",
                tag_id=tag_id,
                conflict_count=len(dq),
                window_s=self._cfg.swap_conflict_window_s,
            )

        state = self._active_by_ph.pop(ph_id, None)
        if state is None:
            return []
        closed = replace(
            state.segment,
            ended_at=at,
            attestation_grade=WearableAttestationGrade.NONE,
            swap_suspected=state.segment.swap_suspected or swap_suspected,
            close_reason="wearable_face_conflict",
        )
        return [WearableAssociationTransition(action="close", segment=closed)]

    def is_tag_parked(self, tag_id: str) -> bool:
        """Return True if *tag_id* is currently classified as parked."""
        st = self._tag_motion.get(tag_id)
        return bool(st and st.parked)

    def step(
        self,
        *,
        fixes: Sequence[WearableFix],
        open_phs: Sequence[PersonHypothesis],
        closed_ph_ids: Sequence[str],
        ph_history: PHFloorHistory,
        room_polygons: Mapping[str, list[tuple[float, float]]],
        camera_room_map: Mapping[str, str],
        now: datetime,
    ) -> AssociationRoundResult:
        """Evaluate wearable associations for one tracker round."""
        cfg = self._cfg
        transitions: list[WearableAssociationTransition] = []

        # 1. Close segments on PHs that closed this round.
        for closed_id in closed_ph_ids:
            transitions.extend(self.reset_ph(closed_id, now=now, reason="ph_closed"))

        # 2. Compute current room for every open PH and update room occupancy history.
        ph_rooms: dict[str, str] = {}
        room_ph_counts: dict[str, int] = {}
        for ph in open_phs:
            rid = _ph_room_id(ph, room_polygons, camera_room_map)
            ph_rooms[ph.ph_id] = rid
            if rid:
                room_ph_counts[rid] = room_ph_counts.get(rid, 0) + 1
                self._ph_room_entered_at.setdefault((ph.ph_id, rid), ph.born_at)

        for rid, count in room_ph_counts.items():
            if count >= 2:
                self._room_multi_occupant_at[rid] = now

        # Update sole-occupant flags on open room-class segments.
        for state in self._active_by_ph.values():
            if state.room_id and room_ph_counts.get(state.room_id, 0) >= 2:
                state.sole_occupant_throughout = False

        # Determine which rooms have had a sole open PH over the entire sliding window.
        window_start = now - timedelta(seconds=cfg.window_s)
        sole_occupant_rooms: dict[str, bool] = {}
        for rid, count in room_ph_counts.items():
            last_multi = self._room_multi_occupant_at.get(rid)
            sole_occupant_rooms[rid] = count == 1 and (
                last_multi is None or last_multi < window_start
            )

        # 3. Update parked-tag motion state for all observed fixes in chronological order.
        ordered_fixes = sorted(fixes, key=lambda f: (f.measured_at, f.tag_id, f.epoch_seq))
        for fix in ordered_fixes:
            self._update_tag_motion(fix, room_polygons)

        # 4. Filter fixes to the sliding window [now - window_s, now] and group by member/tag.
        window_fixes = [f for f in ordered_fixes if f.measured_at >= window_start]
        outcomes: list[WearableAssociationOutcome] = []

        raw_by_member: dict[str, dict[str, list[WearableFix]]] = {}
        for fix in window_fixes:
            if self.is_tag_parked(fix.tag_id):
                continue
            raw_by_member.setdefault(fix.member_id, {}).setdefault(fix.tag_id, []).append(fix)

        # Count parked tags that had fixes in the window.
        parked_tags_in_window = {f.tag_id for f in window_fixes if self.is_tag_parked(f.tag_id)}
        for _ in parked_tags_in_window:
            outcomes.append(WearableAssociationOutcome.PARKED)

        # 5. Check multi-tag room disagreement per member.
        fixes_by_member: dict[str, dict[str, list[WearableFix]]] = {}
        for member_id, by_tag in raw_by_member.items():
            if self._check_tag_disagreement(member_id, by_tag, room_polygons, now):
                outcomes.append(WearableAssociationOutcome.TAG_DISAGREEMENT)
                continue
            fixes_by_member[member_id] = by_tag

        # 6. Pre-existing-in-room guard for room-class association.
        blocked_tag_ph: set[tuple[str, str]] = set()
        for ph in open_phs:
            rid = ph_rooms.get(ph.ph_id, "")
            if not rid:
                continue
            ph_entered = self._ph_room_entered_at.get((ph.ph_id, rid), ph.born_at)
            for by_tag in fixes_by_member.values():
                for tag_id in by_tag:
                    tm = self._tag_motion.get(tag_id)
                    if tm is None:
                        continue
                    tag_entered = tm.room_entered_at.get(rid)
                    if tag_entered is None:
                        continue
                    tag_before_ph = tag_entered < ph_entered and (
                        tm.last_motion_state == MotionState.STILL
                        or (ph_entered - tag_entered).total_seconds() > cfg.room_entry_window_s
                    )
                    moved_since_ph = (
                        tm.last_moving_at is not None and tm.last_moving_at >= ph_entered
                    )
                    if tag_before_ph and not moved_since_ph:
                        blocked_tag_ph.add((tag_id, ph.ph_id))

        # 7. Score pairs and run joint assignment.
        pair_scores, score_outcomes = score_member_ph_pairs(
            fixes_by_member=fixes_by_member,
            open_phs=open_phs,
            ph_history=ph_history,
            ph_rooms=ph_rooms,
            room_polygons=room_polygons,
            sole_occupant_rooms=sole_occupant_rooms,
            blocked_tag_ph=blocked_tag_ph,
            cfg=cfg,
        )
        outcomes.extend(score_outcomes)

        member_ids = sorted(fixes_by_member.keys())
        ph_ids = [ph.ph_id for ph in open_phs]
        accepted, assign_outcomes = assign_members_to_phs(
            member_ids=member_ids,
            ph_ids=ph_ids,
            pair_scores=pair_scores,
            cfg=cfg,
        )
        outcomes.extend(assign_outcomes)

        # 8. Update segments, compute graded attestation, and build WearableEvidence.
        evidence_by_ph: dict[str, WearableEvidence] = {}
        accepted_ph_ids = set(accepted.keys())

        # Close active segments whose PH was not accepted (or switched identity/tag).
        for ph_id in list(self._active_by_ph.keys()):
            if ph_id not in accepted_ph_ids:
                transitions.extend(self.reset_ph(ph_id, now=now, reason="association_lost"))

        for ph_id, (pair, margin_llr) in accepted.items():
            rid = ph_rooms.get(ph_id, "")
            existing = self._active_by_ph.get(ph_id)
            if existing is not None and (
                existing.segment.identity_id != pair.member_id
                or existing.segment.tag_id != pair.tag_id
            ):
                transitions.extend(self.reset_ph(ph_id, now=now, reason="reassigned"))
                existing = None

            if existing is None:
                seg_id = str(uuid.uuid4())
                traj_times = (
                    set(pair.fix_timestamps)
                    if pair.evidence_class == WearableEvidenceClass.TRAJECTORY
                    else set()
                )
                grade = self._evaluate_attestation_grade(
                    pair=pair,
                    margin_llr=margin_llr,
                    continuous_first_fix_at=pair.earliest_fix_at,
                    distinct_traj_fixes=len(traj_times),
                    room_id=rid,
                    ph_id=ph_id,
                    sole_occupant_throughout=sole_occupant_rooms.get(rid, False),
                    had_face_conflict=False,
                )
                seg = WearableAssociationSegment(
                    segment_id=seg_id,
                    ph_id=ph_id,
                    tag_id=pair.tag_id,
                    identity_id=pair.member_id,
                    started_at=pair.earliest_fix_at,
                    ended_at=None,
                    attestation_grade=grade,
                    window_llr=pair.window_llr,
                    margin_llr=margin_llr,
                    matched_fixes=pair.matched_fixes,
                    mean_residual_m=pair.mean_residual_m,
                    swap_suspected=self._is_swap_suspected(pair.tag_id, now),
                )
                self._active_by_ph[ph_id] = _ActiveSegmentState(
                    segment=seg,
                    evidence_class=pair.evidence_class,
                    continuous_first_fix_at=pair.earliest_fix_at,
                    distinct_traj_fix_times=traj_times,
                    room_id=rid,
                    sole_occupant_throughout=sole_occupant_rooms.get(rid, False),
                )
                transitions.append(WearableAssociationTransition(action="open", segment=seg))
            else:
                if pair.evidence_class == WearableEvidenceClass.TRAJECTORY:
                    existing.distinct_traj_fix_times.update(pair.fix_timestamps)
                if rid and rid != existing.room_id:
                    existing.room_id = rid
                    existing.sole_occupant_throughout = sole_occupant_rooms.get(rid, False)
                grade = self._evaluate_attestation_grade(
                    pair=pair,
                    margin_llr=margin_llr,
                    continuous_first_fix_at=existing.continuous_first_fix_at,
                    distinct_traj_fixes=len(existing.distinct_traj_fix_times),
                    room_id=existing.room_id,
                    ph_id=ph_id,
                    sole_occupant_throughout=existing.sole_occupant_throughout,
                    had_face_conflict=existing.had_face_conflict,
                )
                # Never downgrade an unbroken attested segment's tier within the same association.
                grade = _higher_grade(existing.segment.attestation_grade, grade)
                swap_flag = existing.segment.swap_suspected or self._is_swap_suspected(
                    pair.tag_id, now
                )
                updated_seg = replace(
                    existing.segment,
                    attestation_grade=grade,
                    window_llr=pair.window_llr,
                    margin_llr=margin_llr,
                    matched_fixes=pair.matched_fixes,
                    mean_residual_m=pair.mean_residual_m,
                    swap_suspected=swap_flag,
                )
                grade_changed = updated_seg.attestation_grade != existing.segment.attestation_grade
                swap_changed = updated_seg.swap_suspected != existing.segment.swap_suspected
                existing.segment = updated_seg
                existing.evidence_class = pair.evidence_class
                if grade_changed or swap_changed:
                    transitions.append(
                        WearableAssociationTransition(action="update", segment=updated_seg)
                    )

            active_seg = self._active_by_ph[ph_id].segment
            evidence_by_ph[ph_id] = WearableEvidence(
                ph_id=ph_id,
                tag_id=pair.tag_id,
                identity_id=pair.member_id,
                evidence_class=pair.evidence_class,
                attestation_grade=active_seg.attestation_grade,
                window_llr=pair.window_llr,
                margin_llr=margin_llr,
                matched_fixes=pair.matched_fixes,
                mean_residual_m=pair.mean_residual_m,
                newest_fix_at=pair.newest_fix_at,
            )

            # Observe residual histogram for trajectory-class matches.
            if pair.mean_residual_m is not None:
                _metrics.metrics.cts_wearable_assoc_residual_m.labels(
                    room_id=rid or "unknown"
                ).observe(pair.mean_residual_m)

        # Record association outcome counters.
        for outcome in outcomes:
            _metrics.metrics.cts_wearable_associations_total.labels(result=str(outcome)).inc()

        self._last_evidence_by_ph = evidence_by_ph
        return AssociationRoundResult(
            evidence_by_ph=evidence_by_ph,
            transitions=tuple(transitions),
            outcomes=tuple(outcomes),
        )

    def _is_swap_suspected(self, tag_id: str, now: datetime) -> bool:
        dq = self._tag_conflicts.get(tag_id)
        if not dq:
            return False
        cutoff = now - timedelta(seconds=self._cfg.swap_conflict_window_s)
        while dq and dq[0] < cutoff:
            dq.popleft()
        return len(dq) >= self._cfg.swap_conflict_count

    def _evaluate_attestation_grade(
        self,
        *,
        pair: PairScore,
        margin_llr: float,
        continuous_first_fix_at: datetime,
        distinct_traj_fixes: int,
        room_id: str,
        ph_id: str,
        sole_occupant_throughout: bool,
        had_face_conflict: bool,
    ) -> WearableAttestationGrade:
        if had_face_conflict:
            return WearableAttestationGrade.NONE
        cfg = self._cfg
        span_s = (pair.newest_fix_at - continuous_first_fix_at).total_seconds()

        if pair.evidence_class == WearableEvidenceClass.TRAJECTORY:
            if (
                span_s >= cfg.trajectory_attest_min_duration_s
                and distinct_traj_fixes >= cfg.trajectory_attest_min_fixes
                and pair.window_llr >= cfg.attest_min_llr
                and margin_llr >= cfg.min_margin_llr
            ):
                return WearableAttestationGrade.TRAJECTORY
            return WearableAttestationGrade.NONE

        # Room-class attestation requires sustained sole occupancy, observed motion,
        # and co-arrival within room_entry_window_s.
        if span_s < cfg.room_attest_min_duration_s or not sole_occupant_throughout:
            return WearableAttestationGrade.NONE
        tm = self._tag_motion.get(pair.tag_id)
        if tm is None:
            return WearableAttestationGrade.NONE
        if tm.last_moving_at is None or tm.last_moving_at < continuous_first_fix_at:
            return WearableAttestationGrade.NONE
        tag_entered = tm.room_entered_at.get(room_id)
        ph_entered = self._ph_room_entered_at.get((ph_id, room_id))
        if (
            tag_entered is not None
            and ph_entered is not None
            and abs((tag_entered - ph_entered).total_seconds()) > cfg.room_entry_window_s
        ):
            return WearableAttestationGrade.NONE
        return WearableAttestationGrade.ROOM

    def _update_tag_motion(
        self,
        fix: WearableFix,
        room_polygons: Mapping[str, list[tuple[float, float]]],
    ) -> None:
        st = self._tag_motion.get(fix.tag_id)
        if st is None:
            st = _TagMotionTracker()
            self._tag_motion[fix.tag_id] = st
        if st.last_processed_at is not None and fix.measured_at <= st.last_processed_at:
            return
        st.last_processed_at = fix.measured_at
        st.last_motion_state = fix.motion_state

        # Resolve dominant room for this fix.
        dist = room_distribution_for_fix(
            fix, room_polygons, min_mass=self._cfg.room_polygon_min_mass
        )
        dominant_room = max(dist, key=lambda r: dist[r]) if dist else fix.room_id

        room_changed = bool(
            dominant_room and st.last_room_id is not None and dominant_room != st.last_room_id
        )
        if dominant_room:
            if room_changed or dominant_room not in st.room_entered_at:
                st.room_entered_at[dominant_room] = fix.measured_at
            st.last_room_id = dominant_room

        # Explicit movement or room change clears parked state immediately.
        if fix.motion_state == MotionState.MOVING or room_changed:
            st.parked = False
            st.still_since = None
            st.still_mean_x = None
            st.still_mean_y = None
            st.still_pos_count = 0
            st.last_moving_at = fix.measured_at
            return

        if fix.motion_state == MotionState.STILL:
            if st.still_since is None:
                st.still_since = fix.measured_at
            if (fix.measured_at - st.still_since).total_seconds() >= self._cfg.parked_after_s:
                st.parked = True
            return

        # MOTION_STATE_UNSPECIFIED: test whether solved positions stay within their
        # own 95% covariance ellipse of their running mean.
        if fix.x_m is not None and fix.y_m is not None and fix.cov_m2 is not None:
            if st.still_mean_x is None or st.still_mean_y is None:
                st.still_mean_x = fix.x_m
                st.still_mean_y = fix.y_m
                st.still_pos_count = 1
                st.still_since = fix.measured_at
                return
            dx = fix.x_m - st.still_mean_x
            dy = fix.y_m - st.still_mean_y
            s00, s01, _, s11 = fix.cov_m2
            det = s00 * s11 - s01 * s01
            moved = False
            if det > 1e-12:
                d2 = (s11 * dx * dx - 2.0 * s01 * dx * dy + s00 * dy * dy) / det
                moved = d2 > _STILL_COVARIANCE_CHI2_95
            if moved:
                st.parked = False
                st.still_since = fix.measured_at
                st.still_mean_x = fix.x_m
                st.still_mean_y = fix.y_m
                st.still_pos_count = 1
                st.last_moving_at = fix.measured_at
            else:
                n = st.still_pos_count + 1
                st.still_mean_x = st.still_mean_x + (fix.x_m - st.still_mean_x) / n
                st.still_mean_y = st.still_mean_y + (fix.y_m - st.still_mean_y) / n
                st.still_pos_count = n
                if st.still_since is None:
                    st.still_since = fix.measured_at
                if (fix.measured_at - st.still_since).total_seconds() >= self._cfg.parked_after_s:
                    st.parked = True
        else:
            # Room-only unspecified motion: parked after parked_after_s with no room change.
            if st.still_since is None:
                st.still_since = fix.measured_at
            if (fix.measured_at - st.still_since).total_seconds() >= self._cfg.parked_after_s:
                st.parked = True

    def _check_tag_disagreement(
        self,
        member_id: str,
        by_tag: Mapping[str, Sequence[WearableFix]],
        room_polygons: Mapping[str, list[tuple[float, float]]],
        now: datetime,
    ) -> bool:
        if len(by_tag) < 2:
            self._tag_disagreement_since.pop(member_id, None)
            return False

        dominant_by_tag: dict[str, str] = {}
        earliest_by_tag: dict[str, datetime] = {}
        for tag_id, tag_fixes in by_tag.items():
            room_mass: dict[str, float] = {}
            for fix in tag_fixes:
                dist = room_distribution_for_fix(
                    fix, room_polygons, min_mass=self._cfg.room_polygon_min_mass
                )
                for rid, m in dist.items():
                    room_mass[rid] = room_mass.get(rid, 0.0) + m
            if room_mass:
                dominant_by_tag[tag_id] = max(room_mass, key=lambda r: room_mass[r])
                earliest_by_tag[tag_id] = min(f.measured_at for f in tag_fixes)

        distinct_rooms = set(dominant_by_tag.values())
        if len(distinct_rooms) < 2:
            self._tag_disagreement_since.pop(member_id, None)
            return False

        # Track sustained disagreement starting from the earliest overlapping fix time.
        overlap_start = max(earliest_by_tag.values())
        since = self._tag_disagreement_since.setdefault(member_id, overlap_start)
        if overlap_start < since:
            since = overlap_start
            self._tag_disagreement_since[member_id] = since

        if (now - since).total_seconds() >= self._cfg.tag_disagreement_window_s:
            logger.warning(
                "wearable_tag_disagreement",
                tag_ids=sorted(dominant_by_tag.keys()),
                room_count=len(distinct_rooms),
            )
            return True
        return False


def _higher_grade(
    a: WearableAttestationGrade, b: WearableAttestationGrade
) -> WearableAttestationGrade:
    order = {
        WearableAttestationGrade.NONE: 0,
        WearableAttestationGrade.ROOM: 1,
        WearableAttestationGrade.TRAJECTORY: 2,
    }
    return a if order[a] >= order[b] else b
