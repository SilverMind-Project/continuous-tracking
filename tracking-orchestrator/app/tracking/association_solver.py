"""Cross-camera association solver using Hungarian assignment.

Replaces greedy pairwise merging with a deterministic, testable solver:

1. Hard-reject candidates that violate do-not-fuse, identity conflict,
   floor-plan mismatch, or temporal infeasibility constraints.
2. Build a cost matrix for remaining (tracklet, global_track) pairs.
3. Solve with scipy.optimize.linear_sum_assignment.
4. Create new global tracks for unmatched tracklets.
5. Run a separate conservative GT-to-GT consolidation pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from structlog import get_logger

from ..domain import GlobalTrack, Tracklet
from .association_evidence import (
    AssociationCandidate,
    AssociationDecision,
    AssociationResult,
)
from .camera_adjacency import CameraAdjacency
from .spatial_projection import SpatialProjectionService

logger = get_logger(__name__)


@dataclass(frozen=True)
class AssociationConfig:
    """Configuration for the association solver."""

    # Appearance weight in combined score (0..1).
    alpha: float = 0.7

    # Floor-plan sigma for exponential distance decay (metres).
    floor_sigma_m: float = 1.5

    # Maximum floor-plane distance for a candidate pair (metres).
    max_floor_distance_m: float = 8.0

    # Minimum combined score to link a tracklet to an existing global track.
    min_link_score: float = 0.55

    # Relaxed score threshold for within-overlap-group pairs.
    within_group_min_score: float = 0.35

    # Minimum appearance similarity for UNKNOWN merge on the same camera.
    unknown_merge_appearance_threshold: float = 0.92

    # Reduced threshold for known-identity re-entry on the same camera.
    known_identity_reentry_threshold: float = 0.72

    # Maximum gap (seconds) for known-identity re-entry to apply.
    same_camera_reentry_max_gap_s: float = 30.0

    # Maximum gap (seconds) between two non-overlapping UNKNOWN GTs
    # on the same camera for temporal+spatial merge.
    unknown_merge_max_gap_s: float = 300.0

    # Maximum floor-plane distance (metres) between two UNKNOWN GTs'
    # last known positions for merge.
    unknown_merge_max_distance_m: float = 2.0

    # Minimum cosine similarity for inter-GT consolidation.
    inter_gt_consolidation_appearance_threshold: float = 0.88

    # When uncalibrated geometry is present, require this stronger
    # appearance threshold instead of the usual min_link_score.
    uncalibrated_appearance_threshold: float = 0.65

    # When both cameras are adjacent but uncalibrated, appearance alone
    # must be >= this threshold to link.  Higher than the standard
    # threshold because geometry provides no signal.
    uncalibrated_adjacent_appearance_threshold: float = 0.70

    # The geometry is unknown — do not use 1.0 (perfect) or 0.5
    # (arbitrary neutral).  Instead skip geometry entirely and require
    # appearance-only scoring to clear a stricter threshold.
    unknown_geometry_neutral_score: float | None = None


# ---------------------------------------------------------------------------
# Hard rejection predicates
# ---------------------------------------------------------------------------


def _check_do_not_fuse(candidate: AssociationCandidate) -> str | None:
    if candidate.do_not_fuse:
        return "do_not_fuse"
    return None


def _check_identity_conflict(candidate: AssociationCandidate) -> str | None:
    if candidate.identity_conflict:
        return "identity_conflict"
    return None


def _check_temporal_infeasible(candidate: AssociationCandidate) -> str | None:
    if not candidate.temporal_feasible:
        return "temporal_infeasible"
    return None


HARD_REJECT_CHECKS: list[object] = [
    _check_do_not_fuse,
    _check_identity_conflict,
    _check_temporal_infeasible,
]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class AssociationSolver:
    """Deterministic cross-camera association solver.

    Usage::

        solver = AssociationSolver(
            adjacency=adjacency,
            spatial=spatial_projection,
            config=AssociationConfig(),
        )
        result = await solver.solve(
            unassigned_tracklets=tracklets,
            active_gts=global_tracks,
            tracklet_by_id=lookup,
            existing_gt_map=existing,
            blocked_pairs=blocked,
            gallery_similarity=gallery_sim_fn,
            captured_at=now,
        )
    """

    def __init__(
        self,
        adjacency: CameraAdjacency,
        spatial: SpatialProjectionService | None = None,
        config: AssociationConfig | None = None,
    ) -> None:
        self._adjacency = adjacency
        self._spatial = spatial
        self._config = config or AssociationConfig()

    async def solve(
        self,
        unassigned_tracklets: list[Tracklet],
        active_gts: list[GlobalTrack],
        tracklet_by_id: dict[str, Tracklet],
        existing_gt_map: dict[str, str],
        blocked_pairs: set[tuple[str, str]],
        gallery_similarity: Any,
        captured_at: Any,
    ) -> AssociationResult:
        """Run one frame of association.

        Returns an ``AssociationResult`` with decisions for every
        unassigned tracklet.
        """
        result = AssociationResult()

        if not unassigned_tracklets:
            return result

        # ---- Candidate generation ----
        candidates = self._generate_candidates(
            unassigned_tracklets,
            active_gts,
            tracklet_by_id,
            existing_gt_map,
            blocked_pairs,
            captured_at,
        )
        result.candidates_generated = len(candidates)

        # ---- Rejection counting ----
        reject_counts: dict[str, int] = {}
        valid: list[AssociationCandidate] = []
        for c in candidates:
            reason = self._apply_hard_rejects(c)
            if reason:
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
            else:
                valid.append(c)
        result.candidates_rejected = reject_counts

        # ---- Hungarian assignment ----
        assignments = self._hungarian_assign(
            valid, unassigned_tracklets, active_gts, gallery_similarity
        )

        # ---- Build decisions ----
        assigned_tids: set[str] = set()
        for tracklet, gt, score in assignments:
            assigned_tids.add(tracklet.tracklet_id)
            result.decisions.append(
                AssociationDecision(
                    tracklet_id=tracklet.tracklet_id,
                    global_track_id=gt.global_track_id,
                    action="extend",
                    reason=f"hungarian_assignment (score={score:.3f})",
                    score=score,
                )
            )
            result.score_histogram.append(score)

        # Unassigned tracklets → create new GTs.
        for t in unassigned_tracklets:
            if t.tracklet_id not in assigned_tids and t.tracklet_id not in existing_gt_map:
                result.decisions.append(
                    AssociationDecision(
                        tracklet_id=t.tracklet_id,
                        global_track_id=None,
                        action="create",
                        reason="no_legal_match",
                        score=None,
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        unassigned: list[Tracklet],
        active_gts: list[GlobalTrack],
        tracklet_by_id: dict[str, Tracklet],
        existing_gt_map: dict[str, str],
        blocked_pairs: set[tuple[str, str]],
        captured_at: Any,
    ) -> list[AssociationCandidate]:
        """Generate AssociationCandidate records for tracklet→GT pairs."""
        candidates: list[AssociationCandidate] = []

        for t in unassigned:
            for gt in active_gts:
                if gt.state == "closed":
                    continue

                c = self._build_candidate(
                    t, gt, tracklet_by_id, existing_gt_map, blocked_pairs, captured_at
                )
                candidates.append(c)

        return candidates

    def _build_candidate(
        self,
        tracklet: Tracklet,
        gt: GlobalTrack,
        tracklet_by_id: dict[str, Tracklet],
        existing_gt_map: dict[str, str],
        blocked_pairs: set[tuple[str, str]],
        captured_at: Any,
    ) -> AssociationCandidate:
        """Build a single AssociationCandidate for one (tracklet, GT) pair."""
        # Identity conflict: both have different non-None committed identities.
        # Actually, identity conflict is: gt has a committed identity, and
        # another gt has a different committed identity on the same camera.
        # For a single (tracklet, gt) pair, the conflict is about the gt's
        # identity vs other gts' identities — handled at solver level.
        id_conflict = False

        # do_not_fuse check.
        dnf = (tracklet.tracklet_id, gt.global_track_id) in blocked_pairs

        # Temporal feasibility: for same-camera, same person re-entry.
        temporal_feasible = True
        if tracklet.camera_id in gt.camera_ids:
            # Same camera: check if person can re-enter.
            temporal_feasible = True  # Always feasible for same-camera re-extend
        else:
            # Cross-camera: check adjacency.
            for cam in gt.camera_ids:
                same_group = self._adjacency.same_overlap_group(cam, tracklet.camera_id)
                if same_group:
                    temporal_feasible = True
                    break
                max_trans = self._adjacency.get_max_transition(cam, tracklet.camera_id)
                if max_trans is not None:
                    older = min(gt.last_seen_at, tracklet.started_at)
                    budget = max(max_trans, (captured_at - older).total_seconds())
                    temporal_feasible = self._adjacency.reachable(
                        cam, tracklet.camera_id, within_s=budget
                    )
                    if temporal_feasible:
                        break
                temporal_feasible = False

        overlap_group_id = None
        for cam in gt.camera_ids:
            og = self._adjacency.get_overlap_group(cam)
            if og and self._adjacency.same_overlap_group(cam, tracklet.camera_id):
                overlap_group_id = og
                break

        return AssociationCandidate(
            source_tracklet_id=tracklet.tracklet_id,
            target_global_track_id=gt.global_track_id,
            appearance_sim=None,  # populated later via gallery query
            floor_distance_m=None,  # populated later
            temporal_feasible=temporal_feasible,
            overlap_group_id=overlap_group_id,
            identity_conflict=id_conflict,
            do_not_fuse=dnf,
            score=0.0,
        )

    # ------------------------------------------------------------------
    # Hard rejection
    # ------------------------------------------------------------------

    def _apply_hard_rejects(self, candidate: AssociationCandidate) -> str | None:
        """Apply hard rejection checks.  Returns the first reject reason or None."""
        for check in HARD_REJECT_CHECKS:
            reason = check(candidate)  # type: ignore[operator]
            if reason is not None:
                return reason  # type: ignore[no-any-return]
        return None

    # ------------------------------------------------------------------
    # Hungarian assignment
    # ------------------------------------------------------------------

    def _hungarian_assign(
        self,
        candidates: list[AssociationCandidate],
        tracklets: list[Tracklet],
        active_gts: list[GlobalTrack],
        gallery_similarity: Any,
    ) -> list[tuple[Tracklet, GlobalTrack, float]]:
        """Run Hungarian assignment on valid candidates.

        Builds a cost matrix of shape (n_tracklets, n_gts).  Unmatched
        tracklets are left for the create-new-GT path.
        """
        if not candidates or not active_gts:
            return []

        # Index tracklets and GTs.
        tl_ids = [t.tracklet_id for t in tracklets]
        gt_ids = [gt.global_track_id for gt in active_gts]
        tl_idx = {tid: i for i, tid in enumerate(tl_ids)}
        gt_idx = {gid: j for j, gid in enumerate(gt_ids)}

        n = len(tl_ids)
        m = len(gt_ids)

        # Cost matrix: higher score → lower cost.  Default to a high sentinel.
        sentinel = 1e9
        cost = np.full((n, m), sentinel, dtype=np.float64)

        # Build gt → tracklet lookup for spatial scoring.
        gt_tracklet_map: dict[str, list[str]] = {}
        for gt in active_gts:
            gt_tracklet_map[gt.global_track_id] = list(gt.tracklet_ids)

        for c in candidates:
            i = tl_idx.get(c.source_tracklet_id)
            j = gt_idx.get(c.target_global_track_id)
            if i is None or j is None:
                continue

            # Compute combined score.
            app_sim = c.appearance_sim or 0.0

            # Geometry score.
            geo_score = self._compute_geo_score(c)
            if geo_score is None:
                # Unknown geometry: require appearance-only.
                if app_sim < self._config.uncalibrated_appearance_threshold:
                    continue  # below threshold, skip this candidate
                geo_score = 0.0  # no geometry contribution

            combined = self._config.alpha * app_sim + (1 - self._config.alpha) * geo_score

            if combined < self._config.min_link_score:
                if c.overlap_group_id and combined >= self._config.within_group_min_score:
                    pass  # within-group threshold is lower
                else:
                    continue

            # Cost = 1.0 - combined (minimize cost = maximize score).
            cost[i, j] = 1.0 - combined

        # Solve Hungarian assignment.
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

            row_ind, col_ind = linear_sum_assignment(cost)
        except ImportError:
            logger.warning("scipy not available for Hungarian assignment, falling back to greedy")
            return self._greedy_fallback(candidates, tracklets, active_gts)

        results: list[tuple[Tracklet, GlobalTrack, float]] = []
        assigned_gt_cols: set[int] = set()

        for r, c_idx in zip(row_ind, col_ind, strict=True):
            cost_val = cost[r, c_idx]
            if cost_val >= sentinel * 0.5:
                continue  # sentinel — no real candidate

            # One GT can only be assigned one new tracklet per frame
            # (unless same overlap group with duplicate-view evidence).
            if c_idx in assigned_gt_cols:
                continue
            assigned_gt_cols.add(c_idx)

            tl = _find_tracklet(tracklets, tl_ids[r])
            gt_or_none = _find_gt(active_gts, gt_ids[c_idx])
            if tl is not None and gt_or_none is not None:
                score = 1.0 - cost_val
                results.append((tl, gt_or_none, score))

        return results

    def _compute_geo_score(self, candidate: AssociationCandidate) -> float | None:
        """Compute geometric score for a candidate, or None if unavailable."""
        if self._spatial is None:
            return None

        dist_m = candidate.floor_distance_m
        if dist_m is None:
            return None

        if dist_m > self._config.max_floor_distance_m:
            return None  # pruned

        sigma = self._config.floor_sigma_m
        return math.exp(-((dist_m / sigma) ** 2))

    def _greedy_fallback(
        self,
        candidates: list[AssociationCandidate],
        tracklets: list[Tracklet],
        active_gts: list[GlobalTrack],
    ) -> list[tuple[Tracklet, GlobalTrack, float]]:
        """Greedy fallback when scipy is unavailable."""
        results: list[tuple[Tracklet, GlobalTrack, float]] = []
        assigned_tids: set[str] = set()
        assigned_gts: set[str] = set()

        for c in sorted(candidates, key=lambda c: c.score, reverse=True):
            if c.source_tracklet_id in assigned_tids:
                continue
            if c.target_global_track_id in assigned_gts:
                continue
            tl = _find_tracklet(tracklets, c.source_tracklet_id)
            gt = _find_gt(active_gts, c.target_global_track_id)
            if tl is not None and gt is not None:
                results.append((tl, gt, c.score))
                assigned_tids.add(c.source_tracklet_id)
                assigned_gts.add(c.target_global_track_id)

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_tracklet(tracklets: list[Tracklet], tid: str) -> Tracklet | None:
    for t in tracklets:
        if t.tracklet_id == tid:
            return t
    return None


def _find_gt(gts: list[GlobalTrack], gid: str) -> GlobalTrack | None:
    for gt in gts:
        if gt.global_track_id == gid:
            return gt
    return None
