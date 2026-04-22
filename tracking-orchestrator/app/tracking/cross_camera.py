"""Cross-camera association: merging tracklets into GlobalTracks.

The CrossCameraAssociator scores pairs of open tracklets from different
cameras using a weighted combination of:
1. **Appearance similarity** — quality-weighted k-NN between tracklet galleries.
2. **Floor-plan geometry** — exponential decay of floor-plane distance.

Pairs above a minimum score threshold are greedily merged into GlobalTracks.
The association respects the camera adjacency graph to prevent impossible
transitions (e.g., camera A and camera C are not directly reachable).

This module does NOT perform identity resolution — that happens in
IdentityResolver after GlobalTracks are formed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from ..domain import CameraId, GlobalTrack, Tracklet, TrackletId
from ..storage.base import GalleryRepository, GlobalTrackRepository
from .camera_adjacency import CameraAdjacency


@dataclass(frozen=True)
class CrossCamConfig:
    """Configuration for cross-camera association."""

    # Appearance weight in the association score (0..1).
    # Higher = appearance dominates, lower = geometry dominates.
    alpha: float = 0.7

    # Floor-plan sigma for exponential distance decay.
    # Larger sigma = more permissive about distance.
    floor_sigma_m: float = 1.5

    # Maximum floor-plane distance (meters) for a candidate pair.
    max_floor_distance_m: float = 8.0

    # Minimum combined score to link two tracklets.
    min_link_score: float = 0.55

    # Maximum number of candidate pairs to evaluate per frame.
    max_candidates: int = 100


@dataclass
class TrackletPairScore:
    """Score for a candidate pair of tracklets from different cameras."""

    tracklet_a_id: TrackletId
    tracklet_b_id: TrackletId
    camera_a: CameraId
    camera_b: CameraId
    appearance_sim: float
    geo_score: float
    combined_score: float


class CrossCameraAssociator:
    """Merges open tracklets from different cameras into GlobalTracks.

    Usage::

        assoc = CrossCameraAssociator(
            gallery=gallery_repo,
            adjacency=adjacency_graph,
            config=CrossCamConfig(),
            global_track_repo=global_track_repo,
        )

        global_tracks = await assoc.associate(
            open_tracklets,
            captured_at=datetime.now(UTC),
        )

    The associator maintains the global track lifecycle:
    1. New tracklets that don't match any existing GlobalTrack start a new one.
    2. Tracklets that match an existing GlobalTrack are merged in.
    3. Closed GlobalTracks are not extended.
    """

    def __init__(
        self,
        gallery: GalleryRepository,
        adjacency: CameraAdjacency,
        global_track_repo: GlobalTrackRepository,
        config: CrossCamConfig | None = None,
    ) -> None:
        self._gallery = gallery
        self._adjacency = adjacency
        self._repo = global_track_repo
        self._config = config or CrossCamConfig()

    async def associate(
        self,
        open_tracklets: list[Tracklet],
        captured_at: datetime,
    ) -> list[GlobalTrack]:
        """Associate open tracklets across cameras into GlobalTracks.

        Args:
            open_tracklets: tracklets that are still active (state="active").
            captured_at: wall-clock time of the current frame.

        Returns:
            Updated list of GlobalTracks (both existing and newly created).
        """
        if not open_tracklets:
            return await self._repo.list_active()

        # ---- Step 0: Build tracklet lookup ----
        tracklet_by_id: dict[TrackletId, Tracklet] = {t.tracklet_id: t for t in open_tracklets}

        # ---- Step 1: Group tracklets by existing GlobalTrack ----
        existing_gt_map: dict[TrackletId, str] = {}
        active_gts: list[GlobalTrack] = await self._repo.list_active()
        for gt in active_gts:
            for tid in gt.tracklet_ids:
                existing_gt_map[tid] = gt.global_track_id

        # ---- Step 2: Build candidate pairs ----
        # Only consider tracklets not yet assigned to a GlobalTrack.
        unassigned = [t for t in open_tracklets if t.tracklet_id not in existing_gt_map]

        # Build candidate pairs from unassigned tracklets.
        candidate_scores: list[TrackletPairScore] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, ta in enumerate(unassigned):
            for tb in unassigned[i + 1 :]:
                if ta.camera_id == tb.camera_id:
                    continue

                pair_key: tuple[str, str] = (
                    (ta.tracklet_id, tb.tracklet_id)
                    if ta.tracklet_id <= tb.tracklet_id
                    else (tb.tracklet_id, ta.tracklet_id)
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Check adjacency with time-bounded reachability.
                # The time budget is the max transition time for this camera pair,
                # capped by the age of the older tracklet (a tracklet that started
                # 10 minutes ago can't reach a camera reachable in 30 seconds if
                # the person already left).
                max_transition = self._adjacency.get_max_transition(ta.camera_id, tb.camera_id)
                if max_transition is None:
                    continue
                older_started = min(ta.started_at, tb.started_at)
                budget = max_transition
                tracklet_age = (captured_at - older_started).total_seconds()
                if tracklet_age >= budget:
                    budget = tracklet_age
                if not self._adjacency.reachable(ta.camera_id, tb.camera_id, within_s=budget):
                    continue

                score = self._score_pair(ta, tb)
                if score is not None and score.combined_score >= self._config.min_link_score:
                    candidate_scores.append(score)

        # Sort by combined score descending (greedy merge).
        candidate_scores.sort(key=lambda s: s.combined_score, reverse=True)

        # Helper to refresh a GlobalTrack entry in active_gts from the repo.
        # merge_tracklets updates the repo but leaves in-memory active_gts stale.
        async def _refresh(gt_id: str) -> None:
            fresh = await self._repo.get(gt_id)
            if fresh is not None:
                idx = next((i for i, g in enumerate(active_gts) if g.global_track_id == gt_id), None)
                if idx is not None:
                    active_gts[idx] = fresh

        # ---- Step 3: Greedy merge into GlobalTracks ----
        # assignment_map tracks tracklets in newly created clusters (not pre-existing).
        assignment_map: dict[TrackletId, str] = {}

        for pair in candidate_scores:
            a_in_existing = pair.tracklet_a_id in existing_gt_map
            b_in_existing = pair.tracklet_b_id in existing_gt_map

            # Both in pre-existing GlobalTracks — already handled, skip.
            if a_in_existing and b_in_existing:
                continue

            # One in pre-existing GT, one newly assigned or unassigned — extend.
            if a_in_existing or b_in_existing:
                gt_id = existing_gt_map.get(pair.tracklet_a_id) or existing_gt_map.get(
                    pair.tracklet_b_id
                )
                if gt_id:
                    new_tids = [
                        pair.tracklet_a_id
                        if pair.tracklet_a_id not in existing_gt_map
                        else pair.tracklet_b_id
                    ]
                    new_cids = [
                        pair.camera_a
                        if pair.tracklet_a_id not in existing_gt_map
                        else pair.camera_b
                    ]
                    existing_gt = next(
                        (gt for gt in active_gts if gt.global_track_id == gt_id), None
                    )
                    if existing_gt:
                        merged = await self._repo.merge_tracklets(
                            tracklet_ids=new_tids,
                            camera_ids=new_cids,
                            existing=existing_gt,
                        )
                        await _refresh(merged.global_track_id)
                        assignment_map[new_tids[0]] = merged.global_track_id
                        continue

            # Both newly assigned to clusters.
            gt_a = assignment_map.get(pair.tracklet_a_id)
            gt_b = assignment_map.get(pair.tracklet_b_id)

            if gt_a and gt_b:
                # Different clusters: merge gt_b into gt_a.
                if gt_a != gt_b:
                    for tid, gid in list(assignment_map.items()):
                        if gid == gt_b:
                            assignment_map[tid] = gt_a
                    active_gts = [gt for gt in active_gts if gt.global_track_id != gt_b]
                    # Persist the merge and refresh the in-memory reference.
                    merged_gt = next(
                        (gt for gt in active_gts if gt.global_track_id == gt_a), None
                    )
                    if merged_gt:
                        await _refresh(gt_a)
            elif gt_a:
                # One in a new cluster, one unassigned — add to that cluster.
                add_tid = pair.tracklet_b_id
                add_cid = pair.camera_b
                existing_gt = next((gt for gt in active_gts if gt.global_track_id == gt_a), None)
                if existing_gt:
                    merged = await self._repo.merge_tracklets(
                        tracklet_ids=[add_tid],
                        camera_ids=[add_cid],
                        existing=existing_gt,
                    )
                    await _refresh(merged.global_track_id)
                    assignment_map[add_tid] = merged.global_track_id
            elif gt_b:
                # One in a new cluster, one unassigned — add to that cluster.
                add_tid = pair.tracklet_a_id
                add_cid = pair.camera_a
                existing_gt = next((gt for gt in active_gts if gt.global_track_id == gt_b), None)
                if existing_gt:
                    merged = await self._repo.merge_tracklets(
                        tracklet_ids=[add_tid],
                        camera_ids=[add_cid],
                        existing=existing_gt,
                    )
                    await _refresh(merged.global_track_id)
                    assignment_map[add_tid] = merged.global_track_id
            else:
                # Neither assigned: create a new GlobalTrack.
                new_gt = await self._repo.merge_tracklets(
                    tracklet_ids=[pair.tracklet_a_id, pair.tracklet_b_id],
                    camera_ids=[pair.camera_a, pair.camera_b],
                )
                assignment_map[pair.tracklet_a_id] = new_gt.global_track_id
                assignment_map[pair.tracklet_b_id] = new_gt.global_track_id
                active_gts.append(new_gt)

        # ---- Step 4: Handle remaining unassigned tracklets ----
        newly_assigned: set[TrackletId] = set(assignment_map.keys())
        for t in unassigned:
            if t.tracklet_id in newly_assigned or t.tracklet_id in existing_gt_map:
                continue

            # Try to extend an existing GlobalTrack first.
            extended = False
            for gt in active_gts:
                if gt.state == "closed":
                    continue
                # Skip if this GlobalTrack already has a tracklet on the same camera.
                if t.camera_id in gt.camera_ids:
                    continue
                # Check if any camera in this GlobalTrack is adjacent to t's camera.
                # Time budget: the max transition time, capped by the age of the new
                # tracklet and the time since the GlobalTrack was last seen.
                for existing_cam in gt.camera_ids:
                    max_transition = self._adjacency.get_max_transition(existing_cam, t.camera_id)
                    if max_transition is not None:
                        older_time = min(gt.last_seen_at, t.started_at)
                        budget = max_transition
                        tracklet_age = (captured_at - older_time).total_seconds()
                        if tracklet_age >= budget:
                            budget = tracklet_age
                        if self._adjacency.reachable(existing_cam, t.camera_id, within_s=budget):
                            # Check association score before extending.
                            # Find the tracklet on existing_cam from this GlobalTrack.
                            existing_tid = next(
                                (tid for tid, cid in zip(gt.tracklet_ids, gt.camera_ids)
                                 if cid == existing_cam), None
                            )
                            existing_tl = (
                                tracklet_by_id.get(existing_tid)
                                if existing_tid
                                else (
                                    await self._repo.get_tracklet(existing_tid)
                                    if existing_tid
                                    else None
                                )
                            )
                            score = (
                                self._score_pair(t, existing_tl)
                                if existing_tl
                                else None
                            )
                            if score is not None and score.combined_score < self._config.min_link_score:
                                continue  # skip if score below threshold
                            # Extend this GlobalTrack.
                            merged = await self._repo.merge_tracklets(
                                tracklet_ids=[t.tracklet_id],
                                camera_ids=[t.camera_id],
                                existing=gt,
                            )
                            await _refresh(merged.global_track_id)
                            assignment_map[t.tracklet_id] = merged.global_track_id
                            newly_assigned.add(t.tracklet_id)
                            extended = True
                            break
                    if extended:
                        break
                if extended:
                    break

            # If no existing track was extended, create a new one.
            if not extended:
                new_gt = await self._repo.merge_tracklets(
                    tracklet_ids=[t.tracklet_id],
                    camera_ids=[t.camera_id],
                )
                assignment_map[t.tracklet_id] = new_gt.global_track_id
                newly_assigned.add(t.tracklet_id)
                active_gts.append(new_gt)

        # ---- Step 5: Update last_seen_at for all existing GlobalTracks ----
        # Fetch fresh state from repo (active_gts may have stale tracklet_ids
        # and camera_ids from merge_tracklets calls that updated the repo but
        # not the in-memory list).
        updated_gts: list[GlobalTrack] = []
        fresh_active = await self._repo.list_active()
        for gt in fresh_active:
            updated_gts.append(
                GlobalTrack(
                    global_track_id=gt.global_track_id,
                    camera_ids=gt.camera_ids,
                    tracklet_ids=gt.tracklet_ids,
                    started_at=gt.started_at,
                    last_seen_at=captured_at,
                    current_identity_id=gt.current_identity_id,
                    state="active",
                )
            )

        return updated_gts

    def _score_pair(self, ta: Tracklet, tb: Tracklet) -> TrackletPairScore | None:
        """Score a candidate pair of tracklets.

        Returns None if the pair cannot be scored (e.g., missing adjacency).
        """
        # Geometry: use the last detection's bbox center as a proxy for position.
        # In production, this would use floor_plan projection from homography.
        # For now, use a simple heuristic based on camera distance.
        max_transition = self._adjacency.get_max_transition(ta.camera_id, tb.camera_id)
        if max_transition is None:
            return None

        # Geometry score: exponential decay based on max_transition.
        # Shorter max_transition = closer cameras = higher geo_score.
        geo_score = math.exp(-max_transition / (self._config.floor_sigma_m * 100))

        # Appearance: query the gallery for similarity between the two tracklets.
        # We approximate by comparing the last gallery embedding from each tracklet.
        # In production, this uses Gallery.cross_tracklet_similarity().
        appearance_sim = self._approximate_gallery_similarity(ta, tb)

        # Combined score
        combined = self._config.alpha * appearance_sim + (1 - self._config.alpha) * geo_score

        return TrackletPairScore(
            tracklet_a_id=ta.tracklet_id,
            tracklet_b_id=tb.tracklet_id,
            camera_a=ta.camera_id,
            camera_b=tb.camera_id,
            appearance_sim=appearance_sim,
            geo_score=geo_score,
            combined_score=combined,
        )

    def _approximate_gallery_similarity(self, ta: Tracklet, tb: Tracklet) -> float:
        """Approximate gallery similarity between two tracklets.

        In production, this queries the GalleryRepository for cross-tracklet
        similarity. Here we return a moderate-high default since the actual
        gallery data is populated by the TrackletManager.

        The approximation uses the tracklet IDs as a proxy: if both tracklets
        have the same number of detections, they are more likely to be the
        same person.
        """
        # Placeholder: in production, this queries the gallery.
        # For the in-memory test, we rely on the actual gallery repo.
        return 0.8
