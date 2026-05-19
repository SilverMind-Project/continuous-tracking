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
from datetime import UTC, datetime

from structlog import get_logger

from ..domain import CameraId, GlobalTrack, Tracklet, TrackletId
from ..storage.base import GalleryRepository, GlobalTrackRepository
from .camera_adjacency import CameraAdjacency
from .floor_projector import FloorProjector

logger = get_logger(__name__)


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

    # When two UNKNOWN tracklets have appearance similarity above this
    # threshold they are merged into the same GlobalTrack even when on
    # the same camera.  Prevents proliferation of duplicate UNKNOWN
    # GlobalTracks for the same person across tracklet lifecycles.
    unknown_merge_appearance_threshold: float = 0.92

    # Minimum combined score to link two tracklets from cameras in the
    # same overlap group (cameras sharing a physical field of view).
    # Lower than min_link_score because within-group pairs are always
    # geometrically co-located; appearance similarity alone is sufficient.
    within_group_min_score: float = 0.35

    # Minimum **pure cosine similarity** between two GlobalTracks' mean gallery
    # embeddings before the inter-GT consolidation pass merges them.
    # This is appearance-only (no geo component), so must be higher than the
    # combined within_group_min_score.  Conservative to avoid false merges —
    # the consolidation closes the source GT which is hard to undo.
    inter_gt_consolidation_appearance_threshold: float = 0.88


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
        floor_projector: FloorProjector | None = None,
    ) -> None:
        self._gallery = gallery
        self._adjacency = adjacency
        self._repo = global_track_repo
        self._config = config or CrossCamConfig()
        self._floor_projector = floor_projector

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

                # Within-group pairs (same physical space, different camera angles)
                # skip the transit-time budget and use a relaxed score threshold.
                if self._adjacency.same_overlap_group(ta.camera_id, tb.camera_id):
                    score = await self._score_pair(ta, tb)
                    if (
                        score is not None
                        and score.combined_score >= self._config.within_group_min_score
                    ):
                        candidate_scores.append(score)
                    continue

                max_transition = self._adjacency.get_max_transition(ta.camera_id, tb.camera_id)
                if max_transition is not None:
                    # Check time-bounded reachability only when adjacency is configured.
                    # The time budget is the max transition time for this camera pair,
                    # capped by the age of the older tracklet.
                    older_started = min(ta.started_at, tb.started_at)
                    budget = max(max_transition, (captured_at - older_started).total_seconds())
                    if not self._adjacency.reachable(ta.camera_id, tb.camera_id, within_s=budget):
                        continue

                # When adjacency is not configured, fall through: score based on
                # appearance + geometry using the standard min_link_score threshold.
                score = await self._score_pair(ta, tb)
                if score is not None and score.combined_score >= self._config.min_link_score:
                    candidate_scores.append(score)

        # Sort by combined score descending (greedy merge).
        candidate_scores.sort(key=lambda s: s.combined_score, reverse=True)

        # Helper to refresh a GlobalTrack entry in active_gts from the repo.
        # merge_tracklets updates the repo but leaves in-memory active_gts stale.
        async def _refresh(gt_id: str) -> None:
            fresh = await self._repo.get(gt_id)
            if fresh is not None:
                idx = next(
                    (i for i, g in enumerate(active_gts) if g.global_track_id == gt_id),
                    None,
                )
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
                    merged_gt = await self._repo.merge_global_tracks(into_id=gt_a, from_id=gt_b)
                    if merged_gt is not None:
                        for tid, gid in list(assignment_map.items()):
                            if gid == gt_b:
                                assignment_map[tid] = gt_a
                        active_gts = [gt for gt in active_gts if gt.global_track_id != gt_b]
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
                # Same-camera UNKNOWN merge: when this GT and the new tracklet
                # are both UNKNOWN and on the same camera, merge them if their
                # appearance similarity is very high.  This prevents every
                # tracklet lifecycle from creating a new GlobalTrack UUID.
                if t.camera_id in gt.camera_ids:
                    # Same-camera re-entry: try to merge regardless of identity
                    # status.  A committed identity is *more* deserving of a
                    # merge than UNKNOWN — blocking it here generates a new GT
                    # for every re-entry, creating duplicate tracks.
                    existing_tid_same = next(
                        (
                            tid
                            for tid, cid in zip(gt.tracklet_ids, gt.camera_ids, strict=False)
                            if cid == t.camera_id
                        ),
                        None,
                    )
                    if existing_tid_same is not None:
                        existing_tl_same = tracklet_by_id.get(existing_tid_same)
                        if existing_tl_same is not None:
                            app_sim = await self._approximate_gallery_similarity(
                                t, existing_tl_same
                            )
                        else:
                            # Existing tracklet is no longer active — use
                            # gallery-only similarity (no geo component available).
                            app_sim = await self._gallery_similarity_by_ids(
                                t.tracklet_id, existing_tid_same
                            )
                        if app_sim >= self._config.unknown_merge_appearance_threshold:
                            logger.debug(
                                "same_camera_reentry_merge",
                                new_tracklet=t.tracklet_id,
                                existing_gt=gt.global_track_id,
                                identity=gt.current_identity_id,
                                appearance_sim=round(app_sim, 4),
                                threshold=self._config.unknown_merge_appearance_threshold,
                            )
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
                    continue
                # Check if any camera in this GlobalTrack is adjacent to t's camera
                # or in the same overlap group.
                for existing_cam in gt.camera_ids:
                    # Within-group: skip transit check, use relaxed score threshold.
                    in_same_group = self._adjacency.same_overlap_group(existing_cam, t.camera_id)
                    if not in_same_group:
                        max_transition = self._adjacency.get_max_transition(
                            existing_cam, t.camera_id
                        )
                        if max_transition is not None:
                            # Time-bounded reachability check only when configured.
                            older_time = min(gt.last_seen_at, t.started_at)
                            budget = max(
                                max_transition,
                                (captured_at - older_time).total_seconds(),
                            )
                            if not self._adjacency.reachable(
                                existing_cam, t.camera_id, within_s=budget
                            ):
                                continue
                        # No adjacency configured: fall through to appearance scoring.

                    # Find the tracklet on existing_cam from this GlobalTrack.
                    existing_tid = next(
                        (
                            tid
                            for tid, cid in zip(gt.tracklet_ids, gt.camera_ids, strict=False)
                            if cid == existing_cam
                        ),
                        None,
                    )
                    existing_tl = tracklet_by_id.get(existing_tid) if existing_tid else None
                    score_threshold = (
                        self._config.within_group_min_score
                        if in_same_group
                        else self._config.min_link_score
                    )
                    if existing_tl:
                        score = await self._score_pair(t, existing_tl)
                        logger.debug(
                            "remaining_unassigned_score",
                            tracklet=t.tracklet_id,
                            gt_id=gt.global_track_id,
                            score=score.combined_score if score else None,
                            threshold=score_threshold,
                            within_group=in_same_group,
                        )
                        if score is not None and score.combined_score < score_threshold:
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

            # If no existing track was extended, create a new one.
            if not extended:
                new_gt = await self._repo.merge_tracklets(
                    tracklet_ids=[t.tracklet_id],
                    camera_ids=[t.camera_id],
                )
                assignment_map[t.tracklet_id] = new_gt.global_track_id
                newly_assigned.add(t.tracklet_id)
                active_gts.append(new_gt)

        # ---- Step 4b: Consolidate fragmented GTs within overlap groups ----
        # Healing pass: races between concurrent associate() calls can produce
        # two GTs for the same person from cameras in the same overlap group.
        # Merge pairs whose gallery embeddings confirm it's the same person.
        pre_consolidation = await self._repo.list_active()
        await self._consolidate_overlap_group_gts(pre_consolidation)

        # ---- Step 5: Update last_seen_at for all active GlobalTracks ----
        # Fetch fresh state from repo (active_gts may have stale tracklet_ids
        # and camera_ids from merge_tracklets calls that updated the repo but
        # not the in-memory list; also picks up any consolidation merges).
        fresh_active = await self._repo.list_active()

        # Persist the heartbeat so the 5-minute active-window filter never
        # evicts a GlobalTrack whose tracklet is still alive.  Without this,
        # a tracklet longer than 5 minutes will fall off list_active() and be
        # treated as "unassigned", creating a duplicate GlobalTrack.
        fresh_ids = [gt.global_track_id for gt in fresh_active]
        if fresh_ids:
            # Use wall-clock now() rather than captured_at: merge_tracklets()
            # already sets last_seen_at = now() (processing time), which is
            # always >= captured_at (frame time).  Passing captured_at would
            # make the SQL guard `last_seen_at < $2` perpetually false.
            await self._repo.batch_update_last_seen_at(fresh_ids, datetime.now(UTC))

        updated_gts = [
            GlobalTrack(
                global_track_id=gt.global_track_id,
                camera_ids=gt.camera_ids,
                tracklet_ids=gt.tracklet_ids,
                started_at=gt.started_at,
                last_seen_at=captured_at,
                current_identity_id=gt.current_identity_id,
                state="active",
            )
            for gt in fresh_active
        ]

        return updated_gts

    async def _gt_pair_gallery_similarity(
        self, tids_a: list[str], tids_b: list[str]
    ) -> float:
        """Mean cosine similarity between two GlobalTracks via their gallery embeddings.

        Queries all tracklets in each GT, computes a centroid, and returns cosine
        similarity.  Returns 0.0 when either GT has no gallery entries.
        """
        import numpy as np

        try:
            entries_a = await self._gallery.list_gallery_entries_for_tracklets(
                tracklet_ids=set(tids_a), limit=20
            )
        except Exception:
            entries_a = []

        try:
            entries_b = await self._gallery.list_gallery_entries_for_tracklets(
                tracklet_ids=set(tids_b), limit=20
            )
        except Exception:
            entries_b = []

        if not entries_a or not entries_b:
            return 0.0

        emb_a = np.mean([e.embedding for e in entries_a], axis=0)
        emb_b = np.mean([e.embedding for e in entries_b], axis=0)
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b + 1e-9))

    async def _consolidate_overlap_group_gts(
        self, active_gts: list[GlobalTrack]
    ) -> None:
        """Merge fragmented GlobalTracks within the same overlap group.

        Race condition: two concurrent associate() calls can each see an empty
        list_active() and both create fresh GTs for the same person from two
        cameras in the same overlap group.  This healing pass detects those
        cases and merges the pair when gallery embeddings confirm it's the
        same person.

        Hard guard: two GTs with **different** committed current_identity_id
        values are NEVER merged — they represent different enrolled people.
        """
        threshold = self._config.inter_gt_consolidation_appearance_threshold

        # Group active GTs by overlap group.  A GT may span multiple cameras
        # in the same group; we assign it to the first group we find.
        group_to_gts: dict[str, list[GlobalTrack]] = {}
        for gt in active_gts:
            group_id: str | None = None
            for cam_id in gt.camera_ids:
                group_id = self._adjacency.get_overlap_group(cam_id)
                if group_id is not None:
                    break
            if group_id is None:
                continue
            group_to_gts.setdefault(group_id, []).append(gt)

        for group_id, gts in group_to_gts.items():
            if len(gts) < 2:
                continue

            closed_in_group: set[str] = set()

            for i, gt_a in enumerate(gts):
                if gt_a.global_track_id in closed_in_group:
                    continue

                # current_gt_a accumulates merges so subsequent inner-loop
                # iterations use the latest merged state.
                current_gt_a = gt_a

                for gt_b in gts[i + 1 :]:
                    if gt_b.global_track_id in closed_in_group:
                        continue

                    id_a = current_gt_a.current_identity_id
                    id_b = gt_b.current_identity_id
                    if id_a and id_b and id_a != id_b:
                        logger.debug(
                            "overlap_consolidation_identity_conflict",
                            gt_a=current_gt_a.global_track_id,
                            gt_b=gt_b.global_track_id,
                            identity_a=id_a,
                            identity_b=id_b,
                            group_id=group_id,
                        )
                        continue

                    sim = await self._gt_pair_gallery_similarity(
                        current_gt_a.tracklet_ids, gt_b.tracklet_ids
                    )
                    if sim < threshold:
                        continue

                    merged = await self._repo.merge_global_tracks(
                        into_id=current_gt_a.global_track_id,
                        from_id=gt_b.global_track_id,
                    )
                    if merged is not None:
                        logger.info(
                            "overlap_consolidation_merged",
                            into_gt=current_gt_a.global_track_id,
                            from_gt=gt_b.global_track_id,
                            similarity=round(sim, 4),
                            threshold=threshold,
                            group_id=group_id,
                        )
                        current_gt_a = merged
                        closed_in_group.add(gt_b.global_track_id)

    async def _score_pair(self, ta: Tracklet, tb: Tracklet) -> TrackletPairScore | None:
        """Score a candidate pair of tracklets.

        Returns None only when the floor-distance gate prunes the pair.
        When adjacency is not configured, falls back to appearance-only scoring
        (geo_score=1.0) so deployments without explicit calibration still benefit
        from ReID-based cross-camera linking.
        Uses bidirectional adjacency check since cross-camera association
        works regardless of transition direction.
        """
        # Geometry score: exponential decay of floor-plane distance when both
        # tracklets carry a calibrated last_floor_point.  Falls back to 1.0
        # (binary adjacency gate) when projection is unavailable.
        geo_score = self._geo_score(ta, tb)
        if geo_score is None:
            # Distance exceeds max_floor_distance_m — pair is impossible.
            return None

        # Appearance: real gallery similarity between the two tracklets.
        appearance_sim = await self._approximate_gallery_similarity(ta, tb)

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

    def _geo_score(self, ta: Tracklet, tb: Tracklet) -> float | None:
        """Return a geometry score in [0, 1] for a candidate tracklet pair.

        Returns None when the pair exceeds ``max_floor_distance_m`` (pruned).
        Returns 1.0 when floor projection is unavailable (binary adjacency gate).

        When both tracklets carry calibrated ``last_floor_point`` values the
        score follows the phase-3 formula::

            geo_score = exp(-(dist_m / floor_sigma_m) ** 2)
        """
        if self._floor_projector is None:
            return 1.0

        fp_a = ta.last_floor_point
        fp_b = tb.last_floor_point

        # Project from last_bbox if floor point not yet attached.
        if fp_a is None and ta.last_bbox is not None:
            fp_a = self._floor_projector.project(ta.camera_id, ta.last_bbox)
        if fp_b is None and tb.last_bbox is not None:
            fp_b = self._floor_projector.project(tb.camera_id, tb.last_bbox)

        if fp_a is None or fp_b is None or not fp_a.calibrated or not fp_b.calibrated:
            return 1.0

        dist_m = FloorProjector.distance_m(fp_a, fp_b)
        if dist_m > self._config.max_floor_distance_m:
            return None

        return math.exp(-((dist_m / self._config.floor_sigma_m) ** 2))

    async def _gallery_similarity_by_ids(self, tid_a: str, tid_b: str) -> float:
        """Gallery-only cosine similarity between two tracklets identified by ID.

        Used when the Tracklet domain objects are no longer in the active set
        (e.g., a same-camera re-entry where the prior tracklet has closed).
        Returns 0.0 when neither side has gallery entries.
        """
        import numpy as np

        try:
            entries_a = await self._gallery.list_gallery_entries_for_tracklets(
                tracklet_ids={tid_a}, limit=10
            )
        except Exception:
            entries_a = []

        try:
            entries_b = await self._gallery.list_gallery_entries_for_tracklets(
                tracklet_ids={tid_b}, limit=10
            )
        except Exception:
            entries_b = []

        if not entries_a or not entries_b:
            return 0.0

        emb_a = np.mean([e.embedding for e in entries_a], axis=0)
        emb_b = np.mean([e.embedding for e in entries_b], axis=0)
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b + 1e-9))

    async def _approximate_gallery_similarity(self, ta: Tracklet, tb: Tracklet) -> float:
        """Compute real cosine similarity between two tracklets via gallery.

        Queries the GalleryRepository for recent gallery embeddings from each
        tracklet, computes the mean embedding per tracklet, and returns the
        cosine similarity between those means.

        Returns 0.0 when neither tracklet has gallery entries.
        When only one side has entries, returns a conservative 0.5 to allow
        geometry-based cross-camera association while requiring spatial
        proximity to confirm the link.
        """
        import numpy as np

        try:
            entries_a = await self._gallery.list_gallery_entries_for_tracklets(
                tracklet_ids={ta.tracklet_id},
                limit=10,
            )
        except Exception:
            entries_a = []

        try:
            entries_b = await self._gallery.list_gallery_entries_for_tracklets(
                tracklet_ids={tb.tracklet_id},
                limit=10,
            )
        except Exception:
            entries_b = []

        if not entries_a and not entries_b:
            return 0.0

        # When only one side has gallery evidence, return a conservative
        # non-zero score.  Geometry must carry the pair above the link
        # threshold, preventing false positives from appearance alone.
        if not entries_a or not entries_b:
            return 0.5

        emb_a = np.mean([e.embedding for e in entries_a], axis=0)
        emb_b = np.mean([e.embedding for e in entries_b], axis=0)

        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0

        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b + 1e-9))
