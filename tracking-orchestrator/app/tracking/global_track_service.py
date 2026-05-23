"""GlobalTrackService: cross-camera association lifecycle coordinator.

Wraps the ``AssociationSolver`` with repository calls to manage the full
lifecycle of GlobalTracks: creation, extension, merging, and closure.
Preserves the existing ``CrossCameraAssociator`` public API during
migration so no pipeline changes are required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from structlog import get_logger

from ..domain import GlobalTrack, Tracklet, TrackletId
from ..observability import metrics as _metrics
from ..storage.base import DoNotFuseRepository, GalleryRepository, GlobalTrackRepository

if TYPE_CHECKING:
    from ..pipeline.gallery_cache import GalleryCache
from .association_solver import AssociationConfig, AssociationSolver
from .camera_adjacency import CameraAdjacency
from .spatial_projection import SpatialProjectionService

logger = get_logger(__name__)


class GlobalTrackService:
    """Cross-camera association service using Hungarian assignment.

    Replaces the greedy pairwise merging in ``CrossCameraAssociator``
    with a deterministic solver.  Same public API so the pipeline
    can migrate gradually.
    """

    def __init__(
        self,
        gallery: GalleryRepository,
        adjacency: CameraAdjacency,
        global_track_repo: GlobalTrackRepository,
        config: AssociationConfig | None = None,
        spatial_projection: SpatialProjectionService | None = None,
        dnf_repo: DoNotFuseRepository | None = None,
        gallery_cache: GalleryCache | None = None,
    ) -> None:
        self._gallery = gallery
        self._adjacency = adjacency
        self._repo = global_track_repo
        self._config = config or AssociationConfig()
        self._spatial = spatial_projection
        self._dnf_repo = dnf_repo
        self._gallery_cache = gallery_cache
        self._solver = AssociationSolver(
            adjacency=adjacency,
            spatial=spatial_projection,
            config=config,
        )

    async def _gallery_similarity(self, tids_a: set[str], tids_b: set[str]) -> float:
        if self._gallery_cache is not None:
            return await self._gallery_cache.gallery_similarity(tids_a, tids_b)
        return await self._gallery.gallery_similarity(tids_a, tids_b)

    async def associate(
        self,
        open_tracklets: list[Tracklet],
        captured_at: datetime,
    ) -> list[GlobalTrack]:
        """Associate open tracklets across cameras into GlobalTracks."""
        if not open_tracklets:
            return await self._repo.list_active()

        tracklet_by_id: dict[TrackletId, Tracklet] = {t.tracklet_id: t for t in open_tracklets}

        # Pre-load do-not-fuse hints.
        blocked_pairs: set[tuple[str, str]] = set()
        if self._dnf_repo is not None:
            for tid in tracklet_by_id:
                blocked_gts = await self._dnf_repo.get_hints_for_tracklet(tid)
                for blocked_gt_id in blocked_gts:
                    blocked_pairs.add((tid, blocked_gt_id))

        # Load active GTs and build existing assignment map.
        existing_gt_map: dict[TrackletId, str] = {}
        active_gts: list[GlobalTrack] = await self._repo.list_active()
        for gt in active_gts:
            for tid in gt.tracklet_ids:
                existing_gt_map[tid] = gt.global_track_id

        unassigned = [t for t in open_tracklets if t.tracklet_id not in existing_gt_map]

        # ---- Same-camera re-entry ----
        assignment_map: dict[TrackletId, str] = {}
        newly_assigned: set[TrackletId] = set()

        for t in unassigned:
            extended = await self._try_same_camera_reentry(
                t,
                active_gts,
                tracklet_by_id,
                existing_gt_map,
                blocked_pairs,
                captured_at,
                assignment_map,
                newly_assigned,
            )
            if extended:
                continue

            # Try cross-camera extension.
            extended = await self._try_cross_camera_extend(
                t,
                active_gts,
                tracklet_by_id,
                existing_gt_map,
                blocked_pairs,
                captured_at,
                assignment_map,
                newly_assigned,
            )
            if not extended:
                # Create new GT.
                new_gt = await self._repo.merge_tracklets(
                    tracklet_ids=[t.tracklet_id],
                    camera_ids=[t.camera_id],
                )
                assignment_map[t.tracklet_id] = new_gt.global_track_id
                newly_assigned.add(t.tracklet_id)
                active_gts.append(new_gt)

        # ---- GT consolidation ----
        pre_consolidation = await self._repo.list_active()
        await self._consolidate_overlap_group_gts(pre_consolidation)

        # ---- UNKNOWN GT temporal+spatial merge ----
        if self._config.unknown_merge_max_gap_s > 0:
            pre_unknown = await self._repo.list_active()
            await self._merge_unknown_gts(pre_unknown, captured_at, tracklet_by_id)

        # ---- Update last_seen_at ----
        fresh_active = await self._repo.list_active()
        fresh_ids = [gt.global_track_id for gt in fresh_active]
        if fresh_ids:
            await self._repo.batch_update_last_seen_at(fresh_ids, datetime.now(UTC))

        return [
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

    # ------------------------------------------------------------------
    # Same-camera re-entry
    # ------------------------------------------------------------------

    async def _try_same_camera_reentry(
        self,
        tracklet: Tracklet,
        active_gts: list[GlobalTrack],
        tracklet_by_id: dict[str, Tracklet],
        existing_gt_map: dict[str, str],
        blocked_pairs: set[tuple[str, str]],
        captured_at: datetime,
        assignment_map: dict[str, str],
        newly_assigned: set[str],
    ) -> bool:
        """Try to extend an existing GT on the same camera."""
        for gt in active_gts:
            if gt.state == "closed":
                continue
            if tracklet.camera_id not in gt.camera_ids:
                continue

            if (tracklet.tracklet_id, gt.global_track_id) in blocked_pairs:
                continue

            existing_tid = next(
                (
                    tid
                    for tid, cid in zip(gt.tracklet_ids, gt.camera_ids, strict=False)
                    if cid == tracklet.camera_id
                ),
                None,
            )
            existing_tl = tracklet_by_id.get(existing_tid) if existing_tid else None

            if existing_tl:
                app_sim = await self._gallery_similarity(
                    {tracklet.tracklet_id}, {existing_tl.tracklet_id}
                )
            else:
                app_sim = await self._gallery_similarity(
                    {tracklet.tracklet_id}, {cast("str", existing_tid)}
                )

            gap_s = (captured_at - gt.last_seen_at).total_seconds()
            reentry_threshold = (
                self._config.known_identity_reentry_threshold
                if gt.current_identity_id is not None
                and gap_s <= self._config.same_camera_reentry_max_gap_s
                else self._config.unknown_merge_appearance_threshold
            )

            if app_sim >= reentry_threshold:
                merged = await self._repo.merge_tracklets(
                    tracklet_ids=[tracklet.tracklet_id],
                    camera_ids=[tracklet.camera_id],
                    existing=gt,
                )
                await self._refresh_gt(active_gts, merged.global_track_id)
                assignment_map[tracklet.tracklet_id] = merged.global_track_id
                newly_assigned.add(tracklet.tracklet_id)
                return True

        return False

    # ------------------------------------------------------------------
    # Cross-camera extension
    # ------------------------------------------------------------------

    async def _try_cross_camera_extend(
        self,
        tracklet: Tracklet,
        active_gts: list[GlobalTrack],
        tracklet_by_id: dict[str, Tracklet],
        existing_gt_map: dict[str, str],
        blocked_pairs: set[tuple[str, str]],
        captured_at: datetime,
        assignment_map: dict[str, str],
        newly_assigned: set[str],
    ) -> bool:
        """Try to extend an existing GT from a different camera."""
        for gt in active_gts:
            if gt.state == "closed":
                continue

            # Skip same-camera (handled by re-entry).
            if tracklet.camera_id in gt.camera_ids:
                continue

            if (tracklet.tracklet_id, gt.global_track_id) in blocked_pairs:
                continue

            for existing_cam in gt.camera_ids:
                in_same_group = self._adjacency.same_overlap_group(existing_cam, tracklet.camera_id)
                if not in_same_group:
                    max_transition = self._adjacency.get_max_transition(
                        existing_cam, tracklet.camera_id
                    )
                    if max_transition is not None:
                        older_time = min(gt.last_seen_at, tracklet.started_at)
                        budget = max(max_transition, (captured_at - older_time).total_seconds())
                        if not self._adjacency.reachable(
                            existing_cam, tracklet.camera_id, within_s=budget
                        ):
                            continue

                score_threshold = (
                    self._config.within_group_min_score
                    if in_same_group
                    else self._config.min_link_score
                )

                existing_tid = next(
                    (
                        tid
                        for tid, cid in zip(gt.tracklet_ids, gt.camera_ids, strict=False)
                        if cid == existing_cam
                    ),
                    None,
                )
                existing_tl = tracklet_by_id.get(existing_tid) if existing_tid else None

                if existing_tl:
                    app_sim = await self._gallery_similarity(
                        {tracklet.tracklet_id}, {existing_tl.tracklet_id}
                    )
                    geo_score = self._compute_pair_geo(
                        tracklet, existing_tl, in_same_group, score_threshold, app_sim
                    )
                    if geo_score is False:
                        continue
                    score = (
                        self._config.alpha * app_sim + (1 - self._config.alpha) * geo_score
                        if isinstance(geo_score, float)
                        else app_sim
                    )
                    threshold = score_threshold
                    if isinstance(geo_score, float) and score < threshold:
                        continue
                    if (
                        not isinstance(geo_score, float)
                        and app_sim < self._config.uncalibrated_appearance_threshold
                    ):
                        continue

                merged = await self._repo.merge_tracklets(
                    tracklet_ids=[tracklet.tracklet_id],
                    camera_ids=[tracklet.camera_id],
                    existing=gt,
                )
                await self._refresh_gt(active_gts, merged.global_track_id)
                assignment_map[tracklet.tracklet_id] = merged.global_track_id
                newly_assigned.add(tracklet.tracklet_id)
                return True

        return False

    def _compute_pair_geo(
        self,
        ta: Tracklet,
        tb: Tracklet,
        in_same_group: bool,
        score_threshold: float,
        app_sim: float,
    ) -> float | bool:
        """Compute geometry score for a pair.  Returns:
        - float: valid geometry score
        - False: hard-reject (prune pair)
        - True: skip geometry (use appearance only, pair is viable)
        """
        if self._spatial is None:
            return True

        # Use last_floor_point or project from last_bbox.
        fp_a = ta.last_floor_point
        fp_b = tb.last_floor_point
        if fp_a is None and ta.last_bbox is not None:
            fp_a = self._spatial.project_detection(ta.camera_id, ta.last_bbox)
        if fp_b is None and tb.last_bbox is not None:
            fp_b = self._spatial.project_detection(tb.camera_id, tb.last_bbox)

        if fp_a is None or fp_b is None:
            return True  # no geometry, rely on appearance

        if not fp_a.calibrated or not fp_b.calibrated:
            return True  # uncalibrated, rely on appearance

        if not self._spatial.can_compare(ta.camera_id, tb.camera_id):
            return False  # floor plan mismatch → hard reject

        dist_m = self._spatial.distance_m(fp_a, fp_b)
        if dist_m is None:
            return True

        if dist_m > self._config.max_floor_distance_m:
            return False  # too far → hard reject

        import math

        return math.exp(-((dist_m / self._config.floor_sigma_m) ** 2))

    # ------------------------------------------------------------------
    # Consolidation and merge helpers
    # ------------------------------------------------------------------

    async def _refresh_gt(self, active_gts: list[GlobalTrack], gt_id: str) -> None:
        fresh = await self._repo.get(gt_id)
        if fresh is not None:
            idx = next((i for i, g in enumerate(active_gts) if g.global_track_id == gt_id), None)
            if idx is not None:
                active_gts[idx] = fresh

    async def _consolidate_overlap_group_gts(self, active_gts: list[GlobalTrack]) -> None:
        """Merge fragmented GTs within the same overlap group."""
        threshold = self._config.inter_gt_consolidation_appearance_threshold
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

        for _group_id, gts in group_to_gts.items():
            if len(gts) < 2:
                continue
            closed_in_group: set[str] = set()
            for i, gt_a in enumerate(gts):
                if gt_a.global_track_id in closed_in_group:
                    continue
                current_gt_a = gt_a
                for gt_b in gts[i + 1 :]:
                    if gt_b.global_track_id in closed_in_group:
                        continue
                    id_a = current_gt_a.current_identity_id
                    id_b = gt_b.current_identity_id
                    if id_a and id_b and id_a != id_b:
                        continue  # identity conflict → never merge
                    sim = await self._gallery_similarity(
                        set(current_gt_a.tracklet_ids), set(gt_b.tracklet_ids)
                    )
                    if sim < threshold:
                        continue
                    merged = await self._repo.merge_global_tracks(
                        into_id=current_gt_a.global_track_id, from_id=gt_b.global_track_id
                    )
                    if merged is not None:
                        current_gt_a = merged
                        closed_in_group.add(gt_b.global_track_id)

    async def _merge_unknown_gts(
        self,
        active_gts: list[GlobalTrack],
        captured_at: datetime,
        tracklet_by_id: dict[str, Tracklet],
    ) -> None:
        """Merge UNKNOWN GTs on the same camera with non-overlapping intervals."""
        unknown_gts = [gt for gt in active_gts if gt.current_identity_id is None]
        if len(unknown_gts) < 2:
            return
        max_gap = self._config.unknown_merge_max_gap_s
        max_dist = self._config.unknown_merge_max_distance_m
        merged_ids: set[str] = set()

        for i, gt_a in enumerate(unknown_gts):
            if gt_a.global_track_id in merged_ids:
                continue
            a_start, a_end = gt_a.started_at, gt_a.last_seen_at
            for j, gt_b in enumerate(unknown_gts):
                if j <= i:
                    continue
                if gt_b.global_track_id in merged_ids:
                    continue
                if not (set(gt_a.camera_ids) & set(gt_b.camera_ids)):
                    continue
                b_start, b_end = gt_b.started_at, gt_b.last_seen_at
                if a_start <= b_end and b_start <= a_end:
                    continue  # intervals overlap
                gap_s = (
                    (b_start - a_end).total_seconds()
                    if a_end < b_start
                    else (a_start - b_end).total_seconds()
                )
                if gap_s > max_gap:
                    continue
                fp_a = self._last_floor_point(gt_a, tracklet_by_id)
                fp_b = self._last_floor_point(gt_b, tracklet_by_id)
                if fp_a is not None and fp_b is not None:
                    dist_m = SpatialProjectionService.distance_m(fp_a, fp_b)
                    if dist_m is None or dist_m > max_dist:
                        continue
                merged = await self._repo.merge_global_tracks(
                    into_id=gt_a.global_track_id, from_id=gt_b.global_track_id
                )
                if merged is not None:
                    merged_ids.add(gt_b.global_track_id)
                    gt_a = merged
                    _metrics.metrics.unknown_gts_merged_total.inc()

    def _last_floor_point(
        self, gt: GlobalTrack, tracklet_by_id: dict[str, Tracklet]
    ) -> FloorPoint | None:  # type: ignore[name-defined]  # noqa: F821
        for tid in reversed(gt.tracklet_ids):
            tl = tracklet_by_id.get(tid)
            if (
                tl is not None
                and tl.last_floor_point is not None
                and tl.last_floor_point.calibrated
            ):
                return tl.last_floor_point
        return None

    @property
    def solver(self) -> AssociationSolver:
        return self._solver
