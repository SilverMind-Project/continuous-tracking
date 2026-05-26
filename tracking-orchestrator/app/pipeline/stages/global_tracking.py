"""Global tracking stage: cross-camera association, identity resolution, committer."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from structlog import get_logger

from ...domain import IdentityRevision
from ...observability import metrics as _metrics
from ...storage.base import GalleryRepository, GlobalTrackRepository
from ...tracking.floor_projector import FloorProjector
from ...tracking.global_track_service import GlobalTrackService
from ...tracking.identity_committer import IdentityCommitter
from ...tracking.identity_resolver import IdentityResolver
from ...tracking.tracklet_manager import TrackletManager
from ..frame_context import FrameContext
from .base import FrameStage

logger = get_logger(__name__)


class GlobalTrackingStage(FrameStage):
    name = "global_tracking"

    def __init__(
        self,
        tracklet_manager: TrackletManager | None = None,
        cross_camera: GlobalTrackService | None = None,
        identity_resolver: IdentityResolver | None = None,
        identity_committer: IdentityCommitter | None = None,
        global_track_repo: GlobalTrackRepository | None = None,
        gallery_repo: GalleryRepository | None = None,
        floor_projector: FloorProjector | None = None,
        gallery_identity_backfill_delay_s: float = 30.0,
        last_face_id_by_tracklet: dict[str, datetime] | None = None,
    ) -> None:
        self._tracklet_manager = tracklet_manager
        self._cross_camera = cross_camera
        self._identity_resolver = identity_resolver
        self._identity_committer = identity_committer
        self._global_track_repo = global_track_repo
        self._gallery_repo = gallery_repo
        self._floor_projector = floor_projector
        self._gallery_identity_backfill_delay_s = gallery_identity_backfill_delay_s
        # Shared mutable state (written by FaceIdentityStage, pruned here).
        self._last_face_id_by_tracklet: dict[str, datetime] = (
            last_face_id_by_tracklet if last_face_id_by_tracklet is not None else {}
        )

    async def run(self, ctx: FrameContext) -> None:
        ctx.active_tracklets = (
            self._tracklet_manager.get_active_tracklets() if self._tracklet_manager else []
        )
        if self._tracklet_manager is not None:
            for cam_id, held_count in self._tracklet_manager.get_held_count_by_camera().items():
                _metrics.metrics.tracklets_held_below_stability_gate.labels(camera_id=cam_id).set(
                    held_count
                )

        # Prune face ID cooldown entries for closed tracklets.
        active_tracklet_ids = {tl.tracklet_id for tl in ctx.active_tracklets}
        stale_ids = [
            tl_id for tl_id in self._last_face_id_by_tracklet if tl_id not in active_tracklet_ids
        ]
        for tl_id in stale_ids:
            del self._last_face_id_by_tracklet[tl_id]

        if not ctx.active_tracklets:
            return

        assert self._cross_camera is not None
        assert self._identity_resolver is not None

        # Height evidence is deprecated (M2).  estimate_height_mm() projects
        # top and bottom bbox points through the same ground-plane homography,
        # which produces a floor-plane displacement, not true physical height.
        # When re-introduced, rename to bbox_ground_plane_span_mm.

        ctx.active_global_tracks = await self._cross_camera.associate(
            ctx.active_tracklets, captured_at=ctx.event_time
        )
        outcome = await self._identity_resolver.resolve(
            hypotheses=ctx.active_global_tracks,
            new_face_anchors=ctx.face_anchors,
            captured_at=ctx.event_time,
            ph_heights=None,
            face_evidence=ctx._face_evidence if ctx._face_evidence else None,
        )
        ctx.outcome_decisions = outcome.decisions
        ctx.new_revisions = list(outcome.revisions)

        for gt in ctx.active_global_tracks:
            ctx.committed_ids.setdefault(gt.global_track_id, gt.current_identity_id)

        if self._identity_committer is not None:
            self._committer_ingest(ctx)
            await self._committer_face_fastpath(ctx)
            await self._committer_flush(ctx)

        if self._global_track_repo is not None:
            for decision in ctx.outcome_decisions:
                if decision.evidence_backed and decision.identity_id is not None:
                    await self._global_track_repo.set_identity_committed_at(
                        global_track_id=decision.global_track_id,
                        committed_at=ctx.event_time,
                    )

        for gt in ctx.active_global_tracks:
            if gt.global_track_id not in ctx.committed_ids:
                ctx.committed_ids[gt.global_track_id] = gt.current_identity_id

        if self._gallery_repo is not None and self._gallery_identity_backfill_delay_s > 0:
            for gt in ctx.active_global_tracks:
                committed_id = gt.current_identity_id
                committed_at = gt.current_identity_committed_at
                if not committed_id or committed_at is None:
                    continue
                age_s = (ctx.event_time - committed_at).total_seconds()
                if age_s < self._gallery_identity_backfill_delay_s:
                    _metrics.metrics.gallery_backfills_skipped_total.inc()
                    continue
                tracklet_ids = set(gt.tracklet_ids)
                if tracklet_ids:
                    await self._gallery_repo.update_identity_for_tracklets(
                        tracklet_ids=tracklet_ids,
                        identity_id=committed_id,
                    )
        elif self._gallery_repo is not None:
            for gt in ctx.active_global_tracks:
                committed_id = gt.current_identity_id
                if not committed_id:
                    continue
                tracklet_ids = set(gt.tracklet_ids)
                if tracklet_ids:
                    await self._gallery_repo.update_identity_for_tracklets(
                        tracklet_ids=tracklet_ids,
                        identity_id=committed_id,
                    )

    # ------------------------------------------------------------------
    # committer helpers (private to this stage)
    # ------------------------------------------------------------------

    def _committer_ingest(self, ctx: FrameContext) -> None:
        assert self._identity_committer is not None
        for decision in ctx.outcome_decisions:
            _top_id, top_conf = decision.posterior.top_identity()
            gt_id = decision.global_track_id
            current_id = ctx.committed_ids.get(gt_id)
            self._identity_committer.ingest(
                global_track_id=gt_id,
                identity_id=decision.identity_id,
                confidence=top_conf,
                reason=decision.reason,
                current_identity_id=current_id,
            )

    async def _committer_face_fastpath(self, ctx: FrameContext) -> None:
        assert self._identity_committer is not None
        for fa in ctx.face_anchors:
            gt = next(
                (
                    g
                    for g in ctx.active_global_tracks
                    if any(tid == fa.tracklet_id for tid in g.tracklet_ids)
                ),
                None,
            )
            if gt is None:
                continue
            immediate = self._identity_committer.check_high_confidence_face(
                gt.global_track_id, fa.person_id, fa.confidence, first_seen_at=gt.started_at
            )
            if not immediate or self._global_track_repo is None:
                continue
            await self._global_track_repo.assign_identity(
                global_track_id=immediate.global_track_id,
                identity_id=immediate.identity_id,
            )
            if gt.current_identity_id != immediate.identity_id:
                await self._global_track_repo.set_identity_committed_at(
                    global_track_id=immediate.global_track_id,
                    committed_at=ctx.event_time,
                )
            ctx.committed_ids[immediate.global_track_id] = immediate.identity_id
            syn_rev = IdentityRevision(
                revision_id=str(uuid.uuid4()),
                global_track_id=immediate.global_track_id,
                tracklet_ids=list(gt.tracklet_ids),
                candidates=[],
                map_identity_id=immediate.identity_id or "",
                posterior_entropy=0.0,
                previous_identity_id=gt.current_identity_id,
                new_identity_id=immediate.identity_id,
                reason="face_high_confidence_immediate",
                evidence={"face_confidence": fa.confidence},
                revision_time=datetime.now(UTC),
            )
            ctx.new_revisions.append(syn_rev)

    async def _committer_flush(self, ctx: FrameContext) -> None:
        assert self._identity_committer is not None
        flushed = self._identity_committer.flush()
        if self._global_track_repo is None:
            return
        for commit in flushed:
            if commit.identity_id is not None:
                await self._global_track_repo.assign_identity(
                    global_track_id=commit.global_track_id,
                    identity_id=commit.identity_id,
                )
                if commit.previous_identity_id != commit.identity_id:
                    await self._global_track_repo.set_identity_committed_at(
                        global_track_id=commit.global_track_id,
                        committed_at=ctx.event_time,
                    )
            elif commit.previous_identity_id is not None:
                await self._global_track_repo.assign_identity(
                    global_track_id=commit.global_track_id,
                    identity_id=None,
                )
                await self._global_track_repo.clear_identity_committed_at(
                    global_track_id=commit.global_track_id,
                )
                _metrics.metrics.identity_demotions_total.inc()
            ctx.committed_ids[commit.global_track_id] = commit.identity_id
