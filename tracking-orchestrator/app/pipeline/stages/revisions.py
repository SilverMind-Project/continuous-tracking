"""Revisions stage: publishes identity revisions and runs cross-table rewrites."""

from __future__ import annotations

from ...services.identity_rewriter import IdentityRewriter
from ...storage.base import BboxAnnotationRepository, TrackingRepository
from ...transport.revision_publisher import RevisionPublisher
from ..frame_context import FrameContext
from .base import FrameStage


class RevisionsStage(FrameStage):
    name = "revisions"

    def __init__(
        self,
        revision_publisher: RevisionPublisher | None = None,
        repo: TrackingRepository | None = None,
        identity_rewriter: IdentityRewriter | None = None,
        bbox_repo: BboxAnnotationRepository | None = None,
        identity_rewrite_on_face_commit: bool = True,
    ) -> None:
        self._revision_publisher = revision_publisher
        self._repo = repo
        self._identity_rewriter = identity_rewriter
        self._bbox_repo = bbox_repo
        self._identity_rewrite_on_face_commit = identity_rewrite_on_face_commit

    async def run(self, ctx: FrameContext) -> None:
        if ctx.new_revisions and self._revision_publisher:
            await self._revision_publisher.publish_many(ctx.new_revisions)
            if self._repo:
                for rev in ctx.new_revisions:
                    await self._repo.save_identity_revision(revision=rev)

        if (
            self._identity_rewrite_on_face_commit
            and ctx.new_revisions
            and self._identity_rewriter is not None
        ):
            rewrite_time = ctx.event_time
            gt_start_by_id = {gt.global_track_id: gt.started_at for gt in ctx.active_global_tracks}
            for rev in ctx.new_revisions:
                if rev.previous_identity_id is None or rev.new_identity_id is None:
                    continue
                applies_from = gt_start_by_id.get(rev.ph_id, rewrite_time)
                await self._identity_rewriter.rewrite(
                    revision_id=str(rev.revision_id),
                    global_track_id=str(rev.ph_id),  # N0: ph_id passed as global_track_id
                    old_identity_id=str(rev.previous_identity_id),
                    new_identity_id=str(rev.new_identity_id),
                    applies_from=applies_from,
                    applies_to=rewrite_time,
                )

        # N0: tracklet_ids bbox update removed (tracklets no longer exist)
