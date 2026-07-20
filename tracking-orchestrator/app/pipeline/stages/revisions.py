"""Revisions stage: publishes identity revisions and runs cross-table rewrites."""

from __future__ import annotations

from ...services.identity_rewriter import IdentityRewriter
from ...services.unknown_backfill import UnknownBackfillService
from ...storage.base import BboxAnnotationRepository
from ...transport.revision_publisher import RevisionPublisher
from ..frame_context import FrameContext
from .base import FrameStage


class RevisionsStage(FrameStage):
    name = "revisions"

    def __init__(
        self,
        revision_publisher: RevisionPublisher | None = None,
        identity_rewriter: IdentityRewriter | None = None,
        bbox_repo: BboxAnnotationRepository | None = None,
        identity_rewrite_on_face_commit: bool = True,
        backfill_service: UnknownBackfillService | None = None,
    ) -> None:
        self._revision_publisher = revision_publisher
        self._identity_rewriter = identity_rewriter
        self._bbox_repo = bbox_repo
        self._identity_rewrite_on_face_commit = identity_rewrite_on_face_commit
        self._backfill_service = backfill_service

    async def run(self, ctx: FrameContext) -> None:
        if ctx.new_revisions and self._revision_publisher:
            await self._revision_publisher.publish_many(ctx.new_revisions)

        if (
            self._identity_rewrite_on_face_commit
            and ctx.new_revisions
            and self._identity_rewriter is not None
        ):
            rewrite_time = ctx.event_time
            # ph_born_at_by_id is populated by WorldTrackingStage; fall back to
            # rewrite_time for PHs not present in this frame (e.g. recently closed).
            ph_born_at = ctx.ph_born_at_by_id
            for rev in ctx.new_revisions:
                if rev.previous_identity_id is None or rev.new_identity_id is None:
                    continue
                applies_from = ph_born_at.get(rev.ph_id, rewrite_time)
                await self._identity_rewriter.rewrite(
                    revision_id=str(rev.revision_id),
                    ph_id=str(rev.ph_id),
                    old_identity_id=str(rev.previous_identity_id),
                    new_identity_id=str(rev.new_identity_id),
                    applies_from=applies_from,
                    applies_to=rewrite_time,
                )

        if self._backfill_service is not None and ctx.outcome_decisions:
            await self._backfill_service.process(
                outcome_decisions=ctx.outcome_decisions,
                ph_born_at_by_id=ctx.ph_born_at_by_id,
                event_time=ctx.event_time,
            )
