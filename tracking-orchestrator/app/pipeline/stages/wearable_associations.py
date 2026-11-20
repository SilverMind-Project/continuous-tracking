"""Wearable-association persistence stage: durably writes segment transitions.

Runs in ``post_world_stages`` ahead of ``PublishStage`` so the UI publish rate
limit can never drop segment open/update/close writes. ``WorldTrackingStage``
attaches the round's transitions to the primary context only, guaranteeing each
transition is persisted once per tracker round across direct, per-camera
batched, and cross-camera batched execution paths.
"""

from __future__ import annotations

from structlog import get_logger

from app.pipeline.frame_context import FrameContext
from app.pipeline.stages.base import FrameStage
from app.storage.wearable_associations import WearableAssociationRepository

logger = get_logger(__name__)


class WearableAssociationPersistStage(FrameStage):
    """Persist wearable-to-PH association segment transitions once per round."""

    name = "wearable_association_persist"

    def __init__(self, repo: WearableAssociationRepository) -> None:
        self._repo = repo

    async def run(self, ctx: FrameContext) -> None:
        if not ctx.wearable_association_transitions:
            return
        for transition in ctx.wearable_association_transitions:
            try:
                await self._repo.save_segment(transition.segment)
            except Exception:
                logger.warning(
                    "wearable_association_persist_failed",
                    segment_id=transition.segment.segment_id,
                    ph_id=transition.segment.ph_id,
                    tag_id=transition.segment.tag_id,
                    action=transition.action,
                    exc_info=True,
                )
