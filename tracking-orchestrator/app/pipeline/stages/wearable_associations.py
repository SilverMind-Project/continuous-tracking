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
from app.services.wearable_revocation import WearableTeachingRevocationService
from app.storage.wearable_associations import WearableAssociationRepository

logger = get_logger(__name__)


class WearableAssociationPersistStage(FrameStage):
    """Persist wearable-to-PH association segment transitions once per round.

    When ``auto_on_swap`` is enabled and a segment is first marked ``swap_suspected``,
    triggers :meth:`WearableTeachingRevocationService.revoke_for_swap` after persisting
    the segment so everything the tag taught in the conflict window is demoted and further
    teaching pauses until an operator resumes it.
    """

    name = "wearable_association_persist"

    def __init__(
        self,
        repo: WearableAssociationRepository,
        *,
        revocation_service: WearableTeachingRevocationService | None = None,
        auto_on_swap: bool = False,
        swap_conflict_window_s: float = 600.0,
    ) -> None:
        self._repo = repo
        self._revocation_service = revocation_service
        self._auto_on_swap = auto_on_swap
        self._swap_conflict_window_s = swap_conflict_window_s

    async def run(self, ctx: FrameContext) -> None:
        if not ctx.wearable_association_transitions:
            return
        for transition in ctx.wearable_association_transitions:
            seg = transition.segment
            prev = None
            should_auto_revoke = False
            try:
                if (
                    self._auto_on_swap
                    and self._revocation_service is not None
                    and seg.swap_suspected
                ):
                    prev = await self._repo.get_segment(seg.segment_id)
                    should_auto_revoke = (prev is None or not prev.swap_suspected) and (
                        prev is None or prev.revoked_at is None
                    )
                await self._repo.save_segment(seg)
            except Exception:
                logger.warning(
                    "wearable_association_persist_failed",
                    segment_id=seg.segment_id,
                    ph_id=seg.ph_id,
                    tag_id=seg.tag_id,
                    action=transition.action,
                    exc_info=True,
                )
                continue
            if should_auto_revoke and self._revocation_service is not None:
                try:
                    await self._revocation_service.revoke_for_swap(
                        seg.segment_id,
                        actor="system:swap_guard",
                        window_s=self._swap_conflict_window_s,
                    )
                except Exception:
                    logger.warning(
                        "wearable_swap_auto_revocation_failed",
                        segment_id=seg.segment_id,
                        tag_id=seg.tag_id,
                        exc_info=True,
                    )
