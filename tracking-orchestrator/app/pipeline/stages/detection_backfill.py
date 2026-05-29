"""Detection backfill stage: stamps PH id onto detections via ctx.det_to_ph."""

from __future__ import annotations

from dataclasses import replace

from ..frame_context import FrameContext
from .base import FrameStage


class DetectionBackfillStage(FrameStage):
    """Stamps PH id onto detections so downstream stages can join det → PH."""

    name = "detection_backfill"

    async def run(self, ctx: FrameContext) -> None:
        if not ctx.domain_detections or not ctx.det_to_ph:
            return
        updated = [
            replace(
                det,
                ph_id=ctx.det_to_ph.get(det.detection_id, ""),
            )
            for det in ctx.domain_detections
        ]
        ctx.domain_detections = updated
