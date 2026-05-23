"""Privacy stage: applies privacy zone blur and drops filtered detections."""

from __future__ import annotations

from structlog import get_logger

from ...calibration.state import calibration_state
from ...inference.schemas import DetectionBox
from ...observability import metrics as _metrics
from ..frame_context import FrameContext
from ..privacy import PrivacyZoneFilter
from .base import FrameStage

logger = get_logger(__name__)


class PrivacyStage(FrameStage):
    name = "privacy"

    async def run(self, ctx: FrameContext) -> None:
        privacy_filter = PrivacyZoneFilter.from_state(
            calibration_state,
            ctx.frame.camera_id,
            frame_width=ctx.effective_width,
            frame_height=ctx.effective_height,
        )
        if privacy_filter.is_active():
            ctx.image = privacy_filter.apply_blur_mask(ctx.require_image())

        detections = ctx.raw_detections
        if detections and privacy_filter.is_active():
            kept: list[DetectionBox] = []
            dropped_count = 0
            for det in detections:
                foot_x = (det.x1 + det.x2) / 2.0
                foot_y = det.y2
                if privacy_filter.should_drop((foot_x, foot_y)):
                    dropped_count += 1
                    _metrics.metrics.privacy_detections_dropped_total.labels(
                        camera_id=ctx.frame.camera_id,
                    ).inc()
                else:
                    kept.append(det)
            if dropped_count > 0:
                logger.debug(
                    "privacy_detections_filtered",
                    camera_id=ctx.frame.camera_id,
                    dropped=dropped_count,
                    kept=len(kept),
                )
            ctx.raw_detections = kept
