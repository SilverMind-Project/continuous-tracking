"""Detect stage: runs YOLO person detection and IoU dedup."""

from __future__ import annotations

import uuid

from structlog import get_logger

from ...inference.detector import PersonDetector
from ...inference.schemas import DetectionBox
from ...observability import metrics as _metrics
from ..frame_context import FrameContext
from .base import FrameStage

logger = get_logger(__name__)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_dedup_detections(
    boxes: list[DetectionBox],
    iou_threshold: float,
) -> list[DetectionBox]:
    if len(boxes) <= 1:
        return list(boxes)

    sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: list[DetectionBox] = []
    suppressed_count = 0
    for box in sorted_boxes:
        b_coords = [box.x1, box.y1, box.x2, box.y2]
        if any(_bbox_iou(b_coords, [k.x1, k.y1, k.x2, k.y2]) > iou_threshold for k in kept):
            suppressed_count += 1
        else:
            kept.append(box)
    if suppressed_count > 0:
        _metrics.metrics.detections_suppressed_total.labels(stage="iou_dedup").inc(suppressed_count)
    return kept


class DetectStage(FrameStage):
    name = "detect"

    def __init__(
        self,
        detector: PersonDetector,
        iou_dedup_threshold: float = 0.55,
    ) -> None:
        self._detector = detector
        self._iou_dedup_threshold = iou_dedup_threshold

    async def run(self, ctx: FrameContext) -> None:
        image = ctx.require_image()
        detections = await self._detector.detect(image)
        logger.debug(
            "detections_raw",
            camera_id=ctx.frame.camera_id,
            frame_index=ctx.frame.frame_index,
            count=len(detections),
            image_shape=f"{image.shape[0]}x{image.shape[1]}",
        )
        if detections and self._iou_dedup_threshold < 1.0:
            detections = _iou_dedup_detections(detections, self._iou_dedup_threshold)
        ctx.raw_detections = detections

        # Assign a stable detection_id to each kept detection so evidence
        # records from later stages can cross-reference by the same ID.
        ctx._detection_ids = {idx: str(uuid.uuid4()) for idx in range(len(detections))}
